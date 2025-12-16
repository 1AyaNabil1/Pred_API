"""
MLflow Model Serving Manager

This module provides functionality to manage MLflow model serving processes.
It handles starting, stopping, and monitoring model serving instances.

References:
- MLflow Model Serving: https://mlflow.org/docs/latest/models.html#local-rest-server
- MLflow Model URI: https://mlflow.org/docs/latest/concepts.html#artifact-locations

IMPORTANT: MLflow does NOT automatically serve models from the Model Registry.
You must explicitly start a serving process for each model you want to serve.

Usage:
    # Start serving a model in Production stage
    python model_serving.py start --model-name MyModel --stage Production --port 5002
    
    # Start serving a specific model version
    python model_serving.py start --model-name MyModel --version 1 --port 5002
    
    # Stop serving
    python model_serving.py stop --port 5002
    
    # Check status
    python model_serving.py status --port 5002
"""

import os
import sys
import json
import signal
import subprocess
import time
import argparse
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MLflowServingManager:
    """
    Manages MLflow model serving processes.
    
    This class provides methods to start, stop, and monitor MLflow model serving
    instances. It uses the official `mlflow models serve` CLI command.
    
    Reference: https://mlflow.org/docs/latest/models.html#deploy-mlflow-models
    """
    
    def __init__(
        self,
        mlflow_tracking_uri: str = "http://localhost:5001",
        default_port: int = 5002,
        no_conda: bool = True
    ):
        """
        Initialize the serving manager.
        
        Args:
            mlflow_tracking_uri: MLflow tracking server URI
            default_port: Default port for model serving
            no_conda: Whether to use --no-conda flag (recommended for Docker/venv)
        """
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.default_port = default_port
        self.no_conda = no_conda
        self.serving_processes: Dict[int, subprocess.Popen] = {}
        self.pid_file_dir = Path.home() / ".mlflow_serving"
        self.pid_file_dir.mkdir(exist_ok=True)
    
    def build_model_uri(
        self,
        model_name: str,
        stage: Optional[str] = None,
        version: Optional[str] = None
    ) -> str:
        """
        Build an MLflow model URI.
        
        Reference: https://mlflow.org/docs/latest/concepts.html#artifact-locations
        
        Supported formats:
        - models:/<model_name>/<stage> (e.g., models:/MyModel/Production)
        - models:/<model_name>/<version> (e.g., models:/MyModel/1)
        
        Args:
            model_name: Name of the registered model
            stage: Model stage (Production, Staging, Archived, None)
            version: Model version number
            
        Returns:
            MLflow model URI string
            
        Raises:
            ValueError: If neither stage nor version is specified
        """
        if stage:
            return f"models:/{model_name}/{stage}"
        elif version:
            return f"models:/{model_name}/{version}"
        else:
            raise ValueError("Either stage or version must be specified")
    
    def get_pid_file(self, port: int) -> Path:
        """Get the PID file path for a serving port."""
        return self.pid_file_dir / f"serving_{port}.pid"
    
    def is_port_in_use(self, port: int) -> bool:
        """Check if a port is already in use."""
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def wait_for_server(self, port: int, timeout: int = 60) -> bool:
        """
        Wait for the serving server to be ready.
        
        Args:
            port: Port to check
            timeout: Maximum seconds to wait
            
        Returns:
            True if server is ready, False if timeout
        """
        start_time = time.time()
        health_url = f"http://localhost:{port}/health"
        invocations_url = f"http://localhost:{port}/invocations"
        
        while time.time() - start_time < timeout:
            try:
                # Try health endpoint first
                response = requests.get(health_url, timeout=2)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            
            try:
                # Some versions may not have /health, try /invocations with empty body
                response = requests.post(
                    invocations_url,
                    json={"inputs": []},
                    timeout=2
                )
                # Even if it returns an error, the server is up
                if response.status_code in [200, 400, 422]:
                    return True
            except requests.RequestException:
                pass
            
            time.sleep(1)
        
        return False
    
    def start_serving(
        self,
        model_name: str,
        stage: Optional[str] = None,
        version: Optional[str] = None,
        port: Optional[int] = None,
        host: str = "0.0.0.0",
        workers: int = 1,
        env_manager: str = "local"
    ) -> Dict[str, Any]:
        """
        Start MLflow model serving.
        
        This uses the official `mlflow models serve` CLI command.
        Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
        
        Args:
            model_name: Name of the registered model
            stage: Model stage (Production, Staging, etc.)
            version: Model version number
            port: Port to serve on (default: 5002)
            host: Host to bind to (default: 0.0.0.0)
            workers: Number of worker processes
            env_manager: Environment manager (local, conda, virtualenv)
            
        Returns:
            Dict with serving information including PID and URL
            
        Raises:
            RuntimeError: If serving fails to start
        """
        port = port or self.default_port
        
        # Check if port is already in use
        if self.is_port_in_use(port):
            raise RuntimeError(f"Port {port} is already in use")
        
        # Build model URI
        model_uri = self.build_model_uri(model_name, stage, version)
        logger.info(f"Starting model serving for: {model_uri}")
        
        # Build command
        # Reference: mlflow models serve --help
        cmd = [
            "mlflow", "models", "serve",
            "--model-uri", model_uri,
            "--host", host,
            "--port", str(port),
            "--workers", str(workers),
            "--env-manager", env_manager,
        ]
        
        if self.no_conda:
            cmd.append("--no-conda")
        
        # Set environment
        env = os.environ.copy()
        env["MLFLOW_TRACKING_URI"] = self.mlflow_tracking_uri
        
        logger.info(f"Running command: {' '.join(cmd)}")
        
        # Start the process
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                start_new_session=True  # Detach from parent
            )
        except FileNotFoundError:
            raise RuntimeError(
                "MLflow CLI not found. Please install MLflow: pip install mlflow"
            )
        
        # Save PID
        pid_file = self.get_pid_file(port)
        with open(pid_file, 'w') as f:
            json.dump({
                'pid': process.pid,
                'model_uri': model_uri,
                'port': port,
                'host': host,
                'started_at': time.time()
            }, f)
        
        self.serving_processes[port] = process
        
        # Wait for server to be ready
        logger.info(f"Waiting for server to be ready on port {port}...")
        if self.wait_for_server(port, timeout=60):
            logger.info(f"✓ Model serving started successfully!")
            logger.info(f"  Model URI: {model_uri}")
            logger.info(f"  Endpoint: http://{host}:{port}/invocations")
            
            return {
                "status": "running",
                "pid": process.pid,
                "model_uri": model_uri,
                "endpoint": f"http://{host}:{port}/invocations",
                "port": port
            }
        else:
            # Check if process died
            if process.poll() is not None:
                stdout, stderr = process.communicate()
                error_msg = stderr.decode() if stderr else stdout.decode()
                raise RuntimeError(f"Model serving failed to start: {error_msg}")
            else:
                raise RuntimeError(
                    f"Model serving started but not responding. "
                    f"Check logs or try: curl http://localhost:{port}/health"
                )
    
    def stop_serving(self, port: Optional[int] = None) -> bool:
        """
        Stop MLflow model serving on a port.
        
        Args:
            port: Port to stop serving on (default: 5002)
            
        Returns:
            True if stopped successfully, False otherwise
        """
        port = port or self.default_port
        pid_file = self.get_pid_file(port)
        
        if pid_file.exists():
            with open(pid_file) as f:
                info = json.load(f)
            
            pid = info['pid']
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"Sent SIGTERM to PID {pid}")
                
                # Wait for process to terminate
                for _ in range(10):
                    try:
                        os.kill(pid, 0)  # Check if still running
                        time.sleep(0.5)
                    except OSError:
                        break
                else:
                    # Force kill if still running
                    os.kill(pid, signal.SIGKILL)
                    logger.info(f"Sent SIGKILL to PID {pid}")
                
                pid_file.unlink()
                logger.info(f"✓ Stopped model serving on port {port}")
                return True
                
            except OSError as e:
                logger.warning(f"Process {pid} not found: {e}")
                pid_file.unlink()
                return False
        else:
            logger.warning(f"No serving process found for port {port}")
            return False
    
    def get_status(self, port: Optional[int] = None) -> Dict[str, Any]:
        """
        Get the status of model serving on a port.
        
        Args:
            port: Port to check (default: 5002)
            
        Returns:
            Dict with status information
        """
        port = port or self.default_port
        pid_file = self.get_pid_file(port)
        
        result = {
            "port": port,
            "status": "stopped",
            "healthy": False
        }
        
        if pid_file.exists():
            with open(pid_file) as f:
                info = json.load(f)
            
            result.update(info)
            
            # Check if process is still running
            try:
                os.kill(info['pid'], 0)
                result["status"] = "running"
            except OSError:
                result["status"] = "crashed"
                return result
            
            # Check if server is responding
            try:
                response = requests.get(
                    f"http://localhost:{port}/health",
                    timeout=5
                )
                result["healthy"] = response.status_code == 200
            except requests.RequestException:
                # Try invocations endpoint
                try:
                    response = requests.post(
                        f"http://localhost:{port}/invocations",
                        json={"inputs": []},
                        timeout=5
                    )
                    result["healthy"] = True
                except requests.RequestException:
                    result["healthy"] = False
        
        return result
    
    def test_invocation(
        self,
        inputs: Any,
        port: Optional[int] = None,
        input_format: str = "auto"
    ) -> Dict[str, Any]:
        """
        Test model invocation.
        
        Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
        
        Supported input formats:
        - "inputs": {"inputs": [...]} - For tensor-based models
        - "dataframe_split": {"dataframe_split": {"columns": [...], "data": [...]}}
        - "instances": {"instances": [...]} - For record-oriented data
        - "auto": Automatically detect based on input structure
        
        Args:
            inputs: Input data for prediction
            port: Port to invoke (default: 5002)
            input_format: Input format to use
            
        Returns:
            Dict with prediction results and timing
        """
        port = port or self.default_port
        url = f"http://localhost:{port}/invocations"
        
        # Format payload based on input format
        if input_format == "auto":
            if isinstance(inputs, dict) and any(k in inputs for k in ['inputs', 'dataframe_split', 'instances']):
                # Already formatted
                payload = inputs
            elif isinstance(inputs, list) and all(isinstance(i, dict) for i in inputs):
                # List of records
                payload = {"instances": inputs}
            elif isinstance(inputs, dict):
                # Single record
                payload = {"dataframe_records": [inputs]}
            elif isinstance(inputs, list):
                # Array data
                payload = {"inputs": inputs}
            else:
                payload = {"inputs": inputs}
        else:
            payload = {input_format: inputs}
        
        logger.info(f"Invoking model with format: {list(payload.keys())[0]}")
        
        start_time = time.time()
        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                return {
                    "status": "success",
                    "predictions": response.json(),
                    "latency_ms": round(latency_ms, 2)
                }
            else:
                return {
                    "status": "error",
                    "status_code": response.status_code,
                    "error": response.text,
                    "latency_ms": round(latency_ms, 2)
                }
                
        except requests.RequestException as e:
            return {
                "status": "error",
                "error": str(e)
            }


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="MLflow Model Serving Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start serving a Production model
  python model_serving.py start --model-name MyModel --stage Production
  
  # Start serving a specific version
  python model_serving.py start --model-name MyModel --version 1 --port 5003
  
  # Check status
  python model_serving.py status
  
  # Stop serving
  python model_serving.py stop
  
  # Test invocation
  python model_serving.py test '{"feature1": 1.0, "feature2": 2.0}'

Reference: https://mlflow.org/docs/latest/models.html#local-rest-server
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start model serving')
    start_parser.add_argument('--model-name', '-m', required=True, help='Model name')
    start_parser.add_argument('--stage', '-s', help='Model stage (Production, Staging, etc.)')
    start_parser.add_argument('--version', '-v', help='Model version number')
    start_parser.add_argument('--port', '-p', type=int, default=5002, help='Port (default: 5002)')
    start_parser.add_argument('--host', default='0.0.0.0', help='Host (default: 0.0.0.0)')
    start_parser.add_argument('--tracking-uri', default='http://localhost:5001', help='MLflow tracking URI')
    start_parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    
    # Stop command
    stop_parser = subparsers.add_parser('stop', help='Stop model serving')
    stop_parser.add_argument('--port', '-p', type=int, default=5002, help='Port (default: 5002)')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Check serving status')
    status_parser.add_argument('--port', '-p', type=int, default=5002, help='Port (default: 5002)')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test model invocation')
    test_parser.add_argument('inputs', help='JSON input data')
    test_parser.add_argument('--port', '-p', type=int, default=5002, help='Port (default: 5002)')
    test_parser.add_argument('--format', '-f', default='auto', 
                            choices=['auto', 'inputs', 'dataframe_split', 'instances'],
                            help='Input format')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = MLflowServingManager(
        mlflow_tracking_uri=getattr(args, 'tracking_uri', 'http://localhost:5001')
    )
    
    if args.command == 'start':
        if not args.stage and not args.version:
            print("Error: Either --stage or --version must be specified")
            sys.exit(1)
        
        try:
            result = manager.start_serving(
                model_name=args.model_name,
                stage=args.stage,
                version=args.version,
                port=args.port,
                host=args.host,
                workers=args.workers
            )
            print(json.dumps(result, indent=2))
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == 'stop':
        success = manager.stop_serving(port=args.port)
        sys.exit(0 if success else 1)
    
    elif args.command == 'status':
        result = manager.get_status(port=args.port)
        print(json.dumps(result, indent=2))
    
    elif args.command == 'test':
        try:
            inputs = json.loads(args.inputs)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON inputs: {e}")
            sys.exit(1)
        
        result = manager.test_invocation(
            inputs=inputs,
            port=args.port,
            input_format=args.format
        )
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
