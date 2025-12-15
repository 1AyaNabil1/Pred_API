import sys
import os
import joblib

MODEL_PATH = "model.pkl"

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: {MODEL_PATH} not found in project root", file=sys.stderr)
        sys.exit(1)
    
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully: {type(model).__name__}")
        
        if hasattr(model, 'predict'):
            print("Model has predict method - OK")
        else:
            print("Warning: Model does not have predict method", file=sys.stderr)
            sys.exit(1)
        
        joblib.dump(model, MODEL_PATH)
        print(f"Model verified and saved to {MODEL_PATH}")
        
    except Exception as e:
        print(f"Error loading or verifying model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()