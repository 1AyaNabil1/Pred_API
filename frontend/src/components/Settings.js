import React, { useState, useEffect } from 'react';
import './Settings.css';

function Settings({ config, onConfigUpdate }) {
  const [localConfig, setLocalConfig] = useState(config);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    setLocalConfig(config);
  }, [config]);

  const handleChange = (e) => {
    setLocalConfig({
      ...localConfig,
      [e.target.name]: e.target.value
    });
    setSaved(false);
  };

  const handleSave = () => {
    onConfigUpdate(localConfig);
    setSaved(true);
    setTimeout(() => setSaved(false), 3000);
  };

  const handleReset = () => {
    const defaultConfig = {
      apiBaseUrl: 'http://localhost:8000',
      mlflowBaseUrl: 'http://localhost:5000',
      modelServingUrl: '',
      authToken: ''
    };
    setLocalConfig(defaultConfig);
    onConfigUpdate(defaultConfig);
    setSaved(false);
  };

  return (
    <div className="settings">
      <div className="card">
        <h2>Settings</h2>
        <p className="settings-description">
          Configure the connection settings for MLflow and model serving endpoints.
        </p>

        {saved && <div className="success">Settings saved successfully!</div>}

        <div className="settings-form">
          <div className="form-group">
            <label htmlFor="apiBaseUrl">FastAPI Backend URL</label>
            <input
              type="text"
              id="apiBaseUrl"
              name="apiBaseUrl"
              value={localConfig.apiBaseUrl}
              onChange={handleChange}
              placeholder="http://localhost:8000"
            />
            <small>URL of the FastAPI proxy server</small>
          </div>

          <div className="form-group">
            <label htmlFor="mlflowBaseUrl">MLflow Server URL</label>
            <input
              type="text"
              id="mlflowBaseUrl"
              name="mlflowBaseUrl"
              value={localConfig.mlflowBaseUrl}
              onChange={handleChange}
              placeholder="http://localhost:5000"
            />
            <small>URL of your MLflow tracking server</small>
          </div>

          <div className="form-group">
            <label htmlFor="modelServingUrl">Model Serving URL</label>
            <input
              type="text"
              id="modelServingUrl"
              name="modelServingUrl"
              value={localConfig.modelServingUrl}
              onChange={handleChange}
              placeholder="http://localhost:5001"
            />
            <small>Base URL for model serving endpoint (for Try It feature)</small>
          </div>

          <div className="form-group">
            <label htmlFor="authToken">Authentication Token (Optional)</label>
            <input
              type="password"
              id="authToken"
              name="authToken"
              value={localConfig.authToken}
              onChange={handleChange}
              placeholder="Bearer token or API key"
            />
            <small>Optional authentication token for MLflow API</small>
          </div>

          <div className="settings-actions">
            <button className="btn btn-primary" onClick={handleSave}>
              Save Settings
            </button>
            <button className="btn btn-secondary" onClick={handleReset}>
              Reset to Defaults
            </button>
          </div>
        </div>

        <div className="settings-info">
          <h3>Configuration Guide</h3>
          <ul>
            <li>
              <strong>FastAPI Backend URL:</strong> The URL where your FastAPI proxy server is running 
              (default: http://localhost:8000)
            </li>
            <li>
              <strong>MLflow Server URL:</strong> Your MLflow tracking server URL. Set the 
              MLFLOW_BASE_URL environment variable in the backend to match this.
            </li>
            <li>
              <strong>Model Serving URL:</strong> The base URL for your model serving endpoint. 
              This is used for the "Try It" feature to invoke models.
            </li>
            <li>
              <strong>Auth Token:</strong> If your MLflow server requires authentication, 
              provide the bearer token here.
            </li>
          </ul>
        </div>
      </div>
    </div>
  );
}

export default Settings;
