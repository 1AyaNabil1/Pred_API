import React, { useState } from 'react';
import './UploadForm.css';

function UploadForm({ config, onSuccess }) {
  const [uploadMode, setUploadMode] = useState('existing'); // 'existing' or 'file'
  const [formData, setFormData] = useState({
    name: '',
    source: '',
    run_id: '',
    description: '',
    tags: []
  });
  const [fileFormData, setFileFormData] = useState({
    name: '',
    description: '',
    input_example: '',
    file: null
  });
  const [tagInput, setTagInput] = useState({ key: '', value: '' });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

  const addTag = () => {
    if (tagInput.key && tagInput.value) {
      setFormData({
        ...formData,
        tags: [...formData.tags, { key: tagInput.key, value: tagInput.value }]
      });
      setTagInput({ key: '', value: '' });
    }
  };

  const removeTag = (index) => {
    setFormData({
      ...formData,
      tags: formData.tags.filter((_, i) => i !== index)
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!formData.name || !formData.source) {
      setError('Model name and source path are required');
      return;
    }

    // Validate source format
    const validSourcePattern = /^(runs:\/|models:\/|mlflow-artifacts:\/|s3:\/|gs:\/|wasbs:\/|dbfs:\/)/;
    if (!validSourcePattern.test(formData.source)) {
      setError(
        'Invalid source format. Source must be a valid MLflow URI.\n' +
        'Examples:\n' +
        '• runs:/<run-id>/model\n' +
        '• models:/<model-name>/<version>\n' +
        '• mlflow-artifacts:/path/to/model'
      );
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const payload = {
        name: formData.name,
        source: formData.source
      };

      if (formData.run_id) payload.run_id = formData.run_id;
      if (formData.description) payload.description = formData.description;
      if (formData.tags.length > 0) payload.tags = formData.tags;

      const response = await fetch(`${config.apiBaseUrl}/mlflow/model-versions/create`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to register model version');
      }

      const data = await response.json();
      setSuccess(`Successfully registered ${formData.name} version ${data.model_version?.version || 'new'}`);
      
      // Reset form
      setFormData({
        name: '',
        source: '',
        run_id: '',
        description: '',
        tags: []
      });

      // Notify parent after a short delay
      setTimeout(() => {
        if (onSuccess) onSuccess();
      }, 2000);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (!file.name.endsWith('.pkl')) {
        setError('Please select a .pkl file');
        return;
      }
      setFileFormData({
        ...fileFormData,
        file: file
      });
      setError(null);
    }
  };

  const handleFileFormChange = (e) => {
    setFileFormData({
      ...fileFormData,
      [e.target.name]: e.target.value
    });
  };

  const handleFileUpload = async (e) => {
    e.preventDefault();
    
    if (!fileFormData.name || !fileFormData.file) {
      setError('Model name and file are required');
      return;
    }

    setLoading(true);
    setError(null);
    setSuccess(null);

    try {
      const formDataToSend = new FormData();
      formDataToSend.append('file', fileFormData.file);
      formDataToSend.append('model_name', fileFormData.name);
      if (fileFormData.description) {
        formDataToSend.append('description', fileFormData.description);
      }
      if (fileFormData.input_example) {
        formDataToSend.append('input_example', fileFormData.input_example);
      }

      const response = await fetch(`${config.apiBaseUrl}/mlflow/upload-pkl-model`, {
        method: 'POST',
        headers: {
          ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
        },
        body: formDataToSend
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to upload model');
      }

      const data = await response.json();
      setSuccess(`Successfully uploaded and registered ${fileFormData.name}`);
      
      // Reset form
      setFileFormData({
        name: '',
        description: '',
        input_example: '',
        file: null
      });
      
      // Reset file input
      const fileInput = document.getElementById('pkl-file');
      if (fileInput) fileInput.value = '';

      // Notify parent after a short delay
      setTimeout(() => {
        if (onSuccess) onSuccess();
      }, 2000);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="upload-form">
      <div className="card">
        <h2>Upload Model</h2>
        
        <div className="upload-mode-tabs">
          <button
            className={`tab-button ${uploadMode === 'file' ? 'active' : ''}`}
            onClick={() => {
              setUploadMode('file');
              setError(null);
              setSuccess(null);
            }}
          >
            Upload .pkl File
          </button>
          <button
            className={`tab-button ${uploadMode === 'existing' ? 'active' : ''}`}
            onClick={() => {
              setUploadMode('existing');
              setError(null);
              setSuccess(null);
            }}
          >
            Register Existing Model
          </button>
        </div>

        {error && <div className="error" style={{ whiteSpace: 'pre-line' }}>{error}</div>}
        {success && <div className="success">{success}</div>}

        {uploadMode === 'file' ? (
          <form onSubmit={handleFileUpload}>
            <p className="form-description">
              Upload a .pkl file containing a trained model and register it in MLflow.
            </p>

            <div className="form-group">
              <label htmlFor="file-name">Model Name *</label>
              <input
                type="text"
                id="file-name"
                name="name"
                value={fileFormData.name}
                onChange={handleFileFormChange}
                placeholder="my-model"
                required
              />
              <small>Name for the registered model</small>
            </div>

            <div className="form-group">
              <label htmlFor="pkl-file">Model File (.pkl) *</label>
              <input
                type="file"
                id="pkl-file"
                accept=".pkl"
                onChange={handleFileChange}
                required
              />
              {fileFormData.file && (
                <small style={{ color: 'var(--success)' }}>
                  {fileFormData.file.name} selected
                </small>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="file-description">Description (Optional)</label>
              <textarea
                id="file-description"
                name="description"
                value={fileFormData.description}
                onChange={handleFileFormChange}
                placeholder="Description of this model..."
                rows="3"
              />
            </div>

            <div className="form-group">
              <label htmlFor="file-input-example">Input Example (Optional)</label>
              <textarea
                id="file-input-example"
                name="input_example"
                value={fileFormData.input_example}
                onChange={handleFileFormChange}
                placeholder='{"feature1": 1.0, "feature2": 2.0}'
                rows="3"
              />
              <small>JSON object representing a sample input for the model</small>
            </div>

            <button
              type="submit"
              className="btn btn-primary btn-full"
              disabled={loading}
            >
              {loading ? 'Uploading...' : 'Upload & Register Model'}
            </button>
          </form>
        ) : (

          <form onSubmit={handleSubmit}>
            <p className="form-description">
              Register a new version of a model. The artifact must already be logged to MLflow.
            </p>

            <div className="form-group">
              <label htmlFor="name">Model Name *</label>
            <input
              type="text"
              id="name"
              name="name"
              value={formData.name}
              onChange={handleChange}
              placeholder="my-model"
              required
            />
            <small>Name of the registered model (will be created if doesn't exist)</small>
          </div>

          <div className="form-group">
            <label htmlFor="source">Source Path *</label>
            <input
              type="text"
              id="source"
              name="source"
              value={formData.source}
              onChange={handleChange}
              placeholder="runs:/<run-id>/model or models:/<model-name>/<version>"
              required
            />
            <small>
              URI of the model artifacts. Examples:<br/>
              • <code>runs:/&lt;run-id&gt;/model</code> - Model from a specific run<br/>
              • <code>models:/&lt;model-name&gt;/&lt;version&gt;</code> - Copy from existing model<br/>
              • <code>mlflow-artifacts:/path/to/model</code> - Direct artifact path
            </small>
          </div>

          <div className="form-group">
            <label htmlFor="run_id">Run ID (Optional)</label>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
              <input
                type="text"
                id="run_id"
                name="run_id"
                value={formData.run_id}
                onChange={handleChange}
                placeholder="abc123def456..."
                style={{ flex: 1 }}
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => {
                  if (formData.run_id) {
                    setFormData({
                      ...formData,
                      source: `runs:/${formData.run_id}/model`
                    });
                  }
                }}
                disabled={!formData.run_id}
                title="Auto-fill source from Run ID"
              >
                → Source
              </button>
            </div>
            <small>MLflow run ID that generated this model. Click "→ Source" to auto-fill the source path.</small>
          </div>

          <div className="form-group">
            <label htmlFor="description">Description (Optional)</label>
            <textarea
              id="description"
              name="description"
              value={formData.description}
              onChange={handleChange}
              placeholder="Description of this model version..."
              rows="4"
            />
          </div>

          <div className="form-group">
            <label>Tags (Optional)</label>
            <div className="tags-input">
              <input
                type="text"
                placeholder="Key"
                value={tagInput.key}
                onChange={(e) => setTagInput({ ...tagInput, key: e.target.value })}
              />
              <input
                type="text"
                placeholder="Value"
                value={tagInput.value}
                onChange={(e) => setTagInput({ ...tagInput, value: e.target.value })}
              />
              <button
                type="button"
                className="btn btn-secondary"
                onClick={addTag}
              >
                Add Tag
              </button>
            </div>

            {formData.tags.length > 0 && (
              <div className="tags-list">
                {formData.tags.map((tag, idx) => (
                  <div key={idx} className="tag-item">
                    <span>{tag.key}: {tag.value}</span>
                    <button
                      type="button"
                      onClick={() => removeTag(idx)}
                      className="tag-remove"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <button
            type="submit"
            className="btn btn-primary btn-full"
            disabled={loading}
          >
              {loading ? 'Registering...' : 'Register Version'}
            </button>
          </form>
        )}
      </div>
    </div>
  );
}

export default UploadForm;
