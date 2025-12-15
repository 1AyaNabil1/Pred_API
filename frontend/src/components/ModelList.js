import React, { useState, useEffect } from 'react';
import ConfirmModal from './ConfirmModal';
import './ModelList.css';

function ModelList({ config, onModelSelect, refreshTrigger }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [deleteModel, setDeleteModel] = useState(null);

  useEffect(() => {
    fetchModels();
  }, [config, refreshTrigger]);

  const fetchModels = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${config.apiBaseUrl}/mlflow/registered-models/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
        },
        body: JSON.stringify({
          max_results: 100
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch models');
      }

      const data = await response.json();
      setModels(data.registered_models || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (modelName) => {
    try {
      const response = await fetch(`${config.apiBaseUrl}/mlflow/registered-models/delete`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json',
          ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
        },
        body: JSON.stringify({ name: modelName })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to delete model');
      }

      fetchModels();
      setDeleteModel(null);
    } catch (err) {
      setError(err.message);
      setDeleteModel(null);
    }
  };

  const getLatestVersion = (model) => {
    if (!model.latest_versions || model.latest_versions.length === 0) {
      return null;
    }
    return model.latest_versions.reduce((latest, current) => {
      return parseInt(current.version) > parseInt(latest.version) ? current : latest;
    });
  };

  const getStageBadge = (stage) => {
    const stageMap = {
      'Production': 'badge-production',
      'Staging': 'badge-staging',
      'Archived': 'badge-archived',
      'None': 'badge-none'
    };
    return stageMap[stage] || 'badge-none';
  };

  if (loading) {
    return <div className="loading">Loading models...</div>;
  }

  return (
    <div className="model-list">
      <div className="card">
        <h2>Registered Models</h2>
        {error && <div className="error">{error}</div>}
        
        {models.length === 0 ? (
          <p style={{ color: 'var(--text-secondary)' }}>No registered models found.</p>
        ) : (
          <div className="models-grid">
            {models.map((model) => {
              const latestVersion = getLatestVersion(model);
              return (
                <div key={model.name} className="model-card">
                  <div className="model-header">
                    <h3>{model.name}</h3>
                    {latestVersion && (
                      <span className={`badge ${getStageBadge(latestVersion.current_stage)}`}>
                        {latestVersion.current_stage}
                      </span>
                    )}
                  </div>
                  
                  {model.description && (
                    <p className="model-description">{model.description}</p>
                  )}
                  
                  {latestVersion && (
                    <div className="model-meta">
                      <span>Latest Version: {latestVersion.version}</span>
                    </div>
                  )}
                  
                  {model.tags && model.tags.length > 0 && (
                    <div className="model-tags">
                      {model.tags.map((tag, idx) => (
                        <span key={idx} className="tag">{tag.key}: {tag.value}</span>
                      ))}
                    </div>
                  )}
                  
                  <div className="model-actions">
                    <button 
                      className="btn btn-primary"
                      onClick={() => onModelSelect(model)}
                    >
                      View Details
                    </button>
                    <button 
                      className="btn btn-danger"
                      onClick={() => setDeleteModel(model)}
                    >
                      Delete
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {deleteModel && (
        <ConfirmModal
          title="Delete Model"
          message={`Are you sure you want to delete "${deleteModel.name}"? This action cannot be undone.`}
          onConfirm={() => handleDelete(deleteModel.name)}
          onCancel={() => setDeleteModel(null)}
        />
      )}
    </div>
  );
}

export default ModelList;
