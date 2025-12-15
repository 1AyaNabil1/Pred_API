import React, { useState, useEffect } from 'react';
import SchemaEditor from './SchemaEditor';
import TryItPanel from './TryItPanel';
import ConfirmModal from './ConfirmModal';
import './ModelDetail.css';

function ModelDetail({ model, config, onBack }) {
  const [versions, setVersions] = useState([]);
  const [selectedVersion, setSelectedVersion] = useState(null);
  const [versionMetadata, setVersionMetadata] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);
  const [showUpdateModal, setShowUpdateModal] = useState(false);
  const [updateDescription, setUpdateDescription] = useState('');
  const [transitionModal, setTransitionModal] = useState(null);

  useEffect(() => {
    if (model.latest_versions) {
      setVersions(model.latest_versions);
      if (model.latest_versions.length > 0) {
        const latest = model.latest_versions.reduce((prev, current) => 
          parseInt(current.version) > parseInt(prev.version) ? current : prev
        );
        setSelectedVersion(latest);
      }
    }
  }, [model]);

  useEffect(() => {
    if (selectedVersion) {
      fetchVersionMetadata();
    }
  }, [selectedVersion]);

  const fetchVersionMetadata = async () => {
    if (!selectedVersion) return;

    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `${config.apiBaseUrl}/mlflow/model-versions/get?name=${encodeURIComponent(model.name)}&version=${selectedVersion.version}`,
        {
          headers: {
            ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
          }
        }
      );

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch version metadata');
      }

      const data = await response.json();
      setVersionMetadata(data.model_version);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleStageTransition = async (newStage) => {
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(`${config.apiBaseUrl}/mlflow/model-versions/transition-stage`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
        },
        body: JSON.stringify({
          name: model.name,
          version: selectedVersion.version,
          stage: newStage,
          archive_existing_versions: false
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to transition stage');
      }

      setSuccess(`Successfully transitioned to ${newStage}`);
      fetchVersionMetadata();
      setTransitionModal(null);
    } catch (err) {
      setError(err.message);
      setTransitionModal(null);
    }
  };

  const handleUpdateDescription = async () => {
    setError(null);
    setSuccess(null);

    try {
      const response = await fetch(`${config.apiBaseUrl}/mlflow/registered-models/update`, {
        method: 'PATCH',
        headers: {
          'Content-Type': 'application/json',
          ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
        },
        body: JSON.stringify({
          name: model.name,
          description: updateDescription
        })
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to update description');
      }

      setSuccess('Successfully updated description');
      setShowUpdateModal(false);
      model.description = updateDescription;
    } catch (err) {
      setError(err.message);
    }
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

  return (
    <div className="model-detail">
      <div className="detail-header">
        <button className="btn btn-secondary" onClick={onBack}>
          ← Back to Models
        </button>
        <h2>{model.name}</h2>
      </div>

      {error && <div className="error">{error}</div>}
      {success && <div className="success">{success}</div>}

      <div className="detail-layout">
        <div className="detail-main">
          <div className="card">
            <h3>Model Information</h3>
            <div className="info-grid">
              <div className="info-item">
                <label>Name:</label>
                <span>{model.name}</span>
              </div>
              <div className="info-item">
                <label>Description:</label>
                <span>{model.description || 'No description'}</span>
                <button 
                  className="btn btn-secondary btn-sm"
                  onClick={() => {
                    setUpdateDescription(model.description || '');
                    setShowUpdateModal(true);
                  }}
                >
                  Edit
                </button>
              </div>
            </div>
          </div>

          <div className="card">
            <h3>Versions</h3>
            <div className="versions-list">
              {versions.map((version) => (
                <div
                  key={version.version}
                  className={`version-item ${selectedVersion?.version === version.version ? 'active' : ''}`}
                  onClick={() => setSelectedVersion(version)}
                >
                  <div className="version-info">
                    <strong>Version {version.version}</strong>
                    <span className={`badge ${getStageBadge(version.current_stage)}`}>
                      {version.current_stage}
                    </span>
                  </div>
                  {version.description && (
                    <p className="version-desc">{version.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>

          {selectedVersion && (
            <div className="card">
              <h3>Version {selectedVersion.version} Details</h3>
              {loading ? (
                <div className="loading">Loading metadata...</div>
              ) : versionMetadata ? (
                <>
                  <div className="info-grid">
                    <div className="info-item">
                      <label>Current Stage:</label>
                      <span className={`badge ${getStageBadge(versionMetadata.current_stage)}`}>
                        {versionMetadata.current_stage}
                      </span>
                    </div>
                    <div className="info-item">
                      <label>Source:</label>
                      <span className="source-path">{versionMetadata.source}</span>
                    </div>
                    {versionMetadata.run_id && (
                      <div className="info-item">
                        <label>Run ID:</label>
                        <span>{versionMetadata.run_id}</span>
                      </div>
                    )}
                  </div>

                  <div className="stage-actions">
                    <h4>Model Lifecycle Actions:</h4>
                    <div className="stage-buttons">
                      <button 
                        className="btn btn-success"
                        onClick={() => setTransitionModal('Production')}
                        disabled={versionMetadata.current_stage === 'Production'}
                        title="Activate model for production use"
                      >
                        {versionMetadata.current_stage === 'Production' ? 'Active (Production)' : 'Activate (Production)'}
                      </button>
                      <button 
                        className="btn btn-warning"
                        onClick={() => setTransitionModal('Staging')}
                        disabled={versionMetadata.current_stage === 'Staging'}
                        title="Move to staging for testing"
                      >
                        {versionMetadata.current_stage === 'Staging' ? 'In Staging' : 'Move to Staging'}
                      </button>
                      <button 
                        className="btn btn-secondary"
                        onClick={() => setTransitionModal('Archived')}
                        disabled={versionMetadata.current_stage === 'Archived'}
                        title="Deactivate and archive model"
                      >
                        {versionMetadata.current_stage === 'Archived' ? 'Archived' : 'Deactivate (Archive)'}
                      </button>
                      <button 
                        className="btn btn-outline"
                        onClick={() => setTransitionModal('None')}
                        disabled={versionMetadata.current_stage === 'None'}
                        title="Remove from all stages"
                      >
                        {versionMetadata.current_stage === 'None' ? 'No Stage' : 'Remove Stage'}
                      </button>
                    </div>
                    <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginTop: '1rem' }}>
                      <strong>Current Status:</strong> {
                        versionMetadata.current_stage === 'Production' ? 'Active in Production' :
                        versionMetadata.current_stage === 'Staging' ? 'Testing in Staging' :
                        versionMetadata.current_stage === 'Archived' ? 'Deactivated (Archived)' :
                        'Not Deployed'
                      }
                    </p>
                  </div>
                </>
              ) : null}
            </div>
          )}
        </div>

        <div className="detail-sidebar">
          {selectedVersion && versionMetadata && (
            <>
              <SchemaEditor
                modelName={model.name}
                version={selectedVersion.version}
                metadata={versionMetadata}
                config={config}
              />
              <TryItPanel
                modelName={model.name}
                version={selectedVersion.version}
                metadata={versionMetadata}
                config={config}
              />
            </>
          )}
        </div>
      </div>

      {showUpdateModal && (
        <div className="modal-overlay">
          <div className="modal-content">
            <div className="modal-header">
              <h3>Update Description</h3>
            </div>
            <div className="form-group">
              <label>Description:</label>
              <textarea
                value={updateDescription}
                onChange={(e) => setUpdateDescription(e.target.value)}
                rows="4"
              />
            </div>
            <div className="modal-actions">
              <button className="btn btn-secondary" onClick={() => setShowUpdateModal(false)}>
                Cancel
              </button>
              <button className="btn btn-primary" onClick={handleUpdateDescription}>
                Save
              </button>
            </div>
          </div>
        </div>
      )}

      {transitionModal && (
        <ConfirmModal
          title={`Transition to ${transitionModal}`}
          message={`Are you sure you want to transition version ${selectedVersion.version} to ${transitionModal}?`}
          onConfirm={() => handleStageTransition(transitionModal)}
          onCancel={() => setTransitionModal(null)}
        />
      )}
    </div>
  );
}

export default ModelDetail;
