import React, { useState } from 'react';
import ModelList from './components/ModelList';
import ModelDetail from './components/ModelDetail';
import UploadForm from './components/UploadForm';
import Settings from './components/Settings';
import './App.css';

function App() {
  const [currentView, setCurrentView] = useState('home');
  const [selectedModel, setSelectedModel] = useState(null);
  const [config, setConfig] = useState({
    apiBaseUrl: 'http://localhost:8000',
    mlflowBaseUrl: 'http://localhost:5001',
    modelServingUrl: '',
    authToken: ''
  });
  const [refreshTrigger, setRefreshTrigger] = useState(0);

  const handleModelSelect = (model) => {
    setSelectedModel(model);
    setCurrentView('detail');
  };

  const handleBackToList = () => {
    setSelectedModel(null);
    setCurrentView('models');
    setRefreshTrigger(prev => prev + 1);
  };

  const handleConfigUpdate = (newConfig) => {
    setConfig(newConfig);
  };

  const handleNavigation = (view) => {
    setCurrentView(view);
    if (view !== 'detail') {
      setSelectedModel(null);
    }
  };

  return (
    <div className="app-container">
      <div className="app-content">
        <header className="app-header">
          <div className="logo">
            <div className="logo-icon">PX</div>
            <h1>Pixonal</h1>
          </div>
          
          <nav>
            <button
              className={currentView === 'home' ? 'active' : ''}
              onClick={() => handleNavigation('home')}
            >
              Home
            </button>
            <button
              className={currentView === 'models' || currentView === 'detail' ? 'active' : ''}
              onClick={() => handleNavigation('models')}
            >
              Models
            </button>
            <button
              className={currentView === 'upload' ? 'active' : ''}
              onClick={() => handleNavigation('upload')}
            >
              Upload
            </button>
            <button
              className={currentView === 'settings' ? 'active' : ''}
              onClick={() => handleNavigation('settings')}
            >
              Settings
            </button>
          </nav>

          <button className="login-btn">Login</button>
        </header>

        <div className="app-layout">
          <main className="main-content">
            {currentView === 'home' && (
              <div className="card">
                <h2>Welcome to Pixonal Studio</h2>
                <p style={{ color: 'var(--text-secondary)', lineHeight: '1.6' }}>
                  Manage machine learning models with MLflow. Upload new models,
                  view model details, test predictions, and manage model lifecycles all in one place.
                </p>
                <button 
                  className="btn btn-primary"
                  style={{ marginTop: '1.5rem' }}
                  onClick={() => handleNavigation('models')}
                >
                  View Models
                </button>
              </div>
            )}
            {currentView === 'models' && (
              <ModelList
                config={config}
                onModelSelect={handleModelSelect}
                refreshTrigger={refreshTrigger}
              />
            )}
            {currentView === 'detail' && selectedModel && (
              <ModelDetail
                model={selectedModel}
                config={config}
                onBack={handleBackToList}
              />
            )}
            {currentView === 'upload' && (
              <UploadForm
                config={config}
                onSuccess={() => {
                  setCurrentView('models');
                  setRefreshTrigger(prev => prev + 1);
                }}
              />
            )}
            {currentView === 'settings' && (
              <Settings
                config={config}
                onConfigUpdate={handleConfigUpdate}
              />
            )}
          </main>
        </div>
      </div>
    </div>
  );
}

export default App;
