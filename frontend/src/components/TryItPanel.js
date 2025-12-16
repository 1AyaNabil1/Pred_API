import React, { useState, useEffect } from 'react';
import './TryItPanel.css';

/**
 * TryItPanel - Model Invocation Component
 *
 * CANONICAL INPUT CONTRACT (NON-NEGOTIABLE):
 * ==========================================
 * This component sends predictions in exactly this format:
 *
 * {
 *   "inputs": [
 *     [x1, x2, x3, x4],
 *     [y1, y2, y3, y4]
 *   ]
 * }
 *
 * Where:
 * - "inputs" is a 2D list (list of lists)
 * - Each row has exactly 4 numeric values
 * - No objects, strings, or alternate keys
 *
 * This component enforces the contract by:
 * 1. Accepting only a 2D JSON array as input
 * 2. Validating structure before submission
 * 3. Blocking invocation if validation fails
 */
function TryItPanel({ modelName, version, metadata, config }) {
  // State for 2D array input as text
  const [inputText, setInputText] = useState('[\n  [5.1, 3.5, 1.4, 0.2],\n  [6.0, 2.7, 5.1, 1.6]\n]');
  const [validationError, setValidationError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);
  const [latency, setLatency] = useState(null);
  const [showCurl, setShowCurl] = useState(false);

  // Validate the input text whenever it changes
  useEffect(() => {
    validateInput(inputText);
  }, [inputText]);

  /**
   * Validates the input text against the canonical contract.
   * 
   * Requirements:
   * 1. Must be valid JSON
   * 2. Must be a 2D array (list of lists)
   * 3. Each row must have exactly 4 values
   * 4. All values must be numeric
   * 
   * @param {string} text - The input text to validate
   * @returns {Array|null} - The parsed array if valid, null otherwise
   */
  const validateInput = (text) => {
    try {
      const parsed = JSON.parse(text);
      
      // Must be an array
      if (!Array.isArray(parsed)) {
        setValidationError('Input must be a 2D array. Expected: [[x1, x2, x3, x4], ...]');
        return null;
      }
      
      // Must not be empty
      if (parsed.length === 0) {
        setValidationError('Input array cannot be empty. Provide at least one row.');
        return null;
      }
      
      // Each element must be an array of exactly 4 numbers
      for (let i = 0; i < parsed.length; i++) {
        const row = parsed[i];
        
        if (!Array.isArray(row)) {
          setValidationError(`Row ${i} is not an array. Each row must be [x1, x2, x3, x4].`);
          return null;
        }
        
        if (row.length !== 4) {
          setValidationError(`Row ${i} has ${row.length} values. Each row must have exactly 4 numeric values.`);
          return null;
        }
        
        for (let j = 0; j < row.length; j++) {
          const value = row[j];
          if (typeof value !== 'number' || isNaN(value) || !isFinite(value)) {
            setValidationError(`Value at row ${i}, column ${j} is not a valid number.`);
            return null;
          }
        }
      }
      
      setValidationError(null);
      return parsed;
    } catch (e) {
      setValidationError(`Invalid JSON: ${e.message}`);
      return null;
    }
  };

  /**
   * Handles the model invocation.
   * 
   * Sends the request in the canonical format:
   * { "inputs": [[x1, x2, x3, x4], ...] }
   */
  const handleInvoke = async () => {
    const parsedInputs = validateInput(inputText);
    
    if (!parsedInputs) {
      setError('Cannot invoke: Input validation failed. Please fix the errors above.');
      return;
    }
    
    if (!config.apiBaseUrl) {
      setError('API Base URL not configured. Please set it in Settings.');
      return;
    }

    setLoading(true);
    setError(null);
    setResponse(null);
    setLatency(null);

    try {
      // Send the canonical format - NO WRAPPING, NO TRANSFORMATION
      const apiResponse = await fetch(`${config.apiBaseUrl}/predict/${modelName}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(config.authToken && { 'Authorization': `Bearer ${config.authToken}` })
        },
        body: JSON.stringify({
          inputs: parsedInputs  // Direct pass-through, no wrapping
        })
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.detail || 'Invocation failed');
      }

      const data = await apiResponse.json();
      setResponse(data.predictions);
      setLatency(data.latency_ms);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Generates a curl command for the canonical request format.
   */
  const generateCurlCommand = () => {
    const apiUrl = `${config.apiBaseUrl}/predict/${modelName}`;
    const headers = ['-H "Content-Type: application/json"'];
    
    if (config.authToken) {
      headers.push(`-H "Authorization: Bearer ${config.authToken}"`);
    }

    // Parse and reformat for clean output
    let payload;
    try {
      const parsed = JSON.parse(inputText);
      payload = JSON.stringify({ inputs: parsed }, null, 2);
    } catch (e) {
      payload = '{"inputs": [[5.1, 3.5, 1.4, 0.2]]}';
    }

    return `curl -X POST "${apiUrl}" \\
${headers.join(' \\\n')} \\
-d '${payload}'`;
  };

  /**
   * Generates a JavaScript fetch snippet for the canonical request format.
   */
  const generateFetchSnippet = () => {
    const apiUrl = `${config.apiBaseUrl}/predict/${modelName}`;
    
    let inputsStr;
    try {
      const parsed = JSON.parse(inputText);
      inputsStr = JSON.stringify(parsed);
    } catch (e) {
      inputsStr = '[[5.1, 3.5, 1.4, 0.2]]';
    }

    return `// CANONICAL FORMAT: inputs is a 2D array, each row has exactly 4 values
fetch('${apiUrl}', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
${config.authToken ? `    'Authorization': 'Bearer ${config.authToken}',\n` : ''}  },
  body: JSON.stringify({
    inputs: ${inputsStr}
  })
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));`;
  };

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text);
    alert('Copied to clipboard!');
  };

  const getApiUrl = () => {
    if (!config.apiBaseUrl) return 'Not configured';
    return `${config.apiBaseUrl}/predict/${modelName}`;
  };

  const copyEndpoint = () => {
    navigator.clipboard.writeText(getApiUrl());
    alert('Endpoint copied to clipboard!');
  };

  /**
   * Resets the input to the default example.
   */
  const resetToDefault = () => {
    setInputText('[\n  [5.1, 3.5, 1.4, 0.2],\n  [6.0, 2.7, 5.1, 1.6]\n]');
  };

  return (
    <div className="card try-it-card">
      <h3>Try It Out</h3>

      <div className="api-info-section">
        <div className="api-info-header">
          <h4>API Endpoint</h4>
          {config.apiBaseUrl ? (
            <button className="btn btn-sm btn-secondary" onClick={copyEndpoint}>
              Copy
            </button>
          ) : (
            <span className="warning-badge">Not Configured</span>
          )}
        </div>
        <div className="endpoint-container">
          <div className="endpoint-url">{getApiUrl()}</div>
          {!config.apiBaseUrl && (
            <small className="endpoint-hint">
              Configure the API Base URL in Settings to enable testing
            </small>
          )}
        </div>

        <div className="api-params">
          <h4>Required Parameters</h4>
          <div className="params-list">
            <div className="param-item">
              <span className="param-label">Method:</span>
              <code className="param-value">POST</code>
            </div>
            <div className="param-item">
              <span className="param-label">Content-Type:</span>
              <code className="param-value">application/json</code>
            </div>
            {config.authToken && (
              <div className="param-item">
                <span className="param-label">Authorization:</span>
                <code className="param-value">Bearer {config.authToken.substring(0, 20)}...</code>
              </div>
            )}
            <div className="param-item">
              <span className="param-label">Body Format:</span>
              <code className="param-value">{`{"inputs": [[x1, x2, x3, x4], ...]}`}</code>
            </div>
          </div>
        </div>
      </div>

      <div className="inputs-section">
        <div className="section-header">
          <h4>Input Data (2D Array)</h4>
          <button className="btn btn-sm btn-secondary" onClick={resetToDefault}>
            Reset to Example
          </button>
        </div>
        
        <div className="input-format-help">
          <p><strong>Required Format:</strong> A 2D array where each row has exactly 4 numeric values.</p>
          <p><em>Example: [[5.1, 3.5, 1.4, 0.2], [6.0, 2.7, 5.1, 1.6]]</em></p>
        </div>

        <textarea
          className={`input-textarea ${validationError ? 'input-error' : 'input-valid'}`}
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder='Enter a 2D array: [[x1, x2, x3, x4], ...]'
          rows={6}
          spellCheck={false}
        />
        
        {validationError && (
          <div className="validation-error">
            {validationError}
          </div>
        )}
        
        {!validationError && inputText && (
          <div className="validation-success">
            Valid input format
          </div>
        )}
      </div>

      <button
        className="btn btn-primary btn-full invoke-button"
        onClick={handleInvoke}
        disabled={loading || !config.apiBaseUrl || validationError}
        title={
          !config.apiBaseUrl 
            ? 'Please configure API Base URL in Settings first' 
            : validationError 
              ? 'Fix validation errors before invoking' 
              : 'Click to invoke the model'
        }
      >
        {loading ? 'Invoking...' : 'Invoke Model'}
      </button>

      {!config.apiBaseUrl && (
        <div className="warning-message">
          <strong>Configuration Required:</strong> Please configure the API Base URL in Settings to enable model invocation.
        </div>
      )}

      {error && <div className="error"><strong>Error:</strong> {error}</div>}

      {response && (
        <div className="response-section">
          <div className="response-header">
            <h4>Response</h4>
            {latency && <span className="latency">⚡ {latency}ms</span>}
          </div>
          <pre className="response-code">
            {JSON.stringify(response, null, 2)}
          </pre>
        </div>
      )}

      <div className="code-snippets">
        <button
          className="btn btn-secondary btn-full"
          onClick={() => setShowCurl(!showCurl)}
        >
          {showCurl ? 'Hide' : 'Show'} Code Snippets
        </button>

        {showCurl && (
          <div className="snippets-container">
            <div className="snippet">
              <div className="snippet-header">
                <h5>cURL</h5>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => copyToClipboard(generateCurlCommand())}
                >
                  Copy
                </button>
              </div>
              <pre className="snippet-code">{generateCurlCommand()}</pre>
            </div>

            <div className="snippet">
              <div className="snippet-header">
                <h5>JavaScript (fetch)</h5>
                <button
                  className="btn btn-sm btn-secondary"
                  onClick={() => copyToClipboard(generateFetchSnippet())}
                >
                  Copy
                </button>
              </div>
              <pre className="snippet-code">{generateFetchSnippet()}</pre>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TryItPanel;
