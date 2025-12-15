# Pixonal - MLflow Model Management

## Features

- **View All Models** - Browse all registered models in MLflow
- **Upload .pkl Files** - Direct upload of pickle files to MLflow
- **Register Models** - Register existing model artifacts from MLflow runs
- **Schema Visualization** - View input/output schemas with visual indicators
- **Test Models** - Make sample API calls with interactive input management
- **Lifecycle Management** - Activate/deactivate models (Production, Staging, Archived)
- **Update Models** - Edit model descriptions and metadata
- **Delete Models** - Remove models with confirmation
- **Code Snippets** - Generate cURL and JavaScript code for API calls

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 14+
- npm or yarn

### Installation

1. **Clone the repository** (if not already done)
   ```bash
   cd /path/to/Pred_API
   ```

2. **Set up the backend**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Set up the frontend**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Configure environment** (optional)
   ```bash
   # Copy the example env file
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the Application

#### Option 1: Using the Startup Script (Recommended)

```bash
# Start MLflow and Backend
./start_services.sh

# In a new terminal, start the frontend
cd frontend
npm start
```

#### Option 2: Manual Start

**Terminal 1 - MLflow Server:**
```bash
source backend/venv/bin/activate
mlflow server --host 0.0.0.0 --port 5001 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

**Terminal 2 - Backend API:**
```bash
cd backend
source venv/bin/activate
uvicorn main_secure:app --host 0.0.0.0 --port 8000 --reload
```

**Terminal 3 - Frontend:**
```bash
cd frontend
npm start
```

### Stopping the Application

```bash
./stop_services.sh
```

## Access Points

- **Frontend UI**: http://localhost:3000
- **MLflow UI**: http://localhost:5001
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## Configuration

Navigate to **Settings** in the UI to configure:

- **API Base URL**: Backend API endpoint (default: `http://localhost:8000`)
- **MLflow Base URL**: MLflow tracking server (default: `http://localhost:5001`)
- **Model Serving URL**: URL for model inference endpoint (configure as needed)
- **Auth Token**: Optional authentication token

## Usage Guide

### Uploading Models

#### Upload .pkl File
1. Navigate to **Upload** section
2. Click on **Upload .pkl File** tab
3. Fill in the model name
4. Select your .pkl file
5. Optionally add description and input example
6. Click **Upload & Register Model**

#### Register Existing Model
1. Navigate to **Upload** section
2. Click on **Register Existing Model** tab
3. Enter model name and source URI
4. Format: `runs:/<run-id>/model` or `models:/<name>/<version>`
5. Add optional metadata (description, tags)
6. Click **Register Version**

### Managing Models

#### View Model Details
- Click **View Details** on any model card
- View versions, schema, and metadata
- Test the model with sample inputs

#### Activate/Deactivate
In the model detail page:
- **Activate (Production)** - Deploy to production
- **Move to Staging** - Move to testing
- **Deactivate (Archive)** - Archive the model
- **Remove Stage** - Remove from all stages

#### Update Model
- Click **Edit** next to the description
- Modify the text
- Click **Save**

#### Delete Model
- Click **Delete** on the model card
- Confirm the deletion in the modal

### Testing Models

1. Go to model detail page
2. Scroll to **Try It Out** panel
3. Add input fields or use auto-detected schema
4. Enter test values
5. Click **Invoke Model**
6. View response and latency
7. Generate code snippets (cURL, JavaScript)

## Project Structure

```
Pred_API/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── requirements.txt     # Python dependencies
│   └── venv/               # Virtual environment
├── frontend/
│   ├── src/
│   │   ├── components/     # React components
│   │   │   ├── ModelList.js
│   │   │   ├── ModelDetail.js
│   │   │   ├── UploadForm.js
│   │   │   ├── SchemaEditor.js
│   │   │   ├── TryItPanel.js
│   │   │   └── Settings.js
│   │   ├── App.js
│   │   └── index.js
│   ├── package.json
│   └── public/
├── mlruns/                 # MLflow artifacts
├── mlflow.db              # MLflow metadata database
├── start_services.sh      # Service startup script
├── stop_services.sh       # Service stop script
└── README.md             # This file
```

## Backend API Endpoints

### Model Registry
- `POST /mlflow/registered-models/search` - Search models
- `POST /mlflow/model-versions/create` - Register model version
- `GET /mlflow/model-versions/get` - Get version metadata
- `POST /mlflow/model-versions/transition-stage` - Change stage
- `PATCH /mlflow/registered-models/update` - Update model
- `DELETE /mlflow/registered-models/delete` - Delete model
- `POST /mlflow/upload-pkl-model` - Upload .pkl file

### Model Serving
- `POST /serve/invoke` - Invoke model endpoint

### Health
- `GET /health` - Check service health

## Design

- **Theme**: Dark mode with purple/blue accents
- **Layout**: Card-based responsive design
- **Icons**: Emoji-enhanced for better UX
- **Colors**:
  - Primary: Purple/Blue (#6366f1)
  - Success: Green (#10b981)
  - Warning: Yellow (#f59e0b)
  - Error: Red (#ef4444)

## Troubleshooting

### Backend returns 500 errors
**Solution**: Make sure MLflow is running on port 5001
```bash
curl http://localhost:5001/health
```

### Frontend can't connect to backend
**Solution**: Check if backend is running on port 8000
```bash
curl http://localhost:8000/health
```

### MLflow database errors
**Solution**: Delete and recreate the database
```bash
rm mlflow.db
# Restart MLflow server
```

### Port already in use
**Solution**: Stop existing services
```bash
./stop_services.sh
# Or manually kill processes
lsof -ti:5001 | xargs kill -9
lsof -ti:8000 | xargs kill -9
lsof -ti:3000 | xargs kill -9
```

## Dependencies

### Backend
- FastAPI - Web framework
- uvicorn - ASGI server
- httpx - HTTP client
- mlflow - Model tracking
- scikit-learn - ML support
- python-multipart - File uploads

### Frontend
- React - UI framework
- Modern CSS - Styling

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of Pixonal Studio.

## Tips

- Use the **Settings** page to configure URLs before using other features
- Check the MLflow UI (http://localhost:5001) to verify model artifacts
- Use the **API Docs** (http://localhost:8000/docs) to test endpoints directly
- Models must be in Production or Staging stage to be served
- Always test models in Staging before promoting to Production

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review logs: `mlflow.log` and `backend.log`
3. Verify all services are running
4. Check browser console for frontend errors