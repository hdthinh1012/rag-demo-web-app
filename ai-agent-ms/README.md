# AI Agent Microservice - RAG with Speech Generation

A Flask microservice that implements Retrieval Augmented Generation (RAG) with speech generation capabilities using Google's Gemini 2.5 Live API.

## Features

- **File Upload & Processing**: Upload PDF files via form-data requests
- **RAG Pipeline**: Automatic document indexing and retrieval
- **Speech Generation**: Generate audio responses using Gemini 2.5 Live API
- **Vector Database**: In-memory vector database for document similarity search
- **REST API**: Simple REST endpoints for interaction

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Copy the environment template and configure your Google Cloud settings:

```bash
cp env.template .env
```

Edit `.env` file with your Google Cloud project details:
```bash
PROJECT_ID=your-google-cloud-project-id
LOCATION=asia-southeast1
```

### 3. Google Vertex AI Authentication

This application uses the `google.genai` library with Vertex AI for generative AI capabilities. Service account authentication is required for production use.

#### 3.1. Create Service Account

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to **IAM & Admin** > **Service Accounts**
3. Click **Create Service Account**
4. Fill in the service account details:
   - **Name**: `ai-agent-service-account` (or your preferred name)
   - **Description**: Service account for AI Agent RAG application
5. Click **Create and Continue**

#### 3.2. Grant Required Permissions

Assign the following roles to your service account:
- **Vertex AI User** (`roles/aiplatform.user`)
- **ML Developer** (`roles/ml.developer`)
- **Service Usage Consumer** (`roles/serviceusage.serviceUsageConsumer`)

#### 3.3. Download Service Account Key

1. In the **Service Accounts** page, find your newly created service account
2. Click on the service account email
3. Go to the **Keys** tab
4. Click **Add Key** > **Create new key**
5. Select **JSON** format
6. Click **Create** - the JSON key file will be downloaded

#### 3.4. Configure Authentication

1. Create a `credentials` folder in your project directory:
   ```bash
   mkdir credentials
   ```

2. Move the downloaded JSON key file to the credentials folder:
   ```bash
   mv ~/Downloads/your-service-account-key.json ./credentials/
   ```

3. Set the `GOOGLE_APPLICATION_CREDENTIALS` environment variable:

   **Windows (PowerShell):**
   ```powershell
   $env:GOOGLE_APPLICATION_CREDENTIALS="./credentials/your-service-account-key.json"
   ```

   **Linux/macOS:**
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="./credentials/your-service-account-key.json"
   ```

4. Update your `.env` file:
   ```bash
   PROJECT_ID=your-google-cloud-project-id
   LOCATION=asia-southeast1
   SERVICE_ACCOUNT_CREDENTIALS=./credentials/your-service-account-key.json
   ```

#### 3.5. Enable Required APIs

Make sure the following APIs are enabled in your Google Cloud project:
```bash
gcloud services enable aiplatform.googleapis.com
gcloud services enable generativelanguage.googleapis.com
```

## Usage

### Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints

#### POST `/generate-speech`

Generate speech response based on uploaded documents and query.

**Request:**
- Method: POST
- Content-Type: multipart/form-data
- Parameters:
  - `query` (string): The question to ask
  - `files` (file[]): PDF files to upload and process

**Response:**
- Content-Type: audio/wav
- Returns: Audio file with the generated speech response

**Example using curl:**

```bash
curl -X POST http://localhost:5000/generate-speech \
  -F "query=What is the main topic of the document?" \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf" \
  -o response.wav
```

#### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "rag_initialized": true,
  "documents_indexed": true,
  "upload_folder": "./files-db",
  "pdf_files_count": 2
}
```

#### GET `/files`

List all uploaded files.

**Response:**
```json
{
  "files": [
    {
      "filename": "1234567890_document.pdf",
      "size": 1024000,
      "modified": 1703123456.789
    }
  ],
  "total_count": 1
}
```

## Architecture

### RAG Pipeline

1. **Document Processing**: PDF files are processed and split into chunks
2. **Embedding Generation**: Text chunks are converted to embeddings using Gemini embedding model
3. **Vector Database**: Embeddings are stored in an in-memory pandas DataFrame
4. **Retrieval**: Query embeddings are matched against document embeddings using cosine similarity
5. **Generation**: Relevant context is used to generate audio responses via Gemini 2.5 Live API

### File Management

- Uploaded files are stored in `./files-db/` directory
- Files are automatically timestamped to prevent naming conflicts
- Supported formats: PDF
- Maximum file size: 16MB per file

## Configuration

### Environment Variables

- `PROJECT_ID`: Google Cloud project ID
- `LOCATION`: Google Cloud region (default: asia-southeast1)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to service account key (optional)

### Flask Settings

- Debug mode: Enabled by default
- Host: 0.0.0.0 (accepts external connections)
- Port: 5000
- Max file size: 16MB

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `400`: Bad request (missing query, empty query, no documents)
- `500`: Internal server error (API quota issues, processing failures)

## Dependencies

- Flask: Web framework
- google-genai: Google Generative AI SDK
- PyPDF2: PDF processing
- pandas: Data manipulation
- scikit-learn: Similarity calculations
- numpy: Numerical operations
- tenacity: Retry mechanisms

## Limitations

- In-memory vector database (not persistent)
- PDF files only
- Single language support
- Requires Google Cloud authentication
- API quota limitations may apply