# CORS and Path Fixes for Flask RAG Agent

## Overview

This document explains the fixes applied to resolve:
1. **CORS issues** preventing frontend-backend communication
2. **Path issues** causing errors when running the app from different directories

## 🛠️ Path Fixes Applied

### Problem
The original code used relative paths that broke when running from different directories:
```python
# OLD - Relative paths (problematic)
UPLOAD_FOLDER = './files-db'  # Current working directory
storage_dir = "./vector_db_storage"  # Current working directory
```

**Issue**: When running `python app.py` from different directories, these paths would point to wrong locations.

### Solution
Updated to use absolute paths relative to the script location:

#### In `app.py`:
```python
# NEW - Robust path handling
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(SCRIPT_DIR, 'files-db')
VECTOR_DB_STORAGE = os.path.join(SCRIPT_DIR, 'vector_db_storage')

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_DB_STORAGE, exist_ok=True)
```

#### In `setup.py`:
```python
# Updated function signatures to accept explicit paths
def load_or_build_vector_db(
    upload_folder: str,
    embedding_client: Any,
    embedding_model: str,
    storage_dir: str = None,  # Changed from "./vector_db_storage"
    # ... other parameters
):
    # Set default storage directory if not provided (relative to this script)
    if storage_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        storage_dir = os.path.join(script_dir, "vector_db_storage")
```

### Benefits
✅ **Works from any directory**: Run `python app.py` or `python ai-agent-ms/app.py`
✅ **Absolute paths**: No more relative path confusion
✅ **Auto-creation**: Directories are created if they don't exist
✅ **Consistent**: Same behavior regardless of working directory

## 🌐 CORS Fixes Applied

### Problem
Frontend (Next.js on port 3000) couldn't communicate with backend (Flask on port 5000) due to CORS restrictions.

### Solution
Added Flask-CORS configuration:

```python
from flask_cors import CORS

# CORS Configuration - Allow frontend to communicate with backend
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",  # Next.js development server
            "http://127.0.0.1:3000",  # Alternative localhost
            "http://192.168.1.26:3000"  # Network access (adjust IP as needed)
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True
    }
})
```

### Installation Required
Add Flask-CORS to requirements.txt:
```text
Flask-CORS==4.0.0
```

Install with:
```bash
pip install Flask-CORS==4.0.0
```

## 📁 Directory Structure

After fixes, the structure looks like:
```
ai-agent-ms/
├── app.py                      # Flask backend with CORS
├── setup.py                    # RAG functions with robust paths
├── files-db/                   # PDF uploads (auto-created)
├── vector_db_storage/          # Persistent embeddings (auto-created)
│   ├── vector_db.pkl          # Pickled DataFrame
│   └── file_metadata.json     # File change tracking
├── requirements.txt            # Including Flask-CORS
└── test_path_fix.py           # Test script to verify fixes
```

## 🧪 Testing the Fixes

### Path Testing
```bash
# Test 1: Run from ai-agent-ms directory
cd ai-agent-ms
python app.py

# Test 2: Run from parent directory  
cd ..
python ai-agent-ms/app.py

# Test 3: Run the verification script
cd ai-agent-ms
python test_path_fix.py
```

### CORS Testing
1. Start Flask backend: `python app.py`
2. Start Next.js frontend: `npm run dev` (in personal-rag-agent-ui)
3. Test file upload and chat in browser at `http://localhost:3000`

## 🔧 Configuration Options

### Customizing CORS Origins
Update the origins list in `app.py` for your network:
```python
"origins": [
    "http://localhost:3000",     # Development
    "http://your-domain.com",    # Production
    "http://192.168.1.100:3000"  # Your specific IP
],
```

### Customizing Storage Paths
Pass explicit paths when calling functions:
```python
# Custom storage location
vector_db = load_or_build_vector_db(
    upload_folder="/custom/upload/path",
    storage_dir="/custom/storage/path",
    # ... other parameters
)
```

## 🚨 Troubleshooting

### Path Issues
- **Error**: "No such file or directory"
  **Solution**: Ensure you've updated to the latest code with path fixes

- **Error**: "Permission denied creating directory"
  **Solution**: Check write permissions in the script directory

### CORS Issues
- **Error**: "Access to fetch blocked by CORS policy"
  **Solution**: Ensure Flask-CORS is installed and backend is running

- **Error**: "Network Error" in frontend
  **Solution**: Check that Flask backend is running on port 5000

### Import Issues
- **Error**: "ModuleNotFoundError: No module named 'flask_cors'"
  **Solution**: Install Flask-CORS: `pip install Flask-CORS==4.0.0`

## 📝 Summary of Changes

### Files Modified:
1. **`app.py`**: Added CORS configuration, robust path handling
2. **`setup.py`**: Updated default path parameters, added path validation
3. **`requirements.txt`**: Added Flask-CORS dependency

### New Files:
1. **`test_path_fix.py`**: Verification script for path fixes
2. **`CORS_AND_PATH_FIXES.md`**: This documentation

### Benefits:
- ✅ Frontend-backend communication works
- ✅ App runs from any directory
- ✅ Robust error handling
- ✅ Auto-directory creation
- ✅ Better development experience

Your RAG system is now ready for seamless development and deployment! 🎉