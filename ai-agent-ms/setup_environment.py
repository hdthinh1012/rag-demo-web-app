#!/usr/bin/env python3
"""
Environment setup script for the Flask RAG Agent
Handles Flask-CORS installation and path verification
"""

import subprocess
import sys
import os

def check_and_install_flask_cors():
    """Check if Flask-CORS is installed, install if needed"""
    print("🔍 Checking Flask-CORS installation...")
    
    try:
        import flask_cors
        print("✅ Flask-CORS is already installed")
        return True
    except ImportError:
        print("❌ Flask-CORS not found. Installing...")
        
        try:
            # Try different installation methods
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Flask-CORS==4.0.0"])
            print("✅ Flask-CORS installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install Flask-CORS: {e}")
            print("\n💡 Manual installation options:")
            print("1. Run: pip install Flask-CORS")
            print("2. Run: conda install -c conda-forge flask-cors")
            print("3. Or install using your preferred package manager")
            return False

def verify_paths():
    """Verify that the path fixes are working"""
    print("\n📁 Verifying path configuration...")
    
    try:
        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        upload_folder = os.path.join(script_dir, 'files-db')
        storage_folder = os.path.join(script_dir, 'vector_db_storage')
        
        print(f"Script directory: {script_dir}")
        print(f"Upload folder: {upload_folder}")
        print(f"Storage folder: {storage_folder}")
        
        # Create directories if they don't exist
        os.makedirs(upload_folder, exist_ok=True)
        os.makedirs(storage_folder, exist_ok=True)
        
        print("✅ Directories created/verified successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error setting up directories: {e}")
        return False

def check_dependencies():
    """Check all required dependencies"""
    print("\n📦 Checking required dependencies...")
    
    required_packages = [
        'flask',
        'google-genai', 
        'PyPDF2',
        'numpy',
        'pandas',
        'scikit-learn',
        'tenacity',
        'python-dotenv'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed!")
        return True

def main():
    """Main setup function"""
    print("🚀 Flask RAG Agent Environment Setup")
    print("=" * 50)
    
    success = True
    
    # Check and install Flask-CORS
    if not check_and_install_flask_cors():
        success = False
    
    # Verify paths
    if not verify_paths():
        success = False
    
    # Check other dependencies
    if not check_dependencies():
        success = False
    
    print("\n" + "=" * 50)
    
    if success:
        print("🎉 Environment setup completed successfully!")
        print("\nYour Flask RAG Agent is ready to run:")
        print("1. Start backend: python app.py")
        print("2. Start frontend: cd ../personal-rag-agent-ui && npm run dev")
        print("3. Open browser: http://localhost:3000")
    else:
        print("❌ Environment setup had some issues.")
        print("Please resolve the issues above and run this script again.")
        print("\nFor manual setup:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. Install Flask-CORS: pip install Flask-CORS")
        print("3. Check file permissions in the project directory")

if __name__ == "__main__":
    main()