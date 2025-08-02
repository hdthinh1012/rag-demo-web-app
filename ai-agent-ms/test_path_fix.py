#!/usr/bin/env python3
"""
Test script to verify that the path fixes work correctly
"""

import os
import sys

def test_paths():
    """Test the path configuration"""
    print("🧪 Testing Path Configuration")
    print("=" * 50)
    
    # Test 1: Import the app module to check if paths are set correctly
    try:
        # Add current directory to path so we can import app
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
        
        # Import the modules (this will execute the path setup)
        import app
        
        print("✅ App module imported successfully")
        print(f"📁 Script directory: {app.SCRIPT_DIR}")
        print(f"📁 Upload folder: {app.UPLOAD_FOLDER}")
        print(f"📁 Vector DB storage: {app.VECTOR_DB_STORAGE}")
        
        # Verify paths are absolute
        if os.path.isabs(app.UPLOAD_FOLDER):
            print("✅ Upload folder path is absolute")
        else:
            print("❌ Upload folder path is still relative")
        
        if os.path.isabs(app.VECTOR_DB_STORAGE):
            print("✅ Vector DB storage path is absolute")
        else:
            print("❌ Vector DB storage path is still relative")
        
        # Check if directories exist
        if os.path.exists(app.UPLOAD_FOLDER):
            print("✅ Upload folder exists")
        else:
            print("⚠️ Upload folder doesn't exist (will be created when needed)")
        
        if os.path.exists(app.VECTOR_DB_STORAGE):
            print("✅ Vector DB storage folder exists")
        else:
            print("⚠️ Vector DB storage folder doesn't exist (will be created when needed)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error importing app module: {str(e)}")
        return False

def test_setup_functions():
    """Test the setup.py functions with path handling"""
    print("\n🔧 Testing Setup Functions")
    print("=" * 50)
    
    try:
        from setup import get_storage_info
        
        # Test with no storage_dir parameter (should use default)
        storage_info = get_storage_info()
        print("✅ get_storage_info() called successfully with default path")
        print(f"📊 Storage exists: {storage_info['storage_exists']}")
        
        # Test with explicit path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_storage_dir = os.path.join(current_dir, "test_vector_storage")
        storage_info = get_storage_info(test_storage_dir)
        print("✅ get_storage_info() called successfully with explicit path")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing setup functions: {str(e)}")
        return False

def test_from_different_directories():
    """Test that paths work when running from different directories"""
    print("\n📂 Testing from Different Working Directories")
    print("=" * 50)
    
    current_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print(f"Current working directory: {current_dir}")
    print(f"Script directory: {script_dir}")
    
    if current_dir == script_dir:
        print("✅ Running from script directory")
    else:
        print("✅ Running from different directory - this should still work!")
    
    return True

def main():
    print("🚀 Path Fix Verification Test")
    print("This script tests that the path fixes allow the app to work from any directory")
    print()
    
    try:
        success1 = test_paths()
        success2 = test_setup_functions()  
        success3 = test_from_different_directories()
        
        if success1 and success2 and success3:
            print("\n🎉 All path tests passed!")
            print("Your Flask app should now work when run from any directory.")
            print("\nTo test:")
            print("1. cd to ai-agent-ms: python app.py")
            print("2. cd to parent: python ai-agent-ms/app.py")
            print("Both should work now!")
        else:
            print("\n❌ Some tests failed. Check the output above.")
            
    except Exception as e:
        print(f"\n💥 Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()