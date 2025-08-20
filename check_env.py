# Quick environment debug script - save as check_env.py and run it

import sys
import os

print("PYTHON ENVIRONMENT DEBUG")
print("=" * 50)
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")  # Show first 3 paths

print("\nTESTING ONNX RUNTIME IMPORT")
print("=" * 50)

try:
    import onnxruntime as ort
    print("✓ ONNX Runtime imported successfully!")
    print(f"  Version: {ort.__version__}")
    print(f"  Available providers: {ort.get_available_providers()}")
    
    # Test basic functionality
    print("\nTesting ONNX Runtime functionality...")
    providers = ort.get_available_providers()
    if 'CPUExecutionProvider' in providers:
        print("✓ CPU Execution Provider available")
    else:
        print("✗ CPU Execution Provider not available")
        
except ImportError as e:
    print(f"✗ ONNX Runtime import failed: {e}")
    print("This suggests the package is installed in a different environment")
    
except Exception as e:
    print(f"✗ ONNX Runtime error: {e}")

print("\nCHECKING OBJECT DETECTION IMPORT")
print("=" * 50)

try:
    # Add current directory to path
    current_dir = os.getcwd()
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Try to import the object detection module
    sys.path.append('modules/object_detection')
    from object_detection import ObjectDetector, YOLOv8
    
    print("✓ Object detection module imported successfully")
    
    # Test creating an ObjectDetector
    detector = ObjectDetector()
    print("✓ ObjectDetector created successfully")
    
    # Check if it's using ONNX Runtime
    if hasattr(detector.detector, 'session') and detector.detector.session:
        print("✓ Detector is using ONNX Runtime!")
    elif hasattr(detector.detector, 'fallback_mode') and detector.detector.fallback_mode:
        print("⚠️  Detector is in fallback mode")
    else:
        print("⚠️  Detector status unclear")
        
except Exception as e:
    print(f"✗ Object detection import failed: {e}")
    import traceback
    print(traceback.format_exc())

print("\nRECOMMENDATIONS")
print("=" * 50)

# Check if running in Anaconda
if 'anaconda' in sys.executable.lower() or 'conda' in sys.executable.lower():
    print("You're using Anaconda/Conda environment")
    print("Try these commands:")
    print("1. conda install onnxruntime")
    print("2. Or: conda install -c conda-forge onnxruntime")
else:
    print("You're using system Python or virtual environment")
    print("Try: pip install --upgrade onnxruntime")

print("\nIf ONNX Runtime is working but detection still fails:")
print("1. Restart your Flask application completely")
print("2. Check that you're running Flask with the same Python environment")
print("3. Verify model file compatibility")