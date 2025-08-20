# Step-by-Step Debug Script for Object Detection Model
# Save this as debug_model.py and run it

import os
import sys
import cv2
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def check_model_file():
    """Check if the YOLO model file exists and is valid"""
    print("=" * 50)
    print("CHECKING MODEL FILE")
    print("=" * 50)
    
    possible_paths = [
        r"C:\Users\Hp\CVPlatform\modules\object_detection\models\yolov8m.onnx",
        "modules/object_detection/models/yolov8m.onnx",
        "models/yolov8m.onnx",
        "yolov8m.onnx"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"✓ Found model: {path}")
            print(f"  Size: {size_mb:.1f} MB")
            
            # Check if file is corrupted (YOLO models should be >10MB)
            if size_mb < 10:
                print(f"⚠️  Warning: Model file seems too small ({size_mb:.1f} MB)")
                print("   This might be a corrupted or incomplete download")
            
            return path
        else:
            print(f"✗ Not found: {path}")
    
    print("\n❌ No model file found!")
    return None

def test_model_loading():
    """Test if we can load the model with different methods"""
    print("\n" + "=" * 50)
    print("TESTING MODEL LOADING")
    print("=" * 50)
    
    model_path = check_model_file()
    if not model_path:
        print("Cannot test model loading - no model file found")
        return False
    
    # Test 1: Try ONNX Runtime
    try:
        import onnxruntime as ort
        print("✓ ONNX Runtime available")
        
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        output_names = [output.name for output in session.get_outputs()]
        
        print(f"✓ ONNX model loaded successfully")
        print(f"  Input name: {input_name}")
        print(f"  Input shape: {input_shape}")
        print(f"  Output names: {output_names}")
        
        return True
        
    except ImportError:
        print("✗ ONNX Runtime not available")
    except Exception as e:
        print(f"✗ ONNX model loading failed: {str(e)}")
    
    # Test 2: Try OpenCV DNN
    try:
        net = cv2.dnn.readNetFromONNX(model_path)
        print("✓ OpenCV DNN loaded model successfully")
        return True
    except Exception as e:
        print(f"✗ OpenCV DNN loading failed: {str(e)}")
    
    return False

def test_inference_pipeline():
    """Test the complete inference pipeline"""
    print("\n" + "=" * 50)
    print("TESTING INFERENCE PIPELINE")
    print("=" * 50)
    
    try:
        # Add the modules path
        sys.path.append('modules/object_detection')
        from object_detection import ObjectDetector
        
        print("✓ ObjectDetector imported successfully")
        
        # Initialize detector
        detector = ObjectDetector()
        print("✓ ObjectDetector initialized")
        
        # Create a test image with clear objects
        test_image = create_test_image()
        
        # Test detection
        results = detector.detect_objects_frame(test_image)
        print(f"✓ Detection completed")
        print(f"  Results: {len(results)} objects detected")
        
        if len(results) == 0:
            print("⚠️  No objects detected in test image")
            print("   This suggests the model inference is not working properly")
            
            # Test with lower confidence threshold
            print("   Testing with different confidence thresholds...")
            if hasattr(detector.detector, 'conf_threshold'):
                original_conf = detector.detector.conf_threshold
                detector.detector.conf_threshold = 0.1  # Very low threshold
                results_low = detector.detect_objects_frame(test_image)
                print(f"   Low threshold (0.1): {len(results_low)} objects")
                detector.detector.conf_threshold = original_conf
        
        return len(results) > 0
        
    except Exception as e:
        print(f"✗ Inference pipeline test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def create_test_image():
    """Create a test image with clear geometric shapes"""
    # Create a 640x640 image
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    img.fill(50)  # Dark gray background
    
    # Add some colored rectangles and circles that should be easy to detect
    cv2.rectangle(img, (100, 100), (200, 200), (255, 0, 0), -1)  # Blue rectangle
    cv2.rectangle(img, (300, 300), (400, 400), (0, 255, 0), -1)  # Green rectangle
    cv2.circle(img, (500, 150), 60, (0, 0, 255), -1)  # Red circle
    cv2.rectangle(img, (450, 350), (550, 450), (255, 255, 0), -1)  # Cyan rectangle
    
    return img

def test_with_real_image():
    """Test with a real uploaded image if available"""
    print("\n" + "=" * 50)
    print("TESTING WITH REAL IMAGE")
    print("=" * 50)
    
    # Look for uploaded images
    upload_dirs = ['uploads', 'storage/uploads']
    
    for upload_dir in upload_dirs:
        if os.path.exists(upload_dir):
            image_files = [f for f in os.listdir(upload_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if image_files:
                # Use the most recent image
                latest_image = max(image_files, 
                                 key=lambda f: os.path.getctime(os.path.join(upload_dir, f)))
                image_path = os.path.join(upload_dir, latest_image)
                
                print(f"Testing with: {image_path}")
                
                try:
                    sys.path.append('modules/object_detection')
                    from object_detection import ObjectDetector
                    
                    detector = ObjectDetector()
                    results = detector.detect_objects(image_path)
                    
                    print(f"✓ Real image test completed")
                    print(f"  Found {len(results)} objects")
                    
                    for i, result in enumerate(results):
                        print(f"  Object {i+1}: {result.get('class_name', 'unknown')} "
                              f"(confidence: {result.get('confidence', 0):.2f})")
                    
                    return len(results)
                    
                except Exception as e:
                    print(f"✗ Real image test failed: {str(e)}")
                    return 0
    
    print("No uploaded images found to test with")
    return 0

def suggest_fixes():
    """Suggest fixes based on test results"""
    print("\n" + "=" * 50)
    print("SUGGESTED FIXES")
    print("=" * 50)
    
    print("Based on the tests above, try these solutions:")
    print()
    
    print("1. DOWNLOAD PROPER MODEL:")
    print("   python -c \"from ultralytics import YOLO; YOLO('yolov8m.pt').export(format='onnx')\"")
    print("   # This downloads ~50MB yolov8m.pt and converts to ONNX")
    print()
    
    print("2. INSTALL MISSING DEPENDENCIES:")
    print("   pip install ultralytics onnxruntime")
    print()
    
    print("3. CHECK MODEL PATH IN CODE:")
    print("   Make sure ObjectDetector can find the model file")
    print("   Update the model path in object_detection.py if needed")
    print()
    
    print("4. LOWER CONFIDENCE THRESHOLD:")
    print("   Try setting confidence threshold to 0.1 or 0.2 in the model")
    print()
    
    print("5. CHECK FLASK LOGS:")
    print("   Look for error messages when processing images")
    print("   Add more logging to see what's happening")

def main():
    """Run all diagnostic tests"""
    print("OBJECT DETECTION MODEL DIAGNOSTIC")
    print("=" * 50)
    
    # Test 1: Check model file
    model_exists = check_model_file() is not None
    
    # Test 2: Test model loading
    model_loads = test_model_loading() if model_exists else False
    
    # Test 3: Test inference pipeline
    inference_works = test_inference_pipeline() if model_loads else False
    
    # Test 4: Test with real image
    real_image_detections = test_with_real_image() if inference_works else 0
    
    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)
    
    print(f"Model file exists: {'✓' if model_exists else '✗'}")
    print(f"Model loads properly: {'✓' if model_loads else '✗'}")
    print(f"Inference pipeline works: {'✓' if inference_works else '✗'}")
    print(f"Real image detections: {real_image_detections}")
    
    if not inference_works:
        suggest_fixes()
    elif real_image_detections == 0:
        print("\nThe model loads but isn't detecting objects.")
        print("This could be due to:")
        print("- Confidence threshold too high")
        print("- Image preprocessing issues")
        print("- Model not properly trained/loaded")

if __name__ == "__main__":
    main()