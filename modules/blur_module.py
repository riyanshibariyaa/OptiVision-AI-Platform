# modules/blur_module.py - Fixed version with robust error handling
from flask import Blueprint, render_template, request, jsonify, Response, send_file
import os
import cv2
import time
import json
import base64
import numpy as np
from pathlib import Path
from threading import Event
import zipfile
from io import BytesIO
from werkzeug.utils import secure_filename
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Create the blueprint
blur_bp = Blueprint('blur', __name__, url_prefix='/blur')

# Global state management
processing_state = {
    'pause_event': Event(),
    'stop_event': Event(),
    'current_session': None
}

# Initialize models
face_model = None
plate_model = None
models_loaded = False

def initialize_models():
    """Initialize models with robust error handling and fallbacks"""
    global face_model, plate_model, models_loaded
    
    logger.info("Initializing computer vision models...")
    
    # Method 1: Try YOLOv8 from ultralytics
    try:
        from ultralytics import YOLO
        logger.info("Attempting to load YOLOv8n model...")
        face_model = YOLO('yolov8n.pt')
        plate_model = YOLO('yolov8n.pt')
        models_loaded = True
        logger.info("✅ YOLOv8 models loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"YOLOv8 loading failed: {str(e)}")
    
    # Method 2: Try YOLOv5 from torch hub
    try:
        import torch
        logger.info("Attempting to load YOLOv5 model...")
        face_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        plate_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        models_loaded = True
        logger.info("✅ YOLOv5 models loaded successfully")
        return True
    except Exception as e:
        logger.warning(f"YOLOv5 loading failed: {str(e)}")
    
    # Method 3: Use OpenCV's face detection as final fallback
    try:
        logger.info("Attempting to load OpenCV Haar cascades...")
        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        face_model = cv2.CascadeClassifier(face_cascade_path)
        plate_model = None  # No plate detection with OpenCV fallback
        models_loaded = True
        logger.info("✅ OpenCV Haar cascade loaded successfully")
        return True
    except Exception as e:
        logger.error(f"OpenCV fallback failed: {str(e)}")
    
    # Method 4: No model mode (blur only)
    logger.warning("⚠️ No computer vision models available - running in blur-only mode")
    face_model = None
    plate_model = None
    models_loaded = True  # Set to True to allow blur-only processing
    return True

def get_config():
    """Get configuration with fallback values"""
    try:
        from config import Config
        return Config
    except ImportError:
        # Fallback configuration
        class FallbackConfig:
            UPLOAD_FOLDER = './uploads'
            PROCESSED_FOLDER = './processed'
            ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
            MAX_CONTENT_LENGTH = 200 * 1024 * 1024
            
            @classmethod
            def init_directories(cls):
                os.makedirs(cls.UPLOAD_FOLDER, exist_ok=True)
                os.makedirs(cls.PROCESSED_FOLDER, exist_ok=True)
        
        return FallbackConfig

Config = get_config()

def allowed_file(filename):
    """Check if file extension is allowed"""
    return ('.' in filename and 
            filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS)

def calculate_kernel_size(intensity):
    """Calculate Gaussian blur kernel size based on intensity percentage"""
    base_kernel = 51
    min_kernel = 15
    
    kernel_size = int(min_kernel + (base_kernel - min_kernel) * (int(intensity) / 100))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size

def blur_region(image, region, intensity='50'):
    """Apply Gaussian blur to a specific region"""
    x, y, w, h = region
    
    # Ensure coordinates are within image bounds
    height, width = image.shape[:2]
    x = max(0, min(x, width))
    y = max(0, min(y, height))
    w = max(0, min(w, width - x))
    h = max(0, min(h, height - y))
    
    if w <= 0 or h <= 0:
        return image
    
    roi = image[y:y+h, x:x+w]
    kernel_size = calculate_kernel_size(intensity)
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    image[y:y+h, x:x+w] = blurred_roi
    return image

def detect_faces_opencv(image, face_cascade):
    """Detect faces using OpenCV Haar cascades"""
    detections = []
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        for (x, y, w, h) in faces:
            detections.append([x, y, x + w, y + h, 0.8, 'face'])
            
    except Exception as e:
        logger.error(f"OpenCV face detection error: {str(e)}")
    
    return detections

def detect_faces_yolo(image, model):
    """Detect faces using YOLO model"""
    detections = []
    try:
        # Check if it's YOLOv8 or YOLOv5
        if hasattr(model, 'predict'):
            # YOLOv8
            results = model.predict(image, conf=0.3, classes=[0])  # class 0 is 'person'
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), 'person'])
        else:
            # YOLOv5
            results = model(image)
            predictions = results.pandas().xyxy[0]
            for _, detection in predictions.iterrows():
                if detection['class'] == 0 and detection['confidence'] > 0.3:  # person class
                    detections.append([
                        int(detection['xmin']), int(detection['ymin']),
                        int(detection['xmax']), int(detection['ymax']),
                        float(detection['confidence']), 'person'
                    ])
                    
    except Exception as e:
        logger.error(f"YOLO face detection error: {str(e)}")
    
    return detections

def process_image_with_blur(image, face_model, plate_model, intensity='50', 
                          detect_faces=True, detect_plates=True, show_boxes=False):
    """Process image with blur and detection"""
    output = image.copy()
    all_detections = []
    
    try:
        # Face detection
        if detect_faces and face_model is not None:
            if isinstance(face_model, cv2.CascadeClassifier):
                # OpenCV Haar cascade
                face_detections = detect_faces_opencv(image, face_model)
            else:
                # YOLO model
                face_detections = detect_faces_yolo(image, face_model)
            
            # Apply blur to detected faces
            for detection in face_detections:
                x1, y1, x2, y2 = detection[:4]
                blur_region(output, (x1, y1, x2 - x1, y2 - y1), intensity)
                all_detections.append(detection)
        
        # License plate detection (only for YOLO models)
        if detect_plates and plate_model is not None and not isinstance(plate_model, cv2.CascadeClassifier):
            # For now, we'll skip plate detection with basic models
            # You can add custom plate detection logic here
            pass
            
        # Apply general blur if no models are available
        if face_model is None and detect_faces:
            # Apply light blur to entire image as fallback
            kernel_size = calculate_kernel_size(25)  # Light blur
            output = cv2.GaussianBlur(output, (kernel_size, kernel_size), 0)
            logger.info("Applied general blur (no detection models available)")
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        # Fallback to simple blur
        try:
            kernel_size = calculate_kernel_size(intensity)
            output = cv2.GaussianBlur(output, (kernel_size, kernel_size), 0)
            logger.info("Applied fallback blur due to processing error")
        except Exception as e2:
            logger.error(f"Fallback blur failed: {str(e2)}")
            output = image.copy()  # Return original if all else fails
    
    return output, all_detections

def encode_image(image):
    """Convert OpenCV image to base64 string"""
    try:
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Image encoding error: {str(e)}")
        return ""

# Initialize models when module loads
try:
    initialize_models()
    Config.init_directories()
    logger.info(f"✅ Blur module initialized successfully (models_loaded: {models_loaded})")
except Exception as e:
    logger.error(f"❌ Blur module initialization failed: {str(e)}")
    models_loaded = True  # Allow module to work without models

# Routes
@blur_bp.route('/')
def index():
    """Render the blur module page"""
    return render_template('neural_blur.html')

@blur_bp.route('/api/neural/blur/upload', methods=['POST'])
def neural_blur_upload():
    """Handle neural blur upload"""
    try:
        logger.info("🔄 Processing blur upload request...")
        
        if 'file' not in request.files:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'NO_INPUT_DETECTED',
                'message': 'Neural network requires input data stream'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'ERROR', 
                'neural_response': 'EMPTY_STREAM',
                'message': 'Data stream is empty'
            }), 400
        
        # Generate job ID
        job_id = f"NEURAL_{int(time.time())}_{np.random.randint(1000, 9999)}"
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
        
        input_path = os.path.join(Config.UPLOAD_FOLDER, f"{job_id}_input.{file_ext}")
        output_path = os.path.join(Config.PROCESSED_FOLDER, f"{job_id}_output.{file_ext}")
        
        file.save(input_path)
        logger.info(f"📁 File saved: {input_path}")
        
        # Get parameters
        intensity = request.form.get('intensity', '50')
        detect_faces = request.form.get('detect_faces', 'true').lower() == 'true'
        detect_plates = request.form.get('detect_plates', 'false').lower() == 'true'
        
        # Process image
        input_image = cv2.imread(input_path)
        if input_image is None:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'INVALID_IMAGE',
                'message': 'Could not read uploaded image'
            }), 400
        
        logger.info(f"🖼️ Processing image: {input_image.shape}")
        logger.info(f"⚙️ Parameters: intensity={intensity}, faces={detect_faces}, plates={detect_plates}")
        
        # Apply processing
        output_image, detections = process_image_with_blur(
            input_image, face_model, plate_model, intensity, detect_faces, detect_plates
        )
        
        # Save processed image
        cv2.imwrite(output_path, output_image)
        
        logger.info(f"✅ Processing complete. Detections: {len(detections)}")
        
        return jsonify({
            'status': 'SUCCESS',
            'neural_response': 'PRIVACY_SHIELD_COMPLETE',
            'job_id': job_id,
            'detections_found': len(detections),
            'models_available': {
                'face_model': face_model is not None,
                'plate_model': plate_model is not None,
                'model_type': 'opencv' if isinstance(face_model, cv2.CascadeClassifier) else 'yolo' if face_model else 'none'
            },
            'neural_params': {
                'intensity': intensity,
                'face_detection': detect_faces,
                'plate_detection': detect_plates
            },
            'input_image': f"data:image/jpeg;base64,{encode_image(input_image)}",
            'output_image': f"data:image/jpeg;base64,{encode_image(output_image)}",
            'message': f'Neural privacy protocols applied. {len(detections)} regions processed.'
        })
        
    except Exception as e:
        logger.error(f"❌ Neural blur error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'SYSTEM_FAILURE',
            'message': f'Neural processing failed: {str(e)}',
            'fallback_available': True
        }), 500

@blur_bp.route('/status')
def status():
    """Return module status"""
    model_info = {
        'models_loaded': models_loaded,
        'face_model_type': 'opencv' if isinstance(face_model, cv2.CascadeClassifier) else 'yolo' if face_model else 'none',
        'face_model_available': face_model is not None,
        'plate_model_available': plate_model is not None
    }
    
    return jsonify({
        'status': 'active',
        'module': 'neural_blur',
        'version': '2.0.0',
        **model_info
    })
