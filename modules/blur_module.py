from flask import Blueprint, render_template, request, jsonify, Response, send_file
import os
import torch
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
from ultralytics import YOLO
import requests

# Import configuration
from config import Config

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

def initialize_models():
    """Initialize YOLOv8 models for face and license plate detection"""
    global face_model, plate_model
    
    try:
        # Initialize YOLOv8 models
        # For face detection, we'll use a pre-trained model and filter for 'person' class
        face_model = YOLO('yolov8n.pt')
        
        # For license plate detection, we'll use the same model and look for specific patterns
        # You can replace this with a custom trained license plate model
        plate_model = YOLO('yolov8n.pt')
        
        logger.info("Models initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing models: {e}")
        try:
            # Fallback to YOLOv5 if YOLOv8 fails
            face_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            plate_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            logger.info("Fallback to YOLOv5 models successful")
            return True
        except Exception as e2:
            logger.error(f"Error loading fallback models: {e2}")
            return False

# Initialize models on module load
models_loaded = initialize_models()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def calculate_kernel_size(intensity):
    """Calculate Gaussian blur kernel size based on intensity percentage"""
    base_kernel = 51  # Maximum kernel size (100% intensity)
    min_kernel = 15   # Minimum kernel size (25% intensity)
    
    kernel_size = int(min_kernel + (base_kernel - min_kernel) * (int(intensity) / 100))
    if kernel_size % 2 == 0:
        kernel_size += 1
    return kernel_size

def blur_region(image, region, intensity='25'):
    """Apply Gaussian blur to a specific region with given intensity"""
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

def draw_boxes(image, detections):
    """Draw bounding boxes around detected regions"""
    image_with_boxes = image.copy()
    for detection in detections:
        if len(detection) >= 6:
            x1, y1, x2, y2, confidence, cls = detection[:6]
        else:
            x1, y1, x2, y2, confidence = detection[:5]
            
        if confidence > 0.4:
            cv2.rectangle(image_with_boxes, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (0, 255, 0), 2)
            # Add confidence text
            cv2.putText(image_with_boxes, 
                       f'{confidence:.2f}', 
                       (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
    return image_with_boxes

def detect_faces(image, model, confidence_threshold=0.3):
    """Detect faces in image using YOLO model"""
    detections = []
    try:
        # Use YOLOv8 or YOLOv5 based on model type
        if hasattr(model, 'predict'):
            # YOLOv8
            results = model.predict(image, conf=confidence_threshold, verbose=False)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Filter for person class (class 0 in COCO dataset)
                        if int(box.cls) == 0:  # person class
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            
                            # Estimate face region (upper portion of person detection)
                            face_height = (y2 - y1) * 0.3
                            face_y2 = y1 + face_height
                            
                            detections.append([x1, y1, x2, face_y2, conf, 0])
        else:
            # YOLOv5
            results = model(image)
            pred = results.xyxy[0].cpu().numpy()
            
            for detection in pred:
                x1, y1, x2, y2, conf, cls = detection
                # Filter for person class
                if int(cls) == 0 and conf > confidence_threshold:
                    # Estimate face region
                    face_height = (y2 - y1) * 0.3
                    face_y2 = y1 + face_height
                    detections.append([x1, y1, x2, face_y2, conf, cls])
                    
    except Exception as e:
        logger.error(f"Error in face detection: {e}")
        
    return detections

def detect_license_plates(image, model, confidence_threshold=0.3):
    """Detect license plates in image"""
    detections = []
    try:
        # Convert image to grayscale for plate detection preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use edge detection to find potential plate regions
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Calculate contour area and bounding rectangle
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # License plates typically have aspect ratio between 2:1 and 5:1
                if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                    # Add some padding
                    padding = 5
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(image.shape[1] - x, w + 2*padding)
                    h = min(image.shape[0] - y, h + 2*padding)
                    
                    detections.append([x, y, x+w, y+h, 0.7, 999])  # Use custom class 999 for plates
                    
    except Exception as e:
        logger.error(f"Error in license plate detection: {e}")
        
    return detections

def process_image_with_blur(image, face_model, plate_model, blur_intensity='25', detect_faces_flag=True, detect_plates_flag=True, show_boxes=False):
    """Process image with face and plate detection and blurring"""
    output = image.copy()
    all_detections = []
    
    try:
        # Face detection
        if detect_faces_flag and face_model:
            face_detections = detect_faces(image, face_model)
            
            # Apply blur to faces
            for detection in face_detections:
                x1, y1, x2, y2, confidence, cls = detection
                if confidence > 0.3:
                    output = blur_region(output, 
                                      (int(x1), int(y1), int(x2-x1), int(y2-y1)), 
                                      blur_intensity)
                    all_detections.append(detection)
        
        # License plate detection
        if detect_plates_flag and plate_model:
            plate_detections = detect_license_plates(image, plate_model)
            
            # Apply blur to plates
            for detection in plate_detections:
                x1, y1, x2, y2, confidence, cls = detection
                output = blur_region(output, 
                                  (int(x1), int(y1), int(x2-x1), int(y2-y1)), 
                                  blur_intensity)
                all_detections.append(detection)
        
        # Draw boxes if requested
        if show_boxes and len(all_detections) > 0:
            output = draw_boxes(output, all_detections)
            
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        
    return output, all_detections

def encode_image(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Routes

@blur_bp.route('/')
def index():
    """Render the blur module page"""
    return render_template('neural_blur.html')

@blur_bp.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    
    # Ensure upload directory exists
    os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            uploaded_files.append(filename)
            logger.info(f"File uploaded: {filename}")
    
    return jsonify({'uploaded': uploaded_files})

@blur_bp.route('/process-stream')
def process_stream():
    """Stream processing of uploaded images with real-time updates"""
    if not models_loaded:
        return jsonify({'error': 'Models not loaded properly'}), 500
        
    blur_intensity = request.args.get('blurIntensity', '25')
    session_id = request.args.get('sessionId')
    show_boxes = request.args.get('showBoxes', 'false').lower() == 'true'
    detect_faces_flag = request.args.get('detectFaces', 'true').lower() == 'true'
    detect_plates_flag = request.args.get('detectPlates', 'true').lower() == 'true'

    # Reset processing state
    processing_state['pause_event'].set()
    processing_state['stop_event'].clear()
    processing_state['current_session'] = session_id

    logger.info(f"Starting processing stream with intensity {blur_intensity}")

    def generate():
        if not os.path.exists(Config.UPLOAD_FOLDER):
            yield f"data: {json.dumps({'error': 'Upload folder not found'})}\n\n"
            return
            
        images = [f for f in os.listdir(Config.UPLOAD_FOLDER) if allowed_file(f)]
        total_images = len(images)
        processed_count = 0

        logger.info(f"Found {total_images} images to process")
        
        if total_images == 0:
            yield f"data: {json.dumps({'error': 'No images found to process'})}\n\n"
            return

        # Ensure processed folder exists
        os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True)

        for filename in images:
            if processing_state['stop_event'].is_set():
                logger.info("Processing stopped by user")
                yield f"data: {json.dumps({'processing_stopped': True})}\n\n"
                return

            processing_state['pause_event'].wait()

            input_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            if not os.path.exists(input_path):
                logger.warning(f"File not found: {input_path}")
                continue

            try:
                input_image = cv2.imread(input_path)
                if input_image is None:
                    logger.error(f"Could not read image: {filename}")
                    yield f"data: {json.dumps({'error': f'Could not read image: {filename}'})}\n\n"
                    continue

                # Process image with blur
                output, detections = process_image_with_blur(
                    input_image, 
                    face_model, 
                    plate_model, 
                    blur_intensity,
                    detect_faces_flag,
                    detect_plates_flag,
                    show_boxes
                )

                # Save processed image
                output_path = os.path.join(Config.PROCESSED_FOLDER, filename)
                cv2.imwrite(output_path, output)

                progress_data = {
                    'filename': filename,
                    'processed': processed_count + 1,
                    'total': total_images,
                    'input_image': f"data:image/jpeg;base64,{encode_image(input_image)}",
                    'output_image': f"data:image/jpeg;base64,{encode_image(output)}",
                    'detections_count': len(detections)
                }

                processed_count += 1
                logger.info(f"Processed {processed_count}/{total_images}: {filename}")
                yield f"data: {json.dumps(progress_data)}\n\n"
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {e}")
                yield f"data: {json.dumps({'error': f'Error processing {filename}: {str(e)}'})}\n\n"

        logger.info("Processing completed")
        yield f"data: {json.dumps({'processing_complete': True})}\n\n"

    return Response(generate(), mimetype='text/event-stream')

# API endpoint for neural_blur.html integration
@blur_bp.route('/api/neural/blur/upload', methods=['POST'])
def neural_blur_upload():
    """Handle single file upload from neural_blur.html interface"""
    try:
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
        
        if not models_loaded:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'MODEL_ERROR',
                'message': 'Neural models not loaded properly'
            }), 500
        
        # Generate neural job ID
        import random
        job_id = f"NEURAL_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        processed_filename = f"{job_id}.{file_ext}"
        
        # Ensure directories exist
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True)
        
        input_path = os.path.join(Config.UPLOAD_FOLDER, processed_filename)
        file.save(input_path)
        
        # Get neural parameters
        intensity = request.form.get('intensity', '50')
        detect_faces = request.form.get('detect_faces', 'true') == 'true'
        detect_plates = request.form.get('detect_plates', 'true') == 'true'
        
        # Process the image
        input_image = cv2.imread(input_path)
        if input_image is None:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'INVALID_IMAGE',
                'message': 'Could not read uploaded image'
            }), 400
        
        # Apply privacy shield processing
        output_image, detections = process_image_with_blur(
            input_image,
            face_model,
            plate_model,
            intensity,
            detect_faces,
            detect_plates,
            False
        )
        
        # Save processed image
        output_path = os.path.join(Config.PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(output_path, output_image)
        
        return jsonify({
            'status': 'COMPLETE',
            'neural_response': 'PRIVACY_SHIELD_COMPLETE',
            'job_id': job_id,
            'processed_filename': processed_filename,
            'detections_found': len(detections),
            'neural_params': {
                'intensity': intensity,
                'face_detection': detect_faces,
                'plate_detection': detect_plates
            },
            'input_image': f"data:image/jpeg;base64,{encode_image(input_image)}",
            'output_image': f"data:image/jpeg;base64,{encode_image(output_image)}",
            'download_url': f'/blur/download/{processed_filename}',
            'message': 'Neural privacy protocols successfully applied'
        })
        
    except Exception as e:
        logger.error(f"Neural blur error: {str(e)}")
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'SYSTEM_FAILURE',
            'message': f'Neural network encountered critical error: {str(e)}'
        }), 500

@blur_bp.route('/download/<filename>')
def download_file(filename):
    """Download a single processed image"""
    try:
        filepath = os.path.join(Config.PROCESSED_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
            
        return send_file(
            filepath,
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': f'Download failed: {str(e)}'}), 500

@blur_bp.route('/status')
def status():
    """Return module status information"""
    return jsonify({
        'status': 'active',
        'models_loaded': models_loaded,
        'upload_directory': Config.UPLOAD_FOLDER,
        'processed_directory': Config.PROCESSED_FOLDER,
        'file_count': {
            'uploads': len([f for f in os.listdir(Config.UPLOAD_FOLDER) if allowed_file(f)]) if os.path.exists(Config.UPLOAD_FOLDER) else 0,
            'processed': len([f for f in os.listdir(Config.PROCESSED_FOLDER) if allowed_file(f)]) if os.path.exists(Config.PROCESSED_FOLDER) else 0
        }
    })

@blur_bp.route('/control-processing', methods=['POST'])
def control_processing():
    """Control processing state (pause/resume/stop)"""
    command = request.json.get('command')
    
    if command == 'pause':
        processing_state['pause_event'].clear()
    elif command == 'resume':
        processing_state['pause_event'].set()
    elif command == 'stop':
        processing_state['stop_event'].set()
        processing_state['pause_event'].set()
    
    return jsonify({'status': 'success', 'command': command})

@blur_bp.route('/clean', methods=['POST'])
def clean_directories():
    """Clean upload and processed directories"""
    try:
        # Clean upload directory
        if os.path.exists(Config.UPLOAD_FOLDER):
            for filename in os.listdir(Config.UPLOAD_FOLDER):
                if allowed_file(filename):
                    os.remove(os.path.join(Config.UPLOAD_FOLDER, filename))
        
        # Clean processed directory
        if os.path.exists(Config.PROCESSED_FOLDER):
            for filename in os.listdir(Config.PROCESSED_FOLDER):
                if allowed_file(filename):
                    os.remove(os.path.join(Config.PROCESSED_FOLDER, filename))
        
        return jsonify({'status': 'success', 'message': 'Directories cleaned'})
    except Exception as e:
        logger.error(f"Clean error: {str(e)}")
        return jsonify({'error': f'Clean failed: {str(e)}'}), 500