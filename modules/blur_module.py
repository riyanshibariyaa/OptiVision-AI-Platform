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

# Load YOLOv5 models for face and plate detection
try:
    face_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    plate_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {e}")

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
    roi = image[y:y+h, x:x+w]
    kernel_size = calculate_kernel_size(intensity)
    blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
    image[y:y+h, x:x+w] = blurred_roi
    return image

def draw_boxes(image, detections):
    """Draw bounding boxes around detected regions"""
    image_with_boxes = image.copy()
    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if confidence > 0.4:
            cv2.rectangle(image_with_boxes, 
                         (int(x1), int(y1)), 
                         (int(x2), int(y2)), 
                         (255, 0, 0), 2)
    return image_with_boxes

def detect_and_blur(image, model, blur_intensity='25'):
    """Detect objects and apply blur with specified intensity"""
    results = model(image)
    detections = results.xyxy[0].numpy()
    
    detected_regions = []
    
    for detection in detections:
        x1, y1, x2, y2, confidence, cls = detection
        if confidence > 0.4:
            image = blur_region(image, 
                              (int(x1), int(y1), int(x2-x1), int(y2-y1)), 
                              blur_intensity)
            detected_regions.append(detection)
    
    return image, detected_regions

def encode_image(image):
    """Convert OpenCV image to base64 string"""
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

@blur_bp.route('/')
def index():
    """Render the blur module page"""
    return render_template('blur_module.html')

@blur_bp.route('/upload', methods=['POST'])
def upload_files():
    """Handle file uploads"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files uploaded'}), 400
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
            file.save(filepath)
            uploaded_files.append(filename)
            logger.info(f"File uploaded: {filename}")
    
    return jsonify({'uploaded': uploaded_files})

@blur_bp.route('/download/<filename>')
def download_file(filename):
    """Download a single processed image"""
    try:
        return send_file(
            os.path.join(Config.PROCESSED_FOLDER, filename),
            as_attachment=True,
            download_name=filename
        )
    except Exception as e:
        logger.error(f"Download error: {e}")
        return jsonify({'error': str(e)}), 404

@blur_bp.route('/download-zip')
def download_zip():
    """Download all processed images as ZIP"""
    memory_file = BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for filename in os.listdir(Config.PROCESSED_FOLDER):
            if allowed_file(filename):
                file_path = os.path.join(Config.PROCESSED_FOLDER, filename)
                zf.write(file_path, filename)
    
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='processed_images.zip'
    )

@blur_bp.route('/control-processing', methods=['POST'])
def control_processing():
    """Handle processing control commands (pause/resume/stop)"""
    command = request.json.get('command')
    session_id = request.json.get('sessionId')
    
    if session_id != processing_state['current_session']:
        return jsonify({'error': 'Invalid session ID'})
    
    if command == 'pause':
        processing_state['pause_event'].clear()
        logger.info("Processing paused")
    elif command == 'resume':
        processing_state['pause_event'].set()
        logger.info("Processing resumed")
    elif command == 'stop':
        processing_state['stop_event'].set()
        logger.info("Processing stopped")
    
    return jsonify({'status': 'success'})

@blur_bp.route('/process-stream')
def process_stream():
    """Process images and stream results back to the client."""
    blur_intensity = request.args.get('blurIntensity', '25')
    session_id = request.args.get('sessionId')
    show_boxes = request.args.get('showBoxes', 'false').lower() == 'true'  


    # Reset processing state
    processing_state['pause_event'].set()
    processing_state['stop_event'].clear()
    processing_state['current_session'] = session_id

    logger.info(f"Starting processing stream with intensity {blur_intensity}")

    def generate():
        images = [f for f in os.listdir(Config.UPLOAD_FOLDER) if allowed_file(f)]
        total_images = len(images)
        processed_count = 0

        logger.info(f"Found {total_images} images to process")

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

            input_image = cv2.imread(input_path)
            if input_image is None:
                logger.error(f"Could not read image: {filename}")
                yield f"data: {json.dumps({'error': f'Could not read image: {filename}'})}\n\n"
                continue

            try:
                output = input_image.copy()

                # First, perform detection
                face_results = face_model(input_image.copy())
                face_detections = face_results.xyxy[0].numpy()
                
                plate_results = plate_model(input_image.copy())
                plate_detections = plate_results.xyxy[0].numpy()
                
                # Apply blur
                for detection in face_detections:
                    x1, y1, x2, y2, confidence, cls = detection
                    if confidence > 0.4:
                        output = blur_region(output, 
                                          (int(x1), int(y1), int(x2-x1), int(y2-y1)), 
                                          blur_intensity)
                
                for detection in plate_detections:
                    x1, y1, x2, y2, confidence, cls = detection
                    if confidence > 0.4:
                        output = blur_region(output, 
                                          (int(x1), int(y1), int(x2-x1), int(y2-y1)), 
                                          blur_intensity)
                
                # If show_boxes is enabled, draw the detection boxes
                if show_boxes:
                    # Combine all detections for drawing
                    all_detections = np.vstack([face_detections, plate_detections]) if len(face_detections) > 0 and len(plate_detections) > 0 else \
                                   face_detections if len(face_detections) > 0 else \
                                   plate_detections if len(plate_detections) > 0 else np.array([])
                    
                    if len(all_detections) > 0:
                        output = draw_boxes(output, all_detections)

                        
                output, face_detections = detect_and_blur(output, face_model, blur_intensity)
                output, plate_detections = detect_and_blur(output, plate_model, blur_intensity)

                output_path = os.path.join(Config.PROCESSED_FOLDER, filename)
                cv2.imwrite(output_path, output)

                progress_data = {
                    'filename': filename,
                    'processed': processed_count + 1,
                    'total': total_images,
                    'input_image': f"data:image/jpeg;base64,{encode_image(input_image)}",
                    'output_image': f"data:image/jpeg;base64,{encode_image(output)}"
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

# Additional utility routes
@blur_bp.route('/status')
def status():
    """Return module status information"""
    return jsonify({
        'status': 'active',
        'models_loaded': face_model is not None and plate_model is not None,
        'upload_directory': Config.UPLOAD_FOLDER,
        'processed_directory': Config.PROCESSED_FOLDER,
        'file_count': {
            'uploads': len([f for f in os.listdir(Config.UPLOAD_FOLDER) if allowed_file(f)]),
            'processed': len([f for f in os.listdir(Config.PROCESSED_FOLDER) if allowed_file(f)])
        }
    })

@blur_bp.route('/clean/<folder>')
def clean_folder(folder):
    """Clean upload or processed folders"""
    if folder not in ['uploads', 'processed']:
        return jsonify({'error': 'Invalid folder specified'}), 400
    
    target_folder = Config.UPLOAD_FOLDER if folder == 'uploads' else Config.PROCESSED_FOLDER
    
    count = 0
    for filename in os.listdir(target_folder):
        if allowed_file(filename):
            file_path = os.path.join(target_folder, filename)
            os.remove(file_path)
            count += 1
    
    logger.info(f"Cleaned {count} files from {folder} folder")
    return jsonify({'status': 'success', 'files_removed': count})

@blur_bp.route('/fetch-local')
def fetch_local_images():
    """Fetch images from the local repository folder"""
    try:
        local_images = [f for f in os.listdir(Config.LOCAL_FOLDER) if allowed_file(f)]
        
        # Copy local images to the upload folder for processing
        for filename in local_images:
            source_path = os.path.join(Config.LOCAL_FOLDER, filename)
            dest_path = os.path.join(Config.UPLOAD_FOLDER, filename)
            if os.path.exists(source_path) and not os.path.exists(dest_path):
                import shutil
                shutil.copy2(source_path, dest_path)
        
        return jsonify({'images': local_images})
    except Exception as e:
        logger.error(f"Error fetching local images: {e}")
        return jsonify({'error': str(e), 'images': []}), 500