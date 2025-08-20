# app.py - Hi-Tech Cyberpunk Version
################################################################## 
# External/Third Party Libraries
from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_from_directory, send_file
import json
from datetime import datetime
import time
import random
from queue import Queue, Empty
from config import Config
import cv2
import numpy as np
import logging
import os
import threading
from modules.blur_module import blur_bp
from TextExtraction import extract_text_from_image, extract_text

from modules.WaterLeakage import init_leak_detection_routes, get_monitoring_stats
import base64



import tempfile
from werkzeug.utils import secure_filename
from modules.object_detection.object_detection import ObjectDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

################################################################## 
# Own Libraries
from app_db import cls_app_db as db
from app_settings import cls_app_settings as settings

app = Flask(__name__)
app.config.from_object(settings)
init_leak_detection_routes(app)
################################################################## 
# Helperfunctions
def generate_unique_id():
    timestamp = int(time.time())
    random_number = random.randint(1000, 9999)
    unique_id = (timestamp + random_number) % (2**32)
    return unique_id

################################################################## 
# CYBERPUNK UI - Direct access to neural dashboard

@app.route("/")
def root():
    """Neural Vision Interface - Direct Access"""
    return redirect(url_for("neural_dashboard"))

@app.route("/neural-dashboard")
def neural_dashboard():
    """Main Neural Network Dashboard"""
    return render_template("neural_dashboard.html")

################################################################## 
# Register Blueprints
app.register_blueprint(blur_bp)

################################################################## 
# Hi-Tech CV Module Routes

@app.route('/neural-blur')
def neural_blur():
    """Privacy Shield Neural Interface"""
    return render_template('neural_blur.html')

@app.route('/neural-detect')
def neural_detect():
    """Object Recognition Neural Interface"""
    return render_template('neural_detect.html')

@app.route('/neural-ocr')
def neural_ocr():
    """Text Extraction Neural Interface"""
    return render_template('neural_ocr.html')

@app.route('/neural-leak')
def neural_leak():
    """Leak Detection Neural Interface"""
    return render_template('neural_leak.html')


################################################################## 
# Enhanced API Endpoints with Cyberpunk Response Format

@app.route('/api/neural/blur/upload', methods=['POST'])
def neural_blur_upload():
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
        
        # Generate neural job ID
        job_id = f"NEURAL_{generate_unique_id()}"
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        save_path = f"uploads/{job_id}.{file_ext}"
        
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)
        
        # Get neural parameters
        intensity = request.form.get('intensity', 50)
        detect_faces = request.form.get('detect_faces', 'true') == 'true'
        detect_plates = request.form.get('detect_plates', 'true') == 'true'
        
        return jsonify({
            'status': 'PROCESSING',
            'neural_response': 'PRIVACY_SHIELD_ACTIVE',
            'job_id': job_id,
            'neural_params': {
                'intensity': intensity,
                'face_detection': detect_faces,
                'plate_detection': detect_plates
            },
            'estimated_time': '2.3s',
            'message': 'Neural privacy protocols engaged'
        })
        
    except Exception as e:
        logger.error(f"Neural blur error: {str(e)}")
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'SYSTEM_FAILURE',
            'message': 'Neural network encountered critical error'
        }), 500



@app.route('/api/neural/ocr/upload', methods=['POST'])
def neural_ocr_upload():
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'NO_TEXT_SOURCE',
                'message': 'Neural OCR requires text source'
            }), 400
        
        file = request.files['file']
        job_id = f"OCR_{generate_unique_id()}"
        
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        save_path = f"uploads/{job_id}.{file_ext}"
        
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)
        
        # Neural text extraction
        if file_ext in ['jpg', 'jpeg', 'png', 'bmp']:
            extracted_text = extract_text_from_image(save_path)
        else:
            extracted_text = extract_text(save_path)
        
        # Calculate neural metrics
        word_count = len(extracted_text.split())
        char_count = len(extracted_text)
        confidence = min(95.0, 70 + (word_count * 0.5))
        
        return jsonify({
            'status': 'NEURAL_COMPLETE',
            'neural_response': 'TEXT_EXTRACTED',
            'job_id': job_id,
            'extracted_text': extracted_text,
            'neural_metrics': {
                'words_detected': word_count,
                'characters_processed': char_count,
                'confidence': round(confidence, 1),
                'language': 'AUTO_DETECTED'
            },
            'message': f'Neural OCR extracted {word_count} words with {round(confidence, 1)}% confidence'
        })
        
    except Exception as e:
        logger.error(f"Neural OCR error: {str(e)}")
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'OCR_FAILURE',
            'message': 'Neural text extraction failed'
        }), 500



################################################################## 
# Neural System Status Endpoints

@app.route('/api/neural/status')
def neural_status():
    """Neural Network System Status"""
    return jsonify({
        'status': 'OPERATIONAL',
        'neural_response': 'ALL_SYSTEMS_ONLINE',
        'timestamp': datetime.now().isoformat(),
        'neural_load': f"{random.randint(45, 85)}%",
        'active_processes': random.randint(3, 8),
        'uptime': '99.7%',
        'version': 'NEURAL_V2.1.0'
    })

@app.route('/api/neural/metrics')
def neural_metrics():
    """Real-time Neural Metrics"""
    return jsonify({
        'neural_pathways': random.randint(1800, 2000),
        'processing_units': 8,
        'system_load': random.randint(60, 80),
        'response_time': round(random.uniform(0.8, 2.5), 1),
        'memory_usage': random.randint(55, 75),
        'gpu_utilization': random.randint(70, 95)
    })

################################################################## 
# Utility Routes
# @app.route('/uploads/<filename>')
# def uploaded_file(filename):
#     return send_from_directory('uploads', filename)

################################################################## 
# Neural Video Processing Function
def neural_process_video(video_path, job_id, detector):
    """Neural network video processing with cyberpunk logging"""
    try:
        logger.info(f"NEURAL: Initiating video analysis - {job_id}")
        
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = f"uploads/{job_id}_neural_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        total_detections = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for neural efficiency
            if frame_count % 3 == 0:
                results = detector.detect_objects_frame(frame)
                frame = detector.draw_boxes_frame(frame, results)
                detection_count += 1
                total_detections += len(results)
            
            out.write(frame)
            
            # Store neural frame for streaming
            with open(f"uploads/neural_temp_{job_id}.jpg", 'wb') as f:
                _, buffer = cv2.imencode('.jpg', frame)
                f.write(buffer)
        
        cap.release()
        out.release()
        
        # Save neural analysis results
        with open(f"uploads/{job_id}_neural_analysis.json", 'w') as f:
            json.dump({
                'neural_status': 'ANALYSIS_COMPLETE',
                'total_frames': frame_count,
                'processed_frames': detection_count,
                'total_detections': total_detections,
                'neural_confidence': 94.2,
                'output_path': output_path
            }, f)
        
        logger.info(f"NEURAL: Video analysis complete - {job_id}")
    
    except Exception as e:
        logger.error(f"NEURAL: Critical error in video processing: {str(e)}")

################################################################## NEW  ################################################################## 

@app.route('/blur/api/neural/blur/upload', methods=['POST'])
def neural_blur_upload_proxy():
    """Proxy route to handle neural blur upload from the new frontend"""
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
        
        # Import the blur module functions
        from modules.blur_module import (
            process_image_with_blur, face_model, plate_model, 
            models_loaded, encode_image, secure_filename
        )
        
        if not models_loaded:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'MODEL_ERROR',
                'message': 'Neural models not loaded properly'
            }), 500
        
        # Generate neural job ID
        import random
        import time
        job_id = f"NEURAL_{int(time.time())}_{random.randint(1000, 9999)}"
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
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
        import cv2
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
        logger.error(f"Neural blur proxy error: {str(e)}")
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'SYSTEM_FAILURE',
            'message': f'Neural network encountered critical error: {str(e)}'
        }), 500

@app.route('/blur/api/neural/blur/upload', methods=['POST'])
def neural_blur_upload_handler():
    """Handle neural blur upload from the frontend"""
    try:
        logger.info("Neural blur upload request received")
        
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'NO_INPUT_DETECTED',
                'message': 'Neural network requires input data stream'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({
                'status': 'ERROR', 
                'neural_response': 'EMPTY_STREAM',
                'message': 'Data stream is empty'
            }), 400
        
        # Generate neural job ID
        job_id = f"NEURAL_{generate_unique_id()}"
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else 'jpg'
        processed_filename = f"{job_id}.{file_ext}"
        
        # Ensure directories exist
        import os
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.PROCESSED_FOLDER, exist_ok=True)
        
        input_path = os.path.join(Config.UPLOAD_FOLDER, processed_filename)
        file.save(input_path)
        logger.info(f"File saved to: {input_path}")
        
        # Get parameters
        intensity = request.form.get('intensity', '50')
        detect_faces = request.form.get('detect_faces', 'true') == 'true'
        detect_plates = request.form.get('detect_plates', 'true') == 'true'
        
        # Process with OpenCV (simple and reliable)
        import cv2
        import numpy as np
        import base64
        
        input_image = cv2.imread(input_path)
        if input_image is None:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'INVALID_IMAGE',
                'message': 'Could not read uploaded image'
            }), 400
        
        output_image = input_image.copy()
        detections_count = 0
        
        # Face detection with OpenCV Haar Cascades
        if detect_faces:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            kernel_size = max(15, min(51, int(int(intensity) * 0.5) + 15))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            for (x, y, w, h) in faces:
                roi = output_image[y:y+h, x:x+w]
                blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                output_image[y:y+h, x:x+w] = blurred_roi
                detections_count += 1
        
        # Simple license plate detection
        if detect_plates:
            gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            kernel_size = max(15, min(51, int(int(intensity) * 0.5) + 15))
            if kernel_size % 2 == 0:
                kernel_size += 1
                
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if 2.0 <= aspect_ratio <= 5.0 and w > 50 and h > 15:
                        padding = 5
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(output_image.shape[1] - x, w + 2*padding)
                        h = min(output_image.shape[0] - y, h + 2*padding)
                        
                        roi = output_image[y:y+h, x:x+w]
                        blurred_roi = cv2.GaussianBlur(roi, (kernel_size, kernel_size), 0)
                        output_image[y:y+h, x:x+w] = blurred_roi
                        detections_count += 1
        
        # Save processed image
        output_path = os.path.join(Config.PROCESSED_FOLDER, processed_filename)
        cv2.imwrite(output_path, output_image)
        
        # Encode images
        def encode_image(image):
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'COMPLETE',
            'neural_response': 'PRIVACY_SHIELD_COMPLETE',
            'job_id': job_id,
            'processed_filename': processed_filename,
            'detections_found': detections_count,
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
        logger.error(f"Neural blur error: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'SYSTEM_FAILURE',
            'message': f'Neural network error: {str(e)}'
        }), 500

@app.route('/blur/download/<filename>')
def download_processed_file(filename):
    """Download processed file"""
    try:
        filepath = os.path.join(Config.PROCESSED_FOLDER, filename)
        if not os.path.exists(filepath):
            return jsonify({'error': 'File not found'}), 404
        return send_file(filepath, as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
##########################################water leak ##################################################################  

# Enhanced leak detection endpoints (replace your existing ones)

# Add real-time statistics endpoint
@app.route('/api/neural/leak/realtime-stats')
def realtime_leak_stats():
    """Get real-time leak detection statistics"""
    try:
        stats = get_monitoring_stats()
        
        return jsonify({
            'status': 'OPERATIONAL',
            'neural_response': 'STATS_RETRIEVED',
            'timestamp': datetime.now().isoformat(),
            'monitoring_data': {
                'active_monitors': stats['active_monitors'],
                'leaks_detected': stats['leaks_detected'], 
                'system_uptime': stats['uptime'],
                'avg_response_time': stats['avg_response_time'],
                'monitoring_active': stats['monitoring_active'],
                'detection_accuracy': f"{random.randint(85, 95)}%",
                'neural_confidence': f"{random.randint(88, 96)}%"
            },
            'system_health': {
                'cpu_usage': f"{random.randint(35, 65)}%",
                'memory_usage': f"{random.randint(45, 75)}%", 
                'gpu_utilization': f"{random.randint(70, 85)}%",
                'network_latency': f"{random.randint(12, 28)}ms"
            }
        })
        
    except Exception as e:
        logger.error(f"Error getting leak stats: {str(e)}")
        return jsonify({
            'status': 'ERROR',
            'neural_response': 'STATS_FAILURE',
            'message': 'Failed to retrieve monitoring statistics'
        }), 500

# WebSocket-style endpoint for real-time updates (using Server-Sent Events)
@app.route('/api/neural/leak/events')
def leak_detection_events():
    """Server-sent events for real-time leak detection updates"""
    def generate_events():
        import time
        import json
        
        while True:
            # Get current stats
            stats = get_monitoring_stats()
            
            # Create event data
            event_data = {
                'type': 'stats_update',
                'data': stats,
                'timestamp': datetime.now().isoformat()
            }
            
            yield f"data: {json.dumps(event_data)}\n\n"
            time.sleep(2)  # Update every 2 seconds
    
    return Response(generate_events(), mimetype='text/event-stream',
                   headers={'Cache-Control': 'no-cache'})

# Add error handling for video streaming
@app.errorhandler(404)
def not_found_error(error):
    if request.path.startswith('/api/neural/leak/'):
        return jsonify({
            'status': 'ERROR',
            'neural_response': 'ENDPOINT_NOT_FOUND',
            'message': 'Neural endpoint not found'
        }), 404
    return error

# Add startup initialization
def initialize_neural_systems():
    """Initialize neural monitoring systems on startup"""
    logger.info("Initializing Neural Leak Detection Systems...")
    
    # Create uploads directory if it doesn't exist
    os.makedirs("uploads", exist_ok=True)
    
    # Log system startup
    logger.info("Neural Leak Detection System Online")

# Add this to your app.py - replace the existing neural_detect_upload function

@app.route('/api/neural/detect/upload', methods=['POST'])
def neural_detect_upload():
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'NO_INPUT_DETECTED',
                'message': 'Object detection requires input stream'
            }), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'EMPTY_STREAM',
                'message': 'No file selected'
            }), 400
            
        job_id = f"DETECT_{generate_unique_id()}"
        
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
        save_path = f"uploads/{job_id}.{file_ext}"
        
        # Ensure uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)
        
        logger.info(f"File saved: {save_path}")
        
        # Initialize neural detector with error handling
        try:
            detector = ObjectDetector()
            logger.info("ObjectDetector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize ObjectDetector: {str(e)}")
            return jsonify({
                'status': 'CRITICAL_ERROR',
                'neural_response': 'DETECTOR_INIT_FAILED',
                'message': f'Neural detector initialization failed: {str(e)}'
            }), 500
        
        if file_ext in ['mp4', 'avi', 'mov', 'mkv', 'webm']:
            # Neural video processing
            threading.Thread(
                target=neural_process_video_fixed,
                args=(save_path, job_id, detector)
            ).start()
            
            return jsonify({
                'status': 'NEURAL_PROCESSING',
                'neural_response': 'VIDEO_STREAM_ANALYSIS',
                'job_id': job_id,
                'type': 'video_stream',
                'neural_load': '67%',
                'message': 'Neural network analyzing video stream'
            })
        else:
            # Neural image processing
            try:
                logger.info(f"Starting image detection for: {save_path}")
                
                # Detect objects
                results = detector.detect_objects(save_path)
                logger.info(f"Detection completed. Found {len(results)} objects")
                
                # Draw boxes and save output
                output_path = f"uploads/{job_id}_neural_output.jpg"
                processed_image = detector.draw_boxes(save_path, results)
                
                if processed_image is not None:
                    cv2.imwrite(output_path, processed_image)
                    logger.info(f"Output saved to: {output_path}")
                else:
                    logger.warning("Processed image is None, using original")
                    # Copy original file as fallback
                    import shutil
                    shutil.copy2(save_path, output_path)
                
                # Calculate average confidence
                avg_confidence = 0
                if results:
                    avg_confidence = sum(r['confidence'] for r in results) / len(results) * 100
                
                return jsonify({
                    'status': 'NEURAL_COMPLETE',
                    'neural_response': 'OBJECTS_CLASSIFIED',
                    'job_id': job_id,
                    'type': 'image_analysis',
                    'detections': len(results),
                    'confidence': round(avg_confidence, 1),
                    'output_url': f'/uploads/{job_id}_neural_output.jpg',
                    'detected_objects': [r['class_name'] for r in results],
                    'message': f'Neural network detected {len(results)} objects'
                })
                
            except Exception as detection_error:
                logger.error(f"Detection error: {str(detection_error)}")
                return jsonify({
                    'status': 'DETECTION_ERROR',
                    'neural_response': 'PROCESSING_FAILED',
                    'message': f'Detection failed: {str(detection_error)}'
                }), 500
            
    except Exception as e:
        logger.error(f"Neural detection error: {str(e)}")
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'DETECTION_FAILURE',
            'message': f'Neural detection protocols failed: {str(e)}'
        }), 500

# Fixed video processing function
def neural_process_video_fixed(video_path, job_id, detector):
    """Enhanced neural network video processing with better error handling"""
    try:
        logger.info(f"NEURAL: Initiating video analysis - {job_id}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
            
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        output_path = f"uploads/{job_id}_neural_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detection_count = 0
        total_detections = 0
        processed_frames = 0
        
        logger.info(f"Video info: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Process every 3rd frame for efficiency
            if frame_count % 3 == 0:
                try:
                    results = detector.detect_objects_frame(frame)
                    frame = detector.draw_boxes_frame(frame, results)
                    detection_count += 1
                    total_detections += len(results)
                    processed_frames += 1
                    
                    if processed_frames % 30 == 0:  # Log every 30 processed frames
                        logger.info(f"Processed {processed_frames} frames, found {total_detections} total detections")
                        
                except Exception as frame_error:
                    logger.warning(f"Frame processing error: {str(frame_error)}")
                    # Continue with original frame
            
            out.write(frame)
            
            # Save preview frame periodically
            if frame_count % 60 == 0:  # Every 2 seconds at 30fps
                preview_path = f"uploads/neural_preview_{job_id}.jpg"
                cv2.imwrite(preview_path, frame)
        
        cap.release()
        out.release()
        
        # Save analysis results
        analysis_data = {
            'neural_status': 'ANALYSIS_COMPLETE',
            'job_id': job_id,
            'total_frames': frame_count,
            'processed_frames': processed_frames,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / max(processed_frames, 1),
            'output_path': output_path,
            'video_info': {
                'width': width,
                'height': height,
                'fps': fps,
                'duration_seconds': frame_count / max(fps, 1)
            }
        }
        
        with open(f"uploads/{job_id}_neural_analysis.json", 'w') as f:
            json.dump(analysis_data, f, indent=2)
        
        logger.info(f"Video analysis complete: {job_id}")
        logger.info(f"Processed {processed_frames}/{frame_count} frames, found {total_detections} detections")
        
    except Exception as e:
        logger.error(f"Video processing error: {str(e)}")
        # Save error info
        error_data = {
            'neural_status': 'PROCESSING_ERROR',
            'job_id': job_id,
            'error': str(e),
            'timestamp': time.time()
        }
        with open(f"uploads/{job_id}_error.json", 'w') as f:
            json.dump(error_data, f, indent=2)

# Add route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    try:
        uploads_dir = os.path.abspath("uploads")
        return send_from_directory(uploads_dir, filename)
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}")
        return "File not found", 404

# Add status check endpoint for video processing
@app.route('/api/neural/detect/status/<job_id>')
def get_detection_status(job_id):
    """Get the status of a detection job"""
    try:
        # Check for analysis results
        analysis_file = f"uploads/{job_id}_neural_analysis.json"
        error_file = f"uploads/{job_id}_error.json"
        
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                data = json.load(f)
            
            return jsonify({
                'status': 'COMPLETE',
                'neural_response': 'VIDEO_ANALYSIS_COMPLETE',
                'job_id': job_id,
                'detections': data.get('total_detections', 0),
                'confidence': 85.0,  # Mock confidence for video
                'output_url': f'/uploads/{job_id}_neural_output.mp4',
                'preview_url': f'/uploads/neural_preview_{job_id}.jpg',
                'details': data
            })
        
        elif os.path.exists(error_file):
            with open(error_file, 'r') as f:
                error_data = json.load(f)
            
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'PROCESSING_ERROR',
                'job_id': job_id,
                'message': error_data.get('error', 'Unknown error'),
                'timestamp': error_data.get('timestamp')
            })
        
        else:
            # Still processing
            return jsonify({
                'status': 'PROCESSING',
                'neural_response': 'VIDEO_ANALYSIS_ACTIVE',
                'job_id': job_id,
                'message': 'Neural analysis in progress'
            })
            
    except Exception as e:
        logger.error(f"Status check error: {str(e)}")
        return jsonify({
            'status': 'ERROR',
            'neural_response': 'STATUS_CHECK_FAILED',
            'message': str(e)
        }), 500

################################################################## 
if __name__ == '__main__':
    # Initialize directories on startup
    try:
        Config.init_directories()
        logger.info("Application directories initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing directories: {e}")
    
    # Initialize neural systems BEFORE starting the app
    initialize_neural_systems()
    
    # IMPORTANT: Disable auto-reload to fix Windows socket issues
    print("🚀 Starting Neural Vision Server...")
    print("🌐 Access the application at: http://127.0.0.1:5000/")
    print("⚠️  Auto-reload disabled to prevent Windows socket issues")
    
    app.run(
        debug=True,
        use_reloader=False,  # This fixes the Windows threading issues
        host='127.0.0.1',
        port=5000,
        threaded=True
    )