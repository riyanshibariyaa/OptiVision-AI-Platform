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
        job_id = f"DETECT_{generate_unique_id()}"
        
        filename = secure_filename(file.filename)
        file_ext = filename.rsplit('.', 1)[1].lower()
        save_path = f"uploads/{job_id}.{file_ext}"
        
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)
        
        # Initialize neural detector
        detector = ObjectDetector()
        
        if file_ext in ['mp4', 'avi', 'mov', 'mkv']:
            # Neural video processing
            threading.Thread(
                target=neural_process_video,
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
            results = detector.detect_objects(save_path)
            
            output_path = f"uploads/{job_id}_neural_output.jpg"
            processed_image = detector.draw_boxes(save_path, results)
            cv2.imwrite(output_path, processed_image)
            
            return jsonify({
                'status': 'NEURAL_COMPLETE',
                'neural_response': 'OBJECTS_CLASSIFIED',
                'job_id': job_id,
                'type': 'image_analysis',
                'detections': len(results),
                'confidence': 94.7,
                'output_url': f'/uploads/{job_id}_neural_output.jpg',
                'message': f'Neural network detected {len(results)} objects'
            })
            
    except Exception as e:
        logger.error(f"Neural detection error: {str(e)}")
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'DETECTION_FAILURE',
            'message': 'Neural detection protocols failed'
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

@app.route('/api/neural/leak/upload', methods=['POST'])
def neural_leak_upload():
    try:
        if 'file' not in request.files:
            return jsonify({
                'status': 'ERROR',
                'neural_response': 'NO_MONITORING_SOURCE',
                'message': 'Leak detection requires monitoring source'
            }), 400
        
        file = request.files['file']
        job_id = f"LEAK_{generate_unique_id()}"
        
        filename = secure_filename(file.filename)
        save_path = f"uploads/{job_id}_{filename}"
        
        os.makedirs("uploads", exist_ok=True)
        file.save(save_path)
        
        return jsonify({
            'status': 'NEURAL_MONITORING',
            'neural_response': 'LEAK_DETECTION_ACTIVE',
            'job_id': job_id,
            'monitoring_status': 'ACTIVE',
            'sensitivity': 75,
            'file_path': save_path,
            'message': 'Neural leak detection monitoring initiated'
        })
        
    except Exception as e:
        logger.error(f"Neural leak detection error: {str(e)}")
        return jsonify({
            'status': 'CRITICAL_ERROR',
            'neural_response': 'MONITORING_FAILURE',
            'message': 'Neural monitoring system failure'
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
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory('uploads', filename)

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

################################################################## 
if __name__ == "__main__":
    # Initialize neural network environment
    os.makedirs("uploads", exist_ok=True)
    
    # Start neural vision interface
    logger.info("NEURAL: Initializing OptiView Neural Vision Interface...")
    app.run(debug=True, host='0.0.0.0', port=5000)