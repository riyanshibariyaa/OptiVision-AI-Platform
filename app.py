# app.py
################################################################## 
# External/Third Party Libraries
from pymongo import MongoClient
from flask import Flask, render_template, request, jsonify, redirect, url_for, Response, send_from_directory,  send_file
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity, verify_jwt_in_request
import json
import bcrypt
from werkzeug.security import check_password_hash
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
jwt = JWTManager(app)

################################################################## 
# Helperfunctions
def generate_unique_id():
    # Get the current timestamp in seconds (using int32 range)
    timestamp = int(time.time())  # This will give a 32-bit integer timestamp
    
    # Add a random component to avoid collisions
    random_number = random.randint(1000, 9999)
    
    # Combine the timestamp and random number, and ensure it's within int32 range
    unique_id = (timestamp + random_number) % (2**32)  # Ensures it fits in a 32-bit range
    
    return unique_id

################################################################## 
# Root route - redirect to home if authenticated, else go to login
@app.route("/")
def root():
    try:
        # Attempt to verify JWT token without enforcing it
        verify_jwt_in_request(optional = True)
        if get_jwt_identity():
            return redirect(url_for("home"))
    except:
        pass
    return render_template("login.html")

################################################################## 
# Authentication route
@app.route("/login", methods=["POST"])
def login():
    req_data = request.json
    user_record = db.coll_fw_users.find_one({"user_id": req_data["UserId"].lower()})

    if user_record and bcrypt.checkpw(req_data["Password"].encode('utf-8'), user_record["password"]):
        token = create_access_token(identity=req_data["UserId"])
        return jsonify(access_token = token), 200
    return jsonify(message = "Invalid credentials"), 401

################################################################## 
# Home route
@app.route("/home")
def home():
    return render_template("home.html")

# Function for dynamic views
@app.route('/home/<string:ViewName>', methods=['GET', 'POST'])
def views(ViewName):
    return render_template(f"{ViewName}.html")

################################################################## 
# Get menus based on user rights for a specified unit
@app.route("/get_menus", methods=["POST"])
@jwt_required()
def get_menus():
    current_user = get_jwt_identity()
    data = request.get_json()  # Get the JSON data from the POST request
    print(data)
    selected_unit_id = int(data.get("GlbSelectedUnitId"))  # Extract the selected unit ID from the POST data

    if not selected_unit_id:
        return jsonify({"error": "No unit selected"}), 400

    user = db.coll_fw_users.find_one({"user_id": current_user})
    
    if user:
        accessible_units = user.get("accessible_units", [])
        menu_ids = set()

        # Find menus for the selected unit
        for unit in accessible_units:
            if unit.get("unit_id") == selected_unit_id:
                menu_ids.update(unit.get("menus", []))
        print(menu_ids)
        if menu_ids:
            # Fetch the menu details for these menu IDs
            menus = list(db.coll_fw_menus.find({"_id": {"$in": list(menu_ids)}}))
            return jsonify(menus), 200
        else:
            return jsonify({"error": "No menus found for this unit"}), 404

    return jsonify({"error": "User not found"}), 404

################################################################## 
# Get units based on user rights    
@app.route("/get_units", methods=["POST"])
@jwt_required()
def get_units():
    current_user = get_jwt_identity()
    user = db.coll_fw_users.find_one({"user_id": current_user})
    if user:
        accessible_units = user.get("accessible_units", [])

        # Extract the unit IDs that the user has access to
        unit_ids = [unit.get("unit_id") for unit in accessible_units if unit.get("unit_id")]

        # Fetch the unit details for these unit IDs
        units = list(db.coll_fw_units.find({"_id": {"$in": unit_ids}}))
        return units
    return jsonify([]), 200

################################################################## 
# Units Master
@app.route('/AllUnitCount', methods=['POST'])
@jwt_required()
def get_units_count():
    ReturnData = {
        "Data": [],
        "ErrorMessage": "",
        "HasError": False,
        "HasRows": False,
        "RowsAffected": 0,
        "InsertId": 0
    }

    units_collection = db.coll_fw_units

    Params = request.get_json()
    search_text = Params[0]

    query = {"is_deleted": False}  # Base query to exclude deleted records

    if search_text:
        search_regex = {"$regex": search_text, "$options": "i"}  # Case-insensitive search
        query["$or"] = [
            {"name": search_regex},
            {"location": search_regex},
            {"address": search_regex}
        ]
    
    count = units_collection.count_documents(query)

    ReturnData["HasRows"] = count > 0
    ReturnData["Data"] = {"TotalRecords": count}
    ReturnData["RowsAffected"] = count

    return jsonify(ReturnData), 200

@app.route('/SelectAllUnits', methods=['POST'])
@jwt_required()
def get_all_units():
    ReturnData = {
        "Data": [],
        "ErrorMessage": "",
        "HasError": False,
        "HasRows": False,
        "RowsAffected": 0,
        "InsertId": 0
    }

    units_collection = db.coll_fw_units

    # Get the parameters from the request
    Params = request.get_json()
    search_text = Params[0]
    sort_field = Params[1]
    sort_direction = Params[2]
    page_offset = int(Params[3])
    page_size = int(Params[4])

    query = {}

    # Add search functionality if search text is provided
    if search_text:
        search_regex = {"$regex": search_text, "$options": "i"}  # Case-insensitive search
        query["$or"] = [
            {"name": search_regex},
            {"location": search_regex},
            {"address": search_regex}
        ]


    # Sorting
    sort_order = 1 if sort_direction.lower() == 'asc' else -1
    sort = [(sort_field, sort_order)] if sort_field else [("_id", -1)]  # Default to sorting by _id

    # Pagination (skip and limit)
    units_cursor = units_collection.find(query).sort(sort).skip(page_offset).limit(page_size)
    
    units = list(units_cursor)
    total_count = units_collection.count_documents(query)

    # Prepare the return data
    ReturnData["HasRows"] = total_count > 0
    ReturnData["Data"] = units
    ReturnData["RowsAffected"] = len(units)
    ReturnData["TotalRecords"] = total_count


    print(type(ReturnData))
    return jsonify(ReturnData), 200

@app.route('/SelectUnit', methods=['POST'])
@jwt_required()
def select_unit():
    ReturnData = {
        "Data": [],
        "ErrorMessage": "",
        "HasError": False,
        "HasRows": False,
        "RowsAffected": 0,
        "InsertId": 0
    }

    units_collection = db.coll_fw_units

    # Get the parameters from the request
    Params = request.get_json()
    id_selected = Params[0]  # Assuming "This.IdSelected" is the ID of the unit to be selected

    # Query to find the unit based on IdSelected
    unit = units_collection.find_one({"_id": id_selected, "is_deleted": False})

    if unit:
        ReturnData["HasRows"] = True
        ReturnData["Data"] = unit
        ReturnData["RowsAffected"] = 1
    else:
        ReturnData["HasRows"] = False
        ReturnData["ErrorMessage"] = "Unit not found"
        ReturnData["HasError"] = True

    return jsonify(ReturnData), 200
    
@app.route('/UpdateUnit', methods=['POST'])
@jwt_required()
def update_unit():
    ReturnData = {
        "Data": [],
        "ErrorMessage": "",
        "HasError": False,
        "HasRows": False,
        "RowsAffected": 0,
        "InsertId": 0
    }

    units_collection = db.coll_fw_units

    # Get the parameters from the request
    Params = request.get_json()

    # Extract the parameters
    new_name = Params[0]  # "#TextBoxName"
    new_location = Params[1]  # "#TextBoxLocation"
    new_address = Params[2]  # "#TextBoxAddress"
    id_selected = Params[3]  # "This.IdSelected"

    # Define the fields to update
    update_fields = {
        "name": new_name,
        "location": new_location,
        "address": new_address,
        "modified": get_jwt_identity()+","+datetime.now().strftime("%Y/%m/%d %H:%M")
    }

    # Perform the update operation
    result = units_collection.update_one(
        {"_id": id_selected, "is_deleted": False},  # Match by IdSelected and ensure it's not deleted
        {"$set": update_fields}  # Fields to update
    )

    if result.matched_count > 0:
        ReturnData["HasRows"] = True
        ReturnData["RowsAffected"] = result.modified_count
        ReturnData["Data"] = {"UpdatedFields": update_fields}
    else:
        ReturnData["HasRows"] = False
        ReturnData["ErrorMessage"] = "Unit not found or already deleted"
        ReturnData["HasError"] = True

    return jsonify(ReturnData), 200

@app.route('/fw_CreateNewUnit', methods=['POST'])
@jwt_required()
def create_new_unit():
    ReturnData = {
        "Data": [],
        "ErrorMessage": "",
        "HasError": False,
        "HasRows": False,
        "RowsAffected": 0,
        "InsertId": 0
    }

    units_collection = db.coll_fw_units

    # Get the parameters from the request
    Params = request.get_json()

    # Extract the parameters
    new_name = Params[0]  # "#TextBoxName"
    new_location = Params[1]  # "#TextBoxLocation"
    new_address = Params[2]  # "#TextBoxAddress"
    id_selected = Params[3]  # "This.IdSelected"

    next_id = generate_unique_id()

    # Define the new unit data
    new_unit = {
        "_id":next_id,
        "name": new_name,
        "location": new_location,
        "address": new_address,
        "is_deleted": False,
        "modified": get_jwt_identity()+","+datetime.now().strftime("%Y/%m/%d %H:%M"),  # Set current timestamp
    }

    # Insert the new unit into the collection
    result = units_collection.insert_one(new_unit)

    if result.inserted_id:
        ReturnData["HasRows"] = True
        ReturnData["RowsAffected"] = 1
        ReturnData["InsertId"] = str(result.inserted_id)  # Return the inserted ID
        ReturnData["Data"] = {"InsertedUnit": new_unit}
    else:
        ReturnData["HasError"] = True
        ReturnData["ErrorMessage"] = "Failed to insert unit"
    
    return jsonify(ReturnData), 200

@app.route('/fw_DeleteUnit', methods=['POST'])
@jwt_required()
def fw_delete_unit():
    # Initialize response structure
    ReturnData = {
        "Data": [],
        "ErrorMessage": "",
        "HasError": False,
        "HasRows": False,
        "RowsAffected": 0,
        "InsertId": 0
    }

    # Get parameters from the request
    Params = request.get_json()
    unit_id = Params[0]  # This is the Id of the selected unit

    print(unit_id)
    # Validate the ID
    if not unit_id:
        ReturnData["HasError"] = True
        ReturnData["ErrorMessage"] = "Unit ID is required"
        return jsonify(ReturnData), 400

    try:

        # Get the units collection from the database
        units_collection = db.coll_fw_units

        # Update the unit's 'is_deleted' field to True
        update_result = units_collection.update_one(
            {"_id": int(unit_id)},
            {"$set": {"is_deleted": True, "modified": get_jwt_identity()+","+datetime.now().strftime("%Y/%m/%d %H:%M")}}
        )

        print(update_result)
        # Check if a document was updated
        if update_result.modified_count > 0:
            ReturnData["HasRows"] = True
            ReturnData["RowsAffected"] = update_result.modified_count
        else:
            ReturnData["HasError"] = True
            ReturnData["ErrorMessage"] = "No matching unit found"

    except Exception as e:
        ReturnData["HasError"] = True
        ReturnData["ErrorMessage"] = str(e)

    # Return the result
    return jsonify(ReturnData), 200
    
@app.route('/fw_RestoreUnit', methods=['POST'])
@jwt_required()
def fw_restore_unit():
    # Initialize response structure
    ReturnData = {
        "Data": [],
        "ErrorMessage": "",
        "HasError": False,
        "HasRows": False,
        "RowsAffected": 0,
        "InsertId": 0
    }

    # Get parameters from the request
    Params = request.get_json()
    unit_id = Params[0]  # This is the Id of the selected unit

    # Validate the ID
    if not unit_id:
        ReturnData["HasError"] = True
        ReturnData["ErrorMessage"] = "Unit ID is required"
        return jsonify(ReturnData), 400

    try:

        # Get the units collection from the database
        units_collection = db.coll_fw_units

        # Update the unit's 'is_deleted' field to False (restoring it)
        update_result = units_collection.update_one(
            {"_id": int(unit_id)},
            {"$set": {"is_deleted": False, "modified": get_jwt_identity()+","+datetime.now().strftime("%Y/%m/%d %H:%M")}}
        )

        # Check if a document was updated
        if update_result.modified_count > 0:
            ReturnData["HasRows"] = True
            ReturnData["RowsAffected"] = update_result.modified_count
        else:
            ReturnData["HasError"] = True
            ReturnData["ErrorMessage"] = "No matching unit found"

    except Exception as e:
        ReturnData["HasError"] = True
        ReturnData["ErrorMessage"] = str(e)

    # Return the result
    return jsonify(ReturnData), 200


################################################################## 
# Water Leakage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Global variables to control video input and processing
video_folder = 'D:/Projects/Dishwasher Water Leak Data/Video'
video_source = None
is_upload = False
# Global variables to manage video state
is_processing = False
video_thread = None
# Queue for log messages
log_queue = Queue()

# Custom handler to capture logs and put them in queue
class QueueHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        log_queue.put(log_entry)

# Add queue handler to logger
queue_handler = QueueHandler()
logger.addHandler(queue_handler)

@app.route('/stream_logs')
def stream_logs():
    def generate():
        while True:
            # Get log from queue if available, otherwise wait
            try:
                log_message = log_queue.get(timeout=1)
                yield f"data: {json.dumps({'log': log_message})}\n\n"
            except Empty:
                # No log available, send heartbeat
                yield f"data: {json.dumps({'heartbeat': True})}\n\n"
            
    return Response(generate(), mimetype='text/event-stream')

@app.route('/fetch_local_videos', methods=['GET'])
def fetch_local_videos():
    global video_folder
    
    # Get current index from request params
    current_index = int(request.args.get('currentIndex', 1))
    
    # Get list of video files and sort them
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith(('.MP4', '.avi', '.mov'))])
    total_videos = len(video_files)

    # Validate current_index
    if current_index < 1:
        current_index = 1
    elif current_index > total_videos:
        current_index = total_videos

    logger.info(f"Fetching video {current_index} of {total_videos}")
    
    # Get current video filename
    current_video = video_files[current_index-1] if video_files else 'No file selected'
    
    # Build video URLs
    video_path = os.path.join(video_folder, current_video)
    raw_video_url = f'/raw_video_feed?video_path={video_path}'
    processed_video_url = f'/processed_video_feed?video_path={video_path}'

    # Get logs from processing history if available
    logs = []
    log_file = os.path.join(video_folder, f'{current_video}.log')
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            logs = f.readlines()
    else:
        logs = ["Ready to process video..."]

    return jsonify({
        'totalVideos': total_videos,
        'currentVideo': current_video,
        'rawVideoUrl': raw_video_url, 
        'processedVideoUrl': processed_video_url,
        'currentIndex': current_index,
        'logs': logs
    })

@app.route('/process_video', methods=['POST'])
def process_video():
    global video_source, is_upload, is_processing, video_thread, video_folder

    input_type = request.form.get('input_type')
    action = request.form.get('action')  # This will be either 'start' or 'stop'
    logger.info(f"Processing video - Action: {action}, Input type: {input_type}")

    if action == 'start':
        if not is_processing:  # Start processing only if it's not already running
            if input_type == 'local':
                video_source = os.path.join(video_folder, request.form.get('current_file'))
                logger.info(f"Starting processing of local video: {video_source}")
                is_upload = False
            
            is_processing = True
            # Start a thread to handle video processing asynchronously
            video_thread = threading.Thread(target=start_video_processing)
            video_thread.start()

            return jsonify({'message': 'Video processing started.', 'video_source': video_source})
        else:
            return jsonify({'message': 'Video is already being processed.'})   
    elif action == 'stop':
        if is_processing:
            is_processing = False
            video_thread.join()  # Stop the processing thread
            logger.info("Video processing stopped")
            return jsonify({'message': 'Video processing stopped.'})
        else:
            return jsonify({'message': 'No video is being processed.'})

def start_video_processing():
    # Logic to start the video feed (detect_water_leak) processing
    global video_source
    if video_source:
        logger.info("Starting water leak detection")
        for frame in detect_water_leak(video_source):
            if not is_processing:
                logger.info("Stopping water leak detection")
                break
    else:
        logger.error("No video source provided")

@app.route('/raw_video_feed')
def raw_video_feed():
    video_path = request.args.get('video_path')
    if video_path:
        return Response(stream_raw_video(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        logger.error("No video path provided")
        return jsonify({'error': 'No video path provided'}), 400

@app.route('/processed_video_feed')
def processed_video_feed():
    video_path = request.args.get('video_path')
    if video_path and is_processing:
        return Response(detect_water_leak(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')
    else:
        logger.error("No video path or processing not started")
        return jsonify({'error': 'No video path or processing not started'}), 400

def stream_raw_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error("Could not open video file")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

def detect_water_leak(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        logger.error("Could not open video file")
        return
    
    ret, prev_frame = cap.read()
    if not ret:
        logger.error("Could not read the first frame")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.uint8)
    
    frame_count = 0
    total_motion = 0
    max_motion = 0
    frames_to_analyze = 120

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(prev_gray, gray)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        motion_pixels = sum(cv2.contourArea(cnt) for cnt in contours)
        total_motion += motion_pixels
        max_motion = max(max_motion, motion_pixels)

        for contour in contours:
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x-10, y-10), (x + w + 10, y + h+ 10), (0, 0, 255), 8)

        if frame_count >= frames_to_analyze:
            avg_motion = total_motion / frame_count
            # if avg_motion > 100 or max_motion > 10000:
            #     logger.info("LEAK DETECTED")
            #     cv2.putText(frame, "LEAK DETECTED", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # else:
            #     logger.info("NO LEAK DETECTED") 
            #     cv2.putText(frame, "NO LEAK", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        prev_gray = gray.copy()

    cap.release()




###################################################################################################################################
# Image blurring 

Config.init_directories()
    
    # Register blueprints
app.register_blueprint(blur_bp)
    
    # Configure app
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['PROCESSED_FOLDER'] = Config.PROCESSED_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Main route
@app.route('/')
def index():
    return render_template('index.html')

# Serve uploaded files directly (for development only)
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Serve processed files directly
@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

# Error handlers
@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return render_template('500.html'), 500

######################################################################################################################################
# text extraction  
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/text_extraction")
def text_extraction_page():
    return render_template("TextExtraction.html")

@app.route("/extract_text", methods=["POST"])
def extract_text_route():
    
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    extracted_text = extract_text(file_path)
    if not extracted_text:
        extracted_text = "No text found in the uploaded file."

    return jsonify({"text": extracted_text})

###################################################################################################################################### 
#object detection

################################################################## 
# Object Detection Routes

@app.route('/object_detection')
@jwt_required()
def object_detection():
    return render_template("object_detection.html")

# Set up object detection settings
detection_settings = {
    'confidence_threshold': 0.5,
    'show_boxes': True,
    'show_labels': True,
    'show_confidence': True,
    'enable_tracking': False,
    'class_filter': ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'dog', 'cat', 'bottle', 'chair']
}

# Store the current model
current_model = 'YOLOv8'

# Global variables for object detection
object_detector = None

def initialize_object_detector():
    global object_detector
    if object_detector is None:
        try:
            object_detector = ObjectDetector(model_name=current_model)
            logger.info(f"Object detector initialized with model: {current_model}")
        except Exception as e:
            logger.error(f"Failed to initialize object detector: {str(e)}")
            return False
    return True

@app.route('/api/object_detection/models', methods=['GET'])
@jwt_required()
def get_available_models():
    # Return list of available models
    models = ['YOLOv8', 'YOLOv5', 'SSD MobileNet', 'Faster R-CNN']
    return jsonify({'models': models})

@app.route('/api/object_detection/settings', methods=['GET', 'POST'])
@jwt_required()
def handle_detection_settings():
    global detection_settings
    
    if request.method == 'POST':
        data = request.get_json()
        # Update global settings with provided values
        for key, value in data.items():
            if key in detection_settings:
                detection_settings[key] = value
        return jsonify({'success': True, 'settings': detection_settings})
    else:
        return jsonify(detection_settings)

@app.route('/api/object_detection/upload', methods=['POST'])
@jwt_required()
def upload_for_detection():
    # Ensure upload directory exists
    os.makedirs('uploads', exist_ok=True)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    allowed_extensions = {'png', 'jpg', 'jpeg', 'mp4', 'avi', 'mov'}
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in allowed_extensions:
        # Generate a unique filename with timestamp and uuid
        unique_id = str(uuid.uuid4())
        filename = secure_filename(f"{unique_id}_{file.filename}")
        file_path = os.path.join('uploads', filename)
        file.save(file_path)
        
        # Determine if it's an image or video
        is_video = file.filename.lower().endswith(('mp4', 'avi', 'mov'))
        
        return jsonify({
            'success': True, 
            'filename': filename, 
            'file_path': file_path,
            'type': 'video' if is_video else 'image'
        })
    
    return jsonify({'error': 'File type not allowed'}), 400

@app.route('/api/object_detection/detect/image', methods=['POST'])
@jwt_required()
def detect_objects_in_image():
    if not initialize_object_detector():
        return jsonify({'error': 'Failed to initialize object detector'}), 500
    
    data = request.get_json()
    image_path = data.get('image_path')
    
    if not image_path or not os.path.exists(image_path):
        return jsonify({'error': 'Invalid image path'}), 400
    
    try:
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            return jsonify({'error': 'Failed to read image'}), 400
        
        # Apply detection with settings
        detections = object_detector.detect(
            image, 
            conf_threshold=detection_settings['confidence_threshold'],
            class_filter=detection_settings['class_filter'] if detection_settings['class_filter'] else None
        )
        
        # Draw detections on image
        annotated_image = object_detector.draw_detections(
            image, 
            detections, 
            show_boxes=detection_settings['show_boxes'],
            show_labels=detection_settings['show_labels'],
            show_confidence=detection_settings['show_confidence']
        )
        
        # Save the annotated image
        result_path = f"{image_path}_detected.jpg"
        cv2.imwrite(result_path, annotated_image)
        
        # Prepare results to return
        results = {
            'success': True,
            'detected_objects': len(detections),
            'result_path': result_path,
            'detections': [
                {
                    'class': det['class'],
                    'confidence': float(det['confidence']),
                    'bbox': det['bbox']
                } for det in detections
            ]
        }
        
        return jsonify(results)
    
    except Exception as e:
        logger.error(f"Error during detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/object_detection/detect/video', methods=['POST'])
@jwt_required()
def start_video_detection():
    if not initialize_object_detector():
        return jsonify({'error': 'Failed to initialize object detector'}), 500
    
    data = request.get_json()
    video_path = data.get('video_path')
    
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'Invalid video path'}), 400
    
    try:
        # Generate a unique job ID
        job_id = f"detection_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Start video processing in background thread
        video_thread = threading.Thread(
            target=process_video_detection,
            args=(video_path, job_id, detection_settings.copy())
        )
        video_thread.daemon = True
        video_thread.start()
        
        return jsonify({
            'success': True, 
            'job_id': job_id,
            'stream_url': f'/api/object_detection/video_feed?job_id={job_id}'
        })
    
    except Exception as e:
        logger.error(f"Error starting video detection: {str(e)}")
        return jsonify({'error': str(e)}), 500

def process_video_detection(video_path, job_id, settings):
    """Process video detection in a background thread"""
    global object_detector
    
    try:
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output directory if it doesn't exist
        output_dir = os.path.join('uploads', 'processed')
        os.makedirs(output_dir, exist_ok=True)
        
        # Create output video writer
        output_path = os.path.join(output_dir, f"{job_id}_output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        detection_count = 0
        total_detections = 0
        
        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # For performance, detect only every few frames
            if frame_count % 3 == 0:  # Process every 3rd frame
                detections = object_detector.detect(
                    frame, 
                    conf_threshold=settings['confidence_threshold'],
                    class_filter=settings['class_filter'] if settings['class_filter'] else None
                )
                
                detection_count += 1
                total_detections += len(detections)
                
                # Draw detections
                frame = object_detector.draw_detections(
                    frame, 
                    detections, 
                    show_boxes=settings['show_boxes'],
                    show_labels=settings['show_labels'],
                    show_confidence=settings['show_confidence']
                )
            
            # Write the frame
            out.write(frame)
            
            # Store the latest frame for streaming
            with open(f"uploads/temp_{job_id}.jpg", 'wb') as f:
                _, buffer = cv2.imencode('.jpg', frame)
                f.write(buffer)
        
        # Clean up
        cap.release()
        out.release()
        
        # Save detection stats
        with open(f"uploads/{job_id}_stats.json", 'w') as f:
            json.dump({
                'total_frames': frame_count,
                'processed_frames': detection_count,
                'total_detections': total_detections,
                'output_path': output_path
            }, f)
        
        logger.info(f"Video processing complete: {job_id}")
    
    except Exception as e:
        logger.error(f"Error in video processing: {str(e)}")

@app.route('/api/object_detection/video_feed')
def video_detection_feed():
    job_id = request.args.get('job_id')
    
    if not job_id:
        return jsonify({'error': 'No job ID provided'}), 400
    
    def generate():
        while True:
            # Check if the job is still running
            temp_frame_path = f"uploads/temp_{job_id}.jpg"
            stats_path = f"uploads/{job_id}_stats.json"
            
            if os.path.exists(temp_frame_path):
                # Job is running, return latest frame
                with open(temp_frame_path, 'rb') as f:
                    frame_data = f.read()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
                
                # Short sleep to prevent high CPU usage
                time.sleep(0.03)
            
            elif os.path.exists(stats_path):
                # Job is complete
                # Send completion message
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n'
                       b'Processing complete\r\n')
                break
            else:
                # Job not found or not started
                yield (b'--frame\r\n'
                       b'Content-Type: text/plain\r\n\r\n'
                       b'Waiting for processing to start...\r\n')
                time.sleep(1)
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/object_detection/job_status/<job_id>')
@jwt_required()
def get_detection_job_status(job_id):
    stats_path = f"uploads/{job_id}_stats.json"
    
    if os.path.exists(stats_path):
        # Job is complete, return stats
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        
        return jsonify({
            'status': 'complete',
            'stats': stats
        })
    
    elif os.path.exists(f"uploads/temp_{job_id}.jpg"):
        # Job is still running
        return jsonify({
            'status': 'processing'
        })
    
    else:
        # Job not found
        return jsonify({
            'status': 'not_found'
        }), 404

@app.route('/api/object_detection/webcam_feed')
def webcam_feed():
    def generate():
        # Access webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'Error: Could not access webcam\r\n')
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Short sleep to control frame rate
            time.sleep(0.03)
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/object_detection/webcam_detect')
def webcam_detection_feed():
    if not initialize_object_detector():
        return jsonify({'error': 'Failed to initialize object detector'}), 500
    
    def generate():
        # Access webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            yield (b'--frame\r\n'
                   b'Content-Type: text/plain\r\n\r\n'
                   b'Error: Could not access webcam\r\n')
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = object_detector.detect(
                frame, 
                conf_threshold=detection_settings['confidence_threshold'],
                class_filter=detection_settings['class_filter'] if detection_settings['class_filter'] else None
            )
            
            # Draw detections
            annotated_frame = object_detector.draw_detections(
                frame, 
                detections, 
                show_boxes=detection_settings['show_boxes'],
                show_labels=detection_settings['show_labels'],
                show_confidence=detection_settings['show_confidence']
            )
            
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
            # Short sleep to control frame rate
            time.sleep(0.03)
        
        cap.release()
    
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/object_detection/download/<path:filename>')
@jwt_required()
def download_processed_file(filename):
    """Download a processed file"""
    directory = os.path.join(app.root_path, 'uploads', 'processed')
    return send_from_directory(directory, filename, as_attachment=True)


##################################################################

if __name__ == "__main__":
    app.run(debug = True, host = "127.0.0.1", port = 5002)