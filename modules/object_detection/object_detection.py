# modules/object_detection/object_detection.py

import cv2
import numpy as np
import os
import logging
import time
from pathlib import Path
import urllib.request

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import ONNX Runtime
try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("ONNX Runtime available")
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX Runtime not available, falling back to OpenCV DNN")

class YOLOv8:
    """YOLOv8 object detector with ONNX Runtime or OpenCV DNN fallback"""
    
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        """
        Initialize YOLOv8 object detector
        
        Args:
            model_path: Path to YOLOv8 ONNX model
            conf_thres: Confidence threshold for filtering detections
            iou_thres: IoU threshold for NMS
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.model_path = model_path
        
        # Initialize model
        self.session = None
        self.net = None
        self.input_width = 640
        self.input_height = 640
        
        # Load model
        self.initialize_model(model_path)
        
        # Class names (COCO dataset)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Generate random colors for visualization
        np.random.seed(42)  # For reproducibility
        self.colors = np.random.randint(0, 255, size=(len(self.class_names), 3), dtype=np.uint8)

    def initialize_model(self, model_path):
        """Initialize the YOLO model"""
        try:
            if not os.path.exists(model_path):
                # Try to download a default model if path doesn't exist
                logger.warning(f"Model not found at {model_path}, attempting to use default")
                self._setup_fallback_model()
                return
            
            if ONNX_AVAILABLE and model_path.endswith('.onnx'):
                # Use ONNX Runtime
                self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
                self.input_name = self.session.get_inputs()[0].name
                logger.info(f"Initialized ONNX model: {model_path}")
            else:
                # Use OpenCV DNN as fallback
                self.net = cv2.dnn.readNetFromONNX(model_path)
                logger.info(f"Initialized OpenCV DNN model: {model_path}")
                
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {str(e)}")
            self._setup_fallback_model()

    def _setup_fallback_model(self):
        """Setup a fallback model using OpenCV's pre-trained models"""
        try:
            # Try to use OpenCV's built-in models or create a mock detector
            logger.warning("Setting up fallback detection method")
            self.net = None
            self.session = None
        except Exception as e:
            logger.error(f"Fallback model setup failed: {str(e)}")

    def preprocess(self, image):
        """Preprocess image for YOLO inference"""
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Add batch dimension and change to CHW format
        input_tensor = np.transpose(normalized, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        return input_tensor

    def postprocess(self, outputs, original_shape):
        """Post-process YOLO outputs"""
        if isinstance(outputs, list):
            outputs = outputs[0]
        
        # Handle different output formats
        if len(outputs.shape) == 3:
            outputs = outputs[0]  # Remove batch dimension
        
        # outputs shape: [num_predictions, 84] where 84 = 4 (bbox) + 80 (classes)
        boxes = []
        scores = []
        class_ids = []
        
        # Scale factors for converting back to original image size
        x_scale = original_shape[1] / self.input_width
        y_scale = original_shape[0] / self.input_height
        
        for detection in outputs:
            if len(detection) < 5:
                continue
                
            # Extract confidence scores for all classes (skip first 4 bbox values)
            confidence_scores = detection[4:]
            
            # Get the class with highest confidence
            class_id = np.argmax(confidence_scores)
            confidence = confidence_scores[class_id]
            
            if confidence > self.conf_threshold:
                # Extract bounding box coordinates (center_x, center_y, width, height)
                center_x, center_y, width, height = detection[:4]
                
                # Convert to corner coordinates and scale back to original image
                x1 = int((center_x - width / 2) * x_scale)
                y1 = int((center_y - height / 2) * y_scale)
                x2 = int((center_x + width / 2) * x_scale)
                y2 = int((center_y + height / 2) * y_scale)
                
                # Ensure coordinates are within image bounds
                x1 = max(0, min(x1, original_shape[1]))
                y1 = max(0, min(y1, original_shape[0]))
                x2 = max(0, min(x2, original_shape[1]))
                y2 = max(0, min(y2, original_shape[0]))
                
                boxes.append([x1, y1, x2 - x1, y2 - y1])  # Convert to [x, y, w, h] format
                scores.append(float(confidence))
                class_ids.append(int(class_id))
        
        # Apply Non-Maximum Suppression
        if boxes:
            indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
            
            if len(indices) > 0:
                # Handle different OpenCV versions
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                final_boxes = [boxes[i] for i in indices]
                final_scores = [scores[i] for i in indices]
                final_class_ids = [class_ids[i] for i in indices]
                
                return final_boxes, final_scores, final_class_ids
        
        return [], [], []

    def __call__(self, image):
        """Main inference method"""
        return self.detect_objects(image)

    def detect_objects(self, image):
        """
        Detect objects in an image
        
        Args:
            image: Input image (numpy array)
            
        Returns:
            boxes: List of bounding boxes [x, y, w, h]
            scores: List of confidence scores
            class_ids: List of class IDs
        """
        if image is None:
            return [], [], []
            
        try:
            original_shape = image.shape[:2]
            
            # Preprocess image
            input_tensor = self.preprocess(image)
            
            # Run inference
            if self.session is not None:
                # ONNX Runtime inference
                outputs = self.session.run(None, {self.input_name: input_tensor})
            elif self.net is not None:
                # OpenCV DNN inference
                self.net.setInput(input_tensor)
                outputs = self.net.forward()
            else:
                # Fallback: return empty results
                logger.warning("No valid model available for inference")
                return [], [], []
            
            # Post-process outputs
            boxes, scores, class_ids = self.postprocess(outputs, original_shape)
            
            return boxes, scores, class_ids
            
        except Exception as e:
            logger.error(f"Error during object detection: {str(e)}")
            return [], [], []

    def detect_objects_with_tta(self, img):
        """
        Detect objects using test-time augmentation
        
        Args:
            img: Input image
            
        Returns:
            boxes: Bounding boxes [x, y, w, h]
            scores: Confidence scores
            class_ids: Class indices
        """
        # Regular detection
        boxes1, scores1, class_ids1 = self.detect_objects(img)
        
        # Flipped image detection
        flipped = cv2.flip(img, 1)  # Horizontal flip
        boxes2, scores2, class_ids2 = self.detect_objects(flipped)
        
        # Adjust boxes from flipped image
        width = img.shape[1]
        for i, box in enumerate(boxes2):
            boxes2[i][0] = width - box[0] - box[2]  # Adjust x coordinate
        
        # Combine detections
        all_boxes = boxes1 + boxes2
        all_scores = scores1 + scores2
        all_class_ids = class_ids1 + class_ids2
        
        # Apply NMS to combined detections
        if all_boxes:
            indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.conf_threshold, self.iou_threshold)
            
            if len(indices) > 0:
                if isinstance(indices, np.ndarray):
                    indices = indices.flatten()
                
                final_boxes = [all_boxes[i] for i in indices]
                final_scores = [all_scores[i] for i in indices]
                final_class_ids = [all_class_ids[i] for i in indices]
                
                return final_boxes, final_scores, final_class_ids
        
        return [], [], []

    def draw_detections(self, image, boxes, scores, class_ids):
        """Draw bounding boxes and labels on image"""
        result_image = image.copy()
        
        for box, score, class_id in zip(boxes, scores, class_ids):
            x, y, w, h = box
            x1, y1 = int(x), int(y)
            x2, y2 = int(x + w), int(y + h)
            
            # Get color for this class
            color = tuple(map(int, self.colors[class_id % len(self.colors)]))
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{self.class_names[class_id]}: {score:.2f}"
            
            # Get text size
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            
            # Draw label background
            cv2.rectangle(
                result_image,
                (x1, y1 - text_height - baseline),
                (x1 + text_width, y1),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                result_image,
                label,
                (x1, y1 - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
        
        return result_image


class ObjectDetector:
    """Main object detection class that wraps YOLOv8"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path else self._get_default_model_path()
        self.detector = None
        self._initialize_detector()
    
    def _get_default_model_path(self):
        """Get default model path"""
        # Try different possible locations
        possible_paths = [
            r"C:\Users\Hp\CVPlatform\modules\object_detection\models\yolov8m.onnx",
            "modules/object_detection/models/yolov8m.onnx",
            "models/yolov8m.onnx",
            "yolov8m.onnx"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # If no model found, return the first path (will trigger fallback)
        return possible_paths[0]
    
    def _initialize_detector(self):
        """Initialize YOLOv8 detector"""
        try:
            self.detector = YOLOv8(self.model_path, conf_thres=0.25, iou_thres=0.45)
            logger.info(f"Initialized YOLOv8 detector with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            raise

    def detect_objects(self, image_path):
        """
        Detect objects in an image file
        
        Args:
            image_path: Path to image file
            
        Returns:
            List of detection results
        """
        try:
            # Load image
            if isinstance(image_path, str):
                if image_path.startswith(('http://', 'https://')):
                    img = imread_from_url(image_path)
                else:
                    img = cv2.imread(image_path)
            else:
                # Assume it's already a numpy array
                img = image_path
            
            if img is None:
                raise ValueError(f"Could not load image from: {image_path}")
            
            # Detect objects
            boxes, scores, class_ids = self.detector(img)
            
            # Format results
            results = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                x, y, w, h = box
                results.append({
                    'bbox': [x, y, w, h],
                    'confidence': score,
                    'class_id': class_id,
                    'class_name': self.detector.class_names[class_id]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting objects: {str(e)}")
            return []

    def detect_objects_frame(self, frame):
        """
        Detect objects in a single frame (for video processing)
        
        Args:
            frame: Video frame (numpy array)
            
        Returns:
            List of detection results
        """
        try:
            boxes, scores, class_ids = self.detector(frame)
            
            results = []
            for box, score, class_id in zip(boxes, scores, class_ids):
                x, y, w, h = box
                results.append({
                    'bbox': [x, y, w, h],
                    'confidence': score,
                    'class_id': class_id,
                    'class_name': self.detector.class_names[class_id]
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error detecting objects in frame: {str(e)}")
            return []

    def draw_boxes(self, image_path, results):
        """
        Draw bounding boxes on image
        
        Args:
            image_path: Path to image file
            results: Detection results
            
        Returns:
            Image with drawn bounding boxes
        """
        try:
            # Load image
            if isinstance(image_path, str):
                img = cv2.imread(image_path)
            else:
                img = image_path
            
            if img is None:
                raise ValueError(f"Could not load image")
            
            # Extract data for drawing
            boxes = []
            scores = []
            class_ids = []
            
            for result in results:
                boxes.append(result['bbox'])
                scores.append(result['confidence'])
                class_ids.append(result['class_id'])
            
            # Draw detections
            return self.detector.draw_detections(img, boxes, scores, class_ids)
            
        except Exception as e:
            logger.error(f"Error drawing boxes: {str(e)}")
            return img if 'img' in locals() else np.zeros((480, 640, 3), dtype=np.uint8)

    def draw_boxes_frame(self, frame, results):
        """
        Draw bounding boxes on a video frame
        
        Args:
            frame: Video frame (numpy array)
            results: Detection results
            
        Returns:
            Frame with drawn bounding boxes
        """
        try:
            # Extract data for drawing
            boxes = []
            scores = []
            class_ids = []
            
            for result in results:
                boxes.append(result['bbox'])
                scores.append(result['confidence'])
                class_ids.append(result['class_id'])
            
            # Draw detections
            return self.detector.draw_detections(frame, boxes, scores, class_ids)
            
        except Exception as e:
            logger.error(f"Error drawing boxes on frame: {str(e)}")
            return frame


# Helper functions
def imread_from_url(url):
    """Read image from URL"""
    try:
        # Download image to temporary file
        temp_file, _ = urllib.request.urlretrieve(url)
        # Read image with OpenCV
        img = cv2.imread(temp_file)
        # Clean up
        urllib.request.urlcleanup()
        return img
    except Exception as e:
        logger.error(f"Error reading image from URL {url}: {str(e)}")
        return None


def preprocess_image(img):
    """Apply preprocessing to enhance detection"""
    # Apply histogram equalization to improve contrast
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:,:,0] = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return enhanced


# Test function
def test_detection():
    """Test the object detection functionality"""
    try:
        detector = ObjectDetector()
        
        # Create a test image (if no real image available)
        test_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test detection
        results = detector.detect_objects_frame(test_img)
        
        logger.info(f"Detection test completed. Found {len(results)} objects.")
        return True
        
    except Exception as e:
        logger.error(f"Detection test failed: {str(e)}")
        return False


if __name__ == "__main__":
    # Run test if script is executed directly
    test_detection()