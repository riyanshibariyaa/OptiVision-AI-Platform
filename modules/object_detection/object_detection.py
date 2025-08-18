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

class YOLOv8:
    """YOLOv8 object detector with ONNX Runtime"""
    
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):  # Modified thresholds
        """
        Initialize YOLOv8 object detector
        
        Args:
            model_path: Path to YOLOv8 ONNX model
            conf_thres: Confidence threshold for filtering detections
            iou_thres: IoU threshold for NMS
        """
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        
        # Load model
        self.initialize_model(model_path)
        
        # Class names
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

    # All existing methods remain the same

    # Add this new method for test-time augmentation
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
        boxes = boxes1 + boxes2
        scores = scores1 + scores2
        class_ids = class_ids1 + class_ids2
        
        # Apply NMS to combined detections
        if boxes:
            boxes, scores, class_ids = apply_nms(boxes, scores, class_ids, self.iou_threshold)
        
        return boxes, scores, class_ids


class ObjectDetector:
    """Main object detection class that wraps YOLOv8"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path if model_path else r"C:\Users\Hp\CVPlatform\modules\object_detection\models\yolov8m.onnx"
        self.detector = None
        self._initialize_detector()
    
    def _initialize_detector(self):
        """Initialize YOLOv8 detector"""
        try:
            # Modified thresholds for better detection
            self.detector = YOLOv8(self.model_path, conf_thres=0.25, iou_thres=0.45)
            logger.info(f"Initialized YOLOv8 detector with model: {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to initialize detector: {str(e)}")
            raise
    
    def detect_image(self, image_path, save_path=None, display=True, enhance=True):
        """
        Detect objects in an image with enhanced accuracy
        
        Args:
            image_path: Path to image or URL
            save_path: Path to save output image (optional)
            display: Whether to display the output image
            enhance: Whether to apply detection enhancements
            
        Returns:
            result_img: Image with drawn detections
            detections: List of (boxes, scores, class_ids)
        """
        # Load image
        if image_path.startswith(('http://', 'https://')):
            img = imread_from_url(image_path)
        else:
            img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not load image from: {image_path}")
        
        # Apply preprocessing if enhancing
        if enhance:
            img_proc = preprocess_image(img)
        else:
            img_proc = img.copy()
        
        # Detect objects (with TTA if enhancing)
        if enhance:
            boxes, scores, class_ids = self.detector.detect_objects_with_tta(img_proc)
        else:
            boxes, scores, class_ids = self.detector(img_proc)
        
        # Refine bounding boxes if enhancing
        if enhance and boxes:
            boxes = refine_boxes(img, boxes)
            boxes = snap_to_edges(img, boxes)
        
        # Draw detections
        result_img = self.detector.draw_detections(img, boxes, scores, class_ids)
        
        # Save if requested
        if save_path:
            cv2.imwrite(save_path, result_img)
            logger.info(f"Saved detection result to: {save_path}")
        
        # Display if requested
        if display:
            cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
            cv2.imshow("Detected Objects", result_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return result_img, (boxes, scores, class_ids)
    
    # Modify the video detection method to use enhanced detection
    def detect_video(self, video_path, save_path=None, start_time=0, display=True, enhance=True):
        """
        Detect objects in a video with enhanced accuracy
        
        Args:
            video_path: Path to video or YouTube URL
            save_path: Path to save output video (optional)
            start_time: Skip first N seconds (optional)
            display: Whether to display the output video
            enhance: Whether to apply detection enhancements
            
        Returns:
            success: Whether video processing completed successfully
        """
        # Load video
        if video_path.startswith(('http://', 'https://')) and 'youtu' in video_path:
            # Load from YouTube
            try:
                from cap_from_youtube import cap_from_youtube
                cap = cap_from_youtube(video_path, resolution='720p')
            except ImportError:
                logger.error("cap_from_youtube not found. Install with: pip install cap-from-youtube")
                raise
        else:
            # Load from file
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Skip first N seconds if requested
        if start_time > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_time * cap.get(cv2.CAP_PROP_FPS)))
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if saving
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Process video
        if display:
            cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        processing_times = []
        
        try:
            while cap.isOpened():
                # Measure processing time
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame if enhancing
                if enhance:
                    frame_proc = preprocess_image(frame)
                else:
                    frame_proc = frame.copy()
                
                # Detect objects (with TTA if enhancing)
                if enhance:
                    # Skip TTA for every other frame to improve speed
                    if frame_count % 2 == 0:
                        boxes, scores, class_ids = self.detector.detect_objects_with_tta(frame_proc)
                    else:
                        boxes, scores, class_ids = self.detector(frame_proc)
                    
                    # Refine boxes
                    if boxes:
                        boxes = refine_boxes(frame, boxes)
                else:
                    boxes, scores, class_ids = self.detector(frame_proc)
                
                # Draw detections
                result_frame = self.detector.draw_detections(frame, boxes, scores, class_ids)
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Calculate FPS
                fps_text = f"FPS: {1.0 / (sum(processing_times) / len(processing_times)):.1f}"
                cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if saving
                if writer:
                    writer.write(result_frame)
                
                # Display frame
                if display:
                    cv2.imshow("Detected Objects", result_frame)
                    
                    # Break on 'q' key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            success = False
        else:
            success = True
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Log processing stats
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            logger.info(f"Average processing time: {avg_time:.4f}s ({1.0/avg_time:.1f} FPS)")
            logger.info(f"Processed {frame_count} frames")
        
        return success
    
    # Similar modifications for detect_webcam method
    def detect_webcam(self, camera_id=0, save_path=None, width=None, height=None, enhance=True):
        """
        Detect objects from webcam with enhanced accuracy
        
        Args:
            camera_id: Webcam ID (default: 0)
            save_path: Path to save output video (optional)
            width: Custom width for webcam capture (optional)
            height: Custom height for webcam capture (optional)
            enhance: Whether to apply detection enhancements
            
        Returns:
            success: Whether webcam processing completed successfully
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam with ID: {camera_id}")
        
        # Set custom resolution if specified
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if saving
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Start processing
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        
        processing_times = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                # Measure processing time
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from webcam")
                    break
                
                # Preprocess frame if enhancing
                if enhance:
                    frame_proc = preprocess_image(frame)
                else:
                    frame_proc = frame
                
                # Detect objects (with enhanced detection every 3 frames to maintain speed)
                if enhance and frame_count % 3 == 0:
                    boxes, scores, class_ids = self.detector.detect_objects_with_tta(frame_proc)
                    if boxes:
                        boxes = refine_boxes(frame, boxes)
                else:
                    boxes, scores, class_ids = self.detector(frame_proc)
                
                # Draw detections
                result_frame = self.detector.draw_detections(frame, boxes, scores, class_ids)
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Calculate FPS
                fps_text = f"FPS: {1.0 / (sum(processing_times) / len(processing_times)):.1f}"
                cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if saving
                if writer:
                    writer.write(result_frame)
                
                # Display frame
                cv2.imshow("Detected Objects", result_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        except Exception as e:
            logger.error(f"Error processing webcam: {str(e)}")
            success = False
        else:
            success = True
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Log processing stats
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            logger.info(f"Average processing time: {avg_time:.4f}s ({1.0/avg_time:.1f} FPS)")
            logger.info(f"Processed {frame_count} frames")
        
        return success
    
    def detect_webcam(self, camera_id=0, save_path=None, width=None, height=None):
        """
        Detect objects from webcam
        
        Args:
            camera_id: Webcam ID (default: 0)
            save_path: Path to save output video (optional)
            width: Custom width for webcam capture (optional)
            height: Custom height for webcam capture (optional)
            
        Returns:
            success: Whether webcam processing completed successfully
        """
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam with ID: {camera_id}")
        
        # Set custom resolution if specified
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Get actual properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create video writer if saving
        writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        
        # Start processing
        cv2.namedWindow("Detected Objects", cv2.WINDOW_NORMAL)
        
        processing_times = []
        frame_count = 0
        
        try:
            while cap.isOpened():
                # Measure processing time
                start_time = time.time()
                
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from webcam")
                    break
                
                # Detect objects
                boxes, scores, class_ids = self.detector(frame)
                
                # Draw detections
                result_frame = self.detector.draw_detections(frame, boxes, scores, class_ids)
                
                # Calculate processing time
                process_time = time.time() - start_time
                processing_times.append(process_time)
                if len(processing_times) > 30:
                    processing_times.pop(0)
                
                # Calculate FPS
                fps_text = f"FPS: {1.0 / (sum(processing_times) / len(processing_times)):.1f}"
                cv2.putText(result_frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Write frame if saving
                if writer:
                    writer.write(result_frame)
                
                # Display frame
                cv2.imshow("Detected Objects", result_frame)
                
                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
        
        except Exception as e:
            logger.error(f"Error processing webcam: {str(e)}")
            success = False
        else:
            success = True
        finally:
            # Clean up
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
        
        # Log processing stats
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            logger.info(f"Average processing time: {avg_time:.4f}s ({1.0/avg_time:.1f} FPS)")
            logger.info(f"Processed {frame_count} frames")
        
        return success


# Helper function to read image from URL
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
        logger.error(f"Error reading image from URL: {str(e)}")
        return None
    
def preprocess_image(img):
    """
    Preprocess image to improve detection quality
    
    Args:
        img: Input image
        
    Returns:
        Preprocessed image
    """
    # Make a copy to avoid modifying the original
    img_proc = img.copy()
    
    # Normalize lighting using histogram equalization
    img_yuv = cv2.cvtColor(img_proc, cv2.COLOR_BGR2YUV)
    img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
    img_proc = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    # Apply slight Gaussian blur to reduce noise
    img_proc = cv2.GaussianBlur(img_proc, (3, 3), 0)
    
    return img_proc

def refine_boxes(img, boxes):
    """
    Refine bounding boxes to improve accuracy
    
    Args:
        img: Input image
        boxes: List of bounding boxes [x, y, w, h]
        
    Returns:
        Refined bounding boxes
    """
    refined_boxes = []
    for box in boxes:
        x, y, w, h = box
        
        # Ensure box is within image boundaries
        x = max(0, x)
        y = max(0, y)
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        
        # Apply aspect ratio constraints if needed
        if w > 0 and h > 0:  # Avoid division by zero
            aspect = w / h
            if aspect > 3 or aspect < 0.33:  # Unrealistic aspect ratio
                # Adjust box dimensions while maintaining center
                center_x, center_y = x + w/2, y + h/2
                if aspect > 3:  # too wide
                    w = h * 3
                else:  # too tall
                    h = w * 3
                x = center_x - w/2
                y = center_y - h/2
        
        refined_boxes.append([int(x), int(y), int(w), int(h)])
    return refined_boxes

def snap_to_edges(img, boxes, padding=5):
    """
    Adjust bounding boxes to snap to object edges
    
    Args:
        img: Input image
        boxes: List of bounding boxes [x, y, w, h]
        padding: Padding around box to look for edges
        
    Returns:
        Edge-aligned bounding boxes
    """
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect edges
    edges = cv2.Canny(gray, 50, 150)
    
    adjusted_boxes = []
    for box in boxes:
        x, y, w, h = box
        
        # Skip if box dimensions are too small
        if w < 10 or h < 10:
            adjusted_boxes.append(box)
            continue
            
        # Define region of interest with padding
        roi_y_min = max(0, y-padding)
        roi_y_max = min(img.shape[0], y+h+padding)
        roi_x_min = max(0, x-padding)
        roi_x_max = min(img.shape[1], x+w+padding)
        
        # Extract ROI from edge image
        roi = edges[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        
        if roi.size == 0:  # Skip if ROI is empty
            adjusted_boxes.append(box)
            continue
            
        # Find contours
        contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour
            largest = max(contours, key=cv2.contourArea)
            # Only adjust if contour is significant
            if cv2.contourArea(largest) > 20:
                x_min, y_min, w_adj, h_adj = cv2.boundingRect(largest)
                # Adjust coordinates relative to original image
                x_adj = roi_x_min + x_min
                y_adj = roi_y_min + y_min
                adjusted_boxes.append([x_adj, y_adj, w_adj, h_adj])
            else:
                adjusted_boxes.append(box)
        else:
            adjusted_boxes.append(box)
            
    return adjusted_boxes

def apply_nms(boxes, scores, class_ids, iou_thresh=0.45):
    """
    Apply Non-Maximum Suppression to remove redundant detections
    
    Args:
        boxes: List of bounding boxes [x, y, w, h]
        scores: List of confidence scores
        class_ids: List of class IDs
        iou_thresh: IoU threshold for NMS
        
    Returns:
        Filtered boxes, scores, and class IDs
    """
    # Convert to numpy arrays
    boxes_np = np.array(boxes)
    scores_np = np.array(scores)
    class_ids_np = np.array(class_ids)
    
    # Sort by score
    indices = np.argsort(scores_np)[::-1]
    boxes_np = boxes_np[indices]
    scores_np = scores_np[indices]
    class_ids_np = class_ids_np[indices]
    
    # Get indices of boxes to keep
    keep_indices = cv2.dnn.NMSBoxes(
        boxes_np.tolist(),
        scores_np.tolist(),
        0.0,  # conf_threshold (already filtered)
        iou_thresh
    )
    
    if len(keep_indices) > 0:
        # Extract kept boxes
        result_boxes = [boxes_np[i].tolist() for i in keep_indices]
        result_scores = [scores_np[i] for i in keep_indices]
        result_class_ids = [class_ids_np[i] for i in keep_indices]
        return result_boxes, result_scores, result_class_ids
    else:
        return [], [], []
    