$(document).ready(function() {
    // Global variables
    let webcamStream = null;
    let detectionInterval = null;
    let isDetecting = false;
    let supportedClasses = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
        'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
        'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ];
    
    // Initialize the page
    initializeClassFilters();
    initializeEventListeners();
    
    // ========================
    // Initialization Functions
    // ========================
    
    function initializeClassFilters() {
        // Clear existing class filters
        $('#class-filter-list').empty();
        
        // Add checkboxes for each supported class
        supportedClasses.forEach(className => {
            const classId = 'class-' + className.replace(/\s+/g, '-');
            const checkboxHtml = `
                <div class="form-check">
                    <input class="form-check-input" type="checkbox" id="${classId}" checked>
                    <label class="form-check-label" for="${classId}">${className}</label>
                </div>
            `;
            $('#class-filter-list').append(checkboxHtml);
        });
    }
    
    function initializeEventListeners() {
        // File upload related
        $('#file-input').on('change', handleFileSelection);
        $('#upload-btn').on('click', uploadFile);
        $('#detect-image-btn').on('click', detectObjectsInImage);
        $('#detect-video-btn').on('click', toggleVideoDetection);
        $('#stop-video-btn').on('click', stopVideoDetection);
        
        // Webcam related
        $('#start-webcam-btn').on('click', startWebcam);
        $('#start-detection-btn').on('click', startWebcamDetection);
        $('#stop-webcam-btn').on('click', stopWebcam);
        
        // Settings related
        $('#confidence-threshold').on('input', function() {
            const value = $(this).val();
            $('#confidence-value').text(value);
        });
        
        $('#webcam-confidence-threshold').on('input', function() {
            const value = $(this).val();
            $('#webcam-confidence-value').text(value);
        });
    }
    
    // ===================
    // File Upload Methods
    // ===================
    
    function handleFileSelection(event) {
        const file = event.target.files[0];
        if (!file) return;
        
        // Reset previous displays
        $('#image-container').hide();
        $('#video-container').hide();
        $('#detection-info').hide();
        $('#download-container').hide();
        
        if (file.type.startsWith('image/')) {
            displaySelectedImage(file);
        } else if (file.type.startsWith('video/')) {
            displaySelectedVideo(file);
        } else {
            logMessage('Unsupported file type. Please upload an image or video.');
        }
    }
    
    function displaySelectedImage(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            $('#image-display').attr('src', e.target.result);
            $('#image-title').text(file.name);
            $('#image-container').show();
        };
        reader.readAsDataURL(file);
    }
    
    function displaySelectedVideo(file) {
        const videoUrl = URL.createObjectURL(file);
        $('#video-display').attr('src', videoUrl);
        $('#video-title').text(file.name);
        $('#video-container').show();
    }
    
    function uploadFile() {
        const fileInput = $('#file-input')[0];
        
        if (fileInput.files.length === 0) {
            logMessage('Please select a file first.');
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);
        
        // Show loading spinner
        $('#upload-spinner').show();
        
        // Simulate file upload with delay (replace with actual AJAX request)
        setTimeout(() => {
            $('#upload-spinner').hide();
            logMessage(`File "${file.name}" uploaded successfully.`);
            
            // In a real application, this would come from the server response
            if (file.type.startsWith('image/')) {
                $('#detect-image-btn').prop('disabled', false);
            } else if (file.type.startsWith('video/')) {
                $('#detect-video-btn').prop('disabled', false);
            }
        }, 1500);
        
        // In a real application, you would use AJAX to upload the file
        /*
        $.ajax({
            url: '/upload',
            type: 'POST',
            data: formData,
            processData: false,
            contentType: false,
            success: function(response) {
                $('#upload-spinner').hide();
                logMessage(`File "${file.name}" uploaded successfully.`);
                // Enable detection buttons based on file type
                if (file.type.startsWith('image/')) {
                    $('#detect-image-btn').prop('disabled', false);
                } else if (file.type.startsWith('video/')) {
                    $('#detect-video-btn').prop('disabled', false);
                }
            },
            error: function(xhr, status, error) {
                $('#upload-spinner').hide();
                logMessage('Error uploading file: ' + error);
            }
        });
        */
    }
    
    // ===================
    // Detection Methods
    // ===================
    
    function detectObjectsInImage() {
        const model = $('#model-select').val();
        const confidenceThreshold = parseFloat($('#confidence-threshold').val());
        const showBoxes = $('#show-boxes').is(':checked');
        const showLabels = $('#show-labels').is(':checked');
        const showConfidence = $('#show-confidence').is(':checked');
        const filterClasses = $('#filter-classes').is(':checked');
        
        let selectedClasses = [];
        if (filterClasses) {
            supportedClasses.forEach(className => {
                const classId = 'class-' + className.replace(/\s+/g, '-');
                if ($('#' + classId).is(':checked')) {
                    selectedClasses.push(className);
                }
            });
        }
        
        logMessage(`Running object detection with ${model}, confidence threshold: ${confidenceThreshold}`);
        
        // Simulate detection process with delay
        $('#image-display').css('opacity', 0.7);
        setTimeout(() => {
            // In a real application, detection would be performed on the server
            $('#image-display').css('opacity', 1);
            
            // Simulate detection results
            const detections = simulateDetections();
            displayDetectionResults(detections);
            
            // Show download option for processed image
            $('#download-container').show();
            $('#download-link').attr('href', $('#image-display').attr('src'));
            $('#download-link').attr('download', 'processed_image.jpg');
            
        }, 2000);
        
        // In a real application, you would use AJAX to send the request to the server
        /*
        $.ajax({
            url: '/detect_objects',
            type: 'POST',
            data: JSON.stringify({
                model: model,
                confidence_threshold: confidenceThreshold,
                show_boxes: showBoxes,
                show_labels: showLabels,
                show_confidence: showConfidence,
                filter_classes: filterClasses,
                selected_classes: selectedClasses
            }),
            contentType: 'application/json',
            success: function(response) {
                displayDetectionResults(response.detections);
                
                // Update the image with bounding boxes
                $('#image-display').attr('src', response.processed_image_url);
                
                // Show download option
                $('#download-container').show();
                $('#download-link').attr('href', response.processed_image_url);
                $('#download-link').attr('download', 'processed_image.jpg');
            },
            error: function(xhr, status, error) {
                logMessage('Error performing detection: ' + error);
            }
        });
        */
    }
    
    function toggleVideoDetection() {
        if (!isDetecting) {
            startVideoDetection();
        } else {
            stopVideoDetection();
        }
    }
    
    function startVideoDetection() {
        const model = $('#model-select').val();
        const confidenceThreshold = parseFloat($('#confidence-threshold').val());
        
        logMessage(`Starting video detection with ${model}, confidence threshold: ${confidenceThreshold}`);
        
        isDetecting = true;
        $('#detect-video-btn').text('Pause Detection');
        $('#stop-video-btn').prop('disabled', false);
        
        // Simulate detection at intervals
        detectionInterval = setInterval(() => {
            // Simulate detection results
            const detections = simulateDetections();
            displayDetectionResults(detections);
        }, 1000);
    }
    
    function stopVideoDetection() {
        if (detectionInterval) {
            clearInterval(detectionInterval);
            detectionInterval = null;
        }
        
        isDetecting = false;
        $('#detect-video-btn').text('Start Detection');
        $('#stop-video-btn').prop('disabled', true);
        logMessage('Video detection stopped.');
    }
    
    // ===================
    // Webcam Methods
    // ===================
    
    function startWebcam() {
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            navigator.mediaDevices.getUserMedia({ video: true })
                .then(function(stream) {
                    webcamStream = stream;
                    const video = document.getElementById('webcam-video');
                    video.srcObject = stream;
                    video.play();
                    
                    // Enable detection button and disable start button
                    $('#start-detection-btn').prop('disabled', false);
                    $('#start-webcam-btn').prop('disabled', true);
                    $('#stop-webcam-btn').prop('disabled', false);
                    
                    logMessage('Webcam started successfully.');
                })
                .catch(function(error) {
                    logMessage('Error accessing webcam: ' + error.message);
                });
        } else {
            logMessage('Your browser does not support webcam access.');
        }
    }
    
    function startWebcamDetection() {
        const model = $('#webcam-model-select').val();
        const confidenceThreshold = parseFloat($('#webcam-confidence-threshold').val());
        
        logMessage(`Starting webcam detection with ${model}, confidence threshold: ${confidenceThreshold}`);
        
        // Toggle button states
        $('#start-detection-btn').prop('disabled', true);
        $('#webcam-detection-info').show();
        
        // Set up canvas for drawing
        const video = document.getElementById('webcam-video');
        const canvas = document.getElementById('webcam-canvas');
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Hide video, show canvas
        $('#webcam-video').hide();
        $('#webcam-canvas').show();
        
        // Start detection loop
        detectionInterval = setInterval(() => {
            const ctx = canvas.getContext('2d');
            
            // Draw the current video frame
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Simulate detection
            const detections = simulateDetections();
            
            // Draw bounding boxes
            if ($('#webcam-show-boxes').is(':checked')) {
                drawDetections(ctx, detections);
            }
            
            // Update stats
            displayWebcamDetectionResults(detections);
        }, 100);
    }
    
    function stopWebcam() {
        if (webcamStream) {
            const tracks = webcamStream.getTracks();
            tracks.forEach(track => track.stop());
            webcamStream = null;
        }
        
        if (detectionInterval) {
            clearInterval(detectionInterval);
            detectionInterval = null;
        }
        
        // Reset UI
        $('#webcam-video').show();
        $('#webcam-canvas').hide();
        $('#start-webcam-btn').prop('disabled', false);
        $('#start-detection-btn').prop('disabled', true);
        $('#stop-webcam-btn').prop('disabled', true);
        $('#webcam-detection-info').hide();
        
        logMessage('Webcam stopped.');
    }
    
    // ===================
    // Utility Methods
    // ===================
    
    function logMessage(message) {
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = `<div>[${timestamp}] ${message}</div>`;
        $('#log-window').append(logEntry);
        
        // Scroll to bottom
        const logWindow = document.getElementById('log-window');
        logWindow.scrollTop = logWindow.scrollHeight;
    }
    
    function simulateDetections() {
        // Generate random detections for demo purposes
        const detections = [];
        const numDetections = Math.floor(Math.random() * 5) + 1;
        
        for (let i = 0; i < numDetections; i++) {
            const classIndex = Math.floor(Math.random() * supportedClasses.length);
            const confidence = (Math.random() * 0.5) + 0.5; // Random confidence between 0.5 and 1.0
            
            detections.push({
                class: supportedClasses[classIndex],
                confidence: confidence,
                box: {
                    x: Math.random() * 0.8,
                    y: Math.random() * 0.8,
                    width: Math.random() * 0.3 + 0.1,
                    height: Math.random() * 0.3 + 0.1
                }
            });
        }
        
        return detections;
    }
    
    function displayDetectionResults(detections) {
        // Display detection statistics
        const classCounts = {};
        
        detections.forEach(detection => {
            if (classCounts[detection.class]) {
                classCounts[detection.class]++;
            } else {
                classCounts[detection.class] = 1;
            }
        });
        
        let statsHtml = `<p>Found ${detections.length} objects:</p><ul>`;
        
        for (const className in classCounts) {
            statsHtml += `<li>${className}: ${classCounts[className]}</li>`;
        }
        
        statsHtml += '</ul>';
        
        $('#detection-stats').html(statsHtml);
        $('#detection-info').show();
    }
    
    function displayWebcamDetectionResults(detections) {
        // Similar to displayDetectionResults but for webcam
        const classCounts = {};
        
        detections.forEach(detection => {
            if (classCounts[detection.class]) {
                classCounts[detection.class]++;
            } else {
                classCounts[detection.class] = 1;
            }
        });
        
        let statsHtml = `<p>Detected ${detections.length} objects:</p><ul>`;
        
        for (const className in classCounts) {
            statsHtml += `<li>${className}: ${classCounts[className]}</li>`;
        }
        
        statsHtml += '</ul>';
        
        $('#webcam-detection-stats').html(statsHtml);
    }
    
    function drawDetections(ctx, detections) {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const showLabels = $('#webcam-show-labels').is(':checked');
        const showConfidence = $('#webcam-show-confidence').is(':checked');
        
        detections.forEach(detection => {
            const x = detection.box.x * width;
            const y = detection.box.y * height;
            const w = detection.box.width * width;
            const h = detection.box.height * height;
            
            // Draw bounding box
            ctx.strokeStyle = 'red';
            ctx.lineWidth = 2;
            ctx.strokeRect(x, y, w, h);
            
            if (showLabels || showConfidence) {
                // Draw label background
                let label = detection.class;
                if (showConfidence) {
                    label += ` ${(detection.confidence * 100).toFixed(1)}%`;
                }
                
                const textMetrics = ctx.measureText(label);
                const labelWidth = textMetrics.width + 6;
                const labelHeight = 20;
                
                ctx.fillStyle = 'rgba(255, 0, 0, 0.7)';
                ctx.fillRect(x, y - labelHeight, labelWidth, labelHeight);
                
                // Draw label text
                ctx.fillStyle = 'white';
                ctx.font = '14px Arial';
                ctx.fillText(label, x + 3, y - 5);
            }
        });
    }
});