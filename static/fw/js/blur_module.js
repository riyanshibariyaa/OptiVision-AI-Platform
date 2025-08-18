$(document).ready(function () {
    // State management
    let currentEventSource = null;
    let processedImages = [];
    let currentImageIndex = 0;
    let isProcessingPaused = false;
    let currentSessionId = null;
    let uploadedFiles = new Set();
    let localImages = [];

    // API endpoints with proper module prefixing
    const API = {
        upload: '/blur/upload',
        processStream: '/blur/process-stream',
        controlProcessing: '/blur/control-processing',
        download: '/blur/download',
        downloadZip: '/blur/download-zip',
        clean: '/blur/clean',
        fetchLocal: '/blur/fetch-local'
    };

    // Utility Functions
    function generateSessionId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    function showError(message) {
        $('#errorDisplay').removeClass('d-none').addClass('show');
        $('#errorMessage').text(message);
    }

    function hideError() {
        $('#errorDisplay').removeClass('show').addClass('d-none');
        $('#errorMessage').text('');
    }

    function addLog(message) {
        console.log(message);
    }

    // File Handling Functions
    function handleDragOver(e) {
        e.preventDefault();
        e.stopPropagation();
        $('#dragDropZone').addClass('dragover');
    }

    function handleDragLeave(e) {
        e.preventDefault();
        e.stopPropagation();
        $('#dragDropZone').removeClass('dragover');
    }

    function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        $('#dragDropZone').removeClass('dragover');
        const files = e.originalEvent.dataTransfer.files;
        handleFiles(files);
    }

    async function handleFiles(files) {
        const formData = new FormData();
        let validFiles = 0;

        for (let file of files) {
            if (file.type.startsWith('image/')) {
                formData.append('files[]', file);
                validFiles++;
            }
        }

        if (validFiles === 0) {
            showError('Please upload only image files.');
            return;
        }

        try {
            console.log("Uploading files to:", API.upload);
            const response = await fetch(API.upload, {
                method: 'POST',
                body: formData
            });

            const data = await response.json();
            console.log("Upload response:", data);

            if (data.error) {
                showError(data.error);
                return;
            }

            data.uploaded.forEach(filename => {
                uploadedFiles.add(filename);
                addFileToList(filename);
            });

            updateUploadStatus();
            hideError();
        } catch (error) {
            console.error("Upload error:", error);
            showError('Upload failed: ' + error.message);
        }
    }

    function addFileToList(filename) {
        const fileElement = $(`
            <div class="uploaded-file" data-filename="${filename}">
                <i class="fas fa-file-image me-2"></i>
                <span>${filename}</span>
                <button class="btn btn-link text-danger remove-file p-0 ms-2">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `);
        $('#uploadedFiles').append(fileElement);
    }

    // UI Update Functions
    function updateUploadStatus() {
        const fileCount = uploadedFiles.size;
        $('#startButton').prop('disabled', fileCount === 0);
        $('#uploadedFiles').toggle(fileCount > 0);
    }

    function updateCarousel() {
        const carousel = $('#imageCarousel');
        carousel.empty();

        processedImages.forEach((image, index) => {
            const item = $(`
                <div class="carousel-item ${index === currentImageIndex ? 'active' : ''}" 
                     data-index="${index}">
                    <img src="${image.output_image}" alt="Processed image ${index + 1}" class="img-fluid">
                </div>
            `);
            carousel.append(item);
        });
    }

    async function loadCurrentImage() {
        if (processedImages.length === 0) {
            resetImageDisplay();
            return;
        }

        const currentImage = processedImages[currentImageIndex];

        try {
            if (currentImage.input_image) {
                $('#inputImageContainer').html(`
                    <img src="${currentImage.input_image}" 
                         alt="Input Image" 
                         class="img-fluid rounded image-preview">`);
                $('#inputFilename').text(currentImage.filename);
            }

            if (currentImage.output_image) {
                $('#outputImageContainer').html(`
                    <img src="${currentImage.output_image}" 
                         alt="Output Image" 
                         class="img-fluid rounded image-preview">`);
                $('#outputFilename').text(currentImage.filename);
            }

            $('#imageIndexDisplay').text(`Image ${currentImageIndex + 1} of ${processedImages.length}`);
            updateNavigationButtons();
            updateCarousel();
        } catch (error) {
            showError("Failed to load image: " + error.message);
        }
    }

    function resetImageDisplay() {
        $('#inputImageContainer').html('<div class="placeholder-image">No Image Selected</div>');
        $('#outputImageContainer').html('<div class="placeholder-image">No Image Processed</div>');
        $('#inputFilename').text('');
        $('#outputFilename').text('');
        $('#imageIndexDisplay').text('No images available');
        $('#navigationButtons').hide();
        $('#imageCarousel').empty();
    }

    function updateNavigationButtons() {
        $('#prevButton').prop('disabled', currentImageIndex === 0);
        $('#nextButton').prop('disabled', currentImageIndex === processedImages.length - 1);
        $('#navigationButtons').toggle(processedImages.length > 0);
    }

    function updateControlButtons(processing = false) {
        $('#startButton').prop('disabled', processing || uploadedFiles.size === 0);
        $('#pauseButton').prop('disabled', !processing);
        $('#stopButton').prop('disabled', !processing);
        $('#downloadAllZip').prop('disabled', processedImages.length === 0);
        $('#downloadCurrent').prop('disabled', processedImages.length === 0);
    }

    // Processing Control Functions
    async function sendProcessingCommand(command) {
        try {
            const response = await fetch(API.controlProcessing, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    command: command,
                    sessionId: currentSessionId
                })
            });
            return await response.json();
        } catch (error) {
            console.error('Failed to send processing command:', error);
            return { error: 'Failed to send command' };
        }
    }

    // Fetch local images
    function fetchLocalImages() {
        addLog('Fetching images from local repository...');
        
        $.ajax({
            url: API.fetchLocal,
            type: 'GET',
            success: function(data) {
                localImages = data.images;
                uploadedFiles = new Set(localImages);
                addLog(`Found ${localImages.length} images in local repository.`);
                updateUploadStatus();
            },
            error: function(xhr, status, error) {
                console.error('Error fetching local images:', error);
                addLog('Error: Could not fetch images from local repository.');
            }
        });
    }

    // Sidebar Functions
    function toggleSidebar() {
        const sidebar = $('#sidebar');
        const mainContent = $('#mainContent');
        const sidebarIcon = $('#sidebarIcon');

        sidebar.toggleClass('collapsed');
        mainContent.toggleClass('sidebar-collapsed');

        if (sidebar.hasClass('collapsed')) {
            sidebarIcon.removeClass('fa-chevron-left').addClass('fa-chevron-right');
            mainContent.addClass('expanded');
        } else {
            sidebarIcon.removeClass('fa-chevron-right').addClass('fa-chevron-left');
            mainContent.removeClass('expanded');
        }
    }

    function checkWindowSize() {
        if ($(window).width() <= 991.98) {
            $('#sidebar').addClass('collapsed');
            $('#mainContent').addClass('sidebar-collapsed expanded');
            $('#sidebarIcon').removeClass('fa-chevron-left').addClass('fa-chevron-right');
        }
    }

    // Source selection handlers
    $('#inputType').on('change', function() {
        const selectedValue = $(this).val();
        
        // Hide all sections first
        $('#s3Details, #gdriveDetails, #dropboxDetails, #uploadSection').hide();
        
        // Reset uploaded files
        uploadedFiles = new Set();
        $('#uploadedFiles').empty();
        
        // Show relevant section
        switch(selectedValue) {
            case 's3':
                $('#s3Details').show();
                break;
            case 'gdrive':
                $('#gdriveDetails').show();
                break;
            case 'dropbox':
                $('#dropboxDetails').show();
                break;
            case 'upload':
                $('#uploadSection').show();
                break;
            case 'local':
                fetchLocalImages();
                break;
        }
        
        updateUploadStatus();
    });

    // Event Listeners
    $('#dragDropZone')
        .on('dragover', handleDragOver)
        .on('dragleave', handleDragLeave)
        .on('drop', handleDrop);

    
$('#fileInput').on('change', function (e) {
    handleFiles(this.files);
    
    // Update the file input display text
    const fileNames = Array.from(this.files).map(file => file.name).join(', ');
    const fileInfoDisplay = $(this).next('.file-info-display');
    
    
});

    $('#uploadedFiles').on('click', '.remove-file', function () {
        const fileElement = $(this).closest('.uploaded-file');
        const filename = fileElement.data('filename');
        uploadedFiles.delete(filename);
        fileElement.remove();
        updateUploadStatus();
    });

    $('#prevButton').click(async function () {
        if (currentImageIndex > 0) {
            currentImageIndex--;
            await loadCurrentImage();
        }
    });

    $('#nextButton').click(async function () {
        if (currentImageIndex < processedImages.length - 1) {
            currentImageIndex++;
            await loadCurrentImage();
        }
    });

    $('#imageCarousel').on('click', '.carousel-item', function () {
        const newIndex = $(this).data('index');
        if (newIndex !== currentImageIndex) {
            currentImageIndex = newIndex;
            loadCurrentImage();
        }
    });

    $('#downloadCurrent').click(async function () {
        if (processedImages.length > 0) {
            const filename = processedImages[currentImageIndex].filename;
            window.location.href = `${API.download}/${encodeURIComponent(filename)}`;
        }
    });

    $('#downloadAllZip').click(function () {
        if (processedImages.length > 0) {
            window.location.href = API.downloadZip;
        }
    });

    $('#pauseButton').click(async function () {
        isProcessingPaused = !isProcessingPaused;
        $(this).html(isProcessingPaused ?
            '<i class="fas fa-play me-2"></i>Resume' :
            '<i class="fas fa-pause me-2"></i>Pause'
        );
        await sendProcessingCommand(isProcessingPaused ? 'pause' : 'resume');
    });

    $('#stopButton').click(async function () {
        await sendProcessingCommand('stop');
        if (currentEventSource) {
            currentEventSource.close();
        }
        updateControlButtons(false);
        isProcessingPaused = false;
        $('#pauseButton').html('<i class="fas fa-pause me-2"></i>Pause');
    });

    $('#startButton').click(function () {
        hideError();
        processedImages = [];
        currentImageIndex = 0;
        isProcessingPaused = false;
        currentSessionId = generateSessionId();

        $('#progressBar').css('width', '0%').text('0%');
        resetImageDisplay();
        $('#pauseButton').html('<i class="fas fa-pause me-2"></i>Pause');

        updateControlButtons(true);

        if (currentEventSource) {
            currentEventSource.close();
        }


        

        const blurIntensity = $('#blurIntensity').val();
        const showBoxes = $('#showBoxesToggle').is(':checked');
        console.log("Starting process stream with intensity:", blurIntensity, "show boxes:", showBoxes);

        currentEventSource = new EventSource(
            `${API.processStream}?blurIntensity=${blurIntensity}&sessionId=${currentSessionId}&showBoxes=${showBoxes}`
        );

        currentEventSource.onmessage = function (event) {
            const data = JSON.parse(event.data);
            console.log("Process update:", data);

            if (data.error) {
                showError(data.error);
                currentEventSource.close();
                updateControlButtons(false);
                return;
            }

            if (data.processing_complete || data.processing_stopped) {
                currentEventSource.close();
                updateControlButtons(false);
                return;
            }

            const progress = Math.round((data.processed / data.total) * 100);
            $('#progressBar').css('width', progress + '%').text(progress + '%');

            processedImages = processedImages.filter(img => img.filename !== data.filename);
            processedImages.push({
                filename: data.filename,
                input_image: data.input_image,
                output_image: data.output_image
            });

            if (currentImageIndex === processedImages.length - 2) {
                currentImageIndex++;
                loadCurrentImage();
            } else if (processedImages.length === 1) {
                loadCurrentImage();
            }
        };

        currentEventSource.onerror = function (error) {
            console.error('EventSource failed:', error);
            showError("Error processing images. Please try again.");
            currentEventSource.close();
            updateControlButtons(false);
        };
    });

    $('#sidebarToggle').click(toggleSidebar);

    // Helper functions for cloud storage
    window.saveS3Details = function() {
        const s3Details = {
            accessKey: $('#s3AccessKey').val(),
            secretKey: $('#s3SecretKey').val(),
            bucketName: $('#s3BucketName').val(),
            region: $('#s3Region').val()
        };
        
        localStorage.setItem('s3Details', JSON.stringify(s3Details));
        addLog('S3 details saved.');
    };
    
    window.saveGDriveDetails = function() {
        const gdriveDetails = {
            email: $('#gdriveEmail').val(),
            apiKey: $('#gdriveApiKey').val(),
            folderId: $('#gdriveFolderId').val()
        };
        
        localStorage.setItem('gdriveDetails', JSON.stringify(gdriveDetails));
        addLog('Google Drive details saved.');
    };
    
    window.saveDropboxDetails = function() {
        const dropboxDetails = {
            accessToken: $('#dropboxAccessToken').val(),
            folderPath: $('#dropboxFolderPath').val()
        };
        
        localStorage.setItem('dropboxDetails', JSON.stringify(dropboxDetails));
        addLog('Dropbox details saved.');
    };
    
    // Load saved details if available
    function loadSavedDetails() {
        try {
            const s3Details = JSON.parse(localStorage.getItem('s3Details'));
            if (s3Details) {
                $('#s3AccessKey').val(s3Details.accessKey);
                $('#s3SecretKey').val(s3Details.secretKey);
                $('#s3BucketName').val(s3Details.bucketName);
                $('#s3Region').val(s3Details.region);
            }
            
            const gdriveDetails = JSON.parse(localStorage.getItem('gdriveDetails'));
            if (gdriveDetails) {
                $('#gdriveEmail').val(gdriveDetails.email);
                $('#gdriveApiKey').val(gdriveDetails.apiKey);
                $('#gdriveFolderId').val(gdriveDetails.folderId);
            }
            
            const dropboxDetails = JSON.parse(localStorage.getItem('dropboxDetails'));
            if (dropboxDetails) {
                $('#dropboxAccessToken').val(dropboxDetails.accessToken);
                $('#dropboxFolderPath').val(dropboxDetails.folderPath);
            }
        } catch (e) {
            console.error('Error loading saved details:', e);
        }
    }

    // Initial Setup
    checkWindowSize();
    $(window).resize(checkWindowSize);
    updateUploadStatus();
    updateControlButtons();
    loadSavedDetails();
    
    // Added: Initialize blur intensity display
    $('#blurIntensity').on('input', function() {
        $('#blurValue').text($(this).val() + '%');
    });
});