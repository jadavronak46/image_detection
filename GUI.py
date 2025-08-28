import os
import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QFileDialog, QStackedWidget, QTextEdit,
                             QProgressBar, QMessageBox, QGroupBox, QRadioButton, QComboBox,
                             QListWidget, QListWidgetItem)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QPixmap, QImage
from main import AIDetector, MODEL_PATH, IMAGE_SIZE, VIDEO_FRAME_SAMPLE_RATE

class TrainingThread(QThread):
    update_progress = pyqtSignal(int, str)
    training_complete = pyqtSignal(float, str)  # accuracy, algorithm
    
    def __init__(self, detector):
        super().__init__()
        self.detector = detector
    
    def run(self):
        self.update_progress.emit(0, "Loading dataset...")
        success = self.detector.load_dataset()
        
        if not success:
            self.update_progress.emit(100, "Error: No samples found in dataset")
            self.training_complete.emit(0.0, "")
            return
        
        self.update_progress.emit(30, f"Training {self.detector.algorithm} model...")
        accuracy = self.detector.train()
        
        self.update_progress.emit(80, "Saving model...")
        self.detector.save_model()
        
        self.update_progress.emit(100, "Training complete!")
        self.training_complete.emit(accuracy, self.detector.algorithm)

class DetectionThread(QThread):
    update_progress = pyqtSignal(int, str)
    result_ready = pyqtSignal(str, str, QPixmap)  # title, result, image
    batch_result_ready = pyqtSignal(list)  # List of (filename, result) tuples
    
    def __init__(self, detector, media_paths, is_video=False, is_batch=False):
        super().__init__()
        self.detector = detector
        self.media_paths = media_paths
        self.is_video = is_video
        self.is_batch = is_batch
    
    def run(self):
        if not self.media_paths:
            self.result_ready.emit("Error", "No files selected", QPixmap())
            return
        
        if not self.is_batch:
            if not os.path.exists(self.media_paths[0]):
                self.result_ready.emit("Error", "File not found", QPixmap())
                return
            
            if not self.is_video:
                self.process_image(self.media_paths[0])
            else:
                self.process_video(self.media_paths[0])
        else:
            self.process_batch_images()
    
    def process_batch_images(self):
        results = []
        total_images = len(self.media_paths)
        
        for i, image_path in enumerate(self.media_paths):
            if not os.path.exists(image_path):
                results.append((os.path.basename(image_path), "Error: File not found"))
                continue
            
            self.update_progress.emit(int((i / total_images) * 100), 
                                    f"Processing image {i+1}/{total_images}...")
            
            label, confidence = self.detector.predict_image(image_path)
            if label is None:
                results.append((os.path.basename(image_path), "Error: Could not process image"))
                continue
            
            result_text = ("Real" if label == 0 else "AI-Generated") + \
                         f" (Confidence: {confidence*100:.1f}%)"
            results.append((os.path.basename(image_path), result_text))
        
        self.update_progress.emit(100, "Batch processing complete!")
        self.batch_result_ready.emit(results)
    
    def process_image(self, image_path):
        self.update_progress.emit(0, "Processing image...")
        label, confidence = self.detector.predict_image(image_path)
        
        if label is None:
            self.result_ready.emit("Error", "Could not process image", QPixmap())
            return
        
        # Load and resize image for display
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img).scaled(400, 400, Qt.KeepAspectRatio)
        
        result_text = ("Real" if label == 0 else "AI-Generated") + \
                     f" (Confidence: {confidence*100:.1f}%)\n" + \
                     f"Algorithm: {self.detector.algorithm}"
        
        self.update_progress.emit(100, "Detection complete")
        self.result_ready.emit("Image Detection Result", result_text, pixmap)
    
    def process_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.result_ready.emit("Error", "Could not open video", QPixmap())
            return
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ai_count = 0
        real_count = 0
        frame_count = 0
        sample_frame = None
        
        self.update_progress.emit(0, f"Processing video ({total_frames} frames)...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_count += 1
            if frame_count % VIDEO_FRAME_SAMPLE_RATE != 0:
                continue
                
            if sample_frame is None:
                sample_frame = frame.copy()
                
            prediction = self.detector.predict_video_frame(frame)
            if prediction is not None:
                if prediction == 1:
                    ai_count += 1
                else:
                    real_count += 1
            
            progress = min(100, int((frame_count / total_frames) * 100))
            self.update_progress.emit(progress, 
                                    f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        
        if (ai_count + real_count) == 0:
            self.result_ready.emit("Error", "No frames processed", QPixmap())
            return
        
        ai_percentage = (ai_count / (ai_count + real_count)) * 100
        overall_label = "AI-Generated" if ai_percentage > 50 else "Real"
        
        # Prepare sample frame for display
        if sample_frame is not None:
            sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
            height, width, channel = sample_frame.shape
            bytes_per_line = 3 * width
            q_img = QImage(sample_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_img).scaled(400, 400, Qt.KeepAspectRatio)
        else:
            pixmap = QPixmap()
        
        result_text = (f"Overall: {overall_label}\n"
                      f"AI-Generated frames: {ai_percentage:.1f}%\n"
                      f"Real frames: {100 - ai_percentage:.1f}%\n"
                      f"Frames analyzed: {ai_count + real_count}\n"
                      f"Algorithm: {self.detector.algorithm}")
        
        self.update_progress.emit(100, "Video analysis complete")
        self.result_ready.emit("Video Analysis Result", result_text, pixmap)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.init_ui()
        self.check_model_exists()
    
    def init_ui(self):
        self.setWindowTitle("AI Media Detection System")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create stacked widget for different screens
        self.stacked_widget = QStackedWidget()
        main_layout.addWidget(self.stacked_widget)
        
        # Create pages
        self.create_home_page()
        self.create_training_page()
        self.create_detection_page()
    
    def create_home_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("AI Media Detection System")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 24px; font-weight: bold; margin: 20px;")
        
        desc = QLabel(
            "This system can detect AI-generated images and videos using machine learning.\n"
            "Train the model with your dataset or use the pre-trained model to detect media."
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setWordWrap(True)
        
        btn_train = QPushButton("Train Model")
        btn_train.clicked.connect(self.show_training_page)
        btn_train.setFixedHeight(40)
        
        btn_detect = QPushButton("Detect Media")
        btn_detect.clicked.connect(self.show_detection_page)
        btn_detect.setFixedHeight(40)
        
        btn_exit = QPushButton("Exit")
        btn_exit.clicked.connect(self.close)
        btn_exit.setFixedHeight(40)
        
        layout.addStretch(1)
        layout.addWidget(title)
        layout.addWidget(desc)
        layout.addSpacing(30)
        layout.addWidget(btn_train)
        layout.addWidget(btn_detect)
        layout.addWidget(btn_exit)
        layout.addStretch(1)
        
        self.stacked_widget.addWidget(page)
    
    def create_training_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Model Training")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        
        # Algorithm selection
        algo_group = QGroupBox("Algorithm Selection")
        algo_layout = QHBoxLayout()
        self.algo_combo = QComboBox()
        self.algo_combo.addItems(["SVM", "Random Forest", "K-Nearest Neighbors", "Naive Bayes"])
        algo_layout.addWidget(QLabel("Algorithm:"))
        algo_layout.addWidget(self.algo_combo)
        algo_group.setLayout(algo_layout)
        
        self.train_status = QTextEdit()
        self.train_status.setReadOnly(True)
        self.train_status.setFixedHeight(100)
        
        self.train_progress = QProgressBar()
        self.train_progress.setRange(0, 100)
        
        btn_start = QPushButton("Start Training")
        btn_start.clicked.connect(self.start_training)
        
        btn_back = QPushButton("Back to Home")
        btn_back.clicked.connect(self.show_home_page)
        
        layout.addWidget(title)
        layout.addWidget(algo_group)
        layout.addWidget(QLabel("Training Status:"))
        layout.addWidget(self.train_status)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.train_progress)
        layout.addWidget(btn_start)
        layout.addWidget(btn_back)
        
        self.stacked_widget.addWidget(page)
    
    def create_detection_page(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        
        title = QLabel("Media Detection")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 20px; font-weight: bold; margin: 10px;")
        
        # Media type selection
        media_type_group = QGroupBox("Media Type")
        media_type_layout = QHBoxLayout()
        self.radio_image = QRadioButton("Image")
        self.radio_video = QRadioButton("Video")
        self.radio_batch = QRadioButton("Batch Images")
        self.radio_image.setChecked(True)
        media_type_layout.addWidget(self.radio_image)
        media_type_layout.addWidget(self.radio_video)
        media_type_layout.addWidget(self.radio_batch)
        media_type_group.setLayout(media_type_layout)
        
        self.btn_select = QPushButton("Select Media File(s)")
        self.btn_select.clicked.connect(self.select_media_file)
        
        # List widget for batch images
        self.batch_list = QListWidget()
        self.batch_list.setVisible(False)
        
        self.media_preview = QLabel()
        self.media_preview.setAlignment(Qt.AlignCenter)
        self.media_preview.setFixedSize(400, 400)
        self.media_preview.setStyleSheet("border: 1px solid #ccc; background: #f0f0f0;")
        
        self.detect_status = QTextEdit()
        self.detect_status.setReadOnly(True)
        self.detect_status.setFixedHeight(100)
        
        self.detect_progress = QProgressBar()
        self.detect_progress.setRange(0, 100)
        
        self.btn_detect = QPushButton("Detect")
        self.btn_detect.clicked.connect(self.start_detection)
        self.btn_detect.setEnabled(False)
        
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("font-size: 14px;")
        
        btn_back = QPushButton("Back to Home")
        btn_back.clicked.connect(self.show_home_page)
        
        # Connect radio buttons to toggle UI elements
        self.radio_image.toggled.connect(self.update_media_selection_ui)
        self.radio_video.toggled.connect(self.update_media_selection_ui)
        self.radio_batch.toggled.connect(self.update_media_selection_ui)
        
        layout.addWidget(title)
        layout.addWidget(media_type_group)
        layout.addWidget(self.btn_select)
        layout.addWidget(self.batch_list)
        layout.addWidget(self.media_preview)
        layout.addWidget(QLabel("Status:"))
        layout.addWidget(self.detect_status)
        layout.addWidget(QLabel("Progress:"))
        layout.addWidget(self.detect_progress)
        layout.addWidget(self.btn_detect)
        layout.addWidget(QLabel("Result:"))
        layout.addWidget(self.result_display)
        layout.addWidget(btn_back)
        
        self.stacked_widget.addWidget(page)
    
    def update_media_selection_ui(self):
        """Update UI elements based on selected media type"""
        is_batch = self.radio_batch.isChecked()
        self.batch_list.setVisible(is_batch)
        self.media_preview.setVisible(not is_batch)
        
        if is_batch:
            self.btn_select.setText("Select Image Files")
        elif self.radio_image.isChecked():
            self.btn_select.setText("Select Image File")
        else:
            self.btn_select.setText("Select Video File")
    
    def check_model_exists(self):
        if os.path.exists(MODEL_PATH):
            self.detector = AIDetector()
            if self.detector.load_model():
                self.train_status.append(f"Pre-trained {self.detector.algorithm} model loaded successfully.")
    
    def show_home_page(self):
        self.stacked_widget.setCurrentIndex(0)
    
    def show_training_page(self):
        self.stacked_widget.setCurrentIndex(1)
    
    def show_detection_page(self):
        self.stacked_widget.setCurrentIndex(2)
    
    def start_training(self):
        algorithm_map = {
            "SVM": "svm",
            "Random Forest": "random_forest",
            "K-Nearest Neighbors": "knn",
            "Naive Bayes": "naive_bayes"
        }
        selected_algo = algorithm_map[self.algo_combo.currentText()]
        
        self.detector = AIDetector(algorithm=selected_algo)
        self.train_status.clear()
        self.train_progress.setValue(0)
        
        self.training_thread = TrainingThread(self.detector)
        self.training_thread.update_progress.connect(self.update_training_progress)
        self.training_thread.training_complete.connect(self.training_completed)
        self.training_thread.start()
    
    def update_training_progress(self, value, message):
        self.train_progress.setValue(value)
        self.train_status.append(message)
    
    def training_completed(self, accuracy, algorithm):
        if accuracy > 0:
            self.train_status.append(f"\nTraining completed with accuracy: {accuracy*100:.2f}%")
            QMessageBox.information(self, "Success", 
                                   f"{algorithm} model trained and saved successfully!")
        else:
            QMessageBox.warning(self, "Warning", "Training completed but no valid model was created.")
    
    def select_media_file(self):
        if self.radio_batch.isChecked():
            file_paths, _ = QFileDialog.getOpenFileNames(
                self, "Select Images", "", 
                "Images (*.png *.jpg *.jpeg)"
            )
            
            if file_paths:
                self.current_media_paths = file_paths
                self.btn_detect.setEnabled(True)
                self.batch_list.clear()
                
                for path in file_paths:
                    item = QListWidgetItem(os.path.basename(path))
                    self.batch_list.addItem(item)
                
                self.detect_status.clear()
                self.detect_progress.setValue(0)
                self.result_display.clear()
        else:
            if self.radio_image.isChecked():
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select Image", "", 
                    "Images (*.png *.jpg *.jpeg)"
                )
            else:
                file_path, _ = QFileDialog.getOpenFileName(
                    self, "Select Video", "", 
                    "Videos (*.mp4 *.avi *.mov *.mkv)"
                )
            
            if file_path:
                self.current_media_paths = [file_path]
                self.btn_detect.setEnabled(True)
                
                # Show preview
                if self.radio_image.isChecked():
                    pixmap = QPixmap(file_path).scaled(400, 400, Qt.KeepAspectRatio)
                    self.media_preview.setPixmap(pixmap)
                else:
                    # For video, show a thumbnail
                    cap = cv2.VideoCapture(file_path)
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        height, width, channel = frame.shape
                        bytes_per_line = 3 * width
                        q_img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(q_img).scaled(400, 400, Qt.KeepAspectRatio)
                        self.media_preview.setPixmap(pixmap)
                    cap.release()
                
                self.detect_status.clear()
                self.detect_progress.setValue(0)
                self.result_display.clear()
    
    def start_detection(self):
        if not hasattr(self, 'current_media_paths') or not self.current_media_paths:
            QMessageBox.warning(self, "Warning", "Please select media file(s) first!")
            return
        
        if self.detector is None or not self.detector.load_model():
            QMessageBox.critical(self, "Error", "No trained model available. Please train a model first.")
            return
        
        self.detect_status.clear()
        self.detect_progress.setValue(0)
        self.result_display.clear()
        
        is_video = self.radio_video.isChecked()
        is_batch = self.radio_batch.isChecked()
        
        self.detection_thread = DetectionThread(
            self.detector, 
            self.current_media_paths, 
            is_video,
            is_batch
        )
        self.detection_thread.update_progress.connect(self.update_detection_progress)
        
        if is_batch:
            self.detection_thread.batch_result_ready.connect(self.show_batch_results)
        else:
            self.detection_thread.result_ready.connect(self.show_detection_result)
            
        self.detection_thread.start()
    
    def update_detection_progress(self, value, message):
        self.detect_progress.setValue(value)
        self.detect_status.append(message)
    
    def show_detection_result(self, title, result, pixmap):
        self.result_display.clear()
        self.result_display.append(f"{title}\n{'-'*50}\n{result}")
        
        if not pixmap.isNull():
            self.media_preview.setPixmap(pixmap)
        
        QMessageBox.information(self, "Detection Complete", "Media analysis finished!")
    
    def show_batch_results(self, results):
        self.result_display.clear()
        self.result_display.append("Batch Detection Results\n" + "-"*50)
        
        for filename, result in results:
            self.result_display.append(f"\n{filename}\n{result}")
        
        QMessageBox.information(self, "Batch Processing Complete", 
                              f"Processed {len(results)} images successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())