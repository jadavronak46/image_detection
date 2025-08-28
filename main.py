import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle
from tqdm import tqdm
from skimage.feature import local_binary_pattern

# Configuration
DATASET_PATH = "dataset"
MODEL_PATH = "ai_detector_model.pkl"
IMAGE_SIZE = (128, 128)
VIDEO_FRAME_SAMPLE_RATE = 10

class AIDetector:
    def __init__(self, algorithm='svm'):
        """Initialize with selected algorithm"""
        self.algorithm = algorithm
        self.models = {
            'svm': SVC(kernel='rbf', probability=True, gamma='scale'),
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5),
            'naive_bayes': GaussianNB()
        }
        self.model = self.models.get(algorithm.lower(), SVC())
        self.features = []
        self.labels = []
    
    def extract_features(self, img):
        """Enhanced feature extraction with multiple features"""
        if img is None:
            return None
            
        # Resize and convert to HSV
        img = cv2.resize(img, IMAGE_SIZE)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Color histogram (HSV space)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Texture features (LBP)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray, P=8, R=1, method='uniform')
        lbp_hist = np.histogram(lbp.ravel(), bins=32, range=(0, 256))[0]
        lbp_hist = lbp_hist.astype("float") / (lbp_hist.sum() + 1e-7)
        
        # Edge features (Canny)
        edges = cv2.Canny(gray, 100, 200)
        edge_hist = np.histogram(edges.ravel(), bins=32, range=(0, 256))[0]
        edge_hist = edge_hist.astype("float") / (edge_hist.sum() + 1e-7)
        
        # Combine all features
        return np.hstack([hist, lbp_hist, edge_hist])
    
    def load_dataset(self):
        """Load dataset with progress tracking"""
        self.features = []
        self.labels = []
        
        print("Loading dataset...")
        for label, folder in enumerate(["real", "fake"]):
            folder_path = os.path.join(DATASET_PATH, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: {folder_path} does not exist")
                continue
                
            for filename in tqdm(os.listdir(folder_path)):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(folder_path, filename)
                    img = cv2.imread(img_path)
                    features = self.extract_features(img)
                    if features is not None:
                        self.features.append(features)
                        self.labels.append(label)
        
        if len(self.labels) == 0:
            print("Error: No images found in dataset folders")
            return False
        
        self.features = np.array(self.features)
        self.labels = np.array(self.labels)
        print(f"Loaded {len(self.labels)} samples")
        print(f"Class distribution: Real={sum(self.labels == 0)}, Fake={sum(self.labels == 1)}")
        return True
    
    def train(self, test_size=0.2):
        """Train the selected classifier with evaluation"""
        if len(self.features) == 0:
            print("Error: No features to train on")
            return 0.0
        
        # Check if we have both classes
        unique_classes = np.unique(self.labels)
        if len(unique_classes) < 2:
            print(f"Error: Need both real and fake samples. Found only {unique_classes}")
            print("Please ensure your dataset contains both real and fake images")
            return 0.0
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.features, self.labels, test_size=test_size, random_state=42
        )
        
        print(f"Training {self.algorithm}...")
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"Test Accuracy: {acc*100:.2f}%")
        
        # Handle classification report for binary classification
        try:
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, 
                                      target_names=['Real', 'AI-Generated'],
                                      zero_division=0))
        except ValueError as e:
            print(f"\nCould not generate full classification report: {str(e)}")
            print(f"Simple accuracy: {acc*100:.2f}%")
        
        return acc
    
    def save_model(self):
        """Save trained model with metadata"""
        model_data = {
            'model': self.model,
            'algorithm': self.algorithm,
            'features_shape': self.features[0].shape if len(self.features) > 0 else None
        }
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {MODEL_PATH}")
    
    def load_model(self):
        """Load pre-trained model with validation"""
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.algorithm = model_data.get('algorithm', 'svm')
            print(f"Loaded {self.algorithm} model from {MODEL_PATH}")
            return True
        print("No trained model found")
        return False
    
    def predict_image(self, img_path):
        """Predict if an image is real or AI-generated with confidence"""
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_path}")
            return None, None
        
        features = self.extract_features(img)
        if features is None:
            print(f"Error: Could not extract features from {img_path}")
            return None, None
        
        proba = self.model.predict_proba([features])[0]
        label = self.model.predict([features])[0]
        return label, max(proba)
    
    def predict_video_frame(self, frame):
        """Predict a single video frame"""
        features = self.extract_features(frame)
        if features is None:
            return None
        return self.model.predict([features])[0]

if __name__ == "__main__":
    print("AI Image Detection System")
    print("Available algorithms: svm, random_forest, knn, naive_bayes")
    algorithm = input("Select algorithm (default: svm): ").strip().lower() or 'svm'
    
    detector = AIDetector(algorithm=algorithm)
    
    print("\n1. Train new model")
    print("2. Detect image/video")
    choice = input("Select option (1-2): ").strip()
    
    if choice == '1':
        if detector.load_dataset():
            detector.train()
            detector.save_model()
    elif choice == '2':
        if detector.load_model():
            media_path = input("Enter path to image/video: ").strip()
            if not os.path.exists(media_path):
                print("Error: File not found")
            else:
                ext = os.path.splitext(media_path)[1].lower()
                if ext in ['.jpg', '.jpeg', '.png']:
                    label, confidence = detector.predict_image(media_path)
                    if label is not None:
                        result = "Real" if label == 0 else "AI-Generated"
                        print(f"Result: {result} (Confidence: {confidence*100:.1f}%)")
                elif ext in ['.mp4', '.avi', '.mov']:
                    print("Video detection requires GUI for better visualization")
                    print("Please run GUI.py for video detection")
                else:
                    print("Unsupported file format")
    else:
        print("Invalid choice")