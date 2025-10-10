"""
Simple and Robust Face Recognition System
"""
import cv2
import numpy as np
import os
import json
import sqlite3
import time

class FaceRecognitionSystem:
    def __init__(self):
        # Create data directory
        os.makedirs("data", exist_ok=True)
        
        # Initialize face cascade
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Load known faces
        self.known_faces = {}
        self.load_faces()
        
        # Initialize database
        self.init_database()
        
        # Tracking for attendance logging
        self.last_logged = {}
    
    def load_faces(self):
        """Load known faces from file"""
        try:
            if os.path.exists("data/faces.json"):
                with open("data/faces.json", "r") as f:
                    data = json.load(f)
                    for name, encodings in data.items():
                        self.known_faces[name] = [np.array(e, dtype=np.float32) for e in encodings]
                print(f"Loaded {len(self.known_faces)} people")
        except Exception as e:
            print(f"Error loading faces: {e}")
            self.known_faces = {}
    
    def save_faces(self):
        """Save known faces to file"""
        try:
            data = {}
            for name, encodings in self.known_faces.items():
                data[name] = [e.tolist() for e in encodings]
            with open("data/faces.json", "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error saving faces: {e}")
    
    def init_database(self):
        """Initialize attendance database"""
        try:
            self.db = sqlite3.connect("data/attendance.db")
            self.db.execute("""CREATE TABLE IF NOT EXISTS attendance 
                             (id INTEGER PRIMARY KEY AUTOINCREMENT,
                              name TEXT NOT NULL,
                              timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)""")
            self.db.commit()
        except Exception as e:
            print(f"Database error: {e}")
    
    def extract_features(self, face_img):
        """Extract simple face features"""
        try:
            # Resize to standard size
            face = cv2.resize(face_img, (32, 32))
            
            # Convert to grayscale if needed
            if len(face.shape) == 3:
                gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            else:
                gray = face
            
            # Normalize and flatten
            features = gray.flatten().astype(np.float32) / 255.0
            return features
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return np.array([], dtype=np.float32)
    
    def add_person(self, name):
        """Add a new person"""
        try:
            # Open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Cannot open camera")
                return False
            
            print(f"Capturing face for {name}. Press SPACE to capture, ESC to cancel.")
            face_img = None
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read from camera")
                    break
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
                
                # Draw rectangles
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                cv2.imshow(f"Add {name}", frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space key
                    if len(faces) > 0:
                        # Take the first face
                        (x, y, w, h) = faces[0]
                        face_img = frame[y:y+h, x:x+w]
                        break
                elif key == 27:  # ESC key
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            
            # Process captured face
            if face_img is not None:
                features = self.extract_features(face_img)
                if features.size > 0:
                    self.known_faces[name] = [features]
                    self.save_faces()
                    print(f"Successfully added {name}!")
                    return True
                else:
                    print("Error: Could not extract features")
            else:
                print("Cancelled or no face detected")
            
            return False
        except Exception as e:
            print(f"Error adding person: {e}")
            return False
    
    def recognize_face(self, face_img):
        """Recognize a face"""
        try:
            if not self.known_faces:
                return "Unknown", 0.0
            
            features = self.extract_features(face_img)
            if features.size == 0:
                return "Unknown", 0.0
            
            best_name, best_conf = "Unknown", 0.0
            
            # Compare with known faces using cosine similarity
            for name, encodings in self.known_faces.items():
                for encoding in encodings:
                    # Calculate cosine similarity
                    dot_product = np.dot(features, encoding)
                    norm_a = np.linalg.norm(features)
                    norm_b = np.linalg.norm(encoding)
                    
                    if norm_a > 0 and norm_b > 0:
                        similarity = dot_product / (norm_a * norm_b)
                        # Convert to 0-1 confidence
                        conf = (similarity + 1) / 2
                        
                        if conf > best_conf:
                            best_conf, best_name = conf, name
            
            # Return result with threshold
            return best_name if best_conf > 0.4 else "Unknown", best_conf
        except Exception as e:
            print(f"Recognition error: {e}")
            return "Unknown", 0.0
    
    def log_attendance(self, name):
        """Log attendance with cooldown"""
        try:
            current_time = time.time()
            
            # Check cooldown (5 minutes)
            if name in self.last_logged:
                if current_time - self.last_logged[name] < 300:  # 5 minutes
                    return False
            
            # Log to database
            self.db.execute("INSERT INTO attendance (name) VALUES (?)", (name,))
            self.db.commit()
            
            # Update last logged time
            self.last_logged[name] = current_time
            print(f"Attendance logged: {name}")
            return True
        except Exception as e:
            print(f"Attendance logging error: {e}")
            return False
    
    def start_recognition(self):
        """Start face recognition loop"""
        try:
            if not self.known_faces:
                print("No people added yet. Please add people first.")
                return
            
            # Open camera
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Cannot open camera")
                return
            
            print("Starting recognition. Press ESC to stop.")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Cannot read from camera")
                    break
                
                # Detect faces
                faces = self.face_cascade.detectMultiScale(frame, 1.1, 5, minSize=(30, 30))
                
                # Process each face
                for (x, y, w, h) in faces:
                    face = frame[y:y+h, x:x+w]
                    name, conf = self.recognize_face(face)
                    
                    # Draw result
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{name} {conf:.2f}", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    # Log attendance if recognized
                    if name != "Unknown" and conf > 0.4:
                        self.log_attendance(name)
                
                cv2.imshow("Face Recognition", frame)
                
                # ESC to exit
                if cv2.waitKey(1) & 0xFF == 27:
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("Recognition stopped.")
        except Exception as e:
            print(f"Recognition error: {e}")
    
    def view_records(self):
        """View attendance records"""
        try:
            cursor = self.db.execute("SELECT name, timestamp FROM attendance ORDER BY timestamp DESC LIMIT 20")
            records = cursor.fetchall()
            
            if records:
                print(f"\nRecent Attendance ({len(records)} records):")
                print("-" * 50)
                for name, timestamp in records:
                    print(f"{name:<15} | {timestamp}")
            else:
                print("No attendance records found.")
        except Exception as e:
            print(f"Error viewing records: {e}")

def main():
    # Create system instance
    frs = FaceRecognitionSystem()
    
    # Main menu
    while True:
        print("\n=== Face Recognition System ===")
        print("1. Add Person")
        print("2. Start Recognition")
        print("3. View Records")
        print("4. Exit")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            name = input("Enter person's name: ").strip()
            if name:
                name = name.replace(" ", "_")
                frs.add_person(name)
            else:
                print("Invalid name!")
        
        elif choice == "2":
            frs.start_recognition()
        
        elif choice == "3":
            frs.view_records()
        
        elif choice == "4":
            print("Goodbye!")
            break
        
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()