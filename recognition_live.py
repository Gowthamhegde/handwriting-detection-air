import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from hand_tracking import HandTracker
from scipy.interpolate import interp1d
import pyttsx3
from textblob import TextBlob
import threading
from collections import deque, Counter

class AirWritingRecognizer:
    def __init__(self, model_path='air_writing_model.h5', encoder_path='label_encoder.npy', sequence_length=100):
        self.model = load_model(model_path)
        self.classes = np.load(encoder_path, allow_pickle=True)
        self.sequence_length = sequence_length
        self.tracker = HandTracker()
        
        # Text-to-speech initialization
        try:
            test_engine = pyttsx3.init('sapi5')
            test_engine.stop()
            del test_engine
            
            self.tts_enabled = True
            print("✓ Text-to-speech initialized successfully (Windows SAPI5)")
        except Exception as e:
            print(f"⚠ Warning: Text-to-speech initialization failed: {e}")
            print("  Recognition will work but without voice feedback")
            self.tts_enabled = False
        
        # State variables
        self.trajectory = []
        self.recording = False
        self.last_prediction = ""
        self.confidence = 0.0
        self.prediction_time = 0
        self.is_speaking = False
        
        # Advanced filtering
        self.prediction_history = deque(maxlen=3)
        self.confidence_threshold = 65.0  # Minimum confidence to accept
        self.min_trajectory_points = 15  # Minimum points for valid trajectory
        
    def normalize_trajectory(self, trajectory):
        """Advanced normalization matching training preprocessing"""
        if len(trajectory) < 2:
            return None
        
        trajectory = np.array(trajectory, dtype=np.float32)
        
        # Remove duplicate consecutive points
        unique_trajectory = [trajectory[0]]
        for i in range(1, len(trajectory)):
            if not np.array_equal(trajectory[i], trajectory[i-1]):
                unique_trajectory.append(trajectory[i])
        trajectory = np.array(unique_trajectory)
        
        if len(trajectory) < 2:
            return None
        
        # Center the trajectory
        center = trajectory.mean(axis=0)
        centered = trajectory - center
        
        # Scale to unit variance
        std = centered.std(axis=0)
        std[std == 0] = 1
        normalized = centered / std
        
        # Clip outliers
        normalized = np.clip(normalized, -3, 3)
        
        # Rescale to [0, 1]
        min_val = normalized.min()
        max_val = normalized.max()
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(normalized) * 0.5
        
        # Resample with cubic interpolation
        if len(normalized) != self.sequence_length:
            old_indices = np.linspace(0, len(normalized) - 1, len(normalized))
            new_indices = np.linspace(0, len(normalized) - 1, self.sequence_length)
            
            interp_x = interp1d(old_indices, normalized[:, 0], kind='cubic', fill_value='extrapolate')
            interp_y = interp1d(old_indices, normalized[:, 1], kind='cubic', fill_value='extrapolate')
            
            normalized = np.column_stack([interp_x(new_indices), interp_y(new_indices)])
            normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def extract_features(self, trajectory):
        """Extract position, velocity, and acceleration features"""
        # Position (x, y)
        positions = trajectory
        
        # Velocity (dx/dt, dy/dt)
        velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        
        # Acceleration (d²x/dt², d²y/dt²)
        accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
        
        # Combine all features
        features = np.concatenate([positions, velocities, accelerations], axis=1)
        return features  # Shape: (sequence_length, 6)

    

    
    def predict_word(self, trajectory):
        """Enhanced prediction with confidence filtering and garbage detection"""
        # Validate trajectory
        if len(trajectory) < self.min_trajectory_points:
            return None, 0.0, []
        
        normalized = self.normalize_trajectory(trajectory)
        if normalized is None:
            return None, 0.0, []
        
        # Extract features (6D: x, y, dx, dy, d²x, d²y)
        features = self.extract_features(normalized)
        
        # Predict
        start_time = time.time()
        X = np.expand_dims(features, axis=0)
        predictions = self.model.predict(X, verbose=0)[0]
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-5:][::-1]
        top_predictions = [(self.classes[i], predictions[i] * 100) for i in top_indices]
        
        predicted_word = self.classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Handle GARBAGE class detection
        if predicted_word == "GARBAGE":
            print(f"🗑 GARBAGE detected: Invalid gesture rejected ({confidence:.1f}%)")
            return "GARBAGE", confidence, top_predictions
        
        # Confidence threshold filtering for valid characters/words
        if confidence < self.confidence_threshold:
            print(f"⚠ Low confidence: {confidence:.1f}% (threshold: {self.confidence_threshold}%)")
            return None, confidence, top_predictions
        
        # Add to prediction history for temporal smoothing (exclude GARBAGE)
        self.prediction_history.append(predicted_word)
        
        # Use majority voting if we have enough history
        if len(self.prediction_history) >= 2:
            # Get most common prediction
            counter = Counter(self.prediction_history)
            most_common = counter.most_common(1)[0]
            
            # If there's strong agreement, use it
            if most_common[1] >= 2:
                predicted_word = most_common[0]
        
        # Auto-correction for words (not single letters)
        if len(predicted_word) > 1 and predicted_word != "GARBAGE":
            corrected_word = str(TextBlob(predicted_word).correct())
            if corrected_word != predicted_word:
                print(f"📝 Auto-corrected: '{predicted_word}' → '{corrected_word}'")
                predicted_word = corrected_word
        
        return predicted_word, confidence, top_predictions
    
    def speak(self, text):
        """Speak the recognized word using Windows SAPI"""
        if not self.tts_enabled:
            print(f"📝 Recognized (no voice): {text}")
            return
        
        def _speak():
            try:
                self.is_speaking = True
                print(f"🔊 Speaking: '{text}'")
                
                # Create new engine instance for each speech
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
                
                self.is_speaking = False
            except Exception as e:
                print(f"⚠ Voice error: {e}")
                self.is_speaking = False
        
        # Run in separate thread
        speech_thread = threading.Thread(target=_speak, daemon=False)
        speech_thread.start()
    
    def run(self):
        """Run real-time recognition"""
        cap = cv2.VideoCapture(0)
        fps_time = time.time()
        fps = 0
        
        print("Real-Time Air Writing Recognition")
        print("Instructions:")
        print("- Open hand and write in the air")
        print("- Close hand (thumb-index together) to recognize")
        print("- Press 'c' to clear screen")
        print("- Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = self.tracker.find_hands(frame)
            
            # Check hand state
            is_hand_open = self.tracker.is_hand_open(frame)
            is_hand_closed = self.tracker.is_hand_closed(frame)
            
            # Get index finger position only when hand is open
            pos = None
            if is_hand_open:
                pos = self.tracker.get_index_finger_position(frame)
            
            # Recording logic - only record when hand is open
            if pos and is_hand_open and not is_hand_closed:
                if not self.recording:
                    self.recording = True
                    self.trajectory = []
                
                self.trajectory.append(pos)
                
                # Draw ultra-smooth trajectory with cubic spline interpolation
                if len(self.trajectory) > 1:
                    if len(self.trajectory) >= 4:
                        # Use cubic spline for smooth curves
                        points = np.array(self.trajectory, dtype=np.float32)
                        t = np.linspace(0, 1, len(points))
                        t_smooth = np.linspace(0, 1, len(points) * 3)  # 3x interpolation
                        
                        try:
                            # Cubic spline interpolation
                            spline_x = interp1d(t, points[:, 0], kind='cubic')
                            spline_y = interp1d(t, points[:, 1], kind='cubic')
                            
                            smooth_points = np.column_stack([
                                spline_x(t_smooth),
                                spline_y(t_smooth)
                            ]).astype(np.int32)
                            
                            # Draw smooth polyline with gradient effect
                            cv2.polylines(frame, [smooth_points], False, (0, 255, 0), 5, cv2.LINE_AA)
                            cv2.polylines(frame, [smooth_points], False, (100, 255, 100), 3, cv2.LINE_AA)
                        except:
                            # Fallback to regular polylines
                            points_int = points.astype(np.int32)
                            cv2.polylines(frame, [points_int], False, (0, 255, 0), 5, cv2.LINE_AA)
                    else:
                        # For first few points, use regular lines
                        for i in range(1, len(self.trajectory)):
                            cv2.line(frame, self.trajectory[i-1], self.trajectory[i], 
                                   (0, 255, 0), 5, cv2.LINE_AA)
                
                # Draw smooth current position with glow effect
                cv2.circle(frame, pos, 12, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.circle(frame, pos, 8, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, pos, 5, (255, 255, 255), -1, cv2.LINE_AA)
            
            elif is_hand_closed and self.recording:
                # Hand closed - recognize word
                if len(self.trajectory) >= self.min_trajectory_points:
                    predicted_word, confidence, top_preds = self.predict_word(self.trajectory)
                    
                    if predicted_word == "GARBAGE":
                        # Handle garbage detection
                        self.last_prediction = "INVALID GESTURE"
                        self.confidence = confidence
                        self.prediction_time = time.time()
                        
                        print(f"\n{'='*60}")
                        print(f"🗑 INVALID GESTURE DETECTED ({confidence:.1f}%)")
                        print(f"{'='*60}")
                        print("This gesture was not recognized as a valid character or word.")
                        print("Try writing more clearly or check if it's a supported character.")
                        print(f"{'='*60}")
                        
                        # Don't speak garbage - just show visual feedback
                        
                    elif predicted_word:
                        self.last_prediction = predicted_word
                        self.confidence = confidence
                        self.prediction_time = time.time()
                        
                        print(f"\n{'='*60}")
                        print(f"✓ RECOGNIZED: '{predicted_word.upper()}' ({confidence:.1f}%)")
                        print(f"{'='*60}")
                        print("Top 5 predictions:")
                        for i, (word, conf) in enumerate(top_preds, 1):
                            marker = "★" if i == 1 else " "
                            print(f"  {marker} {i}. {word}: {conf:.1f}%")
                        print(f"{'='*60}")
                        
                        # Speak the word (only for valid predictions)
                        self.speak(predicted_word)
                    else:
                        print(f"\n⚠ Recognition failed - Low confidence or invalid trajectory")
                        print(f"   Trajectory points: {len(self.trajectory)}")
                        if confidence > 0:
                            print(f"   Best guess: {top_preds[0][0]} ({top_preds[0][1]:.1f}%)")
                else:
                    print(f"\n⚠ Trajectory too short: {len(self.trajectory)} points (minimum: {self.min_trajectory_points})")
                
                self.trajectory = []
                self.recording = False
            
            # Calculate FPS
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            
            # Display UI
            h, w, _ = frame.shape
            
            # Status bar
            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show hand state
            if is_hand_closed:
                cv2.putText(frame, "HAND: CLOSED", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            elif is_hand_open:
                cv2.putText(frame, "HAND: OPEN", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "HAND: NEUTRAL", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if self.recording:
                cv2.putText(frame, "WRITING...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Points: {len(self.trajectory)} | Press X to stop", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Open hand to write | Press C to clear", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Close hand to recognize | Press Q to quit", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display last prediction
            if time.time() - self.prediction_time < 3:  # Show for 3 seconds
                cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 0), -1)
                
                # Different colors for different prediction types
                if self.last_prediction == "INVALID GESTURE":
                    text_color = (0, 0, 255)  # Red for garbage/invalid
                    cv2.putText(frame, f"Result: {self.last_prediction}", (10, h-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 3)
                else:
                    text_color = (0, 255, 0)  # Green for valid predictions
                    cv2.putText(frame, f"Word: {self.last_prediction}", (10, h-60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, text_color, 3)
                
                cv2.putText(frame, f"Confidence: {self.confidence:.1f}%", (10, h-20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Air Writing Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # Clear screen
                self.trajectory = []
                self.recording = False
                self.last_prediction = ""
                self.confidence = 0.0
                print("🗑 Screen cleared!")
            elif key == ord('x'):
                # Stop recording without recognizing
                if self.recording:
                    self.trajectory = []
                    self.recording = False
                    print("⏹ Recording stopped (not recognized)")
                else:
                    self.trajectory = []
                    print("🗑 Trajectory cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        self.tracker.release()

if __name__ == "__main__":
    try:
        recognizer = AirWritingRecognizer()
        recognizer.run()
    except FileNotFoundError:
        print("Error: Model files not found!")
        print("Please train the model first using model_training.py")
