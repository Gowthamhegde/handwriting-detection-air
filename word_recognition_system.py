#!/usr/bin/env python3
"""
Intelligent Air-Writing Recognition System
- Gesture-based activation (index + middle finger joined)
- Word formation with dictionary validation
- Real-time feedback with error detection
- A-Z alphabet recognition only
"""

import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from hand_tracking import HandTracker
from scipy.interpolate import interp1d
import pyttsx3
import threading
from collections import deque
import json
import os

class WordRecognitionSystem:
    def __init__(self, model_path='air_writing_model.h5', encoder_path='label_encoder.npy', 
                 dictionary_path='english_dictionary.json', sequence_length=100):
        # Load model and classes
        self.model = load_model(model_path)
        self.classes = np.load(encoder_path, allow_pickle=True)
        self.sequence_length = sequence_length
        self.tracker = HandTracker()
        
        # Load English dictionary
        self.dictionary = self.load_dictionary(dictionary_path)
        
        # Text-to-speech initialization
        try:
            test_engine = pyttsx3.init('sapi5')
            test_engine.stop()
            del test_engine
            self.tts_enabled = True
            print("✓ Text-to-speech initialized successfully")
        except Exception as e:
            print(f"⚠ Warning: Text-to-speech failed: {e}")
            self.tts_enabled = False
        
        # System state variables
        self.trajectory = []
        self.recording = False
        self.system_active = False  # Gesture-based activation
        self.current_word = ""  # Accumulating word
        self.recognized_letters = []  # Letter sequence
        self.last_prediction = ""
        self.confidence = 0.0
        self.prediction_time = 0
        self.error_message = ""
        self.error_time = 0
        
        # Recognition parameters
        self.confidence_threshold = 70.0  # Higher threshold for letters
        self.min_trajectory_points = 20  # Minimum points for valid letter
        self.letter_timeout = 3.0  # Seconds to wait between letters
        self.last_letter_time = 0
        
        # Filter to only A-Z letters
        self.valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        self.alphabet_classes = [cls for cls in self.classes if cls in self.valid_letters]
        
        print(f"✓ System initialized with {len(self.alphabet_classes)} alphabet classes")
        print(f"✓ Dictionary loaded with {len(self.dictionary)} words")
    
    def load_dictionary(self, dictionary_path):
        """Load English dictionary from JSON file"""
        if os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r') as f:
                    dictionary = json.load(f)
                return set(word.upper() for word in dictionary)  # Convert to uppercase
            except Exception as e:
                print(f"⚠ Error loading dictionary: {e}")
        
        # Create basic dictionary if file doesn't exist
        basic_words = [
            "CAT", "DOG", "SUN", "MOON", "STAR", "TREE", "BOOK", "PEN", "CUP", "HAT",
            "CAR", "BUS", "FISH", "BIRD", "HAND", "FOOT", "HEAD", "EYES", "NOSE", "MOUTH",
            "LOVE", "HOPE", "LIFE", "TIME", "WORK", "HOME", "DOOR", "WINDOW", "CHAIR", "TABLE",
            "WATER", "FIRE", "EARTH", "WIND", "LIGHT", "DARK", "GOOD", "BAD", "BIG", "SMALL",
            "HOT", "COLD", "NEW", "OLD", "FAST", "SLOW", "HIGH", "LOW", "NEAR", "FAR",
            "YES", "NO", "HELLO", "WORLD", "PEACE", "HAPPY", "SMILE", "HEART", "MIND", "SOUL"
        ]
        
        # Save basic dictionary
        try:
            with open(dictionary_path, 'w') as f:
                json.dump(basic_words, f, indent=2)
            print(f"✓ Created basic dictionary with {len(basic_words)} words")
        except Exception as e:
            print(f"⚠ Could not save dictionary: {e}")
        
        return set(basic_words)
    
    def is_gesture_active(self, frame):
        """Check if index and middle fingers are joined together"""
        if not self.tracker.results or not self.tracker.results.multi_hand_landmarks:
            return False
        
        for hand_landmarks in self.tracker.results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            
            # Get fingertip positions
            index_tip = landmarks[8]  # Index finger tip
            middle_tip = landmarks[12]  # Middle finger tip
            
            # Convert to pixel coordinates
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
            
            # Calculate distance between fingertips
            distance = np.sqrt((index_pos[0] - middle_pos[0])**2 + 
                             (index_pos[1] - middle_pos[1])**2)
            
            # Fingers are "joined" if distance is small
            join_threshold = 30  # pixels
            return distance < join_threshold
        
        return False
    
    def normalize_trajectory(self, trajectory):
        """Advanced normalization for letter recognition"""
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
        
        # Resample to fixed length
        if len(normalized) != self.sequence_length:
            old_indices = np.linspace(0, len(normalized) - 1, len(normalized))
            new_indices = np.linspace(0, len(normalized) - 1, self.sequence_length)
            
            interp_x = interp1d(old_indices, normalized[:, 0], kind='cubic', fill_value='extrapolate')
            interp_y = interp1d(old_indices, normalized[:, 1], kind='cubic', fill_value='extrapolate')
            
            normalized = np.column_stack([interp_x(new_indices), interp_y(new_indices)])
            normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def extract_features(self, trajectory):
        """Extract 6D features (position, velocity, acceleration)"""
        positions = trajectory
        velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
        features = np.concatenate([positions, velocities, accelerations], axis=1)
        return features
    
    def predict_letter(self, trajectory):
        """Predict letter from trajectory (A-Z only)"""
        if len(trajectory) < self.min_trajectory_points:
            return None, 0.0, []
        
        normalized = self.normalize_trajectory(trajectory)
        if normalized is None:
            return None, 0.0, []
        
        # Extract features
        features = self.extract_features(normalized)
        X = np.expand_dims(features, axis=0)
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Filter to only alphabet classes
        alphabet_predictions = []
        for i, class_name in enumerate(self.classes):
            if class_name in self.valid_letters:
                alphabet_predictions.append((class_name, predictions[i] * 100))
        
        # Sort by confidence
        alphabet_predictions.sort(key=lambda x: x[1], reverse=True)
        
        if not alphabet_predictions:
            return None, 0.0, []
        
        predicted_letter = alphabet_predictions[0][0]
        confidence = alphabet_predictions[0][1]
        
        # Confidence threshold check
        if confidence < self.confidence_threshold:
            return None, confidence, alphabet_predictions[:5]
        
        return predicted_letter, confidence, alphabet_predictions[:5]
    
    def add_letter_to_word(self, letter):
        """Add recognized letter to current word"""
        self.recognized_letters.append(letter)
        self.current_word = ''.join(self.recognized_letters)
        self.last_letter_time = time.time()
        print(f"📝 Added letter '{letter}' → Current word: '{self.current_word}'")
    
    def validate_word(self):
        """Validate current word against dictionary"""
        if not self.current_word:
            return False
        
        if self.current_word in self.dictionary:
            print(f"✓ VALID WORD: '{self.current_word}' found in dictionary!")
            return True
        else:
            print(f"❌ INVALID: '{self.current_word}' not found in dictionary")
            return False
    
    def reset_word(self):
        """Reset current word and letter sequence"""
        self.current_word = ""
        self.recognized_letters = []
        self.last_letter_time = 0
        print("🔄 Word buffer reset")
    
    def set_error(self, message):
        """Set error message with timestamp"""
        self.error_message = message
        self.error_time = time.time()
        print(f"⚠ ERROR: {message}")
    
    def speak(self, text):
        """Speak the recognized word"""
        if not self.tts_enabled:
            return
        
        def _speak():
            try:
                engine = pyttsx3.init('sapi5')
                engine.setProperty('rate', 150)
                engine.setProperty('volume', 1.0)
                engine.say(text)
                engine.runAndWait()
                engine.stop()
                del engine
            except Exception as e:
                print(f"⚠ Voice error: {e}")
        
        threading.Thread(target=_speak, daemon=True).start()
    
    def run(self):
        """Run the intelligent word recognition system"""
        cap = cv2.VideoCapture(0)
        fps_time = time.time()
        
        print("\n" + "="*60)
        print("INTELLIGENT AIR-WRITING WORD RECOGNITION SYSTEM")
        print("="*60)
        print("Instructions:")
        print("• Join index and middle fingers to activate system")
        print("• Write letters in the air (A-Z only)")
        print("• System will form words and validate against dictionary")
        print("• Release finger join to deactivate")
        print("• Press 'r' to reset current word")
        print("• Press 'q' to quit")
        print("="*60)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = self.tracker.find_hands(frame)
            h, w, _ = frame.shape
            
            # Check gesture activation
            gesture_active = self.is_gesture_active(frame)
            
            # System activation logic
            if gesture_active and not self.system_active:
                self.system_active = True
                print("🟢 SYSTEM ACTIVATED - Ready to recognize letters")
            elif not gesture_active and self.system_active:
                self.system_active = False
                self.recording = False
                self.trajectory = []
                print("🔴 SYSTEM DEACTIVATED")
            
            # Only process when system is active
            if self.system_active:
                # Get index finger position for writing
                pos = self.tracker.get_index_finger_position(frame)
                
                if pos:
                    if not self.recording:
                        self.recording = True
                        self.trajectory = []
                    
                    self.trajectory.append(pos)
                    
                    # Draw trajectory
                    if len(self.trajectory) > 1:
                        points = np.array(self.trajectory, dtype=np.int32)
                        cv2.polylines(frame, [points], False, (0, 255, 0), 3, cv2.LINE_AA)
                    
                    # Draw current position
                    cv2.circle(frame, pos, 8, (0, 255, 0), -1)
                
                elif self.recording:
                    # Finger lifted - recognize letter
                    if len(self.trajectory) >= self.min_trajectory_points:
                        predicted_letter, confidence, top_preds = self.predict_letter(self.trajectory)
                        
                        if predicted_letter:
                            self.add_letter_to_word(predicted_letter)
                            self.last_prediction = predicted_letter
                            self.confidence = confidence
                            self.prediction_time = time.time()
                            
                            print(f"✓ RECOGNIZED LETTER: '{predicted_letter}' ({confidence:.1f}%)")
                            print(f"Top predictions: {[(p[0], f'{p[1]:.1f}%') for p in top_preds[:3]]}")
                        else:
                            self.set_error("Invalid letter or low confidence")
                    
                    self.trajectory = []
                    self.recording = False
            
            # Check for word completion (timeout or manual validation)
            current_time = time.time()
            if (self.current_word and 
                current_time - self.last_letter_time > self.letter_timeout):
                
                if self.validate_word():
                    # Valid word found
                    self.last_prediction = f"WORD: {self.current_word}"
                    self.prediction_time = current_time
                    self.speak(self.current_word)
                    self.reset_word()
                else:
                    # Invalid word
                    self.set_error(f"Word '{self.current_word}' not found in dictionary")
                    self.reset_word()
            
            # Calculate FPS
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            
            # ==================== UI DISPLAY ====================
            
            # Status bar background
            cv2.rectangle(frame, (0, 0), (w, 150), (0, 0, 0), -1)
            
            # System status
            status_color = (0, 255, 0) if self.system_active else (0, 0, 255)
            status_text = "ACTIVE" if self.system_active else "INACTIVE"
            cv2.putText(frame, f"System: {status_text}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # FPS
            cv2.putText(frame, f"FPS: {fps:.1f}", (w-120, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Current word being formed
            if self.current_word:
                cv2.putText(frame, f"Word: {self.current_word}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 2)
                
                # Show individual letters
                letters_text = " + ".join(self.recognized_letters)
                cv2.putText(frame, f"Letters: {letters_text}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            else:
                cv2.putText(frame, "Word: (empty)", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (128, 128, 128), 2)
            
            # Instructions
            if not self.system_active:
                cv2.putText(frame, "Join index + middle fingers to activate", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Write letters in air | Release fingers to deactivate", (10, 140), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show last prediction
            if current_time - self.prediction_time < 3:
                cv2.rectangle(frame, (0, h-100), (w, h), (0, 50, 0), -1)
                cv2.putText(frame, f"{self.last_prediction}", (10, h-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if self.confidence > 0:
                    cv2.putText(frame, f"Confidence: {self.confidence:.1f}%", (10, h-20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Show error message
            if current_time - self.error_time < 3:
                cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 50), -1)
                cv2.putText(frame, f"ERROR: {self.error_message}", (10, h-50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Draw finger join indicator
            if gesture_active:
                cv2.circle(frame, (w-50, 50), 20, (0, 255, 0), -1)
                cv2.putText(frame, "JOINED", (w-100, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            cv2.imshow("Intelligent Air-Writing Word Recognition", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.reset_word()
                print("🔄 Manual word reset")
            elif key == ord('v'):
                # Manual word validation
                if self.current_word:
                    if self.validate_word():
                        self.speak(self.current_word)
                        self.reset_word()
                    else:
                        self.set_error(f"Word '{self.current_word}' not found in dictionary")
                        self.reset_word()
        
        cap.release()
        cv2.destroyAllWindows()
        self.tracker.release()

if __name__ == "__main__":
    try:
        system = WordRecognitionSystem()
        system.run()
    except FileNotFoundError:
        print("❌ Error: Model files not found!")
        print("Please train the model first using model_training.py")
    except Exception as e:
        print(f"❌ Error: {e}")