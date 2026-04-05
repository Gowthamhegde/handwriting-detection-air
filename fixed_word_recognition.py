#!/usr/bin/env python3
"""
FIXED Air-Writing Recognition System
- Proper gesture control (index + middle finger joined)
- One letter per gesture cycle
- Dictionary validation
- Error handling for incorrect letters
- Clean word output without random characters
"""

import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from hand_tracking import HandTracker
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import pyttsx3
import threading
from collections import deque
import json
import os

class FixedWordRecognitionSystem:
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
        
        # FIXED: Proper state management
        self.trajectory = []
        self.recording = False
        self.gesture_active = False
        self.last_gesture_time = 0
        self.gesture_cooldown = 0.5  # Minimum time between gestures
        
        # SMOOTH TRACKING: Advanced smoothing parameters
        self.position_buffer = deque(maxlen=8)  # Buffer for position smoothing
        self.velocity_buffer = deque(maxlen=5)  # Buffer for velocity smoothing
        self.last_position = None
        self.last_velocity = None
        self.smoothing_factor = 0.7  # Exponential smoothing factor
        self.prediction_factor = 0.3  # Velocity prediction factor
        
        # STABLE FINGER DETECTION: Prevent premature ending
        self.finger_detection_buffer = deque(maxlen=5)  # Buffer for finger detection stability
        self.fingers_release_delay = 0.3  # Seconds to wait before confirming finger release
        self.last_fingers_joined_time = 0  # Track when fingers were last detected as joined
        
        # Word building
        self.current_word = ""
        self.recognized_letters = []
        self.letter_trajectories = []
        self.word_completed = False
        
        # Recognition parameters - FIXED: More strict
        self.confidence_threshold = 75.0  # Higher threshold for accuracy
        self.min_trajectory_points = 20   # More points for better recognition
        self.max_trajectory_points = 200  # Prevent overly long trajectories
        
        # Error handling
        self.error_message = ""
        self.error_time = 0
        self.last_prediction = ""
        self.prediction_time = 0
        
        # Filter to only A-Z letters
        self.valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        print(f"✓ System initialized with {len(self.classes)} classes")
        print(f"✓ Dictionary loaded with {len(self.dictionary)} words")
        print(f"✓ Valid letters: {sorted(self.valid_letters)}")
    
    def load_dictionary(self, dictionary_path):
        """Load English dictionary from JSON file"""
        if os.path.exists(dictionary_path):
            try:
                with open(dictionary_path, 'r') as f:
                    dictionary = json.load(f)
                return set(word.upper() for word in dictionary)
            except Exception as e:
                print(f"⚠ Error loading dictionary: {e}")
        
        # Basic dictionary with A-Z letters and common words
        alphabet_letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        basic_words = alphabet_letters + [
            "CAT", "DOG", "SUN", "MOON", "STAR", "HELLO", "WORLD", "BOOK", "PEN",
            "CAR", "BUS", "FISH", "BIRD", "HAND", "FOOT", "HEAD", "EYES", "NOSE",
            "LOVE", "HOPE", "LIFE", "TIME", "WORK", "HOME", "DOOR", "WINDOW",
            "WATER", "FIRE", "EARTH", "WIND", "LIGHT", "DARK", "GOOD", "BAD"
        ]
        return set(basic_words)
    
    def get_middle_finger_position(self, frame):
        """Get middle finger tip position"""
        if not self.tracker.results or not self.tracker.results.multi_hand_landmarks:
            return None
        
        for hand_landmarks in self.tracker.results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            
            middle_tip = landmarks[12]  # Middle finger tip
            return (int(middle_tip.x * w), int(middle_tip.y * h))
        
        return None
    
    def get_joined_fingers_position(self, frame):
        """Get the SMOOTH midpoint position between joined index and middle fingers"""
        if not self.tracker.results or not self.tracker.results.multi_hand_landmarks:
            return None
        
        for hand_landmarks in self.tracker.results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            h, w, _ = frame.shape
            
            # Get both fingertip positions
            index_tip = landmarks[8]   # Index finger tip
            middle_tip = landmarks[12] # Middle finger tip
            
            # Convert to pixel coordinates
            index_pos = (int(index_tip.x * w), int(index_tip.y * h))
            middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
            
            # Calculate raw midpoint between the two fingers
            raw_x = (index_pos[0] + middle_pos[0]) / 2.0
            raw_y = (index_pos[1] + middle_pos[1]) / 2.0
            
            # Apply advanced smoothing
            smooth_pos = self.apply_smooth_tracking(raw_x, raw_y)
            
            return smooth_pos
        
        return None
    
    def apply_smooth_tracking(self, x, y):
        """Apply advanced smoothing with exponential moving average and velocity prediction"""
        current_pos = (x, y)
        
        # Add to position buffer
        self.position_buffer.append(current_pos)
        
        # Calculate velocity if we have previous position
        if self.last_position is not None:
            velocity = (x - self.last_position[0], y - self.last_position[1])
            self.velocity_buffer.append(velocity)
            
            # Calculate average velocity for prediction
            if len(self.velocity_buffer) >= 3:
                velocities = np.array(list(self.velocity_buffer))
                avg_velocity = np.mean(velocities, axis=0)
                self.last_velocity = avg_velocity
        
        # Apply exponential moving average smoothing
        if len(self.position_buffer) >= 3:
            positions = np.array(list(self.position_buffer))
            
            # Create exponential weights (more recent = higher weight)
            weights = np.exp(np.linspace(-1, 0, len(positions)))
            weights = weights / weights.sum()
            
            # Apply weighted average
            smooth_x = np.sum(positions[:, 0] * weights)
            smooth_y = np.sum(positions[:, 1] * weights)
            
            # Add velocity prediction for even smoother motion
            if self.last_velocity is not None:
                smooth_x += self.last_velocity[0] * self.prediction_factor
                smooth_y += self.last_velocity[1] * self.prediction_factor
            
            self.last_position = (smooth_x, smooth_y)
            return (int(smooth_x), int(smooth_y))
        else:
            # Not enough data yet, use raw position
            self.last_position = current_pos
            return (int(x), int(y))
    
    def is_fingers_joined_stable(self, frame):
        """STABLE: Robust detection with buffering to prevent premature ending"""
        current_time = time.time()
        
        if not self.tracker.results or not self.tracker.results.multi_hand_landmarks:
            joined = False
        else:
            joined = False
            for hand_landmarks in self.tracker.results.multi_hand_landmarks:
                landmarks = hand_landmarks.landmark
                h, w, _ = frame.shape
                
                # Get fingertip positions
                index_tip = landmarks[8]   # Index finger tip
                middle_tip = landmarks[12] # Middle finger tip
                
                # Convert to pixel coordinates
                index_pos = (int(index_tip.x * w), int(index_tip.y * h))
                middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
                
                # Calculate distance between fingertips
                distance = np.sqrt((index_pos[0] - middle_pos[0])**2 + 
                                 (index_pos[1] - middle_pos[1])**2)
                
                # STABLE: More forgiving threshold for writing
                join_threshold = 45  # Increased threshold for stability during writing
                joined = distance < join_threshold
                break
        
        # Add to detection buffer
        self.finger_detection_buffer.append(joined)
        
        # Update last joined time if fingers are detected as joined
        if joined:
            self.last_fingers_joined_time = current_time
        
        # STABLE LOGIC: Use majority voting from buffer
        if len(self.finger_detection_buffer) >= 3:
            # Count how many recent detections show fingers joined
            joined_count = sum(self.finger_detection_buffer)
            total_count = len(self.finger_detection_buffer)
            
            # If we're currently writing, be more conservative about ending
            if self.gesture_active and self.recording:
                # During writing: require strong evidence that fingers are separated
                # AND enough time has passed since last detection
                time_since_last_joined = current_time - self.last_fingers_joined_time
                
                if joined_count >= 2:  # At least 2 out of 5 recent detections show joined
                    return True
                elif time_since_last_joined < self.fingers_release_delay:
                    # Not enough time passed, keep writing
                    return True
                else:
                    # Enough time passed and consistent separation detected
                    return False
            else:
                # Not currently writing: use normal detection
                return joined_count >= 2  # Majority voting
        
        # Not enough buffer data, use current detection
        return joined
    
    def is_fingers_joined(self, frame):
        """Legacy method - redirects to stable version"""
        return self.is_fingers_joined_stable(frame)
    
    def is_hand_fist(self, frame):
        """FIXED: Better fist detection for word recognition"""
        if not self.tracker.results or not self.tracker.results.multi_hand_landmarks:
            return False
        
        for hand_landmarks in self.tracker.results.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            
            # Get fingertip and joint positions
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            index_tip = landmarks[8]
            index_pip = landmarks[6]
            middle_tip = landmarks[12]
            middle_pip = landmarks[10]
            ring_tip = landmarks[16]
            ring_pip = landmarks[14]
            pinky_tip = landmarks[20]
            pinky_pip = landmarks[18]
            
            # Count closed fingers (tip below PIP joint)
            closed_fingers = 0
            
            # Thumb (special case - check x-axis for right hand)
            if thumb_tip.x < thumb_ip.x:
                closed_fingers += 1
            
            # Other fingers (check y-axis)
            if index_tip.y > index_pip.y:
                closed_fingers += 1
            if middle_tip.y > middle_pip.y:
                closed_fingers += 1
            if ring_tip.y > ring_pip.y:
                closed_fingers += 1
            if pinky_tip.y > pinky_pip.y:
                closed_fingers += 1
            
            # Hand is fist if 4+ fingers are closed
            return closed_fingers >= 4
        
        return False
    
    def normalize_trajectory(self, trajectory):
        """FIXED: Robust trajectory normalization with smoothing"""
        if len(trajectory) < 2:
            return None
        
        trajectory = np.array(trajectory, dtype=np.float32)
        
        # SMOOTH: Apply Gaussian smoothing to reduce noise
        if len(trajectory) > 5:
            from scipy.ndimage import gaussian_filter1d
            trajectory[:, 0] = gaussian_filter1d(trajectory[:, 0], sigma=1.0)
            trajectory[:, 1] = gaussian_filter1d(trajectory[:, 1], sigma=1.0)
        
        # Remove duplicate consecutive points
        unique_trajectory = [trajectory[0]]
        for i in range(1, len(trajectory)):
            # Use distance threshold instead of exact equality for smoother trajectories
            distance = np.sqrt(np.sum((trajectory[i] - trajectory[i-1])**2))
            if distance > 2.0:  # Minimum distance between points
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
        """FIXED: Robust letter prediction with proper GARBAGE detection"""
        if len(trajectory) < self.min_trajectory_points:
            return None, 0.0, [], f"Too few points: {len(trajectory)} < {self.min_trajectory_points}"
        
        if len(trajectory) > self.max_trajectory_points:
            return None, 0.0, [], f"Too many points: {len(trajectory)} > {self.max_trajectory_points}"
        
        normalized = self.normalize_trajectory(trajectory)
        if normalized is None:
            return None, 0.0, [], "Normalization failed"
        
        # Extract features and predict
        features = self.extract_features(normalized)
        X = np.expand_dims(features, axis=0)
        predictions = self.model.predict(X, verbose=0)[0]
        
        # Get all predictions sorted by confidence
        all_predictions = [(self.classes[i], predictions[i] * 100) for i in range(len(self.classes))]
        all_predictions.sort(key=lambda x: x[1], reverse=True)
        
        # FIXED: Proper GARBAGE handling
        top_prediction = all_predictions[0]
        
        # If GARBAGE is top prediction, reject immediately
        if top_prediction[0] == "GARBAGE":
            return "GARBAGE", top_prediction[1], all_predictions[:5], "Classified as GARBAGE"
        
        # Find best valid letter
        best_letter = None
        best_confidence = 0.0
        garbage_confidence = 0.0
        
        for class_name, confidence in all_predictions:
            if class_name == "GARBAGE":
                garbage_confidence = confidence
            elif class_name in self.valid_letters and best_letter is None:
                best_letter = class_name
                best_confidence = confidence
        
        # FIXED: Strict validation
        if best_letter is None:
            return "GARBAGE", garbage_confidence, all_predictions[:5], "No valid letters found"
        
        # If GARBAGE confidence is too high relative to best letter
        if garbage_confidence > best_confidence * 0.6:
            return "GARBAGE", garbage_confidence, all_predictions[:5], f"GARBAGE too high: {garbage_confidence:.1f}% vs {best_letter}: {best_confidence:.1f}%"
        
        # Check confidence threshold
        if best_confidence < self.confidence_threshold:
            return None, best_confidence, all_predictions[:5], f"Low confidence: {best_confidence:.1f}% < {self.confidence_threshold}%"
        
        return best_letter, best_confidence, all_predictions[:5], "Success"
    
    def add_letter_to_word(self, letter):
        """FIXED: Add letter with validation"""
        if letter in self.valid_letters:
            self.recognized_letters.append(letter)
            self.current_word = ''.join(self.recognized_letters)
            
            # Store trajectory for display (using joined fingers positions)
            if len(self.trajectory) > 1:
                self.letter_trajectories.append(self.trajectory.copy())
            
            print(f"✅ Added letter '{letter}' → Current word: '{self.current_word}'")
            print(f"   Trajectory points: {len(self.trajectory)} (tracked at joined fingers midpoint)")
            return True
        else:
            print(f"❌ Invalid letter '{letter}' - not in A-Z range")
            return False
    
    def validate_word(self):
        """FIXED: Dictionary validation with error handling"""
        if not self.current_word:
            return False, "Empty word"
        
        if self.current_word in self.dictionary:
            return True, f"Valid word: '{self.current_word}'"
        else:
            return False, f"Word '{self.current_word}' not found in dictionary"
    
    def set_error(self, message):
        """Set error message with timestamp"""
        self.error_message = message
        self.error_time = time.time()
        print(f"⚠ ERROR: {message}")
    
    def draw_smooth_trajectory(self, frame, trajectory, color, thickness):
        """Draw ultra-smooth trajectory using cubic spline interpolation"""
        if len(trajectory) < 2:
            return
        
        try:
            if len(trajectory) >= 4:
                # Use cubic spline for ultra-smooth curves
                points = np.array(trajectory, dtype=np.float32)
                
                # Create parameter array for interpolation
                t = np.linspace(0, 1, len(points))
                t_smooth = np.linspace(0, 1, len(points) * 4)  # 4x interpolation for smoothness
                
                # Cubic spline interpolation
                from scipy.interpolate import interp1d
                spline_x = interp1d(t, points[:, 0], kind='cubic', bounds_error=False, fill_value='extrapolate')
                spline_y = interp1d(t, points[:, 1], kind='cubic', bounds_error=False, fill_value='extrapolate')
                
                # Generate smooth points
                smooth_points = np.column_stack([
                    spline_x(t_smooth),
                    spline_y(t_smooth)
                ]).astype(np.int32)
                
                # Draw with gradient effect (multiple layers for smoothness)
                cv2.polylines(frame, [smooth_points], False, color, thickness, cv2.LINE_AA)
                cv2.polylines(frame, [smooth_points], False, 
                             (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50)), 
                             max(1, thickness - 2), cv2.LINE_AA)
                
            else:
                # For first few points, use regular smooth lines
                points = np.array(trajectory, dtype=np.int32)
                cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)
                
        except Exception as e:
            # Fallback to regular polylines if spline fails
            points = np.array(trajectory, dtype=np.int32)
            cv2.polylines(frame, [points], False, color, thickness, cv2.LINE_AA)
    
    def reset_word(self):
        """Reset word and trajectories"""
        self.current_word = ""
        self.recognized_letters = []
        self.letter_trajectories = []
        self.word_completed = False
        self.last_prediction = ""
        self.error_message = ""
        
        # Reset smoothing buffers
        self.position_buffer.clear()
        self.velocity_buffer.clear()
        self.last_position = None
        self.last_velocity = None
        
        # Reset finger detection buffers
        self.finger_detection_buffer.clear()
        self.last_fingers_joined_time = 0
        
        print("🔄 Word reset")
    
    def speak(self, text):
        """Speak text using TTS"""
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
        """FIXED: Main recognition loop with proper gesture control"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        fps_time = time.time()
        
        print("\n" + "="*70)
        print("FIXED AIR-WRITING WORD RECOGNITION SYSTEM")
        print("="*70)
        print("INSTRUCTIONS:")
        print("1. JOIN index and middle fingers to START writing a letter")
        print("2. Write ONE letter while keeping fingers joined")
        print("3. RELEASE fingers to STOP and recognize the letter")
        print("4. Repeat steps 1-3 for each letter")
        print("5. Make a FIST to recognize the complete word")
        print("6. Press 'C' to clear and start over")
        print("7. Press 'Q' to quit")
        print("="*70)
        print("FIXES APPLIED:")
        print("✓ One letter per gesture cycle")
        print("✓ Proper gesture control (join/release)")
        print("✓ GARBAGE detection to prevent random letters")
        print("✓ Dictionary validation")
        print("✓ Error messages for incorrect detection")
        print("✓ Cooldown between gestures")
        print("="*70)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = self.tracker.find_hands(frame)
            h, w, _ = frame.shape
            current_time = time.time()
            
            # STABLE: Robust gesture detection with buffering
            fingers_joined = self.is_fingers_joined_stable(frame)
            hand_fist = self.is_hand_fist(frame)
            
            # STABLE: State machine for gesture control
            if fingers_joined and not self.gesture_active and (current_time - self.last_gesture_time) > self.gesture_cooldown:
                # Start new gesture
                self.gesture_active = True
                self.recording = True
                self.trajectory = []
                
                # Reset smoothing buffers for new gesture
                self.position_buffer.clear()
                self.velocity_buffer.clear()
                self.last_position = None
                self.last_velocity = None
                
                # Reset finger detection buffer for stable tracking
                self.finger_detection_buffer.clear()
                self.last_fingers_joined_time = current_time
                
                print("🟢 GESTURE STARTED - Writing letter...")
            
            elif not fingers_joined and self.gesture_active:
                # End gesture and recognize letter
                self.gesture_active = False
                self.recording = False
                self.last_gesture_time = current_time
                
                if len(self.trajectory) >= self.min_trajectory_points:
                    print(f"🔍 RECOGNIZING letter with {len(self.trajectory)} points...")
                    
                    try:
                        predicted_letter, confidence, top_preds, reason = self.predict_letter(self.trajectory)
                        
                        if predicted_letter == "GARBAGE":
                            self.set_error("Error: Incorrect letter detected. Please rewrite.")
                            self.last_prediction = "INVALID"
                            self.prediction_time = current_time
                            print(f"🗑 GARBAGE DETECTED ({confidence:.1f}%): {reason}")
                            
                        elif predicted_letter:
                            if self.add_letter_to_word(predicted_letter):
                                self.last_prediction = predicted_letter
                                self.prediction_time = current_time
                                print(f"✅ LETTER RECOGNIZED: '{predicted_letter}' ({confidence:.1f}%)")
                                print(f"   Top 3: {[(p[0], f'{p[1]:.1f}%') for p in top_preds[:3]]}")
                            else:
                                self.set_error("Error: Invalid letter format detected.")
                        
                        else:
                            self.set_error("Error: Letter unclear. Please rewrite more clearly.")
                            print(f"❌ RECOGNITION FAILED: {reason}")
                            if top_preds:
                                print(f"   Best guess: {top_preds[0][0]} ({top_preds[0][1]:.1f}%)")
                    
                    except Exception as e:
                        self.set_error("Error: Recognition system failure.")
                        print(f"❌ SYSTEM ERROR: {e}")
                
                else:
                    print(f"⚠ Trajectory too short: {len(self.trajectory)} points")
                
                print("🔴 GESTURE ENDED")
            
            # Record trajectory while gesture is active
            if self.gesture_active and self.recording:
                pos = self.get_joined_fingers_position(frame)
                if pos:
                    self.trajectory.append(pos)
                    
                    # Draw ultra-smooth trajectory with cubic spline interpolation
                    if len(self.trajectory) > 1:
                        self.draw_smooth_trajectory(frame, self.trajectory, (0, 255, 0), 8)
                    
                    # Draw current position (at joined fingers midpoint) with glow effect
                    cv2.circle(frame, pos, 25, (0, 255, 0), 2, cv2.LINE_AA)  # Outer glow
                    cv2.circle(frame, pos, 20, (0, 255, 0), 3, cv2.LINE_AA)  # Main circle
                    cv2.circle(frame, pos, 15, (0, 255, 0), -1)             # Filled center
                    cv2.circle(frame, pos, 8, (255, 255, 255), -1)          # White center
                    
                    # Draw individual finger positions for reference (smaller, dimmed)
                    index_pos = self.tracker.get_index_finger_position(frame)
                    middle_pos = self.get_middle_finger_position(frame)
                    if index_pos:
                        cv2.circle(frame, index_pos, 6, (150, 150, 255), 1)  # Dim blue for index
                    if middle_pos:
                        cv2.circle(frame, middle_pos, 6, (255, 150, 150), 1)  # Dim red for middle
                    
                    # Draw connection line between fingers (very subtle)
                    if index_pos and middle_pos:
                        cv2.line(frame, index_pos, middle_pos, (100, 100, 100), 1)
            
            # FIXED: Word recognition with proper validation
            if hand_fist and self.current_word and not self.gesture_active and not self.word_completed:
                print(f"👊 FIST DETECTED - Validating word: '{self.current_word}'")
                
                is_valid, message = self.validate_word()
                if is_valid:
                    self.last_prediction = f"✅ WORD: {self.current_word}"
                    self.prediction_time = current_time
                    self.word_completed = True
                    self.speak(self.current_word)
                    print(f"🎉 SUCCESS! {message}")
                else:
                    self.set_error(f"Error: {message}")
                    print(f"❌ WORD VALIDATION FAILED: {message}")
            
            # Draw all letter trajectories with smooth rendering
            for i, letter_traj in enumerate(self.letter_trajectories):
                if len(letter_traj) > 1:
                    colors = [(255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
                    color = colors[i % len(colors)]
                    self.draw_smooth_trajectory(frame, letter_traj, color, 6)
            
            # FIXED: Clean UI display
            fps = 1.0 / (time.time() - fps_time)
            fps_time = time.time()
            
            # Create a clean status bar at the top
            cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)  # Black background
            
            # Status indicator with finger detection feedback
            if self.gesture_active:
                status_color = (0, 255, 0)
                status_text = "WRITING"
                # Show trajectory length during writing
                traj_info = f"({len(self.trajectory)} pts)"
            elif (current_time - self.last_gesture_time) < self.gesture_cooldown:
                status_color = (255, 255, 0)
                status_text = "COOLDOWN"
                traj_info = ""
            else:
                status_color = (100, 100, 100)
                status_text = "READY"
                traj_info = ""
            
            cv2.circle(frame, (30, 30), 15, status_color, -1)
            cv2.putText(frame, f"{status_text} {traj_info}", (60, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Finger detection stability indicator (smaller, less intrusive)
            if len(self.finger_detection_buffer) > 0:
                joined_count = sum(self.finger_detection_buffer)
                total_count = len(self.finger_detection_buffer)
                stability = joined_count / total_count
                
                # Show finger detection stability (smaller)
                stability_color = (0, int(255 * stability), int(255 * (1 - stability)))
                cv2.circle(frame, (30, 60), 6, stability_color, -1)
                cv2.putText(frame, f"Stability: {stability:.1f}", (50, 65), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, stability_color, 1)
            
            # Current word display - moved to top-right area
            if self.current_word:
                word_color = (0, 255, 0) if self.word_completed else (255, 255, 0)
                status = "✅" if self.word_completed else "→"
                cv2.putText(frame, f"{status} {self.current_word}", (w-300, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, word_color, 3)
            
            # Feedback area - clean display at top center
            feedback_y = 140
            if current_time - self.prediction_time < 3:
                # Success messages
                if "WORD:" in self.last_prediction:
                    cv2.rectangle(frame, (w//2-200, feedback_y-40), (w//2+200, feedback_y+20), (0, 50, 0), -1)
                    cv2.putText(frame, self.last_prediction, (w//2-180, feedback_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                elif self.last_prediction and self.last_prediction != "INVALID":
                    cv2.rectangle(frame, (w//2-150, feedback_y-40), (w//2+150, feedback_y+20), (0, 50, 0), -1)
                    cv2.putText(frame, f"Letter: {self.last_prediction}", (w//2-130, feedback_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            # Error messages - clean display at top center
            if current_time - self.error_time < 4:
                cv2.rectangle(frame, (w//2-250, feedback_y-40), (w//2+250, feedback_y+40), (50, 0, 0), -1)
                cv2.putText(frame, "ERROR:", (w//2-230, feedback_y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.putText(frame, self.error_message[:40], (w//2-230, feedback_y+25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Instructions - smaller and in status bar
            if not self.current_word and current_time - self.error_time > 4:
                cv2.putText(frame, "Join fingers to write | Make fist to recognize word", (w//2-250, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
            
            # Bottom status bar for additional info
            cv2.rectangle(frame, (0, h-60), (w, h), (0, 0, 0), -1)  # Black background
            
            # Show controls at bottom
            cv2.putText(frame, "Controls: 'C' = Clear | 'Q' = Quit", (20, h-35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 2)
            
            # Show finger positions when not actively writing (for reference) - moved to bottom
            if not self.gesture_active and self.tracker.results:
                index_pos = self.tracker.get_index_finger_position(frame)
                middle_pos = self.get_middle_finger_position(frame)
                joined_pos = self.get_joined_fingers_position(frame)
                
                if index_pos and middle_pos and joined_pos:
                    # Draw individual finger positions (smaller, dimmed)
                    cv2.circle(frame, index_pos, 6, (100, 100, 255), 1)  # Dim blue for index
                    cv2.circle(frame, middle_pos, 6, (100, 100, 255), 1)  # Dim blue for middle
                    
                    # Draw line between fingers
                    cv2.line(frame, index_pos, middle_pos, (100, 100, 100), 1)
                    
                    # Draw midpoint (tracking point)
                    cv2.circle(frame, joined_pos, 8, (255, 255, 0), 2)  # Yellow for midpoint
                    cv2.putText(frame, "Track", (joined_pos[0]-20, joined_pos[1]-15), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                    
                    # Show tracking info at bottom
                    cv2.putText(frame, f"Tracking: Midpoint ({joined_pos[0]}, {joined_pos[1]})", (w-350, h-35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Create window
            cv2.namedWindow("Fixed Air Writing System", cv2.WINDOW_FULLSCREEN)
            cv2.imshow("Fixed Air Writing System", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.reset_word()
                self.trajectory = []
                self.recording = False
                self.gesture_active = False
                
                # Reset all detection buffers
                self.finger_detection_buffer.clear()
                self.position_buffer.clear()
                self.velocity_buffer.clear()
                self.last_position = None
                self.last_velocity = None
                
                print("🗑️ System cleared")
        
        cap.release()
        cv2.destroyAllWindows()
        self.tracker.release()

if __name__ == "__main__":
    try:
        system = FixedWordRecognitionSystem()
        system.run()
    except FileNotFoundError:
        print("❌ Error: Model files not found!")
        print("Please ensure air_writing_model.h5 and label_encoder.npy exist")
    except KeyboardInterrupt:
        print("\n👋 Program stopped by user")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")