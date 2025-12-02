import cv2
import numpy as np
import time
from tensorflow.keras.models import load_model
from hand_tracking import HandTracker
from scipy.interpolate import interp1d
import pyttsx3
from textblob import TextBlob
import threading

class AirWritingRecognizer:
    def __init__(self, model_path='air_writing_model.h5', encoder_path='label_encoder.npy', sequence_length=100):
        self.model = load_model(model_path)
        self.classes = np.load(encoder_path, allow_pickle=True)
        self.sequence_length = sequence_length
        self.tracker = HandTracker()
        
        # Text-to-speech initialization
        try:
            # Test if TTS is available
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
        
    def normalize_trajectory(self, trajectory):
        """Normalize and resample trajectory"""
        if len(trajectory) < 2:
            return None
        
        trajectory = np.array(trajectory)
        
        # Normalize to [0, 1]
        min_vals = trajectory.min(axis=0)
        max_vals = trajectory.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1
        
        normalized = (trajectory - min_vals) / range_vals
        
        # Resample to fixed length
        if len(normalized) == self.sequence_length:
            return normalized
        
        old_indices = np.linspace(0, len(normalized) - 1, len(normalized))
        new_indices = np.linspace(0, len(normalized) - 1, self.sequence_length)
        
        interp_x = interp1d(old_indices, normalized[:, 0], kind='linear')
        interp_y = interp1d(old_indices, normalized[:, 1], kind='linear')
        
        resampled = np.column_stack([interp_x(new_indices), interp_y(new_indices)])
        return resampled
    
    def predict_word(self, trajectory):
        """Predict word from trajectory"""
        normalized = self.normalize_trajectory(trajectory)
        if normalized is None:
            return None, 0.0, []
        
        # Predict
        start_time = time.time()
        X = np.expand_dims(normalized, axis=0)
        predictions = self.model.predict(X, verbose=0)[0]
        inference_time = (time.time() - start_time) * 1000  # ms
        
        # Get top predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [(self.classes[i], predictions[i] * 100) for i in top_indices]
        
        predicted_word = self.classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Auto-correction with TextBlob
        corrected_word = str(TextBlob(predicted_word).correct())
        
        return corrected_word, confidence, top_predictions
    
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
            frame = self.tracker.find_hands(frame, draw=False)
            
            # Get index finger position
            pos = self.tracker.get_index_finger_position(frame)
            is_closed = self.tracker.is_hand_closed(frame)
            
            # Recording logic
            if pos and not is_closed:
                if not self.recording:
                    self.recording = True
                    self.trajectory = []
                
                self.trajectory.append(pos)
                
                # Draw trajectory in green
                for i in range(1, len(self.trajectory)):
                    cv2.line(frame, self.trajectory[i-1], self.trajectory[i], (0, 255, 0), 3)
                
                # Draw current position
                cv2.circle(frame, pos, 8, (0, 255, 0), -1)
            
            elif is_closed and self.recording:
                # Hand closed - recognize word
                if len(self.trajectory) > 10:
                    predicted_word, confidence, top_preds = self.predict_word(self.trajectory)
                    
                    if predicted_word:
                        self.last_prediction = predicted_word
                        self.confidence = confidence
                        self.prediction_time = time.time()
                        
                        print(f"\n{'='*50}")
                        print(f"✓ Recognized: {predicted_word.upper()} ({confidence:.1f}%)")
                        print(f"{'='*50}")
                        print("Top 3 predictions:")
                        for word, conf in top_preds:
                            print(f"  {word}: {conf:.1f}%")
                        
                        # Speak the word EVERY TIME
                        self.speak(predicted_word)
                
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
            
            if self.recording:
                cv2.putText(frame, "WRITING...", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Points: {len(self.trajectory)}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Open hand to write", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(frame, "Close hand to recognize", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display last prediction
            if time.time() - self.prediction_time < 3:  # Show for 3 seconds
                cv2.rectangle(frame, (0, h-100), (w, h), (0, 0, 0), -1)
                cv2.putText(frame, f"Word: {self.last_prediction}", (10, h-60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
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
                print("Screen cleared!")
        
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
