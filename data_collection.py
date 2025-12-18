import cv2
import numpy as np
import os
from hand_tracking import HandTracker
from scipy.interpolate import interp1d

class DataCollector:
    def __init__(self, dataset_path="dataset", sequence_length=100):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.tracker = HandTracker()
        self.trajectory = []
        self.recording = False
        
        os.makedirs(dataset_path, exist_ok=True)
    
    def normalize_trajectory(self, trajectory):
        """Advanced normalization with centering and scaling"""
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
        
        # Clip outliers (3 standard deviations)
        normalized = np.clip(normalized, -3, 3)
        
        # Rescale to [0, 1] range
        min_val = normalized.min()
        max_val = normalized.max()
        if max_val > min_val:
            normalized = (normalized - min_val) / (max_val - min_val)
        else:
            normalized = np.ones_like(normalized) * 0.5
        
        # Resample to fixed length with cubic interpolation
        if len(normalized) != self.sequence_length:
            old_indices = np.linspace(0, len(normalized) - 1, len(normalized))
            new_indices = np.linspace(0, len(normalized) - 1, self.sequence_length)
            
            # Use cubic interpolation for smoother curves
            interp_x = interp1d(old_indices, normalized[:, 0], kind='cubic', fill_value='extrapolate')
            interp_y = interp1d(old_indices, normalized[:, 1], kind='cubic', fill_value='extrapolate')
            
            normalized = np.column_stack([interp_x(new_indices), interp_y(new_indices)])
            normalized = np.clip(normalized, 0, 1)
        
        return normalized
    
    def collect_data(self, word, user_id, num_samples=50):
        """Collect trajectory samples for a specific word"""
        word_path = os.path.join(self.dataset_path, word)
        os.makedirs(word_path, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        sample_count = 0
        
        print(f"Collecting {num_samples} samples for word: {word}")
        print("Instructions:")
        print("- Press SPACE to start recording")
        print("- OPEN your hand (spread fingers) and write in the air")
        print("- CLOSE your hand (make a fist) to automatically save")
        print("- Press 'x' to cancel without saving")
        print("- Press 'c' to clear screen")
        print("- Press 'q' to quit")
        
        while sample_count < num_samples:
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
            
            # Only record when hand is open and moving
            if self.recording and pos and is_hand_open and not is_hand_closed:
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
                            
                            # Draw smooth polyline with anti-aliasing
                            cv2.polylines(frame, [smooth_points], False, (0, 255, 0), 3, cv2.LINE_AA)
                        except:
                            # Fallback to regular polylines
                            points_int = points.astype(np.int32)
                            cv2.polylines(frame, [points_int], False, (0, 255, 0), 3, cv2.LINE_AA)
                    else:
                        # For first few points, use regular lines
                        for i in range(1, len(self.trajectory)):
                            cv2.line(frame, self.trajectory[i-1], self.trajectory[i], 
                                   (0, 255, 0), 3, cv2.LINE_AA)
                
                # Draw smooth current position indicator
                cv2.circle(frame, pos, 8, (0, 255, 0), -1, cv2.LINE_AA)
                cv2.circle(frame, pos, 10, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Check for hand closure to save (outside the recording block)
            if self.recording and is_hand_closed:
                    if len(self.trajectory) > 15:  # Increased minimum points
                        normalized = self.normalize_trajectory(self.trajectory)
                        if normalized is not None:
                            # Validate trajectory quality
                            trajectory_length = np.sum(np.sqrt(np.sum(np.diff(normalized, axis=0)**2, axis=1)))
                            
                            # Only save if trajectory has sufficient movement
                            if trajectory_length > 0.5:
                                filename = f"{user_id}_sample_{sample_count}.npy"
                                filepath = os.path.join(word_path, filename)
                                np.save(filepath, normalized)
                                sample_count += 1
                                print(f"✓ Saved sample {sample_count}/{num_samples} (quality: {trajectory_length:.2f})")
                            else:
                                print(f"⚠ Rejected: trajectory too short/static (quality: {trajectory_length:.2f})")
                    else:
                        print(f"⚠ Rejected: only {len(self.trajectory)} points (need >15)")
                    
                    self.trajectory = []
                    self.recording = False
                    cv2.waitKey(500)  # Brief pause
            
            # Display info
            cv2.putText(frame, f"Word: {word} | Samples: {sample_count}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "SPACE: Start | Open hand: Write | Close hand: Save | X: Cancel", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show hand state
            h, w, _ = frame.shape
            if is_hand_closed:
                cv2.putText(frame, "HAND: CLOSED", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.recording:
                    cv2.putText(frame, ">>> SAVING <<<", (w//2 - 100, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif is_hand_open:
                cv2.putText(frame, "HAND: OPEN", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "HAND: NEUTRAL", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if self.recording:
                cv2.putText(frame, "RECORDING... (Close hand to save)", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not self.recording:
                self.recording = True
                self.trajectory = []
                print("▶ Recording started...")
            elif key == ord('c'):
                # Clear screen
                self.trajectory = []
                self.recording = False
                print("🗑 Screen cleared!")
            elif key == ord('x'):
                # Stop recording without saving
                if self.recording:
                    self.trajectory = []
                    self.recording = False
                    print("⏹ Recording stopped (not saved)")
                else:
                    self.trajectory = []
                    print("🗑 Trajectory cleared")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.tracker.release()
        print(f"Collection complete! Saved {sample_count} samples.")

if __name__ == "__main__":
    collector = DataCollector()
    
    # Comprehensive word list for training - OPTIMIZED FOR ACCURACY
    words = [
        # English uppercase alphabets (26)
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        # English lowercase alphabets (26)
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        # Common 2-letter words (20)
        "hi", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", 
        "so", "to", "up", "us", "we", "at", "be", "do", "go",
        # Common 3-letter words (15)
        "cat", "dog", "sun", "cup", "pen", "box", "car", "hat", "key", "map",
        "yes", "not", "can", "get", "see",
        # Common 4-letter words (15)
        "book", "door", "hand", "love", "tree", "star", "moon", "fish", "bird", "home",
        "good", "time", "work", "life", "help",
        # Common 5-letter words (10)
        "apple", "water", "house", "phone", "happy", "world", "music", "smile", "heart", "peace"
    ]
    
    print(f"\n{'='*60}")
    print(f"AIR WRITING DATA COLLECTION - HIGH ACCURACY MODE")
    print(f"{'='*60}")
    print(f"Total words available: {len(words)}")
    print(f"  - Uppercase letters: 26")
    print(f"  - Lowercase letters: 26")
    print(f"  - 2-letter words: 20")
    print(f"  - 3-letter words: 15")
    print(f"  - 4-letter words: 15")
    print(f"  - 5-letter words: 10")
    print(f"\n💡 TIPS FOR HIGH ACCURACY:")
    print(f"  • Collect 50-100 samples per word for best results")
    print(f"  • Write consistently and clearly")
    print(f"  • Ensure good lighting and camera position")
    print(f"  • Keep hand movements smooth and deliberate")
    print(f"{'='*60}\n")
    
    user_id = input("Enter user ID (e.g., user1): ").strip() or "user1"
    num_samples = int(input("Enter samples per word (recommended 50-100): ").strip() or "50")
    
    # Option to select specific words
    print("\nCollection Options:")
    print("1. Collect ALL words (recommended for best accuracy)")
    print("2. Collect only LETTERS (A-Z, a-z)")
    print("3. Collect only WORDS (2-5 letter words)")
    print("4. Select SPECIFIC words")
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "2":
        words = [w for w in words if len(w) == 1]
        print(f"\n✓ Selected {len(words)} letters")
    elif choice == "3":
        words = [w for w in words if len(w) > 1]
        print(f"\n✓ Selected {len(words)} words")
    elif choice == "4":
        print("\nAvailable words:")
        for i, word in enumerate(words, 1):
            print(f"{i:3d}. {word}")
        selected = input("\nEnter word numbers separated by commas (e.g., 1,3,5): ")
        indices = [int(x.strip()) - 1 for x in selected.split(",")]
        words = [words[i] for i in indices if 0 <= i < len(words)]
        print(f"\n✓ Selected {len(words)} items")
    
    print(f"\n{'='*60}")
    print(f"STARTING COLLECTION")
    print(f"{'='*60}")
    print(f"Items to collect: {len(words)}")
    print(f"Samples per item: {num_samples}")
    print(f"Total samples: {len(words) * num_samples}")
    print(f"{'='*60}\n")
    
    input("Press ENTER to start...")
    
    for i, word in enumerate(words, 1):
        print(f"\n{'='*60}")
        print(f"📝 Word {i}/{len(words)}: '{word.upper()}'")
        print(f"{'='*60}")
        collector.collect_data(word, user_id, num_samples=num_samples)
    
    print(f"\n{'='*60}")
    print(f"✓ DATA COLLECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Total words collected: {len(words)}")
    print(f"Samples per word: {num_samples}")
    print(f"Total samples: {len(words) * num_samples}")
    print(f"\n📊 Next step: Run 'python model_training.py' to train the model")
    print(f"{'='*60}")
