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
        """Normalize and resample trajectory to fixed length"""
        if len(trajectory) < 2:
            return None
        
        trajectory = np.array(trajectory)
        
        # Normalize to [0, 1] range
        min_vals = trajectory.min(axis=0)
        max_vals = trajectory.max(axis=0)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1  # Avoid division by zero
        
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
    
    def collect_data(self, word, user_id, num_samples=50):
        """Collect trajectory samples for a specific word"""
        word_path = os.path.join(self.dataset_path, word)
        os.makedirs(word_path, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        sample_count = 0
        
        print(f"Collecting {num_samples} samples for word: {word}")
        print("Instructions:")
        print("- Press SPACE to start recording")
        print("- Write the word in the air")
        print("- Close your hand (thumb-index together) to stop")
        print("- Press 'c' to clear screen")
        print("- Press 'q' to quit")
        
        while sample_count < num_samples:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            frame = self.tracker.find_hands(frame)
            
            # Get index finger position
            pos = self.tracker.get_index_finger_position(frame)
            
            if self.recording and pos:
                self.trajectory.append(pos)
                # Draw trajectory
                for i in range(1, len(self.trajectory)):
                    cv2.line(frame, self.trajectory[i-1], self.trajectory[i], (0, 255, 0), 2)
                
                # Check for hand closure (end of word)
                if self.tracker.is_hand_closed(frame):
                    if len(self.trajectory) > 10:
                        normalized = self.normalize_trajectory(self.trajectory)
                        if normalized is not None:
                            filename = f"{user_id}_sample_{sample_count}.npy"
                            filepath = os.path.join(word_path, filename)
                            np.save(filepath, normalized)
                            sample_count += 1
                            print(f"Saved sample {sample_count}/{num_samples}")
                    
                    self.trajectory = []
                    self.recording = False
                    cv2.waitKey(500)  # Brief pause
            
            # Display info
            cv2.putText(frame, f"Word: {word} | Samples: {sample_count}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, "SPACE: Start | C: Clear | Close hand: Stop | Q: Quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if self.recording:
                cv2.putText(frame, "RECORDING...", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow("Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not self.recording:
                self.recording = True
                self.trajectory = []
            elif key == ord('c'):
                # Clear screen
                self.trajectory = []
                self.recording = False
                print("Screen cleared!")
            elif key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        self.tracker.release()
        print(f"Collection complete! Saved {sample_count} samples.")

if __name__ == "__main__":
    collector = DataCollector()
    
    # Comprehensive word list for training
    words = [
        # English uppercase alphabets
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
        "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
        # English lowercase alphabets
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
        "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
        # Two letter words
        "hi", "if", "in", "is", "it", "me", "my", "no", "of", "on", "or", 
        "so", "to", "up", "us", "we",
        # Common 3-letter words
        "cat", "dog", "sun", "cup", "pen", "box", "car", "hat", "key", "map",
        # Common 4-letter words
        "book", "door", "hand", "love", "tree", "star", "moon", "fish", "bird", "home",
        # Common 5-letter words
        "apple", "water", "house", "phone", "happy", "world", "music", "smile", "heart", "peace"
    ]
    
    print(f"Total words to collect: {len(words)}")
    print("Recommended: 20-50 samples per word for good accuracy")
    
    user_id = input("Enter user ID: ")
    num_samples = int(input("Enter number of samples per word (default 20): ") or "20")
    
    # Option to select specific words
    print("\nOptions:")
    print("1. Collect all words")
    print("2. Select specific words")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "2":
        print("\nAvailable words:")
        for i, word in enumerate(words, 1):
            print(f"{i}. {word}")
        selected = input("\nEnter word numbers separated by commas (e.g., 1,3,5): ")
        indices = [int(x.strip()) - 1 for x in selected.split(",")]
        words = [words[i] for i in indices if 0 <= i < len(words)]
    
    for i, word in enumerate(words, 1):
        print(f"\n{'='*50}")
        print(f"Word {i}/{len(words)}: {word.upper()}")
        print(f"{'='*50}")
        collector.collect_data(word, user_id, num_samples=num_samples)
    
    print("\n" + "="*50)
    print("Data collection complete!")
    print(f"Total words collected: {len(words)}")
    print(f"Samples per word: {num_samples}")
    print(f"Total samples: {len(words) * num_samples}")
    print("="*50)
