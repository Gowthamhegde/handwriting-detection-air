import cv2
import numpy as np
import os
from hand_tracking import HandTracker
from scipy.interpolate import interp1d

class GarbageDataCollector:
    def __init__(self, dataset_path="dataset", sequence_length=100):
        self.dataset_path = dataset_path
        self.sequence_length = sequence_length
        self.tracker = HandTracker()
        self.trajectory = []
        self.recording = False
        
        # Create GARBAGE folder
        self.garbage_path = os.path.join(dataset_path, "GARBAGE")
        os.makedirs(self.garbage_path, exist_ok=True)
    
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
    
    def collect_garbage_data(self, user_id, num_samples=100):
        """Collect garbage/invalid gesture samples"""
        cap = cv2.VideoCapture(0)
        sample_count = 0
        
        print(f"Collecting {num_samples} GARBAGE samples")
        print("\n" + "="*60)
        print("GARBAGE DATA COLLECTION INSTRUCTIONS:")
        print("="*60)
        print("Collect various INVALID gestures and movements:")
        print("• Random scribbles and meaningless movements")
        print("• Incomplete characters (start but don't finish)")
        print("• Very short movements (dots, tiny lines)")
        print("• Erratic hand movements")
        print("• Accidental gestures")
        print("• Hand movements when not intending to write")
        print("• Partial letters or broken strokes")
        print("• Multiple disconnected strokes")
        print("="*60)
        print("\nControls:")
        print("- Press SPACE to start recording")
        print("- OPEN your hand and make invalid gestures")
        print("- CLOSE your hand to save as garbage")
        print("- Press 'x' to cancel without saving")
        print("- Press 'c' to clear screen")
        print("- Press 'q' to quit")
        print("="*60)
        
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
                
                # Draw trajectory in RED to indicate garbage collection
                if len(self.trajectory) > 1:
                    if len(self.trajectory) >= 4:
                        # Use cubic spline for smooth curves
                        points = np.array(self.trajectory, dtype=np.float32)
                        t = np.linspace(0, 1, len(points))
                        t_smooth = np.linspace(0, 1, len(points) * 3)
                        
                        try:
                            spline_x = interp1d(t, points[:, 0], kind='cubic')
                            spline_y = interp1d(t, points[:, 1], kind='cubic')
                            
                            smooth_points = np.column_stack([
                                spline_x(t_smooth),
                                spline_y(t_smooth)
                            ]).astype(np.int32)
                            
                            # Draw in RED for garbage
                            cv2.polylines(frame, [smooth_points], False, (0, 0, 255), 3, cv2.LINE_AA)
                        except:
                            points_int = points.astype(np.int32)
                            cv2.polylines(frame, [points_int], False, (0, 0, 255), 3, cv2.LINE_AA)
                    else:
                        for i in range(1, len(self.trajectory)):
                            cv2.line(frame, self.trajectory[i-1], self.trajectory[i], 
                                   (0, 0, 255), 3, cv2.LINE_AA)
                
                # Draw current position in RED
                cv2.circle(frame, pos, 8, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, pos, 10, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Check for hand closure to save
            if self.recording and is_hand_closed:
                if len(self.trajectory) > 5:  # Lower threshold for garbage data
                    normalized = self.normalize_trajectory(self.trajectory)
                    if normalized is not None:
                        filename = f"{user_id}_garbage_{sample_count}.npy"
                        filepath = os.path.join(self.garbage_path, filename)
                        np.save(filepath, normalized)
                        sample_count += 1
                        print(f"✓ Saved garbage sample {sample_count}/{num_samples}")
                else:
                    print(f"⚠ Rejected: only {len(self.trajectory)} points (need >5)")
                
                self.trajectory = []
                self.recording = False
                cv2.waitKey(300)  # Brief pause
            
            # Display info with RED theme for garbage collection
            h, w, _ = frame.shape
            cv2.rectangle(frame, (0, 0), (w, 120), (0, 0, 50), -1)  # Dark red background
            
            cv2.putText(frame, f"GARBAGE Collection | Samples: {sample_count}/{num_samples}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Make INVALID gestures: scribbles, incomplete chars, random moves", 
                       (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "SPACE: Start | Open hand: Draw garbage | Close hand: Save", 
                       (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, "X: Cancel | C: Clear | Q: Quit", 
                       (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show hand state
            if is_hand_closed:
                cv2.putText(frame, "HAND: CLOSED", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if self.recording:
                    cv2.putText(frame, ">>> SAVING GARBAGE <<<", (w//2 - 150, h//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            elif is_hand_open:
                cv2.putText(frame, "HAND: OPEN", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "HAND: NEUTRAL", (w - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if self.recording:
                cv2.putText(frame, "RECORDING GARBAGE... (Close hand to save)", (10, h-30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Add suggestions for garbage types
            suggestions = [
                "Try: Random scribbles",
                "Try: Incomplete letters", 
                "Try: Very short lines",
                "Try: Erratic movements",
                "Try: Multiple dots",
                "Try: Broken strokes"
            ]
            suggestion = suggestions[sample_count % len(suggestions)]
            cv2.putText(frame, suggestion, (w - 300, h - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 1)
            
            cv2.imshow("Garbage Data Collection", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(' ') and not self.recording:
                self.recording = True
                self.trajectory = []
                print("▶ Recording garbage gesture...")
            elif key == ord('c'):
                self.trajectory = []
                self.recording = False
                print("🗑 Screen cleared!")
            elif key == ord('x'):
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
        print(f"\n✓ Garbage collection complete! Saved {sample_count} samples.")
        return sample_count

if __name__ == "__main__":
    collector = GarbageDataCollector()
    
    print(f"\n{'='*60}")
    print(f"GARBAGE DATA COLLECTION FOR ROBUST RECOGNITION")
    print(f"{'='*60}")
    print("This will collect samples of INVALID gestures to improve model robustness.")
    print("The model will learn to reject meaningless movements and incomplete characters.")
    print(f"{'='*60}\n")
    
    user_id = input("Enter user ID (e.g., user1): ").strip() or "user1"
    num_samples = int(input("Enter number of garbage samples (recommended 100-200): ").strip() or "100")
    
    print(f"\n{'='*60}")
    print(f"STARTING GARBAGE COLLECTION")
    print(f"{'='*60}")
    print(f"Target samples: {num_samples}")
    print(f"Save location: dataset/GARBAGE/")
    print(f"{'='*60}\n")
    
    input("Press ENTER to start collecting garbage data...")
    
    collected = collector.collect_garbage_data(user_id, num_samples)
    
    print(f"\n{'='*60}")
    print(f"✓ GARBAGE DATA COLLECTION COMPLETE!")
    print(f"{'='*60}")
    print(f"Collected samples: {collected}")
    print(f"Saved to: dataset/GARBAGE/")
    print(f"\n📊 Next steps:")
    print(f"1. Run 'python model_training.py' to retrain with garbage class")
    print(f"2. Test recognition with 'python recognition_live.py'")
    print(f"{'='*60}")