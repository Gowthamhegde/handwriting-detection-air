import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.85, tracking_confidence=0.85):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.hands = None
        self.results = None
        self.smoothing_window = 12  # Larger window for ultra-smooth tracking
        self.position_buffer = deque(maxlen=self.smoothing_window)
        self.velocity_buffer = deque(maxlen=5)  # More velocity samples
        self.acceleration_buffer = deque(maxlen=3)  # Track acceleration
        self.last_position = None
        self.last_velocity = None
        self._initialize_hands()
    
    def _initialize_hands(self):
        """Initialize MediaPipe Hands with optimized settings"""
        if self.hands is not None:
            self.hands.close()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence,
            model_complexity=1  # Use more accurate model
        )
        
    def find_hands(self, frame, draw=True):
        if self.hands is None:
            self._initialize_hands()
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        self.results = self.hands.process(frame_rgb)
        frame_rgb.flags.writeable = True
        
        if draw and self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
        
        return frame
    
    def get_index_finger_position(self, frame):
        """Get ultra-smooth index finger tip position with advanced Kalman-like filtering"""
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            
            index_tip = hand.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            
            # Calculate velocity and acceleration
            if self.last_position is not None:
                velocity = (x - self.last_position[0], y - self.last_position[1])
                self.velocity_buffer.append(velocity)
                
                # Calculate acceleration for even smoother prediction
                if self.last_velocity is not None:
                    acceleration = (velocity[0] - self.last_velocity[0], 
                                  velocity[1] - self.last_velocity[1])
                    self.acceleration_buffer.append(acceleration)
                
                self.last_velocity = velocity
            
            self.last_position = (x, y)
            self.position_buffer.append((x, y))
            
            # Advanced exponential smoothing with Gaussian-like weights
            if len(self.position_buffer) > 0:
                # Create Gaussian-like weights (bell curve centered at recent positions)
                n = len(self.position_buffer)
                indices = np.arange(n)
                weights = np.exp(-((indices - (n-1))**2) / (2 * (n/3)**2))
                weights = weights / weights.sum()
                
                positions = np.array(list(self.position_buffer))
                avg_x = np.sum(positions[:, 0] * weights)
                avg_y = np.sum(positions[:, 1] * weights)
                
                # Advanced velocity smoothing with acceleration compensation
                if len(self.velocity_buffer) >= 3:
                    velocities = np.array(list(self.velocity_buffer))
                    
                    # Exponential smoothing for velocity
                    vel_weights = np.exp(np.linspace(-1, 0, len(velocities)))
                    vel_weights = vel_weights / vel_weights.sum()
                    
                    smooth_vx = np.sum(velocities[:, 0] * vel_weights)
                    smooth_vy = np.sum(velocities[:, 1] * vel_weights)
                    
                    # Apply acceleration compensation for smoother prediction
                    if len(self.acceleration_buffer) > 0:
                        accelerations = np.array(list(self.acceleration_buffer))
                        avg_ax = np.mean(accelerations[:, 0])
                        avg_ay = np.mean(accelerations[:, 1])
                        
                        # Predict with velocity and acceleration
                        prediction_factor = 0.35
                        avg_x = avg_x + smooth_vx * prediction_factor + avg_ax * 0.1
                        avg_y = avg_y + smooth_vy * prediction_factor + avg_ay * 0.1
                    else:
                        # Predict with velocity only
                        avg_x = avg_x + smooth_vx * 0.35
                        avg_y = avg_y + smooth_vy * 0.35
                
                return (int(avg_x), int(avg_y))
        
        return None
    
    def is_hand_open(self, frame, threshold=80):
        """Check if hand is open (fingers spread apart)"""
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            
            # Get fingertip positions
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]
            ring_tip = hand.landmark[16]
            pinky_tip = hand.landmark[20]
            
            # Calculate positions
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
            ring_x, ring_y = int(ring_tip.x * w), int(ring_tip.y * h)
            pinky_x, pinky_y = int(pinky_tip.x * w), int(pinky_tip.y * h)
            
            # Calculate distances between adjacent fingers
            thumb_index_dist = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            index_middle_dist = np.sqrt((index_x - middle_x)**2 + (index_y - middle_y)**2)
            middle_ring_dist = np.sqrt((middle_x - ring_x)**2 + (middle_y - ring_y)**2)
            
            # Hand is open if fingers are spread apart
            return (thumb_index_dist > threshold and 
                    index_middle_dist > threshold * 0.5 and
                    middle_ring_dist > threshold * 0.5)
        
        return False
    
    def is_hand_closed(self, frame, threshold=50):
        """Enhanced hand closure detection - checks if hand is in fist position"""
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            
            # Get fingertip and knuckle positions
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            middle_tip = hand.landmark[12]
            ring_tip = hand.landmark[16]
            
            # Get wrist/palm base
            wrist = hand.landmark[0]
            
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            middle_x, middle_y = int(middle_tip.x * w), int(middle_tip.y * h)
            ring_x, ring_y = int(ring_tip.x * w), int(ring_tip.y * h)
            wrist_x, wrist_y = int(wrist.x * w), int(wrist.y * h)
            
            # Calculate distances
            thumb_index_dist = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            thumb_middle_dist = np.sqrt((thumb_x - middle_x)**2 + (thumb_y - middle_y)**2)
            
            # Check if fingertips are close to wrist (curled fingers)
            index_wrist_dist = np.sqrt((index_x - wrist_x)**2 + (index_y - wrist_y)**2)
            middle_wrist_dist = np.sqrt((middle_x - wrist_x)**2 + (middle_y - wrist_y)**2)
            
            # Hand is closed if:
            # 1. Thumb and index are close together
            # 2. Thumb and middle are close together  
            # 3. Fingertips are relatively close to wrist (curled)
            is_closed = (
                thumb_index_dist < threshold and 
                thumb_middle_dist < threshold * 1.3 and
                index_wrist_dist < threshold * 3.5 and
                middle_wrist_dist < threshold * 3.5
            )
            
            return is_closed
        
        return False
    
    def get_hand_confidence(self):
        """Get detection confidence score"""
        if self.results.multi_hand_landmarks:
            return 1.0  # MediaPipe doesn't expose confidence directly
        return 0.0
    
    def release(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None
