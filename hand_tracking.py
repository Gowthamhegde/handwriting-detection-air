import cv2
import mediapipe as mp
import numpy as np
from collections import deque

class HandTracker:
    def __init__(self, max_hands=1, detection_confidence=0.7, tracking_confidence=0.7):
        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence
        self.hands = None
        self.results = None
        self.smoothing_window = 5
        self.position_buffer = deque(maxlen=self.smoothing_window)
        self._initialize_hands()
    
    def _initialize_hands(self):
        """Initialize MediaPipe Hands"""
        if self.hands is not None:
            self.hands.close()
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
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
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        
        return frame
    
    def get_index_finger_position(self, frame):
        """Get smoothed index finger tip position (landmark 8)"""
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            
            index_tip = hand.landmark[8]
            x, y = int(index_tip.x * w), int(index_tip.y * h)
            
            self.position_buffer.append((x, y))
            
            # Moving average smoothing
            if len(self.position_buffer) > 0:
                avg_x = int(np.mean([p[0] for p in self.position_buffer]))
                avg_y = int(np.mean([p[1] for p in self.position_buffer]))
                return (avg_x, avg_y)
        
        return None
    
    def is_hand_closed(self, frame, threshold=40):
        """Detect hand closure by thumb-index distance"""
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = frame.shape
            
            thumb_tip = hand.landmark[4]
            index_tip = hand.landmark[8]
            
            thumb_x, thumb_y = int(thumb_tip.x * w), int(thumb_tip.y * h)
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            
            distance = np.sqrt((thumb_x - index_x)**2 + (thumb_y - index_y)**2)
            return distance < threshold
        
        return False
    
    def release(self):
        if self.hands is not None:
            self.hands.close()
            self.hands = None
