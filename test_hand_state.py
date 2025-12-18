"""Test hand state detection (open vs closed)"""
import cv2
from hand_tracking import HandTracker

def test_hand_state():
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    print("="*60)
    print("HAND STATE DETECTION TEST")
    print("="*60)
    print("Instructions:")
    print("- Spread your fingers apart = OPEN (GREEN)")
    print("- Make a fist = CLOSED (RED)")
    print("- Relaxed hand = NEUTRAL (YELLOW)")
    print("- Press 'q' to quit")
    print("="*60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = tracker.find_hands(frame)
        
        # Check hand states
        is_open = tracker.is_hand_open(frame)
        is_closed = tracker.is_hand_closed(frame)
        
        h, w, _ = frame.shape
        
        # Display status with color coding
        if is_closed:
            color = (0, 0, 255)  # Red
            state = "CLOSED (FIST)"
            cv2.rectangle(frame, (0, 0), (w, 150), color, -1)
        elif is_open:
            color = (0, 255, 0)  # Green
            state = "OPEN (SPREAD)"
            cv2.rectangle(frame, (0, 0), (w, 150), color, -1)
        else:
            color = (0, 255, 255)  # Yellow
            state = "NEUTRAL"
            cv2.rectangle(frame, (0, 0), (w, 150), (100, 100, 0), -1)
        
        # Display state
        cv2.putText(frame, f"HAND STATE: {state}", (w//2 - 200, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        
        # Instructions
        cv2.putText(frame, "Open = Spread fingers | Closed = Make fist", 
                   (10, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, "Press 'q' to quit", 
                   (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Hand State Test", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    tracker.release()
    
    print(f"\n{'='*60}")
    print("Test complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    test_hand_state()
