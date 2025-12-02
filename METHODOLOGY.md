# Real-Time Air Writing Recognition - Methodology

## MODULE 1: HAND TRACKING (hand_tracking.py)
- MediaPipe Hands for 21-point landmark detection
- Index finger tip (landmark #8) tracking
- Moving average smoothing (window=5)
- Hand closure detection (thumb-index distance ≤40px)
- Output: 2D trajectory coordinates (x, y)

## MODULE 2: DATA COLLECTION (data_collection.py)
- Interactive webcam-based collection
- User writes letters/words in air
- Trajectory normalization to [0,1] range
- Resampling to fixed 100 time steps
- Storage: .npy files (100×2 arrays)
- Requirements: 20-50 samples per class

## MODULE 3: MODEL ARCHITECTURE (model_training.py)
**CNN-LSTM Hybrid Model:**
- Input: (100, 2) trajectory sequence
- Conv1D(64) → MaxPool → Conv1D(128) → MaxPool
- LSTM(128) → LSTM(64)
- Dense(128) → Dropout(0.3) → Dense(classes)
- Optimizer: Adam
- Loss: Categorical cross-entropy
- Training: 50 epochs, batch=32, validation=20%

## MODULE 4: REAL-TIME RECOGNITION (recognition_live.py)
**Inference Pipeline:**
1. Webcam capture → Hand detection
2. Trajectory extraction during writing
3. Hand closure triggers recognition
4. Normalize & resample trajectory
5. CNN-LSTM prediction
6. Display result + confidence
7. Text-to-speech output (pyttsx3)

**Performance:**
- FPS: >20
- Inference: <200ms
- Voice: Queue-based threading

## MODULE 5: SYSTEM WORKFLOW

**Phase 1 - Data Collection:**
```
Webcam → Hand Tracking → Trajectory → Normalize → Save .npy
```

**Phase 2 - Training:**
```
Load Dataset → Encode Labels → Train CNN-LSTM → Save Model
```

**Phase 3 - Inference:**
```
Live Video → Extract Trajectory → Predict → Display + Speak
```

## TECHNICAL STACK
- Python 3.10
- TensorFlow 2.19, Keras 3.12
- OpenCV 4.11, MediaPipe 0.10
- NumPy 1.26, SciPy 1.15
- pyttsx3 2.99 (TTS)

## PERFORMANCE METRICS
- Validation Accuracy: 82-86%
- Real-time FPS: 20-30
- Inference Time: <200ms
- Dataset: 16 classes, 500 samples
- Model Size: ~850KB

## KEY FEATURES
- Single letter & word recognition
- Real-time voice feedback
- Visual trajectory display
- Multi-user support
- Confidence scoring
- Auto spell-correction (TextBlob)
