# Real-Time Air Writing Recognition System - HIGH ACCURACY VERSION

Deep learning-powered air-writing recognition using enhanced CNN-BiLSTM and MediaPipe Hands.

## 🎯 Accuracy Enhancements

This version includes major improvements for **85-95%+ accuracy**:
- **6D Feature Extraction**: Position + Velocity + Acceleration
- **Enhanced Architecture**: CNN-BiLSTM with BatchNormalization
- **Data Augmentation**: 6x multiplication (flip, noise, scale, rotation, time-warp)
- **Advanced Preprocessing**: Cubic interpolation with outlier clipping
- **Improved Hand Tracking**: Kalman-like filtering with velocity prediction
- **Confidence Filtering**: 65% minimum threshold with temporal smoothing
- **Class Balancing**: Weighted loss for imbalanced datasets

## ✨ Features

- **Ultra-Smooth Hand Tracking**: Advanced Kalman-like filtering with acceleration compensation
- **Cubic Spline Interpolation**: Smooth trajectory curves with 3x interpolation
- **Gaussian Smoothing**: Bell-curve weighted position averaging for fluid motion
- **Smart Trajectory Capture**: Quality validation with minimum point requirements
- **Advanced Deep Learning**: CNN-BiLSTM with 6D features (position, velocity, acceleration)
- **Data Augmentation**: Automatic 6x augmentation during training
- **GARBAGE Class Detection**: Robust rejection of invalid gestures and noise
- **Confidence Scoring**: Visual color-coded feedback (Green/Yellow/Orange)
- **Temporal Smoothing**: Majority voting across predictions
- **Voice Feedback**: Text-to-speech with threading
- **Auto-correction**: TextBlob integration for word spelling
- **Comprehensive Vocabulary**: 112 items (52 letters + 60 words)
- **High Accuracy**: 85-95%+ with proper training (50-100 samples/word)

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Collect Training Data

```bash
python data_collection.py
```

- Enter your user ID
- Choose to collect all words or select specific ones
- For each word, press SPACE to start writing
- Write the word in the air with your index finger
- Close your hand (thumb-index together) to save
- Press 'c' to clear and retry
- Repeat for 50-100 samples per word

### 1.5. Collect GARBAGE Data (IMPORTANT!)

```bash
python collect_garbage.py
```

- Collect 100-200 samples of invalid gestures:
  - Random scribbles and meaningless movements
  - Incomplete characters (started but not finished)
  - Very short movements (dots, tiny lines)
  - Erratic or shaky hand movements
  - Accidental gestures
- This dramatically improves recognition accuracy by teaching the model to reject invalid inputs

**Comprehensive word list (112 items):**
- **Uppercase letters (26)**: A-Z
- **Lowercase letters (26)**: a-z
- **2-letter words (20)**: hi, if, in, is, it, me, my, no, of, on, or, so, to, up, us, we, at, be, do, go
- **3-letter words (15)**: cat, dog, sun, cup, pen, box, car, hat, key, map, yes, not, can, get, see
- **4-letter words (15)**: book, door, hand, love, tree, star, moon, fish, bird, home, good, time, work, life, help
- **5-letter words (10)**: apple, water, house, phone, happy, world, music, smile, heart, peace

### 2. Train the Model

```bash
python model_training.py
```

This will:
- Load all collected trajectories
- Train CNN-LSTM model
- Save best model as `air_writing_model.h5`
- Generate training history plot

### 3. Run Real-Time Recognition

```bash
python recognition_live.py
```

- Open your hand and write in the air
- Close your hand to trigger recognition
- The system will display and speak the recognized word

## 🏗️ Architecture

### System Architecture Diagram

![System Architecture](system_architecture_simple.png)

**Enhanced CNN-BiLSTM Model:**
- **Input**: 100 × 6 (time steps × features)
  - Features: x, y, dx/dt, dy/dt, d²x/dt², d²y/dt²
- **CNN Blocks**: 3 layers (64→128→256 filters) with BatchNorm
- **BiLSTM Layers**: 2 layers (128→64 units) for temporal dependencies
- **Dense Layers**: 256→128 units with BatchNorm and Dropout
- **Output**: Softmax classification
- **Optimizer**: Adam with learning rate scheduling
- **Regularization**: Dropout (0.2-0.4) + BatchNormalization

### Generate Architecture Diagrams

```bash
python architecture_diagram.py
```

This generates:
- `system_architecture.png` - Detailed architecture with all components
- `system_architecture_simple.png` - Simplified flow diagram

## Dataset Structure

```
dataset/
├── cat/
│   ├── user1_sample_0.npy
│   ├── user1_sample_1.npy
│   └── ...
├── dog/
├── sun/
├── GARBAGE/          # ← NEW: Invalid gesture samples
│   ├── user1_garbage_0.npy
│   ├── user1_garbage_1.npy
│   └── ...
└── ...
```

Each `.npy` file contains a normalized trajectory of shape (100, 2).

### GARBAGE Class Benefits
- **Reduces False Positives**: Rejects meaningless gestures
- **Improves Accuracy**: Better discrimination between valid/invalid inputs
- **Better UX**: Clear feedback when gestures aren't recognized
- **Robustness**: Handles noise and accidental movements

## 📊 Performance Metrics

**With Proper Training (50-100 samples/word):**
- Model accuracy: 85-95% on validation set
- Top-5 accuracy: 95-98%
- Real-time FPS: 20-30
- Inference latency: <100ms
- Confidence threshold: 65% minimum
- Multi-user support: Yes (collect data from multiple users)

## 💡 Tips for Maximum Accuracy

### Data Collection (MOST IMPORTANT!)
- **Quantity**: Collect 50-100 samples per word (minimum 30)
- **Consistency**: Write the same way every time
- **Quality**: Smooth, continuous movements (no jerky strokes)
- **Environment**: Good lighting, plain background, camera at eye level
- **Speed**: Maintain consistent writing speed across all samples

### Training
- **Balance**: Ensure similar sample counts per class
- **Validation**: Target >85% validation accuracy
- **Patience**: Let early stopping work (usually 40-80 epochs)
- **Monitoring**: Watch for overfitting (train/val accuracy gap)

### Recognition
- **Confidence**: Green (>90%) = excellent, Yellow (75-90%) = good, Orange (<75%) = retry
- **Lighting**: Match training environment conditions
- **Speed**: Write at same speed as training data
- **Clarity**: Deliberate, smooth movements

### Troubleshooting Low Accuracy
1. Collect more data (100+ samples for problematic words)
2. Check data consistency (review saved trajectories)
3. Retrain from scratch (delete old model)
4. Start with fewer words, add gradually
5. Ensure consistent environment (lighting, background)

## Modules

- `hand_tracking.py` - MediaPipe hand detection and tracking
- `data_collection.py` - Interactive trajectory dataset builder
- `collect_garbage.py` - Collect invalid gesture samples for robustness
- `setup_garbage_class.py` - Setup and status check for GARBAGE class
- `model_training.py` - CNN-LSTM training pipeline
- `recognition_live.py` - Real-time inference with UI

## 📚 Documentation

- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Detailed system architecture documentation
  - Component architecture
  - Data flow diagrams
  - Model architecture details
  - GARBAGE class system
  - Performance optimization
  - Troubleshooting guide

## Customization

To add new words, simply collect more data and retrain:

```python
words = ["hello", "world", "python", "code", "test"]
```

Adjust model parameters in `model_training.py` for different accuracy/speed tradeoffs.
