# Real-Time Air Writing Recognition System

Deep learning-powered air-writing recognition using CNN-LSTM and MediaPipe Hands.

## Features

- Real-time hand tracking with MediaPipe (21 landmarks)
- Index finger trajectory extraction with smoothing
- CNN-LSTM model for temporal sequence recognition
- Hand closure detection for word segmentation
- Voice feedback with text-to-speech
- Auto-correction with TextBlob
- High accuracy (≥93% target) with multi-user support

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
- Choose to collect all 30 words or select specific ones
- For each word, press SPACE to start writing
- Write the word in the air with your index finger
- Close your hand (thumb-index together) to save
- Press 'c' to clear and retry
- Repeat for 20-50 samples per word

**Default word list (30 words):**
- 3-letter: cat, dog, sun, cup, pen, box, car, hat, key, map
- 4-letter: book, door, hand, love, tree, star, moon, fish, bird, home
- 5-letter: apple, water, house, phone, happy, world, music, smile, heart, peace

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

## Architecture

**CNN-LSTM Model:**
- Input: 100 × 2 (time steps × coordinates)
- Conv1D layers for spatial feature extraction
- LSTM layers for temporal dynamics
- Dense softmax output for classification

## Dataset Structure

```
dataset/
├── cat/
│   ├── user1_sample_0.npy
│   ├── user1_sample_1.npy
│   └── ...
├── dog/
├── sun/
└── ...
```

Each `.npy` file contains a normalized trajectory of shape (100, 2).

## Performance Targets

- Model accuracy: ≥93% on test set
- Real-time FPS: >20
- Inference latency: <200ms
- Multi-user robustness

## Tips for Best Results

- Collect data from multiple users (5-10)
- Ensure good lighting conditions
- Write clearly with consistent speed
- Collect 50+ samples per word for production use
- Keep hand at consistent distance from camera

## Modules

- `hand_tracking.py` - MediaPipe hand detection and tracking
- `data_collection.py` - Interactive trajectory dataset builder
- `model_training.py` - CNN-LSTM training pipeline
- `recognition_live.py` - Real-time inference with UI

## Customization

To add new words, simply collect more data and retrain:

```python
words = ["hello", "world", "python", "code", "test"]
```

Adjust model parameters in `model_training.py` for different accuracy/speed tradeoffs.
