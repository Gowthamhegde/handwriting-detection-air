# Real-Time Air Writing Recognition System

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

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam
- Windows/Linux/macOS

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gowthamhegde/handwriting-detection-air.git
   cd handwriting-detection-air
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Dependencies
- opencv-python >= 4.8.0 (camera/video processing)
- mediapipe >= 0.10.0 (hand tracking)
- tensorflow >= 2.13.0 (deep learning model)
- numpy >= 1.23.0, < 2.0.0 (numerical operations)
- scikit-learn >= 1.3.0 (data preprocessing)
- pyttsx3 >= 2.90 (text-to-speech)
- textblob >= 0.17.0 (spell checking)
- matplotlib >= 3.7.0 (visualization)
- scipy >= 1.10.0 (signal processing)

## 🎮 Quick Start

### 1. Setup Alphabet Folders (First Time Only)

```bash
python setup_alphabet_folders.py
```

This creates the dataset folder structure for all letters and words.

### 2. Collect Training Data

```bash
python quick_collect.py
```

- Enter your user ID
- Choose to collect all words or select specific ones
- For each word, press SPACE to start writing
- Write the word in the air with your index finger
- Close your hand (thumb-index together) to save
- Press 'c' to clear and retry
- Repeat for 50-100 samples per word

**Comprehensive word list (112 items):**
- **Uppercase letters (26)**: A-Z
- **Lowercase letters (26)**: a-z
- **2-letter words (20)**: hi, if, in, is, it, me, my, no, of, on, or, so, to, up, us, we, at, be, do, go
- **3-letter words (15)**: cat, dog, sun, cup, pen, box, car, hat, key, map, yes, not, can, get, see
- **4-letter words (15)**: book, door, hand, love, tree, star, moon, fish, bird, home, good, time, work, life, help
- **5-letter words (10)**: apple, water, house, phone, happy, world, music, smile, heart, peace

### 3. Train the Model

```bash
python model_training.py
```

This will:
- Load all collected trajectories
- Apply data augmentation (6x multiplication)
- Train CNN-BiLSTM model with 6D features
- Save best model as `air_writing_model.h5`
- Generate training history plot

### 4. Run Real-Time Recognition

```bash
python run_air_writing.py
```

**How to use:**
- Show your hand to the camera
- Write in the air with your index finger
- Close your hand (pinch thumb and index) to trigger recognition
- The system will display and speak the recognized word
- Press 'q' to quit

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

## 📊 Key Modules

- `setup_alphabet_folders.py` - Creates dataset folder structure
- `quick_collect.py` - Interactive trajectory dataset builder
- `model_training.py` - CNN-BiLSTM training pipeline with data augmentation
- `run_air_writing.py` - Main application entry point
- `fixed_word_recognition.py` - Real-time inference engine with hand tracking

## 📁 Project Structure

```
handwriting-detection-air/
├── dataset/                      # Training data (created after setup)
│   ├── A/
│   ├── B/
│   ├── cat/
│   ├── dog/
│   └── ...
├── air_writing_model.h5          # Trained model (created after training)
├── label_encoder.npy             # Label encoder (created after training)
├── training_history.png          # Training metrics plot
├── english_dictionary.json       # Word dictionary for spell checking
├── setup_alphabet_folders.py     # Setup dataset folders
├── quick_collect.py              # Data collection tool
├── model_training.py             # Model training script
├── run_air_writing.py            # Main recognition application
├── fixed_word_recognition.py     # Core recognition logic
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

Each `.npy` file in the dataset contains a normalized trajectory of shape (100, 2).

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

## 🎨 Customization

To add new words, edit the word list in `quick_collect.py` and `fixed_word_recognition.py`:

```python
words = ["hello", "world", "python", "code", "test"]
```

Then collect data and retrain the model. Adjust model parameters in `model_training.py` for different accuracy/speed tradeoffs.

## 🤝 Contributing

Contributions are welcome! Feel free to:
- Report bugs
- Suggest new features
- Submit pull requests
- Improve documentation

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- MediaPipe for hand tracking
- TensorFlow for deep learning framework
- OpenCV for computer vision utilities
