import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Conv1D, MaxPooling1D, LSTM, Dense, Dropout, 
                                      Bidirectional, LayerNormalization, BatchNormalization)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class AirWritingModel:
    def __init__(self, sequence_length=100, num_features=6):  # Increased features
        self.sequence_length = sequence_length
        self.num_features = num_features  # x, y, dx, dy, d²x, d²y
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def extract_features(self, trajectory):
        """Extract position, velocity, and acceleration features"""
        # Ensure trajectory is 2D (x, y)
        if trajectory.shape[1] != 2:
            trajectory = trajectory[:, :2]
        
        # Position (x, y)
        positions = trajectory
        
        # Velocity (dx/dt, dy/dt)
        velocities = np.diff(positions, axis=0, prepend=positions[0:1])
        
        # Acceleration (d²x/dt², d²y/dt²)
        accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
        
        # Combine all features
        features = np.concatenate([positions, velocities, accelerations], axis=1)
        return features  # Shape: (sequence_length, 6)
    
    def augment_trajectory(self, trajectory):
        """Apply data augmentation techniques"""
        augmented = []
        
        # Original
        augmented.append(trajectory)
        
        # Horizontal flip
        flipped = trajectory.copy()
        flipped[:, 0] = 1 - flipped[:, 0]
        augmented.append(flipped)
        
        # Add Gaussian noise
        noisy = trajectory + np.random.normal(0, 0.015, trajectory.shape)
        noisy = np.clip(noisy, 0, 1)
        augmented.append(noisy)
        
        # Scale variations (90% to 110%)
        scale = np.random.uniform(0.92, 1.08)
        center = trajectory.mean(axis=0)
        scaled = (trajectory - center) * scale + center
        scaled = np.clip(scaled, 0, 1)
        augmented.append(scaled)
        
        # Small rotation (-8 to 8 degrees)
        angle = np.random.uniform(-8, 8) * np.pi / 180
        center = trajectory.mean(axis=0)
        rotated = trajectory - center
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated = rotated @ rotation_matrix.T + center
        rotated = np.clip(rotated, 0, 1)
        augmented.append(rotated)
        
        # Time warping (speed variation)
        indices = np.linspace(0, len(trajectory) - 1, len(trajectory))
        warped_indices = indices + np.random.normal(0, 2, len(indices))
        warped_indices = np.clip(warped_indices, 0, len(trajectory) - 1)
        warped_indices = np.sort(warped_indices)
        
        from scipy.interpolate import interp1d
        interp_x = interp1d(indices, trajectory[:, 0], kind='linear', fill_value='extrapolate')
        interp_y = interp1d(indices, trajectory[:, 1], kind='linear', fill_value='extrapolate')
        warped = np.column_stack([interp_x(warped_indices), interp_y(warped_indices)])
        warped = np.clip(warped, 0, 1)
        augmented.append(warped)
        
        return augmented
    
    def load_dataset(self, dataset_path="dataset", augment=True):
        """Load all trajectory samples from dataset folder with augmentation"""
        X, y = [], []
        
        for word in os.listdir(dataset_path):
            word_path = os.path.join(dataset_path, word)
            if not os.path.isdir(word_path):
                continue
            
            for filename in os.listdir(word_path):
                if filename.endswith('.npy'):
                    filepath = os.path.join(word_path, filename)
                    trajectory = np.load(filepath)
                    
                    # Handle both 2D and 6D features
                    if trajectory.shape[0] == self.sequence_length:
                        if trajectory.shape[1] == 2:
                            # Convert 2D to 6D features
                            trajectory = self.extract_features(trajectory)
                        
                        if trajectory.shape == (self.sequence_length, self.num_features):
                            if augment:
                                # Apply augmentation to 2D positions only
                                pos_2d = trajectory[:, :2]
                                augmented_samples = self.augment_trajectory(pos_2d)
                                
                                for aug_sample in augmented_samples:
                                    aug_features = self.extract_features(aug_sample)
                                    X.append(aug_features)
                                    y.append(word)
                            else:
                                X.append(trajectory)
                                y.append(word)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} samples across {len(np.unique(y))} classes")
        print(f"Feature shape: {X.shape}")
        
        # Show class distribution
        unique_classes, counts = np.unique(y, return_counts=True)
        print(f"\nClass distribution:")
        for cls, count in zip(unique_classes, counts):
            print(f"  {cls}: {count} samples")
        
        return X, y
    
    def build_model(self, num_classes):
        """Build enhanced CNN-BiLSTM architecture with attention"""
        model = Sequential([
            # First CNN block - extract local patterns
            Conv1D(64, 5, activation='relu', padding='same', 
                   input_shape=(self.sequence_length, self.num_features)),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # Second CNN block - deeper feature extraction
            Conv1D(128, 5, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(2),
            Dropout(0.2),
            
            # Third CNN block - high-level features
            Conv1D(256, 3, activation='relu', padding='same'),
            BatchNormalization(),
            Dropout(0.2),
            
            # Bidirectional LSTM - capture temporal dependencies
            Bidirectional(LSTM(128, return_sequences=True)),
            Dropout(0.3),
            Bidirectional(LSTM(64)),
            Dropout(0.3),
            
            # Dense layers with regularization
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        # Use Adam optimizer with custom learning rate
        optimizer = Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'top_k_categorical_accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, dataset_path="dataset", epochs=100, batch_size=32, validation_split=0.2):
        """Train the model with advanced techniques"""
        # Load data with augmentation
        X, y = self.load_dataset(dataset_path, augment=True)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(y_encoded),
            y=y_encoded
        )
        class_weight_dict = dict(enumerate(class_weights))
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y_encoded
        )
        
        print(f"\nTraining samples: {len(X_train)}, Validation samples: {len(X_val)}")
        print(f"Number of classes: {len(self.label_encoder.classes_)}")
        print(f"Classes: {sorted(self.label_encoder.classes_)}")
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.build_model(num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Enhanced callbacks
        early_stop = EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train with class weights
        print("\nStarting training...")
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            class_weight=class_weight_dict,
            callbacks=[early_stop, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_accuracy, val_top_k = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\n{'='*60}")
        print(f"Final Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Top-5 Accuracy: {val_top_k * 100:.2f}%")
        print(f"{'='*60}")
        
        # Save model and encoder
        self.model.save('air_writing_model.h5')
        np.save('label_encoder.npy', self.label_encoder.classes_)
        print("\n✓ Model saved as 'air_writing_model.h5'")
        print("✓ Label encoder saved as 'label_encoder.npy'")
        
        # Plot training history
        self.plot_history(history)
        
        return history
    
    def plot_history(self, history):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Train')
        ax1.plot(history.history['val_accuracy'], label='Validation')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(history.history['loss'], label='Train')
        ax2.plot(history.history['val_loss'], label='Validation')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        print("Training history saved to training_history.png")

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"AIR WRITING MODEL TRAINING - HIGH ACCURACY MODE")
    print(f"{'='*60}\n")
    
    # Initialize model with 6 features (x, y, dx, dy, d²x, d²y)
    model = AirWritingModel(sequence_length=100, num_features=6)
    
    print("Configuration:")
    print(f"  • Sequence length: {model.sequence_length}")
    print(f"  • Features: {model.num_features} (position, velocity, acceleration)")
    print(f"  • Data augmentation: ENABLED (6x per sample)")
    print(f"  • Architecture: CNN-BiLSTM with BatchNorm")
    print(f"  • Optimizer: Adam with learning rate scheduling")
    print(f"\n{'='*60}\n")
    
    # Train with optimal parameters
    history = model.train(
        dataset_path="dataset",
        epochs=50,  # Reduced epochs for faster training
        batch_size=16,  # Smaller batch size to avoid memory issues
        validation_split=0.2
    )
    
    print(f"\n{'='*60}")
    print(f"✓ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Model saved as: 'air_writing_model.h5'")
    print(f"📊 Encoder saved as: 'label_encoder.npy'")
    print(f"📊 Training plot saved as: 'training_history.png'")
    print(f"\n🚀 Next step: Run 'python recognition_live.py' to test recognition")
    print(f"{'='*60}\n")
