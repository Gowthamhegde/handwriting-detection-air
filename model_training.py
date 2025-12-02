import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

class AirWritingModel:
    def __init__(self, sequence_length=100, num_features=2):
        self.sequence_length = sequence_length
        self.num_features = num_features
        self.model = None
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, dataset_path="dataset"):
        """Load all trajectory samples from dataset folder"""
        X, y = [], []
        
        for word in os.listdir(dataset_path):
            word_path = os.path.join(dataset_path, word)
            if not os.path.isdir(word_path):
                continue
            
            for filename in os.listdir(word_path):
                if filename.endswith('.npy'):
                    filepath = os.path.join(word_path, filename)
                    trajectory = np.load(filepath)
                    
                    if trajectory.shape == (self.sequence_length, self.num_features):
                        X.append(trajectory)
                        y.append(word)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Loaded {len(X)} samples across {len(np.unique(y))} classes")
        return X, y
    
    def build_model(self, num_classes):
        """Build CNN-LSTM architecture"""
        model = Sequential([
            Conv1D(64, 3, activation='relu', input_shape=(self.sequence_length, self.num_features)),
            MaxPooling1D(2),
            Conv1D(128, 3, activation='relu'),
            MaxPooling1D(2),
            LSTM(128, return_sequences=True),
            LSTM(64),
            Dense(128, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, dataset_path="dataset", epochs=50, batch_size=32, validation_split=0.2):
        """Train the model"""
        # Load data
        X, y = self.load_dataset(dataset_path)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        y_categorical = to_categorical(y_encoded)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_categorical, test_size=validation_split, random_state=42, stratify=y_encoded
        )
        
        print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Build model
        num_classes = len(self.label_encoder.classes_)
        self.build_model(num_classes)
        
        print("\nModel Architecture:")
        self.model.summary()
        
        # Callbacks
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True)
        
        # Train
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, checkpoint],
            verbose=1
        )
        
        # Evaluate
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")
        
        # Save model and encoder
        self.model.save('air_writing_model.h5')
        np.save('label_encoder.npy', self.label_encoder.classes_)
        
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
    model = AirWritingModel()
    model.train(epochs=50, batch_size=32)
