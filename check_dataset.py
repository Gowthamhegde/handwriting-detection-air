"""Check the collected dataset statistics"""
import os
import numpy as np

def check_dataset(dataset_path="dataset"):
    if not os.path.exists(dataset_path):
        print(f"Dataset folder '{dataset_path}' not found!")
        print("Run data_collection.py first to collect data.")
        return
    
    print("="*60)
    print("DATASET STATISTICS")
    print("="*60)
    
    words = []
    total_samples = 0
    
    for word in sorted(os.listdir(dataset_path)):
        word_path = os.path.join(dataset_path, word)
        if not os.path.isdir(word_path):
            continue
        
        samples = [f for f in os.listdir(word_path) if f.endswith('.npy')]
        num_samples = len(samples)
        total_samples += num_samples
        
        words.append((word, num_samples))
        
        # Check sample shape
        if num_samples > 0:
            sample_path = os.path.join(word_path, samples[0])
            sample = np.load(sample_path)
            shape = sample.shape
        else:
            shape = "N/A"
        
        print(f"{word:15s} : {num_samples:3d} samples  (shape: {shape})")
    
    print("="*60)
    print(f"Total words: {len(words)}")
    print(f"Total samples: {total_samples}")
    print(f"Average samples per word: {total_samples/len(words):.1f}" if words else "N/A")
    print("="*60)
    
    if total_samples < 100:
        print("\n⚠ Warning: Less than 100 samples collected!")
        print("Recommendation: Collect at least 20 samples per word")
    elif total_samples < 500:
        print("\n✓ Good start! For better accuracy, collect more samples.")
        print("Recommendation: 30-50 samples per word for production use")
    else:
        print("\n✓ Excellent! You have enough data for training.")
    
    print("\nNext step: Run 'python model_training.py' to train the model")

if __name__ == "__main__":
    check_dataset()
