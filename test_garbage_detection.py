#!/usr/bin/env python3
"""
Test script to demonstrate GARBAGE class detection.
This script loads the trained model and tests it with some sample trajectories.
"""

import numpy as np
import os
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

def create_test_trajectories():
    """Create sample trajectories for testing"""
    trajectories = {}
    
    # Valid letter 'A' - triangle shape
    t = np.linspace(0, 1, 100)
    a_trajectory = np.column_stack([
        0.3 + 0.4 * t,  # x: left to right
        0.8 - 0.6 * np.abs(t - 0.5)  # y: triangle shape
    ])
    trajectories['Valid A'] = a_trajectory
    
    # Valid letter 'O' - circle shape
    angle = np.linspace(0, 2*np.pi, 100)
    o_trajectory = np.column_stack([
        0.5 + 0.3 * np.cos(angle),  # x: circle
        0.5 + 0.3 * np.sin(angle)   # y: circle
    ])
    trajectories['Valid O'] = o_trajectory
    
    # GARBAGE: Random scribble
    np.random.seed(42)
    random_x = np.cumsum(np.random.randn(100) * 0.02) + 0.5
    random_y = np.cumsum(np.random.randn(100) * 0.02) + 0.5
    random_trajectory = np.column_stack([
        np.clip(random_x, 0, 1),
        np.clip(random_y, 0, 1)
    ])
    trajectories['Garbage Scribble'] = random_trajectory
    
    # GARBAGE: Very short line
    short_trajectory = np.column_stack([
        np.linspace(0.45, 0.55, 100),  # very short horizontal line
        np.ones(100) * 0.5
    ])
    trajectories['Garbage Short Line'] = short_trajectory
    
    # GARBAGE: Erratic movement
    erratic_x = 0.5 + 0.1 * np.sin(np.linspace(0, 20*np.pi, 100))
    erratic_y = 0.5 + 0.1 * np.cos(np.linspace(0, 15*np.pi, 100))
    erratic_trajectory = np.column_stack([erratic_x, erratic_y])
    trajectories['Garbage Erratic'] = erratic_trajectory
    
    return trajectories

def extract_features(trajectory):
    """Extract 6D features (position, velocity, acceleration)"""
    # Position (x, y)
    positions = trajectory
    
    # Velocity (dx/dt, dy/dt)
    velocities = np.diff(positions, axis=0, prepend=positions[0:1])
    
    # Acceleration (d²x/dt², d²y/dt²)
    accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
    
    # Combine all features
    features = np.concatenate([positions, velocities, accelerations], axis=1)
    return features

def test_garbage_detection():
    """Test the model's ability to detect garbage vs valid characters"""
    
    # Check if model exists
    if not os.path.exists('air_writing_model.h5'):
        print("❌ Model not found! Please train the model first:")
        print("   1. Run: python collect_garbage.py")
        print("   2. Run: python model_training.py")
        return
    
    if not os.path.exists('label_encoder.npy'):
        print("❌ Label encoder not found! Please train the model first.")
        return
    
    # Load model and classes
    try:
        model = load_model('air_writing_model.h5')
        classes = np.load('label_encoder.npy', allow_pickle=True)
        print(f"✓ Model loaded successfully")
        print(f"✓ Classes: {len(classes)} total")
        
        # Check if GARBAGE class exists
        if 'GARBAGE' not in classes:
            print("⚠ GARBAGE class not found in model!")
            print("   Please collect garbage data and retrain:")
            print("   1. Run: python collect_garbage.py")
            print("   2. Run: python model_training.py")
            return
        else:
            print(f"✓ GARBAGE class found in model")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create test trajectories
    test_trajectories = create_test_trajectories()
    
    print(f"\n{'='*60}")
    print(f"TESTING GARBAGE DETECTION")
    print(f"{'='*60}")
    
    # Test each trajectory
    results = []
    for name, trajectory in test_trajectories.items():
        # Extract features
        features = extract_features(trajectory)
        X = np.expand_dims(features, axis=0)
        
        # Predict
        predictions = model.predict(X, verbose=0)[0]
        predicted_class = classes[np.argmax(predictions)]
        confidence = np.max(predictions) * 100
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_predictions = [(classes[i], predictions[i] * 100) for i in top_indices]
        
        results.append({
            'name': name,
            'trajectory': trajectory,
            'predicted': predicted_class,
            'confidence': confidence,
            'top_predictions': top_predictions
        })
        
        # Display result
        status = "✓" if (name.startswith('Valid') and predicted_class != 'GARBAGE') or \
                       (name.startswith('Garbage') and predicted_class == 'GARBAGE') else "❌"
        
        print(f"\n{status} {name}:")
        print(f"   Predicted: {predicted_class} ({confidence:.1f}%)")
        print(f"   Top 3: {', '.join([f'{cls}({conf:.1f}%)' for cls, conf in top_predictions])}")
    
    # Visualize trajectories
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, result in enumerate(results):
        if i < len(axes):
            ax = axes[i]
            trajectory = result['trajectory']
            
            # Color based on prediction correctness
            is_correct = (result['name'].startswith('Valid') and result['predicted'] != 'GARBAGE') or \
                        (result['name'].startswith('Garbage') and result['predicted'] == 'GARBAGE')
            color = 'green' if is_correct else 'red'
            
            ax.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=2)
            ax.scatter(trajectory[0, 0], trajectory[0, 1], color='blue', s=50, label='Start')
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], color='red', s=50, label='End')
            
            ax.set_title(f"{result['name']}\nPredicted: {result['predicted']} ({result['confidence']:.1f}%)")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend()
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('garbage_detection_test.png', dpi=150, bbox_inches='tight')
    print(f"\n✓ Visualization saved as 'garbage_detection_test.png'")
    
    # Summary
    correct_predictions = sum(1 for result in results if 
                            (result['name'].startswith('Valid') and result['predicted'] != 'GARBAGE') or
                            (result['name'].startswith('Garbage') and result['predicted'] == 'GARBAGE'))
    
    print(f"\n{'='*60}")
    print(f"TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total tests: {len(results)}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {correct_predictions/len(results)*100:.1f}%")
    
    if correct_predictions == len(results):
        print(f"🎉 Perfect! GARBAGE detection is working correctly!")
    elif correct_predictions >= len(results) * 0.8:
        print(f"✓ Good! GARBAGE detection is mostly working.")
    else:
        print(f"⚠ GARBAGE detection needs improvement. Consider:")
        print(f"   • Collecting more diverse garbage samples")
        print(f"   • Retraining the model")
        print(f"   • Adjusting confidence thresholds")
    
    print(f"{'='*60}")

if __name__ == "__main__":
    test_garbage_detection()