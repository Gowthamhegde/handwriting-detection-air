#!/usr/bin/env python3
"""
Test script for the Intelligent Air-Writing Word Recognition System
"""

import json
import os
import numpy as np

def test_dictionary_loading():
    """Test dictionary loading functionality"""
    print("Testing dictionary loading...")
    
    # Test with existing dictionary
    if os.path.exists('english_dictionary.json'):
        with open('english_dictionary.json', 'r') as f:
            dictionary = json.load(f)
        print(f"✓ Dictionary loaded: {len(dictionary)} words")
        
        # Test some common words
        test_words = ["CAT", "DOG", "HELLO", "WORLD", "LOVE"]
        for word in test_words:
            if word in dictionary:
                print(f"✓ '{word}' found in dictionary")
            else:
                print(f"❌ '{word}' NOT found in dictionary")
    else:
        print("❌ Dictionary file not found")

def test_model_files():
    """Test if required model files exist"""
    print("\nTesting model files...")
    
    required_files = [
        'air_writing_model.h5',
        'label_encoder.npy'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ {file} found")
        else:
            print(f"❌ {file} NOT found - Please train the model first")

def test_alphabet_filtering():
    """Test alphabet filtering logic"""
    print("\nTesting alphabet filtering...")
    
    # Simulate model classes
    sample_classes = ['A', 'B', 'C', 'cat', 'dog', 'GARBAGE', 'hello', 'X', 'Y', 'Z']
    valid_letters = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    
    alphabet_classes = [cls for cls in sample_classes if cls in valid_letters]
    print(f"✓ Filtered alphabet classes: {alphabet_classes}")
    print(f"✓ Total alphabet classes: {len(alphabet_classes)}")

def test_word_validation():
    """Test word validation logic"""
    print("\nTesting word validation...")
    
    # Load dictionary
    try:
        with open('english_dictionary.json', 'r') as f:
            dictionary = json.load(f)
        dictionary_set = set(word.upper() for word in dictionary)
        
        # Test words
        test_cases = [
            ("CAT", True),
            ("DOG", True),
            ("HELLO", True),
            ("XYZ", False),
            ("ABCD", False),
            ("LOVE", True)
        ]
        
        for word, expected in test_cases:
            result = word in dictionary_set
            status = "✓" if result == expected else "❌"
            print(f"{status} '{word}' validation: {result} (expected: {expected})")
            
    except Exception as e:
        print(f"❌ Error testing word validation: {e}")

def create_test_trajectory():
    """Create a sample trajectory for testing"""
    print("\nCreating test trajectory...")
    
    # Simple letter 'A' trajectory (triangle shape)
    trajectory = []
    
    # Left stroke (bottom to top)
    for i in range(50):
        x = 0.3 + i * 0.004  # Slight rightward movement
        y = 0.8 - i * 0.012  # Upward movement
        trajectory.append((x, y))
    
    # Right stroke (top to bottom)
    for i in range(50):
        x = 0.5 + i * 0.004  # Rightward movement
        y = 0.2 + i * 0.012  # Downward movement
        trajectory.append((x, y))
    
    print(f"✓ Created test trajectory with {len(trajectory)} points")
    return np.array(trajectory)

def test_feature_extraction():
    """Test feature extraction logic"""
    print("\nTesting feature extraction...")
    
    trajectory = create_test_trajectory()
    
    # Extract features (position, velocity, acceleration)
    positions = trajectory
    velocities = np.diff(positions, axis=0, prepend=positions[0:1])
    accelerations = np.diff(velocities, axis=0, prepend=velocities[0:1])
    features = np.concatenate([positions, velocities, accelerations], axis=1)
    
    print(f"✓ Trajectory shape: {trajectory.shape}")
    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Feature dimensions: {features.shape[1]} (expected: 6)")

def run_all_tests():
    """Run all tests"""
    print("="*60)
    print("INTELLIGENT AIR-WRITING WORD RECOGNITION SYSTEM - TESTS")
    print("="*60)
    
    test_dictionary_loading()
    test_model_files()
    test_alphabet_filtering()
    test_word_validation()
    test_feature_extraction()
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print("If all tests pass (✓), the system should work correctly.")
    print("If any tests fail (❌), please address the issues before running.")
    print("\nTo run the system:")
    print("  python word_recognition_system.py")
    print("="*60)

if __name__ == "__main__":
    run_all_tests()