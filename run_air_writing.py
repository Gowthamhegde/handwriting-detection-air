#!/usr/bin/env python3
"""
Air Writing Recognition System - Main Launcher
Simple launcher for the fixed word recognition system
"""

import sys
import os

def main():
    """Launch the air writing recognition system"""
    print("🚀 Starting Air Writing Recognition System...")
    print("=" * 50)
    
    # Check if required files exist
    required_files = [
        'fixed_word_recognition.py',
        'hand_tracking.py', 
        'air_writing_model.h5',
        'label_encoder.npy',
        'english_dictionary.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Error: Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease ensure all required files are present.")
        return 1
    
    print("✓ All required files found")
    print("✓ Launching air writing system...")
    print("=" * 50)
    
    # Import and run the main system
    try:
        from fixed_word_recognition import FixedWordRecognitionSystem
        system = FixedWordRecognitionSystem()
        system.run()
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("Please install required dependencies: pip install -r requirements.txt")
        return 1
    except FileNotFoundError as e:
        print(f"❌ File Error: {e}")
        print("Please ensure model files are trained and available.")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Program stopped by user")
        return 0
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())