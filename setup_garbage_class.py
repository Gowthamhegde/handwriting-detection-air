#!/usr/bin/env python3
"""
Setup script for adding GARBAGE class to air writing recognition system.
This script helps users understand and set up the garbage class for better model robustness.
"""

import os
import numpy as np

def check_dataset_status():
    """Check current dataset status and provide recommendations"""
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print("❌ Dataset folder not found!")
        print("Please run data_collection.py first to collect character/word data.")
        return False
    
    # Count existing classes and samples
    classes = []
    total_samples = 0
    garbage_samples = 0
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        if os.path.isdir(item_path):
            npy_files = [f for f in os.listdir(item_path) if f.endswith('.npy')]
            sample_count = len(npy_files)
            
            if sample_count > 0:
                classes.append(item)
                total_samples += sample_count
                
                if item == "GARBAGE":
                    garbage_samples = sample_count
    
    print(f"\n{'='*60}")
    print(f"DATASET STATUS REPORT")
    print(f"{'='*60}")
    print(f"Total classes: {len(classes)}")
    print(f"Total samples: {total_samples}")
    
    if "GARBAGE" in classes:
        print(f"✓ GARBAGE class exists: {garbage_samples} samples")
    else:
        print(f"❌ GARBAGE class missing")
    
    print(f"\nExisting classes:")
    for cls in sorted(classes):
        cls_path = os.path.join(dataset_path, cls)
        sample_count = len([f for f in os.listdir(cls_path) if f.endswith('.npy')])
        status = "✓" if sample_count >= 20 else "⚠"
        print(f"  {status} {cls}: {sample_count} samples")
    
    print(f"{'='*60}")
    
    # Provide recommendations
    print(f"\nRECOMMENDATIONS:")
    
    if "GARBAGE" not in classes:
        print(f"🎯 CRITICAL: Add GARBAGE class for robust recognition")
        print(f"   • Run: python collect_garbage.py")
        print(f"   • Collect 100-200 garbage samples")
        print(f"   • Include: random scribbles, incomplete chars, erratic movements")
    elif garbage_samples < 50:
        print(f"🎯 IMPROVE: Collect more GARBAGE samples ({garbage_samples} < 50 recommended)")
        print(f"   • Run: python collect_garbage.py")
        print(f"   • Add {50 - garbage_samples} more samples")
    else:
        print(f"✓ GARBAGE class looks good ({garbage_samples} samples)")
    
    # Check for low sample classes
    low_sample_classes = [cls for cls in classes if cls != "GARBAGE" and 
                         len([f for f in os.listdir(os.path.join(dataset_path, cls)) 
                             if f.endswith('.npy')]) < 20]
    
    if low_sample_classes:
        print(f"🎯 IMPROVE: Some classes have few samples:")
        for cls in low_sample_classes:
            count = len([f for f in os.listdir(os.path.join(dataset_path, cls)) if f.endswith('.npy')])
            print(f"   • {cls}: {count} samples (recommend 50+)")
    
    print(f"\n📊 NEXT STEPS:")
    if "GARBAGE" not in classes or garbage_samples < 50:
        print(f"1. Run: python collect_garbage.py")
        print(f"2. Collect garbage/invalid gesture samples")
        print(f"3. Run: python model_training.py")
        print(f"4. Test: python recognition_live.py")
    else:
        print(f"1. Run: python model_training.py (retrain with GARBAGE class)")
        print(f"2. Test: python recognition_live.py")
    
    print(f"{'='*60}")
    
    return True

def explain_garbage_class():
    """Explain what the GARBAGE class is and why it's important"""
    print(f"\n{'='*60}")
    print(f"WHAT IS THE GARBAGE CLASS?")
    print(f"{'='*60}")
    print(f"The GARBAGE class helps the model recognize and reject invalid gestures:")
    print(f"\n🎯 WHAT TO INCLUDE IN GARBAGE CLASS:")
    print(f"  • Random scribbles and meaningless movements")
    print(f"  • Incomplete characters (started but not finished)")
    print(f"  • Very short movements (dots, tiny lines)")
    print(f"  • Erratic or shaky hand movements")
    print(f"  • Accidental gestures when not intending to write")
    print(f"  • Multiple disconnected strokes")
    print(f"  • Partial letters or broken character attempts")
    print(f"  • Hand movements during pauses or thinking")
    print(f"\n🚀 BENEFITS:")
    print(f"  • Reduces false positive predictions")
    print(f"  • Improves overall recognition accuracy")
    print(f"  • Makes the system more robust to noise")
    print(f"  • Provides better user experience")
    print(f"  • Allows rejection of unclear inputs")
    print(f"\n💡 COLLECTION TIPS:")
    print(f"  • Collect 100-200 garbage samples")
    print(f"  • Make them diverse and realistic")
    print(f"  • Include common mistakes users might make")
    print(f"  • Vary the length and complexity")
    print(f"{'='*60}")

def create_garbage_folder():
    """Create GARBAGE folder if it doesn't exist"""
    garbage_path = os.path.join("dataset", "GARBAGE")
    if not os.path.exists(garbage_path):
        os.makedirs(garbage_path, exist_ok=True)
        print(f"✓ Created GARBAGE folder: {garbage_path}")
        return True
    else:
        print(f"✓ GARBAGE folder already exists: {garbage_path}")
        return False

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print(f"AIR WRITING GARBAGE CLASS SETUP")
    print(f"{'='*60}")
    
    # Explain the concept
    explain_garbage_class()
    
    # Check current status
    if check_dataset_status():
        # Create garbage folder if needed
        create_garbage_folder()
        
        print(f"\n{'='*60}")
        print(f"READY TO COLLECT GARBAGE DATA!")
        print(f"{'='*60}")
        print(f"Run the following command to start collecting:")
        print(f"  python collect_garbage.py")
        print(f"{'='*60}")
    
    input("\nPress ENTER to continue...")