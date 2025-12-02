"""Create folder structure for alphabets in dataset"""
import os

dataset_path = "dataset"

# Create uppercase alphabet folders
uppercase = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
             "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Create lowercase alphabet folders
lowercase = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
             "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]

print("Creating alphabet folders in dataset...")
print("="*60)

created = 0
existing = 0

for letter in uppercase + lowercase:
    folder_path = os.path.join(dataset_path, letter)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"✓ Created: {letter}")
        created += 1
    else:
        print(f"  Exists: {letter}")
        existing += 1

print("="*60)
print(f"✓ Created {created} new folders")
print(f"  {existing} folders already existed")
print(f"  Total alphabet folders: {len(uppercase + lowercase)}")
print("\nNext step: Run data collection to add samples")
print("  python data_collection.py")
