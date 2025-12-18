"""Unified data collection for alphabets and words"""
from data_collection import DataCollector

if __name__ == "__main__":
    collector = DataCollector()
    
    print("="*60)
    print("AIR WRITING DATA COLLECTION")
    print("="*60)
    
    # Define all categories
    uppercase = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                 "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    
    lowercase = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                 "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    
    two_letter = ["hi","on","me"]
    
    three_letter = ["cat", "dog", "sun", "cup", "pen", "box", "car", "hat", "key", "map",
                    "yes", "not", "can", "get", "see"]
    
    four_letter = ["book", "door", "hand", "love", "tree", "star", "moon", "fish", "bird", "home",
                   "good", "time", "work", "life", "help"]
    
    five_letter = ["apple", "water", "house", "phone", "happy", "world", "music", "smile", "heart", "peace"]
    
    print("\nWhat would you like to collect?")
    print("="*60)
    print("ALPHABETS:")
    print("  1. Uppercase letters (A-Z) - 26 items")
    print("  2. Lowercase letters (a-z) - 26 items")
    print("  3. Both uppercase & lowercase - 52 items")
    print("\nWORDS:")
    print("  4. Two-letter words - 20 words")
    print("  5. Three-letter words - 15 words")
    print("  6. Four-letter words - 15 words")
    print("  7. Five-letter words - 10 words")
    print("  8. All words (2-5 letters) - 60 words")
    print("\nCOMBINED:")
    print("  9. Alphabets + Words - 112 items (FULL DATASET)")
    print("="*60)
    
    choice = input("\nEnter choice (1-9): ").strip()
    
    words = []
    if choice == "1":
        words = uppercase
        print(f"\n✓ Selected: Uppercase letters (26)")
    elif choice == "2":
        words = lowercase
        print(f"\n✓ Selected: Lowercase letters (26)")
    elif choice == "3":
        words = uppercase + lowercase
        print(f"\n✓ Selected: All letters (52)")
    elif choice == "4":
        words = two_letter
        print(f"\n✓ Selected: Two-letter words (20)")
    elif choice == "5":
        words = three_letter
        print(f"\n✓ Selected: Three-letter words (15)")
    elif choice == "6":
        words = four_letter
        print(f"\n✓ Selected: Four-letter words (15)")
    elif choice == "7":
        words = five_letter
        print(f"\n✓ Selected: Five-letter words (10)")
    elif choice == "8":
        words = two_letter + three_letter + four_letter + five_letter
        print(f"\n✓ Selected: All words (60)")
    elif choice == "9":
        words = uppercase + lowercase + two_letter + three_letter + four_letter + five_letter
        print(f"\n✓ Selected: FULL DATASET (112 items)")
    else:
        print("Invalid choice!")
        exit()
    
    user_id = input("\nEnter user ID: ").strip() or "user1"
    num_samples = int(input("Enter samples per item (recommended 50): ").strip() or "50")
    
    print("\n" + "="*60)
    print("COLLECTION SUMMARY:")
    print(f"  Items to collect: {len(words)}")
    print(f"  Samples per item: {num_samples}")
    print(f"  Total samples: {len(words) * num_samples}")
    print(f"  User ID: {user_id}")
    print("="*60)
    print("\nINSTRUCTIONS:")
    print("  • Press SPACE to start recording")
    print("  • OPEN your hand (spread fingers) and write in the air")
    print("  • CLOSE your hand (make a fist) to automatically save")
    print("  • Press 'x' to cancel without saving")
    print("  • Press 'c' to clear screen and retry")
    print("  • Press 'q' to skip to next item")
    print("="*60)
    
    input("\nPress ENTER to start...")
    
    for i, word in enumerate(words, 1):
        print(f"\n{'='*60}")
        print(f"Item {i}/{len(words)}: '{word}'")
        print(f"{'='*60}")
        collector.collect_data(word, user_id, num_samples=num_samples)
    
    print("\n" + "="*60)
    print("✓ DATA COLLECTION COMPLETE!")
    print("="*60)
    print(f"✓ Collected: {len(words)} items")
    print(f"✓ Samples per item: {num_samples}")
    print(f"✓ Total samples: {len(words) * num_samples}")
    print(f"✓ User ID: {user_id}")
    print(f"\n📊 Next step: Run 'python model_training.py' to train")
    print("="*60)
