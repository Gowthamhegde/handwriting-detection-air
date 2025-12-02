"""Collect alphabet and 2-letter word data"""
from data_collection import DataCollector

if __name__ == "__main__":
    collector = DataCollector()
    
    print("="*60)
    print("ALPHABET & 2-LETTER WORD DATA COLLECTION")
    print("="*60)
    
    # Choose what to collect
    print("\nWhat would you like to collect?")
    print("1. Uppercase letters (A-Z) - 26 letters")
    print("2. Lowercase letters (a-z) - 26 letters")
    print("3. Two-letter words - 16 words")
    print("4. All of the above - 68 items")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    uppercase = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", 
                 "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    
    lowercase = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
                 "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z"]
    
    two_letter = ["hi", "if", "in", "is", "it", "me", "my", "no", "of", "on", 
                  "or", "so", "to", "up", "us", "we"]
    
    words = []
    if choice == "1":
        words = uppercase
    elif choice == "2":
        words = lowercase
    elif choice == "3":
        words = two_letter
    elif choice == "4":
        words = uppercase + lowercase + two_letter
    else:
        print("Invalid choice!")
        exit()
    
    print(f"\nTotal items to collect: {len(words)}")
    
    user_id = input("Enter user ID: ").strip() or "user1"
    num_samples = int(input("Enter number of samples per item (default 20): ") or "20")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("- Press SPACE to start recording")
    print("- Write clearly in the air")
    print("- Close hand (thumb-index together) to save")
    print("- Press 'c' to clear and retry")
    print("- Press 'q' to skip to next item")
    print("="*60)
    
    for i, word in enumerate(words, 1):
        print(f"\n{'='*60}")
        print(f"Item {i}/{len(words)}: '{word}'")
        print(f"{'='*60}")
        collector.collect_data(word, user_id, num_samples=num_samples)
    
    print("\n" + "="*60)
    print("✓ Data collection complete!")
    print(f"✓ Collected: {len(words)} items × {num_samples} samples")
    print(f"✓ Total samples: {len(words) * num_samples}")
    print(f"✓ User ID: {user_id}")
    print("\nNext step: Run 'python model_training.py' to retrain the model")
    print("="*60)
