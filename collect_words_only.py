"""Collect only word data (no alphabets)"""
from data_collection import DataCollector

if __name__ == "__main__":
    collector = DataCollector()
    
    print("="*60)
    print("WORD DATA COLLECTION")
    print("="*60)
    
    # Word categories
    two_letter = ["hi", "if", "in", "is", "it", "me", "my", "no", "of", "on", 
                  "or", "so", "to", "up", "us", "we"]
    
    three_letter = ["cat", "dog", "sun", "cup", "pen", "box", "car", "hat", "key", "map"]
    
    four_letter = ["book", "door", "hand", "love", "tree", "star", "moon", "fish", "bird", "home"]
    
    five_letter = ["apple", "water", "house", "phone", "happy", "world", "music", "smile", "heart", "peace"]
    
    print("\nWhat would you like to collect?")
    print("1. Two-letter words (16 words)")
    print("2. Three-letter words (10 words)")
    print("3. Four-letter words (10 words)")
    print("4. Five-letter words (10 words)")
    print("5. All words (46 words)")
    
    choice = input("\nEnter choice (1-5): ").strip()
    
    words = []
    if choice == "1":
        words = two_letter
    elif choice == "2":
        words = three_letter
    elif choice == "3":
        words = four_letter
    elif choice == "4":
        words = five_letter
    elif choice == "5":
        words = two_letter + three_letter + four_letter + five_letter
    else:
        print("Invalid choice!")
        exit()
    
    print(f"\nTotal words to collect: {len(words)}")
    
    user_id = input("Enter user ID: ").strip() or "user1"
    num_samples = int(input("Enter number of samples per word (default 20): ") or "20")
    
    print("\n" + "="*60)
    print("INSTRUCTIONS:")
    print("- Press SPACE to start recording")
    print("- Write the word in the air")
    print("- Close hand to save")
    print("- Press 'c' to clear and retry")
    print("="*60)
    
    for i, word in enumerate(words, 1):
        print(f"\n{'='*60}")
        print(f"Word {i}/{len(words)}: {word.upper()}")
        print(f"{'='*60}")
        collector.collect_data(word, user_id, num_samples=num_samples)
    
    print("\n" + "="*60)
    print("✓ Data collection complete!")
    print(f"✓ Collected: {len(words)} words × {num_samples} samples")
    print(f"✓ Total samples: {len(words) * num_samples}")
    print("="*60)
