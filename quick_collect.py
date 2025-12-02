"""Quick data collection helper - collects a subset of words for testing"""
from data_collection import DataCollector

if __name__ == "__main__":
    collector = DataCollector()
    
    # Quick test set - 10 common words
    test_words = ["cat", "dog", "sun", "cup", "pen", "book", "love", "tree", "star", "home"]
    
    print("="*60)
    print("QUICK DATA COLLECTION - 10 Words")
    print("="*60)
    print("Words:", ", ".join(test_words))
    print("\nThis will collect 20 samples per word (200 total samples)")
    print("\nInstructions:")
    print("- Press SPACE to start writing each sample")
    print("- Write clearly in the air with your index finger")
    print("- Close hand (thumb-index together) to save")
    print("- Press 'c' to clear and retry")
    print("- Press 'q' to skip to next word")
    print("="*60)
    
    user_id = input("\nEnter your user ID (e.g., user1): ").strip() or "user1"
    
    for i, word in enumerate(test_words, 1):
        print(f"\n{'='*60}")
        print(f"Word {i}/{len(test_words)}: {word.upper()}")
        print(f"{'='*60}")
        collector.collect_data(word, user_id, num_samples=20)
    
    print("\n" + "="*60)
    print("✓ Data collection complete!")
    print(f"✓ Collected: {len(test_words)} words × 20 samples = 200 total")
    print(f"✓ User ID: {user_id}")
    print("\nNext step: Run 'python model_training.py' to train the model")
    print("="*60)
