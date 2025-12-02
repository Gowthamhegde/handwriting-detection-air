"""Test SAPI5 voice with new engine each time"""
import pyttsx3
import time

print("Testing SAPI5 Voice (New Engine Each Time)")
print("="*50)
print("This creates a fresh engine for each word")
print("="*50)

words = ["cat", "dog", "sun", "book", "tree", "pen", "cup", "hat"]

for i, word in enumerate(words, 1):
    print(f"\n{i}. Speaking: {word}")
    
    try:
        # Create new engine for each word
        engine = pyttsx3.init('sapi5')
        engine.setProperty('rate', 150)
        engine.setProperty('volume', 1.0)
        
        engine.say(word)
        engine.runAndWait()
        engine.stop()
        del engine
        
        print(f"   ✓ Spoke '{word}' successfully")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    time.sleep(0.5)

print("\n" + "="*50)
print("✓ Test complete!")
print("Did you hear ALL 8 words?")
print("="*50)
