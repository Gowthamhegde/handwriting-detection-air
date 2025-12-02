"""Test Windows native speech using win32com"""
import time

try:
    import win32com.client
    
    print("Testing Windows Native Speech (SAPI)")
    print("="*50)
    
    speaker = win32com.client.Dispatch("SAPI.SpVoice")
    
    words = ["cat", "dog", "sun", "book", "tree", "pen", "cup", "hat"]
    
    for i, word in enumerate(words, 1):
        print(f"{i}. Speaking: {word}")
        speaker.Speak(word)
        time.sleep(0.5)
    
    print("\n" + "="*50)
    print("✓ Test complete!")
    print("Did you hear all 8 words?")
    print("="*50)
    
except ImportError:
    print("win32com not available")
    print("Install with: pip install pywin32")
except Exception as e:
    print(f"Error: {e}")
