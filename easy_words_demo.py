#!/usr/bin/env python3
"""
Demo script showing easy words to try with the Air-Writing Word Recognition System
"""

import json
import random

def show_easy_words():
    """Display easy words categorized by length"""
    print("="*60)
    print("EASY WORDS FOR AIR-WRITING PRACTICE")
    print("="*60)
    
    print("\n🔤 2-LETTER WORDS (Super Easy - Great for beginners):")
    print("-" * 50)
    easy_2_letter = [
        "AM", "AN", "AS", "AT", "BE", "BY", "DO", "GO", "HE", "HI",
        "IF", "IN", "IS", "IT", "ME", "MY", "NO", "OF", "OH", "OK",
        "ON", "OR", "SO", "TO", "UP", "US", "WE", "YO"
    ]
    
    for i, word in enumerate(easy_2_letter):
        if i % 7 == 0:
            print()
        print(f"{word:4}", end="")
    print("\n")
    
    print("\n🔤 3-LETTER WORDS (Easy - Simple shapes):")
    print("-" * 50)
    easy_3_letter = [
        "CAT", "DOG", "SUN", "BOX", "CUP", "HAT", "PEN", "BAG", "EGG", "EYE",
        "ARM", "LEG", "EAR", "TOY", "CAR", "BUS", "BAT", "BED", "BOY", "DAD",
        "MOM", "RUN", "SIT", "EAT", "SEE", "BIG", "HOT", "NEW", "OLD", "RED"
    ]
    
    for i, word in enumerate(easy_3_letter):
        if i % 6 == 0:
            print()
        print(f"{word:5}", end="")
    print("\n")
    
    print("\n🔤 4-LETTER WORDS (Medium - Good for practice):")
    print("-" * 50)
    easy_4_letter = [
        "LOVE", "HOPE", "LIFE", "TIME", "HOME", "BOOK", "DOOR", "HAND",
        "BLUE", "GOLD", "FIRE", "MOON", "STAR", "TREE", "BIRD", "FISH",
        "CAKE", "MILK", "FOOD", "GAME", "PLAY", "WORK", "HELP", "GOOD"
    ]
    
    for i, word in enumerate(easy_4_letter):
        if i % 5 == 0:
            print()
        print(f"{word:6}", end="")
    print("\n")

def suggest_practice_words():
    """Suggest words for practice sessions"""
    print("\n" + "="*60)
    print("SUGGESTED PRACTICE PROGRESSION")
    print("="*60)
    
    print("\n1. Beginner (Start here!):")
    print("   💡 Simple letters, easy to write")
    print("   Words: HI → GO → ME → NO → OK → UP → WE → BY")
    
    print("\n2. Basic 3-letter words:")
    print("   💡 Common objects and animals")
    print("   Words: CAT → DOG → SUN → BOX → CUP → HAT → PEN → EYE")
    
    print("\n3. Action words:")
    print("   💡 Verbs - things you can do")
    print("   Words: RUN → SIT → EAT → SEE → WIN → TRY → FLY → DIG")
    
    print("\n4. 4-letter favorites:")
    print("   💡 Beautiful and meaningful words")
    print("   Words: LOVE → HOPE → LIFE → HOME → BOOK → MOON → STAR → BIRD")

def show_writing_tips():
    """Show tips for air writing"""
    print("\n" + "="*60)
    print("AIR-WRITING TIPS FOR SUCCESS")
    print("="*60)
    
    tips = [
        "✋ Join index + middle fingers to activate system",
        "✏️  Write letters clearly and deliberately", 
        "⏱️  Take your time - don't rush",
        "📏 Keep consistent size for all letters",
        "🔄 Practice the same word multiple times",
        "💡 Start with 2-letter words, then progress",
        "🎯 Focus on letter shapes, not speed",
        "📱 Ensure good lighting for hand tracking",
        "👀 Watch the real-time feedback",
        "🔄 Reset with 'R' key if needed"
    ]
    
    for tip in tips:
        print(f"  {tip}")

if __name__ == "__main__":
    show_easy_words()
    suggest_practice_words() 
    show_writing_tips()
    
    print("\n" + "="*60)
    print("Ready to start? Run: python word_recognition_system.py")
    print("="*60)