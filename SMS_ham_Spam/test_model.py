"""
Model Testing Demo Script

This script demonstrates how to load the trained baseline model
and make predictions on new SMS messages.

Usage:
    python test_model.py
"""

import joblib
import numpy as np


def load_model(model_path='models/baseline/baseline_model.pkl'):
    """Load the trained baseline model."""
    print(f"Loading model from '{model_path}'...")
    model = joblib.load(model_path)
    print("✓ Model loaded successfully!\n")
    return model


def predict_spam(model, messages):
    """
    Predict whether messages are spam or ham.
    
    Args:
        model: Trained pipeline
        messages (list): List of SMS messages
        
    Returns:
        predictions and probabilities
    """
    # Get predictions (0=ham, 1=spam)
    predictions = model.predict(messages)
    
    # Get probability of being spam
    probabilities = model.predict_proba(messages)[:, 1]
    
    return predictions, probabilities


def demo():
    """Run a demo with sample messages."""
    
    print("="*70)
    print("SMS SPAM CLASSIFIER - DEMO")
    print("="*70 + "\n")
    
    # Load the trained model
    model = load_model()
    
    # Test messages
    test_messages = [
        "Hi! How are you doing today?",
        "Congratulations! You've won a $1000 gift card. Call now!",
        "Can we meet for lunch tomorrow at 12pm?",
        "FREE entry in 2 a wkly comp to win FA Cup final tkts. Txt WIN to 87077",
        "I'll pick you up at 7pm. See you then!",
        "URGENT! Your account has been compromised. Click here immediately!",
        "Thanks for helping me yesterday. Really appreciate it.",
        "Win cash prizes! Text CASH to 12345 now!",
    ]
    
    print("Testing the model on sample messages:\n")
    print("-"*70)
    
    # Make predictions
    predictions, probabilities = predict_spam(model, test_messages)
    
    # Display results
    for i, (message, pred, prob) in enumerate(zip(test_messages, predictions, probabilities), 1):
        label = "🚫 SPAM" if pred == 1 else "✅ HAM"
        confidence = prob if pred == 1 else (1 - prob)
        
        print(f"\nMessage {i}:")
        print(f"  Text: \"{message}\"")
        print(f"  Prediction: {label}")
        print(f"  Confidence: {confidence*100:.2f}%")
    
    print("\n" + "="*70)
    
    # Interactive mode
    print("\n🔮 Try your own messages! (Type 'quit' to exit)\n")
    
    while True:
        user_message = input("Enter SMS message: ").strip()
        
        if user_message.lower() in ['quit', 'exit', 'q']:
            print("\n👋 Goodbye!")
            break
        
        if not user_message:
            continue
        
        pred, prob = predict_spam(model, [user_message])
        pred_label = "🚫 SPAM" if pred[0] == 1 else "✅ HAM"
        confidence = prob[0] if pred[0] == 1 else (1 - prob[0])
        
        print(f"  → {pred_label} (Confidence: {confidence*100:.2f}%)\n")


if __name__ == "__main__":
    demo()

