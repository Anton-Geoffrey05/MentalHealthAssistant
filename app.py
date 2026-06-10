import gradio as gr
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def respond(message, history):

    crisis_keywords = [
        "suicide",
        "kill myself",
        "end my life",
        "self harm"
    ]

    if any(word in message.lower() for word in crisis_keywords):
        return (
            "⚠️ Please seek immediate support from a trusted person or mental health professional."
        )

    result = classifier(message)

    emotion = result[0]["label"]
    confidence = result[0]["score"]

    responses = {
        "joy": "😊 That's great to hear! Keep enjoying the positive moments.",
        "sadness": "💙 I'm sorry you're feeling sad. Talking to someone you trust may help.",
        "anger": "😌 Take a deep breath and give yourself some time.",
        "fear": "🌱 It's okay to feel worried. Focus on one step at a time.",
        "surprise": "😮 That sounds unexpected. How are you processing it?",
        "love": "❤️ That's a wonderful feeling to experience."
    }

    reply = responses.get(
        emotion,
        "🤝 Thank you for sharing your feelings."
    )

    return (
        f"Detected Emotion: {emotion} "
        f"({confidence:.2%})\n\n"
        f"{reply}"
    )

demo = gr.ChatInterface(
    fn=respond,
    title="🧠 Mental Health Assistant",
    description="AI-powered emotional support chatbot"
)

demo.launch()