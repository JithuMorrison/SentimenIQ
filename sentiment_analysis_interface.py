from transformers import BertForSequenceClassification, BertTokenizer
import torch
import gradio as gr

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the model architecture
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)  # Make sure to set the correct number of labels
model.load_state_dict(torch.load(r"C:\Users\tjtha\Downloads\bert_model.pth"))  # Replace with the path to your PyTorch model file
model.eval()

def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt")
    
    # Forward pass, get logits
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits

    # Apply softmax to get predicted class probabilities
    probabilities = torch.softmax(logits, dim=1).squeeze().tolist()
    
    # Get predicted sentiment label (positive, negative, neutral)
    predicted_label = torch.argmax(logits, dim=1).item()

    return {"predicted_label": predicted_label, "probabilities": dict(zip(["Negative", "Neutral", "Positive"], probabilities))}

# Interface using Gradio
iface = gr.Interface(
    fn=predict_sentiment, 
    inputs="text",
    outputs=gr.outputs.Classification(num_top_classes=3),
    live=True
)

iface.launch()
