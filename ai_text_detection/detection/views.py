from django.shortcuts import render
from django.http import JsonResponse
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import json

# Load the custom trained BERT model from the 'bert_model' folder
model_path = "model/bert_model/"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)

# Make sure the model is in evaluation mode
model.eval()


def index(request):
    return render(request, "detection/index.html")


def detect(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            input_text = data.get("input_text")
        except json.JSONDecodeError:
            return JsonResponse({"error": "Invalid input format"}, status=400)

        if not input_text:
            return JsonResponse({"error": "No text provided"}, status=400)

        # Tokenize and encode input text
        encoded_input = tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=512,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"]

        with torch.no_grad():
            # Perform the prediction
            outputs = model(input_ids, attention_mask=attention_mask)

        # Extract the prediction (logits)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        predicted_label = torch.argmax(probabilities, dim=1).item()
        confidence_score = probabilities[0][predicted_label].item()

        # Format the output
        if predicted_label == 1:
            prediction = "AI-generated"
        else:
            prediction = "Human-written"

        return JsonResponse(
            {"prediction": prediction, "confidence": round(confidence_score * 100, 2)}
        )

    return JsonResponse({"error": "Invalid request method"}, status=405)
