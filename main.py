from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from peft import PeftModel

# Initialize the Flask app
app = Flask(__name__)

id2label = {0: 'bug report', 1: 'feature request', 2: 'rating', 3: 'user experience'}
label2id = {label: id for id, label in id2label.items()}

# Load distillbert model and tokenizer
distillbert_model_dir = "./distillbert_model"
distillbert_model_checkpoint = 'distilbert-base-uncased'
distillbert_tokenizer = AutoTokenizer.from_pretrained(distillbert_model_dir, add_prefix_space=True)
distillbert_adapters = AutoModelForSequenceClassification.from_pretrained(
    distillbert_model_checkpoint, num_labels=4, id2label=id2label, label2id=label2id)
distillbert_model = PeftModel.from_pretrained(distillbert_adapters, distillbert_model_dir)

# Load bert4re model and tokenizer
bert4re_model_dir = "./bert4re_model"
bert4re_model_checkpoint = 'thearod5/bert4re'
bert4re_tokenizer = AutoTokenizer.from_pretrained(bert4re_model_dir, add_prefix_space=True)
bert4re_adapters = AutoModelForSequenceClassification.from_pretrained(
    bert4re_model_checkpoint, num_labels=4, id2label=id2label, label2id=label2id)
bert4re_model = PeftModel.from_pretrained(bert4re_adapters, bert4re_model_dir)

# Load ALBERT model and tokenizer
albert_model_dir = "./albert_model"
albert_model_checkpoint = 'albert-base-v2'
albert_tokenizer = AutoTokenizer.from_pretrained(albert_model_dir, add_prefix_space=True)
albert_adapters = AutoModelForSequenceClassification.from_pretrained(
    albert_model_checkpoint, num_labels=4, id2label=id2label, label2id=label2id)
albert_model = PeftModel.from_pretrained(albert_adapters, albert_model_dir)

id2label = {0: 'bug report', 1: 'feature request', 2: 'rating', 3: 'user experience'}
label2id = {label: id for id, label in id2label.items()}

def preprocess_input(text, tokenizer):
    #tokenize and truncate text
    tokenizer.truncation_side = "left"
    tokenized_inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    return tokenized_inputs
def predict(text, model, tokenizer):
    model.eval()  # Set the model to evaluation mode
    inputs = preprocess_input(text, tokenizer)

    # Move inputs to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1)
        all_probabilities = {id2label[i]: probabilities[0][i].item() for i in range(len(probabilities[0]))}
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()

    return id2label[predicted_class], confidence, all_probabilities

# Route for making predictions with DistillBERT
@app.route('/predict_distillbert', methods=['POST'])
def distillbert_predict_route():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text']
    prediction = predict(text, distillbert_model, distillbert_tokenizer)
    return jsonify({"prediction": prediction})

# Route for making predictions using BERT4RE
@app.route('/predict_bert4re', methods=['POST'])
def bert4re_predict_route():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text']
    prediction = predict(text, bert4re_model, bert4re_tokenizer)
    return jsonify({"prediction": prediction})

# Route for making predictions using ALBERT
@app.route('/predict_albert', methods=['POST'])
def albert_predict_route():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text']
    prediction = predict(text, albert_model, albert_tokenizer)
    return jsonify({"prediction": prediction})

# Route for making predictions using all models
@app.route('/predict', methods=['POST'])
def predict_route():
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400
    text = data['text']
    distillbert_prediction = predict(text, distillbert_model, distillbert_tokenizer)
    bert4re_prediction = predict(text, bert4re_model, bert4re_tokenizer)
    albert_prediction = predict(text, albert_model, albert_tokenizer)
    return jsonify({"distillbert_prediction": distillbert_prediction,
                    "bert4re_prediction": bert4re_prediction,
                    "albert_prediction": albert_prediction})


# Route for serving the HTML page
@app.route('/')
def index():
    return render_template('index.html')


# Main entry point
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
