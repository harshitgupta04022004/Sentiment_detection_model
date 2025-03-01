import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from transformers import T5Tokenizer, TFT5ForConditionalGeneration
import pandas as pd
import numpy as np
import seaborn as sns
import sklearn as sk
import matplotlib.pyplot as plt

# Enable dynamic memory allocation on GPU(s)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

def shift_right(input_ids, pad_token_id, decoder_start_token_id):
    """
    Shifts input ids to the right by one position.
    The first token becomes the decoder_start_token_id.
    """
    batch_size = tf.shape(input_ids)[0]
    start_tokens = tf.fill([batch_size, 1], decoder_start_token_id)
    shifted_input_ids = tf.concat([start_tokens, input_ids[:, :-1]], axis=1)
    return shifted_input_ids

class MultiTaskT5(tf.keras.Model):
    def __init__(self, model_name, num_emotion_labels, num_sentiment_labels):
        super(MultiTaskT5, self).__init__()
        self.t5 = TFT5ForConditionalGeneration.from_pretrained(model_name)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.emotion_classifier = tf.keras.layers.Dense(
            num_emotion_labels, activation='softmax', name="emotion_classifier"
        )
        self.sentiment_classifier = tf.keras.layers.Dense(
            num_sentiment_labels, activation='softmax', name="sentiment_classifier"
        )

    def call(self, inputs, task, training=False, labels=None):
        if task in ['emotion', 'sentiment']:
            encoder_outputs = self.t5.encoder(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                training=training
            )
            pooled_output = tf.reduce_mean(encoder_outputs.last_hidden_state, axis=1)
            pooled_output = self.dropout(pooled_output, training=training)
            if task == 'emotion':
                return self.emotion_classifier(pooled_output)
            else:
                return self.sentiment_classifier(pooled_output)
        elif task == 'summary':
            if labels is None:
                outputs = self.t5(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    training=training
                )
                return outputs.logits
            else:
                pad_token_id = self.t5.config.pad_token_id
                decoder_start_token_id = self.t5.config.decoder_start_token_id
                decoder_input_ids = shift_right(labels, pad_token_id, decoder_start_token_id)
                outputs = self.t5(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    decoder_input_ids=decoder_input_ids,
                    training=training
                )
                return outputs.logits
        else:
            raise ValueError("Unsupported task type. Use 'emotion', 'sentiment', or 'summary'.")

model_name = "t5-small"
num_emotion_labels = 6
num_sentiment_labels = 4  # Updated to 4 output labels for sentiment
model = MultiTaskT5(model_name, num_emotion_labels, num_sentiment_labels)
tokenizer = T5Tokenizer.from_pretrained(model_name)
        
# Load weights saved in H5 format.
model.load_weights('./results/multi_task_T5_tf_weights')
# Load the tokenizer from the saved directory.
tokenizer = T5Tokenizer.from_pretrained("./results/multi_task_T5_tf/")

sentiment_mapping = {3: 'positive', 2: 'neutral', 1: 'negative', 0: 'irrelevant'}
emotion_mapping = {
    0: "anger", 
    1: "fear", 
    2: "joy", 
    3: "love", 
    4: "sadness", 
    5: "surprise"
}

def decode_sentiment(label_code):
    return sentiment_mapping.get(label_code, "Unknown")

def decode_emotion(label_code):
    return emotion_mapping.get(label_code, "Unknown")

def predict(text, task):
    if task in ['emotion', 'sentiment']:
        inputs = tokenizer(text, return_tensors="tf", max_length=128, truncation=True, padding="max_length")
        logits = model(inputs, task=task, training=False)
        pred_idx = int(tf.argmax(logits, axis=1).numpy()[0])
        if task == "sentiment":
            return decode_sentiment(pred_idx)
        elif task == "emotion":
            return decode_emotion(pred_idx)
    elif task == 'summary':
        input_text = "summarize: " + text
        inputs = tokenizer(input_text, return_tensors="tf", max_length=512, truncation=True, padding="max_length")
        generated_ids = model.t5.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )
        summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return summary
    else:
        raise ValueError("Unsupported task type. Use 'emotion', 'sentiment', or 'summary'.")

@app.route('/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    text = data.get('text', '')
    task = data.get('task', '')
    try:
        result = predict(text, task)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    text = request.form.get('text', '')
    task = request.form.get('task', '')
    try:
        result = predict(text, task)
    except Exception as e:
        result = str(e)
    return render_template('result.html', task=task, text=text, result=result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
