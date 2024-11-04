from flask import Flask, render_template, request, jsonify
from transformers import BertForSequenceClassification, BertTokenizer
import torch, requests
import os
from googleapiclient.discovery import build
import re
from langdetect import detect
import sounddevice as sd
import numpy as np
import speech_recognition as sr
import wave

app = Flask(__name__)

tokenizer = BertTokenizer.from_pretrained(r'C:\Users\jithu\OneDrive - SSN-Institute\College\SIH\SIH-web-1.0.2\SIH-web-1.0.2\sih2023int\sih2023int\model - 2\bert_tokenizer')

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

model.load_state_dict(torch.load(r'C:\Users\jithu\OneDrive - SSN-Institute\College\SIH\SIH-web-1.0.2\SIH-web-1.0.2\sih2023int\sih2023int\model - 2\bert_model.pth'))

model.eval()

YOUTUBE_API_KEY = 'API_KEY'
CUSTOM_SEARCH_API_KEY = 'API_KEY'

def youtube_video_search(api_key, query, num_results=5):
    base_url = "https://www.googleapis.com/youtube/v3/search"

    params = {
        'part': 'snippet',
        'q': query,
        'type': 'video',
        'maxResults': num_results,
        'key': api_key,
    }

    response = requests.get(base_url, params=params)
    data = response.json()

    if 'items' in data:
        return [f"https://www.youtube.com/watch?v={item['id']['videoId']}" for item in data['items']]
    else:
        return None
def extract_video_id(video_link):
    match = re.search(r'(?<=v=)[a-zA-Z0-9_-]+', video_link)
    return match.group(0) if match else None

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False
def get_youtube_comments(api_key, video_links):
    comments_list = []

    for video_link in video_links:
        video_id = extract_video_id(video_link)

        if video_id:
            youtube = build('youtube', 'v3', developerKey=api_key)
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                maxResults=100  
            )
            response = request.execute()

            for item in response.get('items', []):
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                if is_english(comment_text):
                    comments_list.append(comment_text)

    return comments_list
def youtube_data_collect(query):
    api_key = YOUTUBE_API_KEY

    num_results = 5

    video_results = youtube_video_search(api_key, query, num_results)
    comments = get_youtube_comments(YOUTUBE_API_KEY, video_results)
    return comments
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/Market research analysis')
def page1():
    return render_template('Page1.html')

@app.route('/Economic disparity solution finder')
def page2():
    return render_template('Page2.html')

@app.route('/Market sentiment contrast analysis')
def page3():
    return render_template('Page3.html')

@app.route('/Influencer sentiment analysis')
def page4():
    return render_template('Page4.html')

@app.route('/Time-tuned sentiment analysis')
def page5():
    return render_template('Page5.html')

@app.route('/Policy communication')
def page6():
    return render_template('Page6.html')

@app.route('/Login')
def page7():
    return render_template('login.html')

@app.route('/Feedback')
def page8():
    return render_template('feedback.html')

@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    if request.method == 'POST':
        text_to_analyze = request.json.get('productName', '')
        if text_to_analyze:
            comments = youtube_data_collect(text_to_analyze)
            probabilities = predict_sentiment_helper(comments)
            result = {
                'sentiment_probabilities': dict(zip(["Negative", "Neutral", "Positive"], probabilities))
            }
            return jsonify(result)
        else:
            return jsonify({'error': 'Empty productName received'})
    else:
        return jsonify({'error': 'Method Not Allowed'})

@app.route('/predict_sentiment_contrast', methods=['POST'])
def predict_sentiment_contrast():
    if request.method == 'POST':
        product_name_1 = request.json.get('productName1', '')
        product_name_2 = request.json.get('productName2', '')

        if product_name_1 and product_name_2:
            probabilities_1 = predict_sentiment_helper(product_name_1)
            probabilities_2 = predict_sentiment_helper(product_name_2)

            result_product_1 = {
                'productName': product_name_1,
                'sentiment_probabilities': probabilities_1
            }

            result_product_2 = {
                'productName': product_name_2,
                'sentiment_probabilities': probabilities_2
            }

            return jsonify({
                'product1': result_product_1,
                'product2': result_product_2
            })
        else:
            return jsonify({'error': 'Empty product names received'})
    else:
        return jsonify({'error': 'Method Not Allowed'})

def predict_sentiment_helper(comments):
    predictions = []
    for comment in comments:
        tokens = tokenizer.encode(comment, max_length=128, truncation=True)

        input_ids = torch.tensor(tokens).unsqueeze(0) 

        with torch.no_grad():
            outputs = model(input_ids)

        prediction = torch.argmax(outputs.logits).item()
        predictions.append(prediction)
    positive = predictions.count(1)
    negative = predictions.count(0)
    neutral = predictions.count(2)
    probabilities = [negative,neutral,positive]
    return probabilities
@app.route('/voice', methods=['POST'])
def voice():

    def record_and_save(filename, duration=5, samplerate=44100):
        print("Recording...")

        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()

        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())

        print(f"Recording saved as {filename}")

    record_and_save("output.wav", duration=10)

    def convert_wav_to_text(wav_file_path):
        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(wav_file_path) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.record(source)

                text = recognizer.recognize_sphinx(audio)
                print("Text from audio: {}".format(text))
                return "Text from audio: {}".format(text)
        except sr.UnknownValueError:
           return "Speech Recognition could not understand audio"
        except sr.RequestError as e:
            return "Error"
    try:
        
        return convert_wav_to_text(r"C:\Users\jithu\OneDrive - SSN-Institute\College\SIH\SIH-web-1.0.2\SIH-web-1.0.2\sih2023int\sih2023int\output.wav")
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
