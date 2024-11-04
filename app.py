# & C:/Python312/python.exe "c:/Users/jithu/OneDrive - SSN-Institute/College/SIH/SIH-web-1.0.2/SIH-web-1.0.2/sih2023int/sih2023int/app.py"
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

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained(r'C:\Users\jithu\OneDrive - SSN-Institute\College\SIH\SIH-web-1.0.2\SIH-web-1.0.2\sih2023int\sih2023int\model - 2\bert_tokenizer')

# Load the model architecture
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load the trained weights
model.load_state_dict(torch.load(r'C:\Users\jithu\OneDrive - SSN-Institute\College\SIH\SIH-web-1.0.2\SIH-web-1.0.2\sih2023int\sih2023int\model - 2\bert_model.pth'))

# Set the model to evaluation mode
model.eval()

YOUTUBE_API_KEY = 'AIzaSyBeVwWYdrruAf-003iQb1iP0Lu0XO8-HYg'
CUSTOM_SEARCH_API_KEY = 'AIzaSyBeVwWYdrruAf-003iQb1iP0Lu0XO8-HYg'

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
                maxResults=100  # Adjust as needed
            )
            response = request.execute()

            for item in response.get('items', []):
                comment_text = item['snippet']['topLevelComment']['snippet']['textDisplay']
                if is_english(comment_text):
                    comments_list.append(comment_text)

    return comments_list
def youtube_data_collect(query):
    # Replace 'YOUR_API_KEY' with your actual YouTube API key
    api_key = YOUTUBE_API_KEY

    # Number of results to retrieve (adjust as needed)
    num_results = 5

    # Perform the video search
    video_results = youtube_video_search(api_key, query, num_results)
    '''
    if video_results:
        print(f"Top {num_results} video links about {query}:")
        for i, link in enumerate(video_results, start=1):
            print(f"{i}. {link}")
    else:
        print("No video results found.")'''
    comments = get_youtube_comments(YOUTUBE_API_KEY, video_results)
    '''
    # Print the relevant comments
    for i, comment in enumerate(comments, start=1):
        print(f"{i}. {comment}")'''
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
        # Retrieve product names from the form
        product_name_1 = request.json.get('productName1', '')
        product_name_2 = request.json.get('productName2', '')

        if product_name_1 and product_name_2:
            # Predict sentiments for both products
            probabilities_1 = predict_sentiment_helper(product_name_1)
            probabilities_2 = predict_sentiment_helper(product_name_2)

            # Create result dictionary for Product 1
            result_product_1 = {
                'productName': product_name_1,
                'sentiment_probabilities': probabilities_1
            }

            # Create result dictionary for Product 2
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
        # Tokenize the comment
        tokens = tokenizer.encode(comment, max_length=128, truncation=True)

        # Convert tokens to PyTorch tensor
        input_ids = torch.tensor(tokens).unsqueeze(0)  # Add batch dimension

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids)

        # Get predicted sentiment (assuming binary classification)
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
        # Record audio
        audio_data = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype=np.int16)
        sd.wait()

        # Save as WAV file
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit audio
            wf.setframerate(samplerate)
            wf.writeframes(audio_data.tobytes())

        print(f"Recording saved as {filename}")

    # Example usage: Record for 10 seconds and save to "output.wav"
    record_and_save("output.wav", duration=10)

    def convert_wav_to_text(wav_file_path):
        recognizer = sr.Recognizer()

        try:
            with sr.AudioFile(wav_file_path) as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.record(source)

                # Use the PocketSphinx recognizer
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
        # Handle the exception and return an error response
        return jsonify({'error': str(e)}), 500



if __name__ == '__main__':
    app.run(debug=True)
