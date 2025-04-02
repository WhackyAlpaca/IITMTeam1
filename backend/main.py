from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from peft import PeftModel
import torch
import os
import csv
from googleapiclient.discovery import build
from urllib.parse import urlparse, parse_qs

app = FastAPI()

# Configure Google API
genai.configure(api_key="YOUR_GOOGLE_API_KEY")
YOUTUBE_API_KEY = "YOUR_YOUTUBE_API_KEY"

# Load models
base_model_name = "google/muril-base-cased"
models = {}
tokenizers = {}

for lang in ["tamil", "kannada", "malayalam"]:
    model = AutoModelForSequenceClassification.from_pretrained(base_model_name)
    adapter_path = f"./models/{lang}-comment-classifier"
    model = PeftModel.from_pretrained(model, adapter_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    models[lang] = model
    tokenizers[lang] = tokenizer

# Helper functions
def detect_language(text):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = f'''
    Classify the following text into one of these languages:
    Tamil (0), Kannada (1), Malayalam (2), English (3), Other (4).
    Text: "{text}"
    Provide only the language ID number, without any additional text. For example: "1"
    '''
    response = model.generate_content(prompt)
    return int(response.text.strip())

def classify_text(text, lang):
    classifier = pipeline("text-classification", model=models[lang], tokenizer=tokenizers[lang])
    result = classifier(text)
    return "Hate" if int(result[0]["label"].split("_")[1]) == 1 else "Not Hate"

def english_hate_classifier(text):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = f'''
    Classify the following English text as offensive speech (1) or not (0).
    Please only respond with '0' for not offensive and '1' for offensive. No other text.
    Text: "{text}"
    Classification (0 or 1):
    '''
    response = model.generate_content(prompt)
    return "Hate" if int(response.text.strip()) == 1 else "Not Hate"

def extract_video_id(url):
    parsed_url = urlparse(url)
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]
    elif parsed_url.netloc == "youtu.be":
        return parsed_url.path.lstrip("/")
    return None

def get_youtube_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None
    
    while True:
        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            pageToken=next_page_token
        ).execute()
        
        comments.extend([
            item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            for item in response.get("items", [])
        ])
        
        next_page_token = response.get("nextPageToken")
        if not next_page_token:
            break
    
    return comments

class YouTubeURL(BaseModel):
    url: str

@app.post("/classify")
async def classify_comments(youtube_url: YouTubeURL):
    video_id = extract_video_id(youtube_url.url)
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    comments = get_youtube_comments(video_id)
    hate_comments = []
    
    for comment in comments:
        lang_id = detect_language(comment)
        if lang_id == 3:  # English
            classification = english_hate_classifier(comment)
        elif lang_id in [0, 1, 2]:  # Tamil, Kannada, Malayalam
            lang = ["tamil", "kannada", "malayalam"][lang_id]
            classification = classify_text(comment, lang)
        else:
            continue
        
        if classification == "Hate":
            hate_comments.append({"comment": comment, "language": ["Tamil", "Kannada", "Malayalam", "English"][lang_id]})
    
    return {"hate_comments": hate_comments}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
