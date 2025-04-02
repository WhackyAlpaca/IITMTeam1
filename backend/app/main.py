# backend/app/main.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch
import os
import re
import csv
from urllib.parse import urlparse, parse_qs
from googleapiclient.discovery import build
import json
import time
from typing import List, Optional

# Initialize FastAPI
app = FastAPI(title="Multilingual Hate Speech Classifier API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure Google API
genai.configure(api_key="AIzaSyDtK8u-GXS5INUJAW38fr2-5ya1RsUwStg")
YOUTUBE_API_KEY = "AIzaSyD7P0g7cyFfdhLxBJ3RayolCsJt6EJfh54"

# Path to your CSV file
csv_file_path = "classified_hate_comments.csv"

# Define input/output models
class YouTubeURL(BaseModel):
    url: str

class ClassificationOutput(BaseModel):
    language: str
    classification: str
    
class Comment(BaseModel):
    text: str
    language: str
    classification: str

class ClassificationResults(BaseModel):
    video_id: str
    hate_comments: List[Comment]
    processing_complete: bool

# Dictionary mappings
dict_of_lang = {0: "Tamil", 1: "Kannada", 2: "Malayalam", 3: "English", 4: "Other"}
dict_of_classification = {0: "Not Hate", 1: "Hate"}

# Global variable to store processing status
processing_status = {}

# Extract video ID from YouTube URL
def extract_video_id(url):
    """Extracts the video ID from a YouTube URL."""
    parsed_url = urlparse(url)
    
    # Standard YouTube URL
    if parsed_url.netloc in ["www.youtube.com", "youtube.com"]:
        query_params = parse_qs(parsed_url.query)
        return query_params.get("v", [None])[0]
    
    # Shortened YouTube URL
    elif parsed_url.netloc in ["youtu.be"]:
        return parsed_url.path.lstrip("/")
    
    return None

# Get YouTube comments
def get_all_youtube_comments(video_id):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    comments = []
    next_page_token = None
    
    try:
        while True:
            response = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,  # Maximum allowed value
                textFormat="plainText",
                pageToken=next_page_token
            ).execute()
            
            comments.extend(
                item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                for item in response.get("items", [])
            )
            
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                break  # No more comments to fetch
                
        return comments
    except Exception as e:
        print(f"Error fetching comments: {e}")
        return []

# Detect language
def detect_language(text):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = f'''
    Classify the following text into one of these languages:
    Tamil (0), Kannada (1), Malayalam (2), English (3), Other (4).
    Text: "{text}"
    Provide only the language ID number, without any additional text. For example: "1"
    '''
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Extract the number (language ID) from the response using regex
        match = re.search(r'^\d+$', response_text)  # Match a string with only digits
        if match:
            return int(match.group(0))  # Return the number found
        else:
            return 4  # Default to "Other" if response format is unexpected
    except Exception as e:
        print(f"Error in language detection: {e}")
        return 4

# Classify English text
def english_hate_classifier(text):
    model = genai.GenerativeModel('models/gemini-2.0-flash')
    prompt = f'''
    Classify the following English text as offensive speech (1) or not (0).
    Please only respond with '0' for not offensive and '1' for offensive. No other text.
    Text: "{text}"
    Classification (0 or 1):
    '''
    
    try:
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Ensure the response is either '0' or '1'
        if response_text == '0' or response_text == '1':
            return int(response_text)  # Return as integer
        else:
            print(f"Unexpected response: {response_text}")
            return 0  # Default to "Not Hate" if response format is unexpected
    except Exception as e:
        print(f"Error in classification: {e}")
        return 0

# Initialize models
models = {}
tokenizers = {}

async def load_models():
    global models, tokenizers
    base_model_name = "google/muril-base-cased"
    
    # For cloud deployment, we'll use Gemini for all languages
    # In a production environment, you would upload and use your fine-tuned models
    print("Models initialized with Gemini fallback")

# Classify text
def classify_text(text):
    language_index = detect_language(text)
    language_detected = dict_of_lang.get(language_index, "Unknown")
    
    # For this deployment example, we'll use Gemini for all languages
    # In production, you would use your fine-tuned models
    if language_detected == "English":
        classification_index = english_hate_classifier(text)
    else:
        # For non-English languages, use Gemini
        model = genai.GenerativeModel('models/gemini-2.0-flash')
        prompt = f'''
        Classify the following text in {language_detected} as hate speech (1) or not (0).
        Please only respond with '0' for not hate speech and '1' for hate speech. No other text.
        Text: "{text}"
        Classification (0 or 1):
        '''
        
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            if response_text == '0' or response_text == '1':
                classification_index = int(response_text)
            else:
                classification_index = 0
        except Exception as e:
            print(f"Error in classification: {e}")
            classification_index = 0
    
    classification = dict_of_classification[classification_index]
    return language_detected, classification

# Write to CSV
def write_to_csv(comment, language, classification):
    # Check if the file exists or not, and write the header if it's the first entry
    file_exists = os.path.isfile(csv_file_path)
    
    with open(csv_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        if not file_exists:
            # Write header if file is new or empty
            writer.writerow(['Comment', 'Language', 'Classification'])
        
        # Write the classified comment
        writer.writerow([comment, language, classification])

# Process YouTube comments
async def process_youtube_comments(video_id: str, background_tasks: BackgroundTasks):
    # Initialize results
    if video_id not in processing_status:
        processing_status[video_id] = {
            "hate_comments": [],
            "processing_complete": False
        }
    
    # Get comments
    comments = get_all_youtube_comments(video_id)
    
    # Process each comment
    for comment in comments:
        language, classification = classify_text(comment)
        
        if classification == "Hate":
            processing_status[video_id]["hate_comments"].append({
                "text": comment,
                "language": language,
                "classification": classification
            })
            write_to_csv(comment, language, classification)
    
    # Mark processing as complete
    processing_status[video_id]["processing_complete"] = True

@app.on_event("startup")
async def startup_event():
    await load_models()

@app.get("/")
async def root():
    return {"message": "Multilingual Hate Speech Classifier API"}

@app.post("/analyze")
async def analyze_youtube_comments(youtube_url: YouTubeURL, background_tasks: BackgroundTasks):
    video_id = extract_video_id(youtube_url.url)
    
    if not video_id:
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # Start processing in the background
    background_tasks.add_task(process_youtube_comments, video_id, background_tasks)
    
    return {"video_id": video_id, "message": "Processing started"}

@app.get("/results/{video_id}")
async def get_results(video_id: str):
    if video_id not in processing_status:
        raise HTTPException(status_code=404, detail="Video ID not found")
    
    return {
        "video_id": video_id,
        "hate_comments": processing_status[video_id]["hate_comments"],
        "processing_complete": processing_status[video_id]["processing_complete"]
    }

@app.get("/download/{video_id}")
async def download_csv(video_id: str):
    if video_id not in processing_status or not processing_status[video_id]["processing_complete"]:
        raise HTTPException(status_code=404, detail="Results not ready or video ID not found")
    
    # In a real implementation, you would generate a CSV file for download
    # For simplicity, we'll just return the hate comments
    return processing_status[video_id]["hate_comments"]

# Run FastAPI
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
