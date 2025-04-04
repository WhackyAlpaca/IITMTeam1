import streamlit as st
import requests
import io

# Set page title and configuration
st.set_page_config(page_title="YouTube Hate Speech Classifier", layout="centered")

# Add header and description
st.title("YouTube Hate Speech Classifier")
st.write("Enter a YouTube URL to analyze comments for hate speech")

# Create input form
youtube_url = st.text_input("YouTube URL", placeholder="Enter YouTube URL (e.g., https://www.youtube.com/watch?v=...)")

# Create analyze button
if st.button("Analyze Comments"):
    if not youtube_url:
        st.error("Please enter a YouTube URL")
    else:
        # Show loading message
        with st.spinner("Processing... This may take a few minutes depending on the number of comments."):
            try:
                # Step 1: Process the comments
                process_response = requests.post(
                    "https://hate-speech-classifier-656879117583.asia-south1.run.app/process",
                    json={"video_url": youtube_url}
                )
                
                if process_response.status_code != 200:
                    st.error(f"Error processing comments: {process_response.status_code} - {process_response.text}")
                else:
                    process_data = process_response.json()
                    hate_count = process_data.get("hate_comments_found", 0)
                    
                    # Step 2: Download the CSV file
                    download_response = requests.get(
                        "https://hate-speech-classifier-656879117583.asia-south1.run.app/download"
                    )
                    
                    if download_response.status_code == 200:
                        # Show success message
                        st.success(f"Analysis complete! Found {hate_count} hate comments. Click below to download the results.")
                        
                        # Create download button for CSV
                        st.download_button(
                            label="Download CSV Results",
                            data=io.BytesIO(download_response.content),
                            file_name="hate_speech_analysis.csv",
                            mime="text/csv"
                        )
                    else:
                        st.error(f"Error downloading CSV: {download_response.status_code} - {download_response.text}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
