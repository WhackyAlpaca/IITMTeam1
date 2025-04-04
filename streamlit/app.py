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
                # Make API request to your FastAPI backend
                # Change 'url' to 'video_url' to match your API's expected parameter
                response = requests.post(
                    "https://hate-speech-classifier-656879117583.asia-south1.run.app/process",
                    json={"video_url": youtube_url}
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    # Show success message
                    st.success("Analysis complete! Click below to download the results.")
                    
                    # Create download button for CSV
                    st.download_button(
                        label="Download CSV Results",
                        data=io.BytesIO(response.content),
                        file_name="hate_speech_analysis.csv",
                        mime="text/csv"
                    )
                else:
                    st.error(f"Error: {response.status_code} - {response.text}")
            
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
