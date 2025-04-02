import streamlit as st
import requests
import pandas as pd

BACKEND_URL = "https://iitmteam1.onrender.com/"  # Replace with your actual backend URL

st.title("YouTube Comment Hate Speech Classifier")

youtube_url = st.text_input("Enter YouTube Video URL:")

if st.button("Classify Comments"):
    if youtube_url:
        with st.spinner("Analyzing comments..."):
            response = requests.post(f"{BACKEND_URL}/classify", json={"url": youtube_url})
            
            if response.status_code == 200:
                hate_comments = response.json()["hate_comments"]
                if hate_comments:
                    st.warning(f"Found {len(hate_comments)} hate comments")
                    df = pd.DataFrame(hate_comments)
                    st.dataframe(df)
                    
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="hate_comments.csv",
                        mime="text/csv",
                    )
                else:
                    st.success("No hate comments found in this video.")
            else:
                st.error("Error occurred while processing the request.")
    else:
        st.error("Please enter a valid YouTube URL.")
