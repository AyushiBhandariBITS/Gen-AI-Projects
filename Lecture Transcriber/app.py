import streamlit as st
import tempfile
import os
from main import transcript_audio
# App title and description
st.title("Audio Transcription App")
st.write("Upload an audio file below to transcribe.")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "m4a", "ogg"])


# Run when file is uploaded
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    st.audio(tmp_path, format='audio/' + uploaded_file.type.split("/")[-1])
    
    if st.button("Transcribe"):
        result = transcript_audio(tmp_path)
        st.subheader("Transcription")
        st.text(result)
