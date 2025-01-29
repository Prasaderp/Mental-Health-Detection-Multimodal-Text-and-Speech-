import streamlit as st
import numpy as np
import librosa
import whisper
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
from transformers import pipeline

# ========================================
# Load Models
# ========================================
@st.cache_resource
def load_models():
    return {
        'whisper': whisper.load_model("base"),
        'emotion_model': pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base"
        )
    }

@st.cache_data
def process_audio(audio_bytes):
    # Convert bytes to audio array
    with BytesIO(audio_bytes) as f:
        audio, sr = librosa.load(f, sr=16000)
    
    # Extract audio features
    features = {
        'pitch': librosa.yin(audio, fmin=50, fmax=2000).mean(),
        'mfcc': librosa.feature.mfcc(y=audio, sr=sr).mean(axis=1)[:5],
        'energy': librosa.feature.rms(y=audio).mean()
    }
    
    # Transcribe audio
    models = load_models()
    result = models['whisper'].transcribe(audio)
    
    return features, result["text"]

# ========================================
# Streamlit UI
# ========================================
def main():
    st.set_page_config(page_title="Speech Analysis", layout="centered")
    
    st.title("Speech Analysis")
    
    # Sidebar Instructions
    with st.sidebar:
        st.markdown("""
        **Instructions:**
        1. Record or upload an audio file.
        2. Wait for processing to complete.
        3. Review speech transcription and insights.
        """)

    # Input Section
    input_col1, input_col2 = st.columns([2, 1])
    with input_col1:
        audio_bytes = audio_recorder(text="Click to Record")
    with input_col2:
        uploaded_file = st.file_uploader("Or upload an audio file", type=["wav", "mp3"])
        if uploaded_file:
            audio_bytes = uploaded_file.read()
    
    # Processing and Results
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        with st.spinner("Processing speech analysis..."):
            features, transcription = process_audio(audio_bytes)
            models = load_models()
            emotions = models['emotion_model'](transcription)[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Speech Features")
            st.metric("Average Pitch", f"{features['pitch']:.1f} Hz")
            st.metric("Energy Level", f"{features['energy']:.2f} dB")

        with col2:
            st.subheader("Transcription")
            st.write(transcription)
            
            st.subheader("Emotion Analysis")
            label, score = emotions['label'], emotions['score']
            st.write(f"Detected Emotion: **{label.title()}** ({score:.1%})")
    
if __name__ == "__main__":
    main()
