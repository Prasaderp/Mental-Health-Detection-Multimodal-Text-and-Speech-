# unified_app.py
import streamlit as st
import numpy as np
import librosa
import whisper
import re
import nltk
import time
from audio_recorder_streamlit import audio_recorder
from io import BytesIO
from transformers import pipeline
from nltk.corpus import stopwords

# ========================================
# Core Functions (Cached)
# ========================================
@st.cache_resource
def initialize_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    return True

@st.cache_resource
def load_models():
    return {
        'whisper': whisper.load_model("base"),
        'emotion_model': pipeline(
            "text-classification",
            model="j-hartmann/emotion-english-distilroberta-base",
            return_all_scores=True
        )
    }

@st.cache_data
def clean_text(text: str) -> str:
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    return ' '.join([word for word in text.split() 
                    if word not in stopwords.words('english')])

@st.cache_data
def process_audio(audio_bytes):
    with BytesIO(audio_bytes) as f:
        audio, sr = librosa.load(f, sr=16000)
    return {
        'pitch': librosa.yin(audio, fmin=50, fmax=2000).mean(),
        'mfcc': librosa.feature.mfcc(y=audio, sr=sr).mean(axis=1)[:5],
        'energy': librosa.feature.rms(y=audio).mean()
    }, audio

# ========================================
# Streamlit Interface
# ========================================
def main():
    st.set_page_config(
        page_title="Multimodal Mental Health Analysis",
        page_icon="🧠",
        layout="centered"
    )
    
    # Initialize NLTK
    _ = initialize_nltk()
    
    # Sidebar Configuration
    with st.sidebar:
        st.subheader("Analysis Mode")
        analysis_mode = st.radio(
            "Select input type:",
            ("📝 Text", "🎙️ Speech"),
            index=0
        )
        
        st.markdown("---")
        st.subheader("Instructions")
        if analysis_mode == "📝 Text":
            st.markdown("""
            1. Enter text directly or upload .txt file
            2. Wait for detailed analysis
            3. Review detailed insights
            """)
        else:
            st.markdown("""
            1. Record audio or upload WAV/MP3
            2. Allow 10-15 seconds for processing
            3. Review vocal metrics & insights
            """)
        st.markdown("---")
        st.caption("**Developed by -** Prasad Somvanshi")

    st.title("Multimodal Mental Health Analysis")

    # Text Analysis Mode
    if analysis_mode == "📝 Text":
        col1, col2 = st.columns([1.5, 2])
        
        with col1:
            input_method = st.radio(
                "Input method:",
                ("Direct Entry", "File Upload"),
                horizontal=True
            )
            
            text = ""
            if input_method == "Direct Entry":
                text = st.text_area("Enter text:", height=200)
            else:
                uploaded_file = st.file_uploader("Upload text file", type=["txt"])
                if uploaded_file:
                    text = uploaded_file.read().decode("utf-8")

        if text:
            with st.status("📊 Analyzing text content...", expanded=True) as status:
                st.write("🧹 Cleaning text input...")
                cleaned_text = clean_text(text)
                time.sleep(1.5)

                st.write("🤖 Running emotion classification...")
                model = load_models()['emotion_model']
                predictions = model(cleaned_text, truncation=True, max_length=512)[0]
                time.sleep(2)

                status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

            with col2:
                st.subheader("Emotional Profile")
                health_map = {
                    "sadness": ("Potential Depression", "#e74c3c"),
                    "fear": ("Anxiety Indicators", "#3498db"),
                    "anger": ("Stress Markers", "#f1c40f"),
                    "neutral": ("Baseline State", "#2ecc71")
                }
                
                for pred in predictions:
                    label = pred['label']
                    score = pred['score']
                    diagnosis, color = health_map.get(label, ("Other", "#95a5a6"))
                    
                    st.markdown(f"""
                    <div style="padding:12px; margin:8px 0; border-radius:8px; 
                                background:{color}10; border-left:4px solid {color}">
                        <div style="font-size:1.1em; color:{color};">
                            {label.title()} ({score:.1%})
                        </div>
                        <div style="font-size:0.9em; color:#666;">
                            {diagnosis}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("---")
                st.subheader("Processed Text")
                st.write(cleaned_text[:500] + "..." if len(cleaned_text) > 500 else cleaned_text)

    # Speech Analysis Mode
    else:
        col1, col2 = st.columns([1.7, 2])
        
        with col1:
            audio_bytes = audio_recorder(
                text="Click to record",
                recording_color="#e74c3c",
                neutral_color="#2ecc71",
                pause_threshold=2.0
            )
            
            uploaded_file = st.file_uploader("Or upload audio file", type=["wav", "mp3"])
            if uploaded_file:
                audio_bytes = uploaded_file.read()

        if audio_bytes:
            with col2:
                st.audio(audio_bytes, format="audio/wav")
                
                with st.status("🔍 Analyzing audio content...", expanded=True) as status:
                    st.write("📡 Extracting acoustic features...")
                    audio_features, raw_audio = process_audio(audio_bytes)
                    time.sleep(2)

                    st.write("🎙️ Transcribing speech content...")
                    models = load_models()
                    result = models['whisper'].transcribe(raw_audio)
                    time.sleep(2)

                    st.write("🧠 Analyzing emotional context...")
                    emotions = models['emotion_model'](result["text"])
                    time.sleep(2)

                    status.update(label="✅ Analysis Complete!", state="complete", expanded=False)

                st.subheader("Analysis Results")
                tab1, tab2 = st.tabs(["Vocal Metrics", "Emotion Profile"])
                
                with tab1:
                    st.metric("Average Pitch", f"{audio_features['pitch']:.1f} Hz")
                    st.metric("MFCC Spread", f"{np.std(audio_features['mfcc']):.2f}")
                    st.metric("Energy Level", f"{audio_features['energy']:.2f} dB")

                    st.markdown((result["text"]))
                
                with tab2:
                    health_map = {
                        "sadness": ("Depression Risk", "#e74c3c"),
                        "fear": ("Anxiety Signs", "#3498db"),
                        "anger": ("Stress Level", "#f1c40f"),
                        "neutral": ("Baseline State", "#2ecc71")
                    }
                    
                    for emotion in emotions[0]:
                        label = emotion['label']
                        score = emotion['score']
                        diagnosis, color = health_map.get(label, ("Other", "#95a5a6"))
                        
                        st.markdown(f"""
                        <div style="padding:12px; margin:8px 0; border-radius:8px; 
                                    background:{color}10; border-left:4px solid {color}">
                            <div style="font-size:1.1em; color:{color};">
                                {label.title()} ({score:.1%})
                            </div>
                            <div style="font-size:0.9em; color:#666;">
                                {diagnosis}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # Universal Disclaimer
    st.markdown("---")
    st.warning("""
    **⚠️ Important Notice** 
                
    This tool offers general insights, not a medical diagnosis. If you're struggling or need support, please consult a qualified mental health professional.
    """)

if __name__ == "__main__":
    main()