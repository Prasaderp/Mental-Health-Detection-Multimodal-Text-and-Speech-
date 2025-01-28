# app.py
import streamlit as st
import re
import nltk
import numpy as np
from nltk.corpus import stopwords
from transformers import pipeline

# ========================================
# NLTK 
# ========================================
st.set_page_config(
    page_title="Mental Health Analyst",
    page_icon="🧠",
    layout="centered"
)

@st.cache_resource
def initialize_nltk():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    return True

_ = initialize_nltk() 

# ========================================
# Cache Processing
# ========================================
@st.cache_data(max_entries=100, show_spinner=False)
def clean_text(text: str) -> str:
    """Cleans text with regex and stopword removal"""
    text = re.sub(r'[^\w\s]', '', text)  # Remove non-alphanumeric
    text = text.lower().strip()
    return ' '.join([word for word in text.split() 
                    if word not in stopwords.words('english')])

@st.cache_resource(show_spinner="Loading AI model...")
def load_emotion_model():
    """Loads and caches the HuggingFace pipeline"""
    return pipeline(
        task="text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        framework="pt",
        device_map="auto",
        return_all_scores=True
    )

# ========================================
# Streamlit UI
# ========================================

def main_interface():
    """Core application interface"""
    st.title("Mental Health Analysis")
    
    with st.sidebar:
        st.markdown("""
        **Instructions:**
        1. Write or upload text
        2. Get instant analysis
        3. Review insights
        """)
        st.markdown("---")
        st.caption("v1.0 | Prasad Somvanshi")

    # Input Section
    input_method = st.radio(
        "Choose input method:",
        ("✍️ Direct Input", "📁 File Upload"),
        horizontal=True
    )
    
    text = ""
    if input_method == "✍️ Direct Input":
        text = st.text_area("Share your thoughts:", height=150)
    else:
        uploaded_file = st.file_uploader("Upload text file", type=["txt"])
        if uploaded_file:
            text = uploaded_file.read().decode("utf-8")

    # Processing Pipeline
    if text:
        with st.spinner("Analyzing emotional patterns..."):
            cleaned_text = clean_text(text)
            model = load_emotion_model()
            predictions = model(cleaned_text, truncation=True, max_length=512)[0]

            # Results Display
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Text Overview")
                st.metric("Processed Length", f"{len(cleaned_text.split())} words")
                st.code(cleaned_text[:200] + "..." if len(cleaned_text) > 200 else cleaned_text)

            with col2:
                st.subheader("Emotional Analysis")
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
                                background:{color}10; border-left:4px solid {color};
                                box-shadow: 0 2px 4px {color}20;">
                        <div style="font-size:1.1em; color:{color};">
                            {label.title()} ({score:.1%})
                        </div>
                        <div style="font-size:0.9em; color:#666;">
                            {diagnosis}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")
        st.warning("""
        **Important Notice**  
        This tool provides preliminary insights only. It is not a substitute for:  
        - Licensed mental health professional evaluation  
        - Clinical diagnosis  
        - Emergency medical services  
        If in crisis, please contact your local emergency services.
        """)

if __name__ == "__main__":
    main_interface()