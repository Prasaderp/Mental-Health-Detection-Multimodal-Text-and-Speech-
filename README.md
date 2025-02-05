<div align="center">

# Mental Health Analyzer

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Web_UI-Streamlit-FF4B4B)](https://streamlit.io)
[![AI](https://img.shields.io/badge/AI-Transformers-orange)](https://huggingface.co)

**Detect emotional patterns through text & speech analysis**

</div>


---

## 🖥️ Preview
![App Interface](https://github.com/Prasaderp/Mental-Health-Detection-Multimodal-Text-and-Speech-/blob/6757b6d02da421cdb56c63b60e0ef460cc32c0dd/Speech-Text-Analysis/Screenshot%20(10).png)  

---

## 🌟 Key Features
- **Dual Input Modes**
  -  Analyze text documents
  -  Process speech recordings
- **AI Insights**
  -  Emotion detection
  -  Stress/Anxiety indicators
  -  Pattern visualization
- **Clinical Support**
  -  Exportable reports
  -  Session history

---

## 🛠️ Quick Start

### 1. Pre requisites
```
# Linux
sudo apt install ffmpeg

# Windows (using Chocolatey)
choco install ffmpeg
```

### 2. Installation
```
git clone https://github.com/Prasaderp/Mental-Health-Detection-Multimodal-Text-and-Speech-.git
cd Speech-Text-Analysis
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

### 3. Launch App
```
streamlit run speech-text.py
```

---

## 🔄 Workflows

### Text Analysis
```
graph LR
    A[Input] --> B(Clean Text)
    B --> C(Detect Emotions)
    C --> D(Generate Report)
```

### Speech Analysis
```
graph LR
    A[Audio] --> B(Extract Features)
    B --> C(Transcribe Speech)
    C --> D(Analyze Patterns)
    D --> E(Visualize Insights)
```

---

## ⚠️ Important Note
This tool provides preliminary insights only - not medical diagnoses.  

---

<div align="center">
📧 Contact: itsprasadsomvanshi@email.com  
   [Report Issues](https://github.com/Prasaderp/Mental-Health-Detection-Multimodal-Text-and-Speech-/issues)  
</div>
