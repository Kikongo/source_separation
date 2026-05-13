import streamlit as st
import requests
import json
import tempfile
import os
from pathlib import Path

# ========== CONFIGURATION ==========
API_URL = os.getenv("API_URL", "http://localhost:8000")
st.set_page_config(page_title="NeuralStem - Music Source Separation", layout="wide", page_icon="🎵")

# ========== CUSTOM CSS (NeuralStem Style) ==========
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background-color: #0a0a0a;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #ffffff;
    }
    
    /* Main title */
    .main-title {
        font-size: 4.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #888888;
        margin-top: 0.5rem;
    }
    
    /* Navigation bar */
    .nav-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem 0;
        border-bottom: 1px solid #222;
        margin-bottom: 2rem;
    }
    
    .logo {
        font-size: 1.5rem;
        font-weight: 700;
        color: #00d4ff;
    }
    
    .nav-links {
        display: flex;
        gap: 2rem;
    }
    
    .nav-links a {
        color: #ccc;
        text-decoration: none;
        transition: color 0.2s;
    }
    
    .nav-links a:hover {
        color: #00d4ff;
    }
    
    /* Cards for examples */
    .example-card {
        background: #1a1a1a;
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        transition: transform 0.2s, border 0.2s;
        border: 1px solid #2a2a2a;
        cursor: pointer;
    }
    
    .example-card:hover {
        transform: translateY(-4px);
        border-color: #00d4ff;
    }
    
    .example-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    
    .example-artist {
        font-size: 0.8rem;
        color: #888;
    }
    
    /* Custom radio buttons (mode selector) */
    div[data-testid="stRadio"] > div {
        gap: 1rem;
    }
    
    div[data-testid="stRadio"] label {
        background: #1a1a1a;
        padding: 0.75rem 1.5rem;
        border-radius: 40px;
        border: 1px solid #2a2a2a;
        transition: all 0.2s;
    }
    
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        border-color: #00d4ff;
    }
    
    div[data-testid="stRadio"] input:checked + div {
        background: #00d4ff;
        color: #000;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background: transparent;
        border-bottom: 1px solid #2a2a2a;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #888;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        color: #00d4ff;
        border-bottom-color: #00d4ff;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        color: white;
        border: none;
        border-radius: 40px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: scale(1.02);
        background: linear-gradient(135deg, #00e4ff 0%, #8b3cdf 100%);
    }
    
    /* Audio player */
    audio {
        width: 100%;
        border-radius: 40px;
    }
    
    /* Divider */
    hr {
        border-color: #2a2a2a;
        margin: 2rem 0;
    }
    
    /* Contact section */
    .contact-section {
        text-align: center;
        padding: 3rem;
        background: #111;
        border-radius: 20px;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== NAVIGATION BAR ==========
col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="logo">🎵 NeuralStem</div>', unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="nav-links" style="justify-content: flex-end;">
        <a href="#">Separate</a>
        <a href="#">Karaoke</a>
        <a href="#">About</a>
        <a href="#">Contact</a>
    </div>
    """, unsafe_allow_html=True)

# ========== HERO SECTION ==========
st.markdown("""
<h1 class="main-title">Music Source Separation<br>& Karaoke</h1>
<p class="subtitle">Extract Vocals & Instrumental from any song</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ========== MODE SELECTION ==========
st.markdown("#### Choose the separation mode")
mode = st.radio(
    "",
    ["🎤 Vocals & Instrumental (Low Quality, Faster)", "🎚️ 4 Sources (Vocals, Drums, Bass, Other)"],
    horizontal=True,
    label_visibility="collapsed"
)
num_sources = 2 if "Faster" in mode else 4

# ========== MAIN CONTENT AREA ==========
tab1, tab2 = st.tabs(["📁 Upload File", "🔗 From URL"])

# ========== TAB 1: UPLOAD FILE ==========
with tab1:
    uploaded_file = st.file_uploader(
        "Drop your audio file here or click to browse",
        type=["wav", "mp3", "flac"],
        help="Supports WAV, MP3, FLAC"
    )
    
    col_upload, col_example = st.columns([1, 1])
    
    with col_upload:
        if uploaded_file is not None:
            if st.button("🎧 Separate", use_container_width=True):
                with st.spinner("Processing..."):
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(
                        f"{API_URL}/separate",
                        files=files,
                        params={"num_sources": num_sources}
                    )
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("Separation complete!")
                    
                    st.markdown("### Results")
                    source_names = data["sources"]
                    cols = st.columns(len(source_names))
                    
                    for i, (col, name, file_url) in enumerate(zip(cols, source_names, data["files"])):
                        with col:
                            st.markdown(f"**{name.capitalize()}**")
                            full_url = f"{API_URL}{file_url}"
                            audio_resp = requests.get(full_url)
                            if audio_resp.status_code == 200:
                                st.audio(audio_resp.content, format="audio/wav")
                                st.download_button(
                                    label="Download",
                                    data=audio_resp.content,
                                    file_name=f"{name}.wav",
                                    mime="audio/wav",
                                    key=f"download_{i}"
                                )
                else:
                    st.error(f"Error: {response.text}")

# ========== TAB 2: FROM URL ==========
with tab2:
    url_input = st.text_input("Enter audio URL", placeholder="https://example.com/song.mp3")
    if url_input and st.button("🔗 Process URL"):
        st.info("URL processing requires additional setup on the backend. Coming soon!")

# ========== EXAMPLES SECTION ==========
st.markdown("---")
st.markdown("#### 🎵 Examples")
st.markdown("Select a sample song and listen to sources separated")

# Примеры песен (можно заменить на реальные аудиофайлы)
examples = [
    {"title": "Blinding Lights", "artist": "The Weeknd", "file": "blinding_lights.mp3"},
    {"title": "Levitating", "artist": "Dua Lipa", "file": "levitating.mp3"},
    {"title": "Flowers", "artist": "Miley Cyrus", "file": "flowers.mp3"},
]

cols = st.columns(3)
for i, (col, example) in enumerate(zip(cols, examples)):
    with col:
        st.markdown(f"""
        <div class="example-card">
            <div class="example-title">{example['title']}</div>
            <div class="example-artist">{example['artist']}</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("▶️ Listen", key=f"example_{i}"):
            st.info(f"Demo for {example['title']} - would load pre-separated stems")

# ========== CONTACT / FOOTER ==========
st.markdown("---")
st.markdown("""
<div class="contact-section">
    <h3>Ready to separate your music?</h3>
    <p style="color: #888">Try NeuralStem now — free and open source</p>
</div>
""", unsafe_allow_html=True)