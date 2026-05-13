import streamlit as st
import requests
import time
import os

API_URL = os.getenv("API_URL", "http://localhost/api")

st.set_page_config(page_title="NeuralStem - Music Source Separation", layout="wide", page_icon="🎵")

st.markdown("""
<style>
    .stApp { background-color: #0a0a0a; }
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #ffffff; }
    .main-title {
        font-size: 4.5rem; font-weight: 700;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 0; line-height: 1.2;
    }
    .subtitle { font-size: 1.2rem; color: #888888; margin-top: 0.5rem; }
    .logo { font-size: 1.5rem; font-weight: 700; color: #00d4ff; }
    .stButton > button {
        background: linear-gradient(135deg, #00d4ff 0%, #7b2cbf 100%);
        color: white; border: none; border-radius: 40px;
        padding: 0.6rem 1.5rem; font-weight: 600;
    }
    audio { width: 100%; border-radius: 40px; }
    hr { border-color: #2a2a2a; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 3])
with col1:
    st.markdown('<div class="logo">🎵 NeuralStem</div>', unsafe_allow_html=True)

st.markdown("""
<h1 class="main-title">Music Source Separation</h1>
<p class="subtitle">Extract Vocals & Instrumental from any song</p>
""", unsafe_allow_html=True)

st.markdown("---")

st.markdown("#### Choose separation mode")
mode = st.radio(
    "",
    ["🎤 Vocals & Instrumental (2 sources)", "🎚️ 4 Sources (Vocals, Drums, Bass, Other)"],
    horizontal=True,
    label_visibility="collapsed",
)
num_sources = 2 if "2 sources" in mode else 4

# ========== MAIN CONTENT AREA ==========
tab1, tab2 = st.tabs(["📁 Upload File", "🔗 From URL"])

# ========== TAB 1: UPLOAD FILE ==========
with tab1:
    uploaded_file = st.file_uploader(
        "Drop your audio file here or click to browse",
        type=["wav", "mp3", "flac"],
        help="Supports WAV, MP3, FLAC (max 50 MB)",
    )
 
    if uploaded_file is not None:
        if st.button("🎧 Separate", use_container_width=True):
            # 1. Submit job
            with st.spinner("Submitting job…"):
                resp = requests.post(
                    f"{API_URL}/separate",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    params={"num_sources": num_sources},
                    timeout=30,
                )
            if resp.status_code != 200:
                st.error(f"Submission failed: {resp.text}")
                st.stop()
 
            job_id = resp.json()["job_id"]
            st.info(f"Job submitted (id: `{job_id}`). Processing…")
 
            # 2. Poll for result
            progress_bar = st.progress(0, text="Waiting in queue…")
            STEPS = {"pending": 0, "processing": 50, "success": 100, "failed": 0}
            step_labels = {
                "loading audio": 20,
                "computing spectrogram": 40,
                "running model": 70,
                "reconstructing audio": 90,
            }
 
            result = None
            for _ in range(120):  # max ~2 minutes
                time.sleep(1)
                status_resp = requests.get(f"{API_URL}/status/{job_id}", timeout=10)
                if status_resp.status_code != 200:
                    continue
                data = status_resp.json()
                state = data.get("state", "pending")
 
                if state == "processing":
                    step = data.get("info", {}).get("step", "")
                    pct = step_labels.get(step, 50)
                    progress_bar.progress(pct, text=f"Processing: {step}…")
                elif state == "success":
                    progress_bar.progress(100, text="Done!")
                    result = data["result"]
                    break
                elif state == "failed":
                    progress_bar.empty()
                    st.error(f"Processing failed: {data.get('error', 'unknown error')}")
                    st.stop()
 
            if result is None:
                st.error("Job timed out. Please try again.")
                st.stop()
 
            # 3. Show results
            st.success("Separation complete!")
            st.markdown("### Results")
            source_names = result["sources"]
            cols = st.columns(len(source_names))
            for i, (col, name, file_url) in enumerate(zip(cols, source_names, result["files"])):
                with col:
                    st.markdown(f"**{name.capitalize()}**")
                    audio_resp = requests.get(f"{API_URL}{file_url}", timeout=60)
                    if audio_resp.status_code == 200:
                        st.audio(audio_resp.content, format="audio/wav")
                        st.download_button(
                            label="⬇ Download",
                            data=audio_resp.content,
                            file_name=f"{name}.wav",
                            mime="audio/wav",
                            key=f"dl_{i}",
                        )
                    else:
                        st.warning(f"Could not load {name}")

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