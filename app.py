"""Streamlit app:  streamlit run app.py"""
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.model import DEFAULT_XCLIP_LABELS, load_model
from src.pipeline import analyze, segment_analysis
from src.viz import plot_timeline

st.set_page_config(page_title="Video Action Segmenter", layout="wide")
st.title("Video Action Segmentation")

@st.cache_resource(show_spinner="Loading AI model (first run downloads weights)...")
def get_model(labels_key: str):
    labels = [l.strip() for l in labels_key.split("|") if l.strip()] or None
    return load_model(labels=labels)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Actions to Detect")
    st.write("Enter the actions you want to find in the video, one per line. The AI will handle the rest.")
    
    txt = st.text_area("Action labels", "\n".join(DEFAULT_XCLIP_LABELS), height=300, label_visibility="collapsed")
    labels_key = "|".join(l.strip() for l in txt.splitlines() if l.strip())
    
    # Hardcoded optimal settings hidden from the user
    hop = 4
    method = "changepoint"
    feature = "probs"
    min_dur = 1.5
    penalty = 1.0
    smooth = 2.0

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv", "webm"])

if uploaded:
    st.video(uploaded)
    # The cache key now only cares if the file or the text labels change
    key = (uploaded.name, uploaded.size, labels_key)
    if st.session_state.get("key") != key:          
        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getbuffer())
            tmp = f.name
        bar = st.progress(0.0, text="Analyzing video...")
        model = get_model(labels_key)
        st.session_state["analysis"] = analyze(tmp, model, hop_frames=hop,
                                               progress_cb=lambda p: bar.progress(p))
        st.session_state["key"] = key
        bar.empty()

    an = st.session_state["analysis"]
    if an.truncated:
        st.warning("Video was longer than 10 minutes; only the first 10 minutes were analysed.")

    # Segmentation happens instantly in the background using the hardcoded optimal values
    segs = segment_analysis(an, method, smooth, min_dur, penalty, feature)
    st.pyplot(plot_timeline(segs, an.duration, title="Segmented Timeline"))

    rows = [s.to_dict() for s in segs]
    df = pd.DataFrame([{
        "start (s)": r["start"], "end (s)": r["end"], "label": r["label"],
        "confidence": r["confidence"],
        "also possible": ", ".join(f"{a['label']} ({a['prob']:.2f})" for a in r["alternatives"]),
    } for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    c1.download_button("Download JSON", json.dumps(rows, indent=2), "segments.json")
    c2.download_button("Download CSV", df.to_csv(index=False), "segments.csv")

    st.subheader("Keyframes")
    cols = st.columns(min(4, max(1, len(segs))))
    for i, s in enumerate(segs):
        t = min(int((s.start + s.end) / 2), len(an.thumbs) - 1)
        cols[i % len(cols)].image(an.thumbs[t], caption=f"{s.start:.0f}-{s.end:.0f}s: {s.label}")