#!/usr/bin/env python3
"""
Streamlit UI for Speaker-Aware Transcription and Sentiment Analysis
"""

import os
import json
import time
from pathlib import Path

import streamlit as st

from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline
from config import RESULTS_DIR

SENTIMENTS_FILE = Path(RESULTS_DIR) / "sentiments_cumulative.json"
SENTIMENTS_FILE.parent.mkdir(parents=True, exist_ok=True)


def read_cumulative():
    if SENTIMENTS_FILE.exists():
        try:
            return json.loads(SENTIMENTS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"entries": []}
    return {"entries": []}


def write_cumulative(data):
    SENTIMENTS_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


@st.cache_resource(show_spinner=False)
def get_pipeline():
    return TranscriptionChunkRiskPipeline()


st.set_page_config(page_title="Audio Analysis", page_icon="🎤", layout="wide")
st.title("Speaker-Aware Transcription & Sentiment Analysis")
st.caption("Upload an audio file to transcribe by speaker, analyze sentiments, and track cumulative results.")

with st.sidebar:
    st.header("Upload")
    uploaded_file = st.file_uploader("Audio file (WAV/MP3/M4A/FLAC)", type=["wav", "mp3", "m4a", "flac"])
    analyze_btn = st.button("Analyze Audio", type="primary", use_container_width=True, disabled=uploaded_file is None)
    st.divider()
    st.header("Cumulative Sentiments")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Refresh Summary", use_container_width=True):
            st.experimental_rerun()
    with col_b:
        if st.button("Reset All", use_container_width=True):
            write_cumulative({"entries": []})
            st.success("Cleared all saved sentiments.")

tab1, tab2, tab3 = st.tabs(["Results", "Cumulative Summary", "About"])

if analyze_btn and uploaded_file is not None:
    # Save to uploads
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    filepath = uploads_dir / uploaded_file.name
    if filepath.exists():
        st.info(f"Using existing file: {filepath.name}")
    else:
        with open(filepath, "wb") as f:
            f.write(uploaded_file.read())

    with st.spinner("Processing audio... This may take a while depending on model sizes."):
        pipe = get_pipeline()
        t0 = time.time()
        result = pipe.process_audio(str(filepath))
        elapsed = time.time() - t0

    st.toast(f"Completed in {elapsed:.1f}s", icon="✅")

    if result:
        with tab1:
            # Summary metrics
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Duration", f"{result.audio_duration:.1f}s")
            m2.metric("Total Chunks", result.total_chunks)
            m3.metric("High Risk", f"{result.risk_summary.get('high_risk_percentage', 0):.1f}%")
            m4.metric("Moderate Risk", f"{result.risk_summary.get('moderate_risk_percentage', 0):.1f}%")
            m5.metric("Overall Risk", result.risk_summary.get("overall_risk_level", "-"))

            # Overall sentiment
            overall = getattr(result, 'overall_raw_sentiment', {}) or {}
            with st.expander("Overall Sentiment", expanded=True):
                st.write({
                    'predicted_emotion': overall.get('predicted_emotion'),
                    'confidence': overall.get('confidence'),
                    'all_scores': overall.get('all_scores')
                })
                # Evidence chunks
                evidence = overall.get('evidence_chunks') or []
                if evidence:
                    st.markdown("**Evidence from transcript:**")
                    for ev in evidence:
                        st.write(f"- [{ev['speaker']}] ({ev['start_time']:.1f}s-{ev['end_time']:.1f}s): {ev['text']}")

            # Speaker transcript (prefer transcript_chunks produced by pipeline)
            speaker_chunks = getattr(result, 'transcript_chunks', []) or getattr(result, 'speaker_chunks', [])
            with st.expander("Transcription by Speaker", expanded=True):
                if speaker_chunks:
                    for c in speaker_chunks:
                        st.write(f"[{c['speaker']}] ({c['start_time']:.1f}s-{c['end_time']:.1f}s): {c['text']}")
                else:
                    st.write(getattr(result, 'raw_transcription', ''))

            # Speaker sentiments
            sp_sents = getattr(result, 'speaker_sentiments', {}) or {}
            if sp_sents:
                st.subheader("Speaker Sentiments")
                st.table([
                    {
                        'Speaker': sp,
                        'Sentiment': v.get('predicted_emotion'),
                        'Confidence': f"{(v.get('confidence', 0)*100):.1f}%"
                    } for sp, v in sp_sents.items()
                ])

            # Unified Risk Chunks table
            st.subheader("Risk Chunks")
            rows = []
            def to_row(c, level_override=None):
                d = c.to_dict() if hasattr(c, 'to_dict') else {
                    'speaker': getattr(c, 'speaker', ''),
                    'text': getattr(c, 'text', ''),
                    'start_time': getattr(c, 'start_time', 0.0),
                    'end_time': getattr(c, 'end_time', 0.0),
                    'emotion': getattr(c, 'emotion', ''),
                    'confidence': getattr(c, 'confidence', 0.0),
                    'risk_level': getattr(c, 'risk_level', level_override or ''),
                    'chunk_index': getattr(c, 'chunk_index', 0),
                }
                return {
                    'Speaker': d['speaker'],
                    'Time': f"{d['start_time']:.1f}s–{d['end_time']:.1f}s",
                    'Emotion': d.get('emotion', ''),
                    'Risk Level': d.get('risk_level', ''),
                    'Confidence': f"{float(d.get('confidence', 0.0))*100:.1f}%",
                    'Excerpt': d.get('text', '')[:140] + ('…' if len(d.get('text',''))>140 else '')
                }
            for c in (getattr(result, 'high_risk_chunks', []) or []):
                # Only include classified as HIGH
                rows.append(to_row(c, 'HIGH'))
            for c in (getattr(result, 'moderate_risk_chunks', []) or []):
                # Only include classified as MODERATE
                rows.append(to_row(c, 'MODERATE'))
            # Sort by start time if available inside the Time string
            try:
                rows.sort(key=lambda r: float(r['Time'].split('s')[0]))
            except Exception:
                pass
            if rows:
                st.table(rows)
            else:
                st.info("No high or moderate risk chunks")

        # Persist cumulative
        data = read_cumulative()
        data.setdefault('entries', []).append({
            'file': Path(result.audio_file).name,
            'overall_raw_sentiment': overall,
            'speaker_sentiments': sp_sents
        })
        write_cumulative(data)

with tab2:
    data = read_cumulative()
    entries = data.get('entries', [])
    st.subheader("Cumulative Summary")
    st.caption(f"Total uploads: {len(entries)}")

    # Aggregate emotions
    counts = {}
    for e in entries:
        o = e.get('overall_raw_sentiment') or {}
        label = (o.get('predicted_emotion') or 'neutral').lower()
        counts[label] = counts.get(label, 0) + 1

    # Display emotion counts in a grid
    st.write("**Emotion Distribution:**")
    
    # Group emotions by type
    positive_emotions = ['happy']
    neutral_emotions = ['neutral', 'confusion']
    risk_emotions = ['anger', 'frustration', 'urgency', 'escalation', 'client_wants_to_leave', 
                     'risk_issue', 'safety_wellbeing', 'financial_distress', 'compliance_privacy']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**😊 Positive**")
        for emotion in positive_emotions:
            count = counts.get(emotion, 0)
            if count > 0:
                st.metric(emotion.replace('_', ' ').title(), count)
        if sum(counts.get(e, 0) for e in positive_emotions) == 0:
            st.caption("No positive emotions")
    
    with col2:
        st.markdown("**😐 Neutral**")
        for emotion in neutral_emotions:
            count = counts.get(emotion, 0)
            if count > 0:
                st.metric(emotion.replace('_', ' ').title(), count)
        if sum(counts.get(e, 0) for e in neutral_emotions) == 0:
            st.caption("No neutral emotions")
    
    with col3:
        st.markdown("**⚠️ Risk/Concern**")
        for emotion in risk_emotions:
            count = counts.get(emotion, 0)
            if count > 0:
                st.metric(emotion.replace('_', ' ').title(), count)
        if sum(counts.get(e, 0) for e in risk_emotions) == 0:
            st.caption("No risk emotions")

    st.divider()
    st.write("Raw Entries")
    if entries:
        st.json(entries)
    else:
        st.info("No entries yet. Upload an audio file to begin.")

with tab3:
    st.markdown("""
    ### About
    This UI uses your existing pipeline to:
    - Diarize speakers (2 speakers)
    - Transcribe per speaker
    - Run fine-tuned sentiment analysis
    - Persist cumulative sentiment summaries
    """)

# (Instructions removed as per request)


