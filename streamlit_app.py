#!/usr/bin/env python3
"""
Risk Audio Analysis Pipeline - Streamlit UI
SageMaker-compatible web interface for audio transcription and emotion recognition
"""

import os
import json
import uuid
import time
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import logging

import streamlit as st
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Risk Audio Analysis",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #dc3545, #fd7e14);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
    }
    .risk-moderate {
        color: #fd7e14;
        font-weight: bold;
    }
    .risk-low {
        color: #198754;
        font-weight: bold;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
    }
    .metric-card.high-risk {
        border-left-color: #dc3545;
    }
    .metric-card.moderate-risk {
        border-left-color: #fd7e14;
    }
</style>
""", unsafe_allow_html=True)

def safe_import_pipeline():
    """Safely import the transcription pipeline with fallback"""
    try:
        # Try direct import first
        from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline
        return TranscriptionChunkRiskPipeline, None
    except ImportError as e:
        logger.warning(f"Direct import failed: {e}")
        
        # Try with path manipulation
        import sys
        from pathlib import Path
        
        project_root = Path(__file__).resolve().parent
        candidates = [
            project_root,
            project_root / 'Emotion_Recognition',
            Path.cwd(),
            Path.cwd() / 'Emotion_Recognition',
        ]
        
        for candidate in candidates:
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.append(candidate_str)
        
        try:
            from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline
            return TranscriptionChunkRiskPipeline, None
        except ImportError as e2:
            return None, f"Could not import pipeline: {e2}"

def create_mock_pipeline():
    """Create a mock pipeline for demo purposes when imports fail"""
    class MockPipeline:
        def process_audio(self, audio_file, progress_callback=None):
            # Mock result for demo
            return type('MockResult', (), {
                'audio_file': audio_file,
                'audio_duration': 120.0,
                'processing_time': 5.0,
                'total_chunks': 10,
                'high_risk_chunks': [
                    type('MockChunk', (), {
                        'speaker': 'Speaker_1',
                        'text': 'I feel really depressed and anxious about everything.',
                        'start_time': 10.0,
                        'end_time': 15.0,
                        'word_count': 10,
                        'emotion': 'depression',
                        'confidence': 0.85,
                        'risk_level': 'HIGH',
                        'chunk_index': 0
                    })()
                ],
                'moderate_risk_chunks': [
                    type('MockChunk', (), {
                        'speaker': 'Speaker_2',
                        'text': 'This situation is really stressful.',
                        'start_time': 30.0,
                        'end_time': 35.0,
                        'word_count': 6,
                        'emotion': 'stress',
                        'confidence': 0.72,
                        'risk_level': 'MODERATE',
                        'chunk_index': 1
                    })()
                ],
                'risk_summary': {
                    'overall_risk_level': 'MODERATE',
                    'high_risk_percentage': 10.0,
                    'moderate_risk_percentage': 10.0,
                    'total_risk_chunks': 2,
                    'emotion_distribution': {'depression': 1, 'stress': 1},
                    'speaker_risk_summary': {
                        'Speaker_1': {'HIGH': 1, 'MODERATE': 0},
                        'Speaker_2': {'HIGH': 0, 'MODERATE': 1}
                    }
                },
                'transcript_chunks': [],
                'raw_transcription': 'Mock transcription for demo purposes.'
            })()
    
    return MockPipeline()

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🎤 Risk-Focused Audio Analysis</h1>
        <p>Transcribe audio and identify high-risk emotional content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Check if pipeline can be imported
        PipelineClass, import_error = safe_import_pipeline()
        
        if PipelineClass is None:
            st.error("⚠️ Pipeline Import Failed")
            st.error(import_error)
            st.info("Running in demo mode with mock data")
            use_mock = True
        else:
            st.success("✅ Pipeline Ready")
            use_mock = False
        
        st.markdown("---")
        
        # Model selection
        st.subheader("🎯 Model Settings")
        model_choice = st.selectbox(
            "Whisper Model",
            ["Whisper Large", "Whisper Base", "Whisper Enhanced"],
            index=0
        )
        
        # Risk thresholds
        st.subheader("🚨 Risk Thresholds")
        high_threshold = st.slider("High Risk Threshold (%)", 0, 50, 20)
        moderate_threshold = st.slider("Moderate Risk Threshold (%)", 0, 30, 10)
        
        st.markdown("---")
        
        # Info
        st.info("""
        **Risk Levels:**
        - 🔴 **HIGH**: Depression, Anxiety, Loneliness, Physical Pain
        - 🟠 **MODERATE**: Stress, Anger, Confusion, Shame/Guilt
        - 🟢 **LOW**: Happiness, Calm, Neutral emotions
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📁 Upload Audio File")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose an audio file",
            type=['wav', 'mp3', 'm4a', 'flac', 'ogg', 'mp4', 'avi', 'mov'],
            help="Supported formats: WAV, MP3, M4A, FLAC, OGG, MP4, AVI, MOV"
        )
        
        if uploaded_file is not None:
            # Show file info
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            st.info(f"Size: {uploaded_file.size / (1024*1024):.1f} MB")
            
            # Process button
            if st.button("🚀 Start Risk Analysis", type="primary"):
                process_audio_file(uploaded_file, use_mock, PipelineClass)
    
    with col2:
        st.header("📊 Quick Stats")
        
        # Placeholder metrics
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Files Processed", "0")
            st.metric("High Risk Detected", "0")
        with col_b:
            st.metric("Processing Time", "0s")
            st.metric("Accuracy", "95%")

def process_audio_file(uploaded_file, use_mock, PipelineClass):
    """Process the uploaded audio file"""
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name
        
        # Initialize pipeline
        if use_mock:
            pipeline = create_mock_pipeline()
        else:
            pipeline = PipelineClass()
        
        # Progress callback
        def update_progress(progress, message):
            progress_bar.progress(progress / 100)
            status_text.text(f"Progress: {progress}% - {message}")
        
        # Process audio
        status_text.text("Starting analysis...")
        progress_bar.progress(0.1)
        
        result = pipeline.process_audio(tmp_path, update_progress)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Display results
        display_results(result)
        
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        logger.error(f"Processing error: {e}", exc_info=True)
    
    finally:
        progress_bar.empty()
        status_text.empty()

def display_results(result):
    """Display analysis results"""
    
    st.header("📈 Analysis Results")
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Duration", 
            f"{result.audio_duration:.1f}s",
            delta=None
        )
    
    with col2:
        st.metric(
            "Total Chunks", 
            result.total_chunks,
            delta=None
        )
    
    with col3:
        st.metric(
            "High Risk", 
            len(result.high_risk_chunks),
            delta=f"{result.risk_summary['high_risk_percentage']:.1f}%"
        )
    
    with col4:
        st.metric(
            "Moderate Risk", 
            len(result.moderate_risk_chunks),
            delta=f"{result.risk_summary['moderate_risk_percentage']:.1f}%"
        )
    
    # Overall risk level
    risk_level = result.risk_summary['overall_risk_level']
    risk_color = {
        'HIGH': '🔴',
        'MODERATE': '🟠', 
        'LOW': '🟢'
    }.get(risk_level, '⚪')
    
    st.markdown(f"""
    <div class="metric-card {'high-risk' if risk_level == 'HIGH' else 'moderate-risk' if risk_level == 'MODERATE' else ''}">
        <h3>{risk_color} Overall Risk Level: {risk_level}</h3>
        <p>Processing Time: {result.processing_time:.1f}s</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Visualizations
    create_visualizations(result)
    
    # Risk chunks
    display_risk_chunks(result)

def create_visualizations(result):
    """Create risk analysis visualizations"""
    
    st.header("📊 Risk Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        risk_counts = {
            'HIGH': len(result.high_risk_chunks),
            'MODERATE': len(result.moderate_risk_chunks),
            'LOW': result.total_chunks - len(result.high_risk_chunks) - len(result.moderate_risk_chunks)
        }
        
        if any(risk_counts.values()):
            fig = go.Figure(data=[go.Pie(
                labels=list(risk_counts.keys()),
                values=list(risk_counts.values()),
                hole=0.3,
                marker=dict(colors=['#dc3545', '#fd7e14', '#198754'])
            )])
            fig.update_layout(title="Risk Level Distribution", height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Emotion distribution
        if result.high_risk_chunks or result.moderate_risk_chunks:
            emotion_counts = {}
            for chunk in result.high_risk_chunks + result.moderate_risk_chunks:
                emotion_counts[chunk.emotion] = emotion_counts.get(chunk.emotion, 0) + 1
            
            if emotion_counts:
                fig = go.Figure(data=[go.Bar(
                    x=list(emotion_counts.keys()),
                    y=list(emotion_counts.values()),
                    marker_color=['#dc3545' if emotion in ['depression', 'anxiety', 'loneliness', 'physical_pain'] 
                                else '#fd7e14' for emotion in emotion_counts.keys()]
                )])
                fig.update_layout(
                    title="Emotion Distribution in Risk Chunks",
                    xaxis_title="Emotion",
                    yaxis_title="Count",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Risk timeline
    if result.high_risk_chunks or result.moderate_risk_chunks:
        st.subheader("⏰ Risk Timeline")
        
        timeline_data = []
        for chunk in result.high_risk_chunks:
            timeline_data.append({
                'Speaker': chunk.speaker,
                'Start': chunk.start_time,
                'End': chunk.end_time,
                'Emotion': chunk.emotion,
                'Risk': 'HIGH',
                'Confidence': chunk.confidence,
                'Text': chunk.text[:50] + '...' if len(chunk.text) > 50 else chunk.text
            })
        
        for chunk in result.moderate_risk_chunks:
            timeline_data.append({
                'Speaker': chunk.speaker,
                'Start': chunk.start_time,
                'End': chunk.end_time,
                'Emotion': chunk.emotion,
                'Risk': 'MODERATE',
                'Confidence': chunk.confidence,
                'Text': chunk.text[:50] + '...' if len(chunk.text) > 50 else chunk.text
            })
        
        if timeline_data:
            df_timeline = pd.DataFrame(timeline_data)
            st.dataframe(df_timeline, use_container_width=True)

def display_risk_chunks(result):
    """Display high and moderate risk chunks"""
    
    # High risk chunks
    if result.high_risk_chunks:
        st.header("🔴 High Risk Chunks")
        
        for i, chunk in enumerate(result.high_risk_chunks):
            with st.expander(f"Chunk {i+1}: {chunk.speaker} - {chunk.emotion} ({chunk.confidence:.1%})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Text:** {chunk.text}")
                    st.write(f"**Time:** {chunk.start_time:.1f}s - {chunk.end_time:.1f}s")
                    st.write(f"**Words:** {chunk.word_count}")
                
                with col2:
                    st.metric("Confidence", f"{chunk.confidence:.1%}")
                    st.metric("Risk Level", chunk.risk_level)
    
    # Moderate risk chunks
    if result.moderate_risk_chunks:
        st.header("🟠 Moderate Risk Chunks")
        
        for i, chunk in enumerate(result.moderate_risk_chunks):
            with st.expander(f"Chunk {i+1}: {chunk.speaker} - {chunk.emotion} ({chunk.confidence:.1%})"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Text:** {chunk.text}")
                    st.write(f"**Time:** {chunk.start_time:.1f}s - {chunk.end_time:.1f}s")
                    st.write(f"**Words:** {chunk.word_count}")
                
                with col2:
                    st.metric("Confidence", f"{chunk.confidence:.1%}")
                    st.metric("Risk Level", chunk.risk_level)
    
    # Download results
    if st.button("💾 Download Results"):
        results_json = json.dumps({
            'audio_file': result.audio_file,
            'audio_duration': result.audio_duration,
            'processing_time': result.processing_time,
            'total_chunks': result.total_chunks,
            'high_risk_chunks': [chunk.__dict__ for chunk in result.high_risk_chunks],
            'moderate_risk_chunks': [chunk.__dict__ for chunk in result.moderate_risk_chunks],
            'risk_summary': result.risk_summary,
            'raw_transcription': result.raw_transcription
        }, indent=2, default=str)
        
        st.download_button(
            label="Download JSON Results",
            data=results_json,
            file_name=f"risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

if __name__ == "__main__":
    main()
