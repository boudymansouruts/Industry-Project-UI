#!/usr/bin/env python3
"""
Risk-Focused Web Interface for Audio Transcription and Emotion Recognition
Flask-based UI that automatically chunks transcripts and flags high/moderate risk segments
"""

import os
import json
import uuid
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import flask
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
import plotly.graph_objs as go
import plotly.utils
import pandas as pd
import numpy as np

from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'risk-focused-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('static/plots', exist_ok=True)

# Global variables for progress tracking
progress_data = {}
pipeline_instance = None

# Allowed file extensions
ALLOWED_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg', '.mp4', '.avi', '.mov'}


def allowed_file(filename):
    """Check if file extension is allowed"""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


def init_pipeline():
    """Initialize the transcription chunk risk pipeline instance"""
    global pipeline_instance
    if pipeline_instance is None:
        try:
            # Initialize with transcription chunks (no chunk size needed)
            pipeline_instance = TranscriptionChunkRiskPipeline()
            app.logger.info("Transcription chunk risk pipeline initialized")
        except Exception as e:
            app.logger.error(f"Failed to initialize pipeline: {e}")
            pipeline_instance = None
    return pipeline_instance is not None


@app.route('/')
def index():
    """Main page"""
    return render_template('risk_index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start risk-focused processing"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file selected'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not supported. Please upload audio/video files.'}), 400
    
    # Get session ID
    session_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_filename = f"{timestamp}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
    file.save(filepath)
    
    # Initialize progress tracking
    progress_data[session_id] = {
        'status': 'starting',
        'progress': 0,
        'message': 'Initializing risk analysis...',
        'filename': filename,
        'filepath': filepath,
        'start_time': time.time(),
        'results': None,
        'error': None
    }
    
    # Start processing in background thread
    thread = threading.Thread(
        target=process_audio_risk_focused,
        args=(session_id, filepath)
    )
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'session_id': session_id,
        'message': 'File uploaded successfully. Risk analysis started.',
        'filename': filename
    })


def process_audio_risk_focused(session_id: str, filepath: str):
    """Process audio file with risk-focused analysis using transcription chunks"""
    try:
        def progress_callback(progress: int, message: str):
            """Update progress"""
            progress_data[session_id].update({
                'progress': progress,
                'message': message,
                'status': 'processing'
            })
        
        # Initialize risk-focused pipeline with transcription chunks
        pipeline = TranscriptionChunkRiskPipeline()
        
        # Process the audio
        result = pipeline.process_audio(filepath, progress_callback)
        
        # Generate risk-focused visualizations
        progress_callback(95, "Generating risk visualizations...")
        plots = generate_risk_visualizations(result, session_id)
        
        # Save results
        results_file = os.path.join(
            app.config['RESULTS_FOLDER'], 
            f"{session_id}_risk_results.json"
        )
        with open(results_file, 'w') as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
        
        # Update progress data
        progress_data[session_id].update({
            'status': 'completed',
            'progress': 100,
            'message': 'Risk analysis completed successfully!',
            'results': result.to_dict(),
            'plots': plots,
            'results_file': results_file,
            'processing_time': time.time() - progress_data[session_id]['start_time']
        })
        
    except Exception as e:
        app.logger.error(f"Error processing audio {session_id}: {str(e)}", exc_info=True)
        progress_data[session_id].update({
            'status': 'error',
            'error': str(e),
            'message': f'Error: {str(e)}'
        })


@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Get processing progress"""
    if session_id not in progress_data:
        return jsonify({'error': 'Session not found'}), 404
    
    data = progress_data[session_id].copy()
    
    # Don't send large results data in progress updates
    if 'results' in data and data['status'] != 'completed':
        data.pop('results', None)
    
    return jsonify(data)


@app.route('/results/<session_id>')
def show_results(session_id):
    """Display risk-focused results page"""
    if session_id not in progress_data:
        return "Session not found", 404
    
    data = progress_data[session_id]
    if data['status'] != 'completed':
        return redirect(url_for('processing', session_id=session_id))
    
    return render_template('risk_results.html', 
                         session_id=session_id, 
                         data=data)


@app.route('/processing/<session_id>')
def processing(session_id):
    """Show processing page with progress"""
    if session_id not in progress_data:
        return "Session not found", 404
    
    return render_template('risk_processing.html', session_id=session_id)


@app.route('/download/<session_id>')
def download_results(session_id):
    """Download risk analysis results as JSON"""
    if session_id not in progress_data:
        return "Session not found", 404
    
    data = progress_data[session_id]
    if 'results_file' not in data:
        return "Results not available", 404
    
    return send_from_directory(
        app.config['RESULTS_FOLDER'],
        os.path.basename(data['results_file']),
        as_attachment=True,
        download_name=f"risk_analysis_{session_id}.json"
    )


@app.route('/transcript/<session_id>')
def show_transcript(session_id):
    """Display full transcript with speaker identification"""
    if session_id not in progress_data:
        return "Session not found", 404
    
    data = progress_data[session_id]
    if data['status'] != 'completed':
        return redirect(url_for('processing', session_id=session_id))
    
    return render_template('risk_transcript.html', 
                         session_id=session_id, 
                         data=data)


def generate_risk_visualizations(result, session_id: str) -> Dict[str, str]:
    """Generate risk-focused visualization plots"""
    plots = {}
    
    try:
        # 1. Risk Level Distribution Pie Chart
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
                textinfo='label+percent',
                marker=dict(colors=['#dc3545', '#fd7e14', '#198754'])  # Red, Orange, Green
            )])
            fig.update_layout(
                title="Risk Level Distribution",
                font=dict(size=12),
                height=400
            )
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            plots['risk_distribution'] = plot_json
        
        # 2. Emotion Distribution for Risk Chunks
        if result.high_risk_chunks or result.moderate_risk_chunks:
            emotion_counts = {}
            for chunk in result.high_risk_chunks + result.moderate_risk_chunks:
                emotion = chunk.emotion
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
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
                    yaxis_title="Number of Chunks",
                    height=400
                )
                plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                plots['emotion_distribution'] = plot_json
        
        # 3. Risk Timeline
        if result.high_risk_chunks or result.moderate_risk_chunks:
            timeline_data = []
            colors = []
            
            for chunk in result.high_risk_chunks:
                timeline_data.append({
                    'x': [chunk.start_time, chunk.end_time],
                    'y': [chunk.speaker, chunk.speaker],
                    'emotion': chunk.emotion,
                    'risk': 'HIGH',
                    'confidence': chunk.confidence,
                    'text': chunk.text[:50] + '...' if len(chunk.text) > 50 else chunk.text
                })
                colors.append('#dc3545')
            
            for chunk in result.moderate_risk_chunks:
                timeline_data.append({
                    'x': [chunk.start_time, chunk.end_time],
                    'y': [chunk.speaker, chunk.speaker],
                    'emotion': chunk.emotion,
                    'risk': 'MODERATE',
                    'confidence': chunk.confidence,
                    'text': chunk.text[:50] + '...' if len(chunk.text) > 50 else chunk.text
                })
                colors.append('#fd7e14')
            
            fig = go.Figure()
            
            for i, data in enumerate(timeline_data):
                fig.add_trace(go.Scatter(
                    x=data['x'],
                    y=data['y'],
                    mode='lines+markers',
                    line=dict(color=colors[i], width=8),
                    name=f"{data['risk']} - {data['emotion']}",
                    hovertemplate=f"<b>{data['y'][0]}</b><br>" +
                                f"Time: {data['x'][0]:.1f}s - {data['x'][1]:.1f}s<br>" +
                                f"Emotion: {data['emotion']}<br>" +
                                f"Risk: {data['risk']}<br>" +
                                f"Confidence: {data['confidence']:.1%}<br>" +
                                f"Text: {data['text']}<extra></extra>"
                ))
            
            fig.update_layout(
                title="Risk Timeline",
                xaxis_title="Time (seconds)",
                yaxis_title="Speaker",
                height=300,
                showlegend=False
            )
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            plots['risk_timeline'] = plot_json
        
        # 4. Speaker Risk Summary
        if result.risk_summary.get('speaker_risk_summary'):
            speakers = list(result.risk_summary['speaker_risk_summary'].keys())
            high_counts = [result.risk_summary['speaker_risk_summary'][s]['HIGH'] for s in speakers]
            moderate_counts = [result.risk_summary['speaker_risk_summary'][s]['MODERATE'] for s in speakers]
            
            fig = go.Figure(data=[
                go.Bar(name='HIGH Risk', x=speakers, y=high_counts, marker_color='#dc3545'),
                go.Bar(name='MODERATE Risk', x=speakers, y=moderate_counts, marker_color='#fd7e14')
            ])
            fig.update_layout(
                title="Risk Distribution by Speaker",
                xaxis_title="Speaker",
                yaxis_title="Number of Risk Chunks",
                barmode='group',
                height=400
            )
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            plots['speaker_risk'] = plot_json
        
        # 5. Confidence Distribution
        if result.high_risk_chunks or result.moderate_risk_chunks:
            high_confidences = [chunk.confidence for chunk in result.high_risk_chunks]
            moderate_confidences = [chunk.confidence for chunk in result.moderate_risk_chunks]
            
            fig = go.Figure()
            if high_confidences:
                fig.add_trace(go.Histogram(
                    x=high_confidences,
                    name='HIGH Risk',
                    opacity=0.7,
                    marker_color='#dc3545',
                    nbinsx=20
                ))
            if moderate_confidences:
                fig.add_trace(go.Histogram(
                    x=moderate_confidences,
                    name='MODERATE Risk',
                    opacity=0.7,
                    marker_color='#fd7e14',
                    nbinsx=20
                ))
            
            fig.update_layout(
                title="Confidence Distribution by Risk Level",
                xaxis_title="Confidence",
                yaxis_title="Count",
                barmode='overlay',
                height=400
            )
            plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
            plots['confidence_dist'] = plot_json
        
    except Exception as e:
        app.logger.error(f"Error generating visualizations: {e}")
    
    return plots


# Import plotly colors (fallback if not available)
try:
    import plotly.express as px
except ImportError:
    class MockPx:
        class colors:
            class qualitative:
                Set3 = ['#8dd3c7', '#ffffb3', '#bebada', '#fb8072', '#80b1d3', '#fdb462']
    px = MockPx()


if __name__ == '__main__':
    # Create templates directory and files
    os.makedirs('templates', exist_ok=True)
    
    # Create risk-focused base template
    base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Risk-Focused Audio Analysis{% endblock %}</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        .progress-container { margin: 20px 0; }
        .risk-high { color: #dc3545; font-weight: bold; }
        .risk-moderate { color: #fd7e14; font-weight: bold; }
        .risk-low { color: #198754; font-weight: bold; }
        .risk-chunk-card { margin-bottom: 15px; border-left: 4px solid #007bff; }
        .risk-chunk-high { border-left-color: #dc3545; }
        .risk-chunk-moderate { border-left-color: #fd7e14; }
        .debug-section { background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .chunk-text { font-family: monospace; background-color: #f8f9fa; padding: 10px; border-radius: 3px; }
        .emotion-badge { margin: 2px; }
        .confidence-bar { height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }
        .confidence-fill { height: 100%; transition: width 0.3s ease; }
        .confidence-high { background-color: #dc3545; }
        .confidence-moderate { background-color: #fd7e14; }
    </style>
</head>
<body>
    <nav class="navbar navbar-expand-lg navbar-dark bg-danger">
        <div class="container">
            <a class="navbar-brand" href="/">
                <i class="fas fa-exclamation-triangle"></i> Risk-Focused Audio Analysis
            </a>
        </div>
    </nav>
    
    <div class="container mt-4">
        {% block content %}{% endblock %}
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
    {% block scripts %}{% endblock %}
</body>
</html>"""
    
    with open('templates/risk_base.html', 'w') as f:
        f.write(base_template)
    
    # Create risk-focused index template
    index_template = """{% extends "risk_base.html" %}

{% block content %}
<div class="row">
    <div class="col-lg-8 offset-lg-2">
        <div class="card">
            <div class="card-header bg-danger text-white">
                <h3><i class="fas fa-upload"></i> Upload Audio for Risk Analysis</h3>
            </div>
            <div class="card-body">
                <form id="uploadForm" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="audioFile" class="form-label">Select Audio/Video File</label>
                        <input type="file" class="form-control" id="audioFile" name="file" 
                               accept=".wav,.mp3,.m4a,.flac,.ogg,.mp4,.avi,.mov" required>
                        <div class="form-text">
                            Supported formats: WAV, MP3, M4A, FLAC, OGG, MP4, AVI, MOV (Max size: 500MB)
                        </div>
                    </div>
                    
                    <div class="row mb-3">
                        <div class="col-md-6">
                            <label for="chunkSize" class="form-label">Words per Chunk</label>
                            <input type="number" class="form-control" id="chunkSize" name="chunk_size" 
                                   value="50" min="10" max="100">
                            <div class="form-text">Number of words to analyze per chunk</div>
                        </div>
                        <div class="col-md-6">
                            <label for="minChunkSize" class="form-label">Minimum Chunk Size</label>
                            <input type="number" class="form-control" id="minChunkSize" name="min_chunk_size" 
                                   value="10" min="5" max="50">
                            <div class="form-text">Minimum words required for analysis</div>
                        </div>
                    </div>
                    
                    <button type="submit" class="btn btn-danger">
                        <i class="fas fa-play"></i> Start Risk Analysis
                    </button>
                </form>
                
                <div id="uploadStatus" class="mt-3" style="display: none;">
                    <div class="alert alert-info">
                        <i class="fas fa-spinner fa-spin"></i> Uploading file...
                    </div>
                </div>
            </div>
        </div>
        
        <div class="card mt-4">
            <div class="card-header">
                <h5><i class="fas fa-info-circle"></i> Risk-Focused Analysis</h5>
            </div>
            <div class="card-body">
                <div class="row">
                    <div class="col-md-6">
                        <h6><i class="fas fa-microphone-alt"></i> Speech Processing</h6>
                        <ul>
                            <li>High-quality transcription using Whisper</li>
                            <li>Automatic speaker identification</li>
                            <li>Intelligent transcript chunking</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6><i class="fas fa-exclamation-triangle"></i> Risk Detection</h6>
                        <ul>
                            <li><span class="risk-high">HIGH RISK</span>: Depression, Anxiety, Loneliness, Physical Pain</li>
                            <li><span class="risk-moderate">MODERATE RISK</span>: Stress, Anger, Confusion, Shame/Guilt</li>
                            <li>Ignores low-risk emotions (happiness, calm, etc.)</li>
                        </ul>
                    </div>
                </div>
                <div class="row mt-3">
                    <div class="col-md-6">
                        <h6><i class="fas fa-chart-line"></i> Risk Visualizations</h6>
                        <ul>
                            <li>Risk level distribution charts</li>
                            <li>Risk timeline with timestamps</li>
                            <li>Speaker risk comparison</li>
                        </ul>
                    </div>
                    <div class="col-md-6">
                        <h6><i class="fas fa-download"></i> Export Options</h6>
                        <ul>
                            <li>Complete JSON results with risk chunks</li>
                            <li>Detailed risk analysis report</li>
                            <li>Processing statistics</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const formData = new FormData();
    const fileInput = document.getElementById('audioFile');
    const file = fileInput.files[0];
    const chunkSize = document.getElementById('chunkSize').value;
    const minChunkSize = document.getElementById('minChunkSize').value;
    
    if (!file) {
        alert('Please select a file');
        return;
    }
    
    formData.append('file', file);
    formData.append('chunk_size', chunkSize);
    formData.append('min_chunk_size', minChunkSize);
    
    // Show upload status
    document.getElementById('uploadStatus').style.display = 'block';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert('Error: ' + data.error);
            document.getElementById('uploadStatus').style.display = 'none';
        } else {
            // Redirect to processing page
            window.location.href = '/processing/' + data.session_id;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Upload failed');
        document.getElementById('uploadStatus').style.display = 'none';
    });
});
</script>
{% endblock %}"""
    
    with open('templates/risk_index.html', 'w') as f:
        f.write(index_template)
    
    # Create risk-focused processing template
    processing_template = """{% extends "risk_base.html" %}

{% block title %}Processing - Risk-Focused Analysis{% endblock %}

{% block content %}
<div class="row">
    <div class="col-lg-8 offset-lg-2">
        <div class="card">
            <div class="card-header bg-danger text-white">
                <h3><i class="fas fa-cog fa-spin"></i> Processing Audio for Risk Analysis</h3>
            </div>
            <div class="card-body">
                <div id="progressContainer">
                    <div class="progress mb-3">
                        <div id="progressBar" class="progress-bar progress-bar-striped progress-bar-animated bg-danger" 
                             role="progressbar" style="width: 0%">0%</div>
                    </div>
                    <p id="statusMessage" class="text-muted">Initializing...</p>
                    
                    <div class="debug-section">
                        <h6><i class="fas fa-info-circle"></i> Processing Details</h6>
                        <div id="debugInfo">
                            <p><strong>File:</strong> <span id="fileName">Loading...</span></p>
                            <p><strong>Status:</strong> <span id="currentStatus">Starting</span></p>
                            <p><strong>Elapsed Time:</strong> <span id="elapsedTime">0s</span></p>
                            <p><strong>Chunk Size:</strong> <span id="chunkSize">50 words</span></p>
                        </div>
                    </div>
                    
                    <div class="alert alert-danger">
                        <h6><i class="fas fa-exclamation-triangle"></i> Risk Analysis Steps</h6>
                        <ul class="mb-0">
                            <li>Loading and preprocessing audio</li>
                            <li>Speech transcription with Whisper</li>
                            <li>Speaker identification and diarization</li>
                            <li>Chunking transcript by word count</li>
                            <li>Analyzing each chunk for HIGH/MODERATE risk</li>
                            <li>Generating risk-focused visualizations</li>
                        </ul>
                    </div>
                </div>
                
                <div id="errorContainer" style="display: none;">
                    <div class="alert alert-danger">
                        <h5><i class="fas fa-exclamation-triangle"></i> Processing Error</h5>
                        <p id="errorMessage"></p>
                        <a href="/" class="btn btn-primary">Try Again</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
const sessionId = '{{ session_id }}';
let startTime = Date.now();

function updateProgress() {
    fetch('/progress/' + sessionId)
    .then(response => response.json())
    .then(data => {
        const progressBar = document.getElementById('progressBar');
        const statusMessage = document.getElementById('statusMessage');
        const currentStatus = document.getElementById('currentStatus');
        const fileName = document.getElementById('fileName');
        const elapsedTime = document.getElementById('elapsedTime');
        const chunkSize = document.getElementById('chunkSize');
        
        // Update progress bar
        progressBar.style.width = data.progress + '%';
        progressBar.textContent = data.progress + '%';
        
        // Update status
        statusMessage.textContent = data.message;
        currentStatus.textContent = data.status;
        fileName.textContent = data.filename || 'Unknown';
        if (data.chunk_size) {
            chunkSize.textContent = data.chunk_size + ' words';
        }
        
        // Update elapsed time
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        elapsedTime.textContent = elapsed + 's';
        
        if (data.status === 'completed') {
            // Redirect to results
            window.location.href = '/results/' + sessionId;
        } else if (data.status === 'error') {
            // Show error
            document.getElementById('progressContainer').style.display = 'none';
            document.getElementById('errorContainer').style.display = 'block';
            document.getElementById('errorMessage').textContent = data.error;
        } else {
            // Continue polling
            setTimeout(updateProgress, 2000);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        setTimeout(updateProgress, 5000); // Retry after longer delay
    });
}

// Start polling
updateProgress();
</script>
{% endblock %}"""
    
    with open('templates/risk_processing.html', 'w') as f:
        f.write(processing_template)
    
    # Create risk-focused results template
    results_template = """{% extends "risk_base.html" %}

{% block title %}Results - Risk-Focused Analysis{% endblock %}

{% block content %}
<div class="row">
    <div class="col-12">
        <div class="card">
            <div class="card-header d-flex justify-content-between align-items-center bg-danger text-white">
                <h3><i class="fas fa-exclamation-triangle"></i> Risk Analysis Results</h3>
                <div>
                    <a href="/download/{{ session_id }}" class="btn btn-success">
                        <i class="fas fa-download"></i> Download Results
                    </a>
                    <a href="/" class="btn btn-light">
                        <i class="fas fa-plus"></i> Analyze Another File
                    </a>
                </div>
            </div>
            <div class="card-body">
                <!-- Summary Section -->
                <div class="row mb-4">
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h5><i class="fas fa-clock"></i> Duration</h5>
                                <h3>{{ "%.1f"|format(data.results.audio_duration) }}s</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-light">
                            <div class="card-body text-center">
                                <h5><i class="fas fa-file-alt"></i> Total Chunks</h5>
                                <h3>{{ data.results.total_chunks }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-danger text-white">
                            <div class="card-body text-center">
                                <h5><i class="fas fa-exclamation-triangle"></i> High Risk</h5>
                                <h3>{{ data.results.high_risk_chunks|length }}</h3>
                            </div>
                        </div>
                    </div>
                    <div class="col-md-3">
                        <div class="card bg-warning text-white">
                            <div class="card-body text-center">
                                <h5><i class="fas fa-exclamation-circle"></i> Moderate Risk</h5>
                                <h3>{{ data.results.moderate_risk_chunks|length }}</h3>
                            </div>
                        </div>
                    </div>
                </div>
                
                <!-- Visualizations -->
                <div class="row">
                    {% if data.plots.risk_distribution %}
                    <div class="col-lg-6 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-chart-pie"></i> Risk Distribution</h5>
                            </div>
                            <div class="card-body">
                                <div id="riskDistribution"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if data.plots.emotion_distribution %}
                    <div class="col-lg-6 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-chart-bar"></i> Emotion Distribution</h5>
                            </div>
                            <div class="card-body">
                                <div id="emotionDistribution"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if data.plots.risk_timeline %}
                    <div class="col-12 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-timeline"></i> Risk Timeline</h5>
                            </div>
                            <div class="card-body">
                                <div id="riskTimeline"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if data.plots.speaker_risk %}
                    <div class="col-lg-6 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-users"></i> Speaker Risk Summary</h5>
                            </div>
                            <div class="card-body">
                                <div id="speakerRisk"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                    
                    {% if data.plots.confidence_dist %}
                    <div class="col-lg-6 mb-4">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-chart-area"></i> Confidence Distribution</h5>
                            </div>
                            <div class="card-body">
                                <div id="confidenceDist"></div>
                            </div>
                        </div>
                    </div>
                    {% endif %}
                </div>
                
                <!-- High Risk Chunks -->
                {% if data.results.high_risk_chunks %}
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-danger text-white">
                                <h5><i class="fas fa-exclamation-triangle"></i> HIGH RISK CHUNKS</h5>
                            </div>
                            <div class="card-body">
                                {% for chunk in data.results.high_risk_chunks %}
                                <div class="risk-chunk-card risk-chunk-high">
                                    <div class="card">
                                        <div class="card-header d-flex justify-content-between align-items-center">
                                            <div>
                                                <strong>{{ chunk.speaker }}</strong>
                                                <span class="badge bg-danger">{{ chunk.emotion }}</span>
                                                <span class="badge bg-secondary">{{ chunk.word_count }} words</span>
                                            </div>
                                            <div>
                                                <small class="text-muted">{{ chunk.start_time:.1f}s - {{ chunk.end_time:.1f}s }}</small>
                                            </div>
                                        </div>
                                        <div class="card-body">
                                            <div class="confidence-bar mb-2">
                                                <div class="confidence-fill confidence-high" style="width: {{ chunk.confidence * 100 }}%"></div>
                                            </div>
                                            <p class="mb-1"><strong>Confidence:</strong> {{ "%.1f"|format(chunk.confidence * 100) }}%</p>
                                            <div class="chunk-text">{{ chunk.text }}</div>
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                <!-- Moderate Risk Chunks -->
                {% if data.results.moderate_risk_chunks %}
                <div class="row">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header bg-warning text-white">
                                <h5><i class="fas fa-exclamation-circle"></i> MODERATE RISK CHUNKS</h5>
                            </div>
                            <div class="card-body">
                                {% for chunk in data.results.moderate_risk_chunks %}
                                <div class="risk-chunk-card risk-chunk-moderate">
                                    <div class="card">
                                        <div class="card-header d-flex justify-content-between align-items-center">
                                            <div>
                                                <strong>{{ chunk.speaker }}</strong>
                                                <span class="badge bg-warning">{{ chunk.emotion }}</span>
                                                <span class="badge bg-secondary">{{ chunk.word_count }} words</span>
                                            </div>
                                            <div>
                                                <small class="text-muted">{{ chunk.start_time:.1f}s - {{ chunk.end_time:.1f}s }}</small>
                                            </div>
                                        </div>
                                        <div class="card-body">
                                            <div class="confidence-bar mb-2">
                                                <div class="confidence-fill confidence-moderate" style="width: {{ chunk.confidence * 100 }}%"></div>
                                            </div>
                                            <p class="mb-1"><strong>Confidence:</strong> {{ "%.1f"|format(chunk.confidence * 100) }}%</p>
                                            <div class="chunk-text">{{ chunk.text }}</div>
                                        </div>
                                    </div>
                                </div>
                                {% endfor %}
                            </div>
                        </div>
                    </div>
                </div>
                {% endif %}
                
                <!-- Risk Summary -->
                <div class="row mt-4">
                    <div class="col-12">
                        <div class="card">
                            <div class="card-header">
                                <h5><i class="fas fa-chart-line"></i> Risk Analysis Summary</h5>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-4">
                                        <strong>Overall Risk Level:</strong> 
                                        <span class="risk-{{ data.results.risk_summary.overall_risk_level.lower() }}">
                                            {{ data.results.risk_summary.overall_risk_level }}
                                        </span>
                                    </div>
                                    <div class="col-md-4">
                                        <strong>High Risk Percentage:</strong> {{ "%.1f"|format(data.results.risk_summary.high_risk_percentage) }}%
                                    </div>
                                    <div class="col-md-4">
                                        <strong>Moderate Risk Percentage:</strong> {{ "%.1f"|format(data.results.risk_summary.moderate_risk_percentage) }}%
                                    </div>
                                </div>
                                <div class="row mt-2">
                                    <div class="col-md-6">
                                        <strong>Processing Time:</strong> {{ "%.2f"|format(data.processing_time) }}s
                                    </div>
                                    <div class="col-md-6">
                                        <strong>Audio File:</strong> {{ data.filename }}
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block scripts %}
<script>
// Render plots
{% if data.plots.risk_distribution %}
Plotly.newPlot('riskDistribution', {{ data.plots.risk_distribution|safe }});
{% endif %}

{% if data.plots.emotion_distribution %}
Plotly.newPlot('emotionDistribution', {{ data.plots.emotion_distribution|safe }});
{% endif %}

{% if data.plots.risk_timeline %}
Plotly.newPlot('riskTimeline', {{ data.plots.risk_timeline|safe }});
{% endif %}

{% if data.plots.speaker_risk %}
Plotly.newPlot('speakerRisk', {{ data.plots.speaker_risk|safe }});
{% endif %}

{% if data.plots.confidence_dist %}
Plotly.newPlot('confidenceDist', {{ data.plots.confidence_dist|safe }});
{% endif %}
</script>
{% endblock %}"""
    
    with open('templates/risk_results.html', 'w') as f:
        f.write(results_template)
    
    print("Risk-focused web application starting...")
    print("Open http://localhost:5000 in your browser")
    print("Templates created successfully!")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
