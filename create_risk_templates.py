#!/usr/bin/env python3
"""
Script to create risk-focused HTML templates
"""

import os
from pathlib import Path

def create_risk_templates():
    """Create all risk-focused HTML templates"""
    
    # Create templates directory
    templates_dir = Path('templates')
    templates_dir.mkdir(exist_ok=True)
    
    # Risk-focused base template
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
    
    # Risk-focused index template
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
    
    # Risk-focused processing template
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
    
    # Risk-focused results template (simplified)
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
console.log('Risk analysis results page loaded');
</script>
{% endblock %}"""
    
    # Write templates
    templates = {
        'risk_base.html': base_template,
        'risk_index.html': index_template,
        'risk_processing.html': processing_template,
        'risk_results.html': results_template
    }
    
    for filename, content in templates.items():
        template_path = templates_dir / filename
        with open(template_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Created: {template_path}")
    
    print("All risk-focused templates created successfully!")

if __name__ == "__main__":
    create_risk_templates()
