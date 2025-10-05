#!/usr/bin/env python3
"""
Flask-based Audio Analysis API
Simple backend for the static HTML UI
"""

from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from pathlib import Path
import json
import os
import sys
from werkzeug.utils import secure_filename

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from transcription_chunk_risk_pipeline import TranscriptionChunkRiskPipeline

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # Enable CORS for API access

# Configuration
UPLOAD_FOLDER = Path('uploads')
UPLOAD_FOLDER.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'm4a', 'flac'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Initialize pipeline
pipeline = None

def get_pipeline():
    """Lazy load the pipeline"""
    global pipeline
    if pipeline is None:
        pipeline = TranscriptionChunkRiskPipeline()
    return pipeline

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Serve the main UI"""
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_audio():
    """Analyze uploaded audio file"""
    try:
        # Check if file was uploaded
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        file = request.files['audio']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: WAV, MP3, M4A, FLAC'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        # Process the audio
        print(f"Processing: {filepath}")
        pipe = get_pipeline()
        result = pipe.process_audio(str(filepath))
        
        if result:
            # Convert result to dictionary
            result_dict = {
                'audio_file': result.audio_file,
                'audio_duration': result.audio_duration,
                'processing_time': result.processing_time,
                'total_chunks': result.total_chunks,
                'high_risk_chunks': result.high_risk_chunks,
                'moderate_risk_chunks': result.moderate_risk_chunks,
                'low_risk_chunks': result.low_risk_chunks,
                'risk_summary': result.risk_summary,
                'raw_transcription': getattr(result, 'raw_transcription', ''),
                'overall_raw_sentiment': getattr(result, 'overall_raw_sentiment', {}),
            }
            
            return jsonify({
                'success': True,
                'result': result_dict
            })
        else:
            return jsonify({'error': 'Processing failed'}), 500
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/files', methods=['GET'])
def list_files():
    """List available audio files"""
    try:
        files = []
        if UPLOAD_FOLDER.exists():
            for file in UPLOAD_FOLDER.glob('*'):
                if file.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac']:
                    files.append({
                        'name': file.name,
                        'size': file.stat().st_size,
                        'path': str(file.relative_to(Path.cwd()))
                    })
        
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status', methods=['GET'])
def status():
    """Check API status"""
    return jsonify({
        'status': 'running',
        'models_loaded': pipeline is not None
    })

if __name__ == '__main__':
    import socket
    
    # Get hostname and IP
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except:
        local_ip = '127.0.0.1'
    
    port = 5000
    
    print("=" * 70)
    print("🎤 Audio Analysis & Risk Detection System")
    print("=" * 70)
    print("\n✅ Server Starting...\n")
    print("📍 Access the application at one of these URLs:\n")
    print(f"   🌐 Local:     http://localhost:{port}")
    print(f"   🌐 Network:   http://{local_ip}:{port}")
    print(f"   🌐 External:  http://0.0.0.0:{port}")
    print("\n" + "=" * 70)
    print("💡 Tips:")
    print("   • Use 'Local' URL if running on your computer")
    print("   • Use 'Network' URL if running on SageMaker or remote server")
    print("   • Press Ctrl+C to stop the server")
    print("=" * 70 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=True)

