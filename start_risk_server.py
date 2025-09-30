#!/usr/bin/env python3
"""
Startup script for Risk-Focused Audio Analysis Pipeline
Automatically opens browser when server starts
"""

import os
import sys
import time
import webbrowser
import threading
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def open_browser():
    """Open browser after a short delay to ensure server is ready"""
    time.sleep(2)  # Wait 2 seconds for server to start
    try:
        webbrowser.open('http://localhost:5000')
        print("🌐 Browser opened automatically!")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print("Please manually open: http://localhost:5000")

def main():
    print("=" * 60)
    print("RISK-FOCUSED AUDIO ANALYSIS PIPELINE")
    print("=" * 60)
    print()
    print("Starting risk-focused web application...")
    print("🌐 Browser will open automatically in 2 seconds...")
    print("   URL: http://localhost:5000")
    print()
    print("Features:")
    print("- Upload audio/video files for risk analysis")
    print("- High-quality transcription with speaker identification")
    print("- Speaker-based natural chunking")
    print("- HIGH and MODERATE risk detection only")
    print("- Real-time processing progress")
    print("- Downloadable risk analysis results")
    print()
    print("Risk Categories:")
    print("🚨 HIGH RISK: Depression, Anxiety, Loneliness, Physical Pain")
    print("⚠️  MODERATE RISK: Stress, Anger, Confusion, Shame/Guilt")
    print("✅ LOW RISK: Ignored (Happiness, Calm, Excitement, etc.)")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    
    # Start browser opening in background thread
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Import and run web app
    try:
        from risk_web_app import app
        app.run(debug=False, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
