#!/usr/bin/env python3
"""
SageMaker Studio - Direct Streamlit Access
No port forwarding needed - uses direct localhost access
"""

import subprocess
import sys
import os
import time
import threading

def start_streamlit_sagemaker():
    """Start Streamlit optimized for SageMaker Studio"""
    
    print("🚀 Starting Risk Audio Analysis Streamlit app...")
    print("📱 SageMaker Studio Access Method:")
    print("   1. Wait for 'Server started' message below")
    print("   2. Open a new browser tab")
    print("   3. Go to: http://localhost:8080")
    print("\n🛑 To stop: Press Ctrl+C")
    
    # Set environment variables
    os.environ['STREAMLIT_SERVER_PORT'] = '8080'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    # Start Streamlit
    try:
        subprocess.run([
            sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
            '--server.port', '8080',
            '--server.address', '0.0.0.0',
            '--server.headless', 'true',
            '--browser.gatherUsageStats', 'false',
            '--server.enableCORS', 'false',
            '--server.enableXsrfProtection', 'false'
        ])
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")

if __name__ == "__main__":
    start_streamlit_sagemaker()
