#!/usr/bin/env python3
"""
SageMaker Studio - Direct Streamlit Access
Uses JupyterLab's proxy system for direct access
"""

import subprocess
import sys
import os
import time
import threading
from IPython.display import display, HTML, clear_output

def start_streamlit_proxy():
    """Start Streamlit with JupyterLab proxy settings"""
    
    # Set environment variables for JupyterLab proxy
    os.environ['STREAMLIT_SERVER_PORT'] = '8080'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    print("🚀 Starting Risk Audio Analysis Streamlit app...")
    print("📱 SageMaker Studio Access Method:")
    print("   1. Wait for 'Server started' message below")
    print("   2. Click the link that appears")
    print("   3. Or open: http://localhost:8080")
    print("\n🛑 To stop: Interrupt this cell (Ctrl+C)")
    
    # Start Streamlit in background
    def run_streamlit():
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
        except Exception as e:
            print(f"Error: {e}")
    
    # Start in background thread
    streamlit_thread = threading.Thread(target=run_streamlit, daemon=True)
    streamlit_thread.start()
    
    # Wait a moment for server to start
    time.sleep(3)
    
    # Display clickable link
    display(HTML("""
    <div style="padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; border-radius: 15px; text-align: center; margin: 20px 0;">
        <h2>🎤 Risk Audio Analysis - Streamlit App</h2>
        <p style="font-size: 18px; margin: 20px 0;">
            <strong>✅ Server Started!</strong>
        </p>
        <div style="background: rgba(255,255,255,0.2); padding: 15px; border-radius: 10px; margin: 20px 0;">
            <h3>🔗 Click to Access:</h3>
            <p style="font-size: 20px;">
                <a href="http://localhost:8080" target="_blank" 
                   style="color: #fff; text-decoration: none; font-weight: bold; 
                          background: rgba(255,255,255,0.3); padding: 10px 20px; 
                          border-radius: 5px; display: inline-block;">
                    🚀 Open Risk Audio Analysis App
                </a>
            </p>
            <p style="margin-top: 15px;">
                <strong>Alternative:</strong> <a href="http://127.0.0.1:8080" target="_blank" 
                                                style="color: #fff;">http://127.0.0.1:8080</a>
            </p>
        </div>
        <p style="font-size: 14px; opacity: 0.8;">
            💡 If the link doesn't work, try opening a new browser tab and typing: <strong>http://localhost:8080</strong>
        </p>
    </div>
    """))
    
    # Keep the cell running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Streamlit server stopped")
        clear_output()

# Run the function
start_streamlit_proxy()
