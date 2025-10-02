#!/usr/bin/env python3
"""
Start Streamlit app for Risk Audio Analysis
SageMaker-compatible startup script
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_streamlit():
    """Check if Streamlit is available"""
    try:
        import streamlit
        logger.info(f"✅ Streamlit {streamlit.__version__} is available")
        return True
    except ImportError:
        logger.error("❌ Streamlit not found")
        return False

def install_streamlit():
    """Install Streamlit if not available"""
    logger.info("🔄 Installing Streamlit...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'streamlit>=1.28.0'])
        logger.info("✅ Streamlit installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install Streamlit: {e}")
        return False

def start_streamlit():
    """Start the Streamlit application"""
    logger.info("🚀 Starting Streamlit app...")
    
    # Set environment variables for SageMaker
    os.environ['STREAMLIT_SERVER_PORT'] = '8080'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    
    # Streamlit command
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
        '--server.port', '8080',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false'
    ]
    
    try:
        logger.info("Starting Streamlit server on port 8080...")
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("👋 Streamlit server stopped")
        return True

def main():
    """Main function"""
    logger.info("=== Risk Audio Analysis - Streamlit Startup ===")
    
    # Check if Streamlit is available
    if not check_streamlit():
        if not install_streamlit():
            logger.error("❌ Cannot start without Streamlit")
            sys.exit(1)
    
    # Start the app
    logger.info("🎉 Starting Risk Audio Analysis Streamlit app...")
    logger.info("📱 Access the app at: http://localhost:8080")
    logger.info("🛑 Press Ctrl+C to stop the server")
    
    start_streamlit()

if __name__ == "__main__":
    main()
