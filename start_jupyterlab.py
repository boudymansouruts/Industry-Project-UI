#!/usr/bin/env python3
"""
SageMaker JupyterLab - Streamlit Startup Guide
Best method for running Risk Audio Analysis in SageMaker Studio
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_environment():
    """Check if we're in SageMaker JupyterLab"""
    if 'SAGEMAKER' in os.environ.get('AWS_DEFAULT_REGION', ''):
        logger.info("✅ Running in SageMaker environment")
        return True
    else:
        logger.info("ℹ️ Running in local environment")
        return False

def install_dependencies():
    """Install required dependencies"""
    logger.info("🔄 Installing dependencies...")
    
    try:
        # Install from requirements.txt
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        return False

def start_streamlit_jupyterlab():
    """Start Streamlit optimized for JupyterLab"""
    logger.info("🚀 Starting Streamlit for JupyterLab...")
    
    # Set environment variables for JupyterLab
    os.environ['STREAMLIT_SERVER_PORT'] = '8080'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    
    # Streamlit command optimized for JupyterLab
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'streamlit_app.py',
        '--server.port', '8080',
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false'
    ]
    
    try:
        logger.info("🎉 Streamlit server starting...")
        logger.info("📱 Access instructions:")
        logger.info("   1. Look for 'Port Forwarding' in JupyterLab left sidebar")
        logger.info("   2. Add port 8080")
        logger.info("   3. Click 'Open in Browser'")
        logger.info("   4. Or try: http://localhost:8080")
        logger.info("🛑 Press Ctrl+C to stop the server")
        
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to start Streamlit: {e}")
        return False
    except KeyboardInterrupt:
        logger.info("👋 Streamlit server stopped")
        return True

def main():
    """Main function for JupyterLab"""
    logger.info("=== Risk Audio Analysis - JupyterLab Startup ===")
    
    # Check environment
    is_sagemaker = check_environment()
    
    # Install dependencies
    if not install_dependencies():
        logger.error("❌ Cannot start without dependencies")
        sys.exit(1)
    
    # Start Streamlit
    logger.info("🎉 Starting Risk Audio Analysis Streamlit app...")
    start_streamlit_jupyterlab()

if __name__ == "__main__":
    main()
