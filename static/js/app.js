// Audio Analysis App - JavaScript
let selectedFile = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const audioFileInput = document.getElementById('audioFile');
const selectedFileDiv = document.getElementById('selectedFile');
const analyzeBtn = document.getElementById('analyzeBtn');
const progressSection = document.getElementById('progressSection');
const progressBar = document.getElementById('progressBar');
const progressText = document.getElementById('progressText');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');

// File Upload Handling
uploadArea.addEventListener('click', () => {
    audioFileInput.click();
});

audioFileInput.addEventListener('change', (e) => {
    handleFileSelect(e.target.files[0]);
});

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('dragover');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('dragover');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('dragover');
    if (e.dataTransfer.files.length) {
        handleFileSelect(e.dataTransfer.files[0]);
    }
});

function handleFileSelect(file) {
    if (!file) return;
    
    const validTypes = ['audio/wav', 'audio/mp3', 'audio/mpeg', 'audio/x-m4a', 'audio/flac'];
    const validExtensions = ['.wav', '.mp3', '.m4a', '.flac'];
    
    const hasValidExtension = validExtensions.some(ext => file.name.toLowerCase().endsWith(ext));
    
    if (!hasValidExtension) {
        alert('Please select a valid audio file (WAV, MP3, M4A, or FLAC)');
        return;
    }
    
    if (file.size > 100 * 1024 * 1024) {
        alert('File size must be less than 100MB');
        return;
    }
    
    selectedFile = file;
    selectedFileDiv.innerHTML = `<strong>Selected:</strong> ${file.name} (${formatFileSize(file.size)})`;
    selectedFileDiv.style.display = 'block';
    analyzeBtn.disabled = false;
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

// Analyze Audio
analyzeBtn.addEventListener('click', async () => {
    if (!selectedFile) return;
    
    // Hide previous results/errors
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
    
    // Show progress
    progressSection.style.display = 'block';
    progressBar.style.width = '0%';
    progressText.textContent = 'Uploading file...';
    analyzeBtn.disabled = true;
    
    try {
        // Create FormData
        const formData = new FormData();
        formData.append('audio', selectedFile);
        
        // Simulate progress for upload
        progressBar.style.width = '30%';
        progressText.textContent = 'Processing audio...';
        
        // Send to API
        const response = await fetch('/api/analyze', {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.error || 'Processing failed');
        }
        
        progressBar.style.width = '100%';
        progressText.textContent = 'Analysis complete!';
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.result);
        } else {
            throw new Error('Processing failed');
        }
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message);
    } finally {
        setTimeout(() => {
            progressSection.style.display = 'none';
            analyzeBtn.disabled = false;
        }, 1000);
    }
});

function displayResults(result) {
    // Update summary cards
    document.getElementById('duration').textContent = `${result.audio_duration.toFixed(1)}s`;
    document.getElementById('totalChunks').textContent = result.total_chunks;
    document.getElementById('highRisk').textContent = result.high_risk_chunks.length;
    document.getElementById('moderateRisk').textContent = result.moderate_risk_chunks.length;
    
    const overallRiskEl = document.getElementById('overallRisk');
    overallRiskEl.textContent = result.risk_summary.overall_risk_level;
    overallRiskEl.className = 'metric';
    
    if (result.risk_summary.overall_risk_level === 'HIGH') {
        overallRiskEl.classList.add('risk-high');
    } else if (result.risk_summary.overall_risk_level === 'MODERATE') {
        overallRiskEl.classList.add('risk-moderate');
    } else {
        overallRiskEl.classList.add('risk-low');
    }
    
    // Display transcript
    const transcriptEl = document.getElementById('transcript');
    transcriptEl.textContent = result.raw_transcription || 'No transcription available';
    
    // Display risk table
    const tableBody = document.getElementById('riskTableBody');
    tableBody.innerHTML = '';
    
    const allRiskChunks = [
        ...result.high_risk_chunks,
        ...result.moderate_risk_chunks,
        ...result.low_risk_chunks
    ];
    
    allRiskChunks.forEach(chunk => {
        const row = document.createElement('tr');
        
        const riskClass = chunk.risk_level === 'HIGH' ? 'badge-high' : 
                         chunk.risk_level === 'MODERATE' ? 'badge-moderate' : 'badge-low';
        
        row.innerHTML = `
            <td>${chunk.speaker || 'Unknown'}</td>
            <td>${chunk.text ? chunk.text.substring(0, 100) + (chunk.text.length > 100 ? '...' : '') : 'N/A'}</td>
            <td>${chunk.emotion || 'Unknown'}</td>
            <td><span class="badge ${riskClass}">${chunk.risk_level || 'UNKNOWN'}</span></td>
            <td>${chunk.confidence ? (chunk.confidence * 100).toFixed(1) + '%' : 'N/A'}</td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Show results
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

function showError(message) {
    errorMessage.textContent = message;
    errorSection.style.display = 'block';
    errorSection.scrollIntoView({ behavior: 'smooth' });
}

// Check API status on load
window.addEventListener('load', async () => {
    try {
        const response = await fetch('/api/status');
        if (!response.ok) {
            console.warn('API might not be available');
        }
    } catch (error) {
        console.warn('Could not connect to API:', error);
    }
});

