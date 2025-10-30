/**
 * Chest X-ray AI Web Application
 * Modern JavaScript application for medical image analysis
 */

class ChestXrayApp {
    constructor() {
        this.currentFile = null;
        this.currentResults = null;
        this.currentChart = null;
        this.diseaseInfo = {};
        
        this.init();
    }
    
    init() {
        this.setupEventListeners();
        this.checkApplicationStatus();
        this.loadDiseaseInfo();
        this.showSection('home');
    }
    
    setupEventListeners() {
        // Navigation
        document.querySelectorAll('.nav-link').forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const section = link.dataset.section;
                this.showSection(section);
                this.updateNavigation(section);
            });
        });
        
        // File upload
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        
        uploadArea.addEventListener('click', () => {
            fileInput.click();
        });
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelect(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                this.handleFileSelect(e.target.files[0]);
            }
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey && e.key === 'u') {
                e.preventDefault();
                this.showSection('analyze');
            }
        });
    }
    
    showSection(sectionId) {
        // Hide all sections
        document.querySelectorAll('.section').forEach(section => {
            section.classList.remove('active');
        });
        
        // Show target section
        const targetSection = document.getElementById(sectionId);
        if (targetSection) {
            targetSection.classList.add('active');
        }
    }
    
    updateNavigation(activeSection) {
        document.querySelectorAll('.nav-link').forEach(link => {
            link.classList.remove('active');
        });
        
        const activeLink = document.querySelector(`[data-section="${activeSection}"]`);
        if (activeLink) {
            activeLink.classList.add('active');
        }
    }
    
    async checkApplicationStatus() {
        try {
            const response = await fetch('/api/status');
            const data = await response.json();
            
            const statusDot = document.querySelector('.status-dot');
            const statusText = document.querySelector('.status-text');
            
            if (statusDot && statusText) {
                statusDot.classList.add('online');
                statusText.textContent = data.model_status;
            }
            
        } catch (error) {
            console.error('Error checking status:', error);
            this.showToast('Connection Error', 'Unable to connect to the server', 'error');
        }
    }
    
    async loadDiseaseInfo() {
        try {
            const response = await fetch('/api/disease-info');
            this.diseaseInfo = await response.json();
        } catch (error) {
            console.error('Error loading disease info:', error);
        }
    }
    
    handleFileSelect(file) {
        // Validate file
        if (!this.validateFile(file)) {
            return;
        }
        
        this.currentFile = file;
        this.previewImage(file);
        
        // Enable analyze button
        const analyzeBtn = document.getElementById('analyzeBtn');
        analyzeBtn.disabled = false;
    }
    
    validateFile(file) {
        const allowedTypes = ['image/png', 'image/jpeg', 'image/jpg'];
        const maxSize = 16 * 1024 * 1024; // 16MB
        
        if (!allowedTypes.includes(file.type)) {
            this.showToast('Invalid File Type', 'Please upload PNG, JPG, or JPEG files only', 'error');
            return false;
        }
        
        if (file.size > maxSize) {
            this.showToast('File Too Large', 'Please upload files smaller than 16MB', 'error');
            return false;
        }
        
        return true;
    }
    
    previewImage(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const uploadArea = document.getElementById('uploadArea');
            const imagePreview = document.getElementById('imagePreview');
            const previewImg = document.getElementById('previewImg');
            const fileName = document.getElementById('fileName');
            const fileSize = document.getElementById('fileSize');
            
            // Hide upload area and show preview
            uploadArea.style.display = 'none';
            imagePreview.style.display = 'block';
            
            // Set image and info
            previewImg.src = e.target.result;
            fileName.textContent = file.name;
            fileSize.textContent = this.formatFileSize(file.size);
        };
        reader.readAsDataURL(file);
    }
    
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    resetUpload() {
        this.currentFile = null;
        
        // Reset UI
        document.getElementById('uploadArea').style.display = 'block';
        document.getElementById('imagePreview').style.display = 'none';
        document.getElementById('fileInput').value = '';
        document.getElementById('analyzeBtn').disabled = true;
        
        // Hide results if showing
        document.getElementById('resultsContainer').style.display = 'none';
    }
    
    async analyzeImage() {
        if (!this.currentFile) {
            this.showToast('No File Selected', 'Please select an image first', 'error');
            return;
        }
        
        // Show loading
        this.showLoading();
        
        // Simulate progress
        this.simulateProgress();
        
        try {
            const formData = new FormData();
            formData.append('file', this.currentFile);
            
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });
            
            console.log('Response status:', response.status);
            console.log('Response headers:', response.headers);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            const data = await response.json();
            console.log('Response data:', data);
            
            if (data.success) {
                this.currentResults = data;
                this.showResults(data);
                this.showToast('Analysis Complete', 'X-ray analysis completed successfully', 'success');
            } else {
                console.error('Analysis failed:', data.error);
                this.showToast('Analysis Failed', data.error || 'Unknown error occurred', 'error');
            }
            
        } catch (error) {
            console.error('Error analyzing image:', error);
            const errorMessage = error.message || 'Unable to analyze image. Please try again.';
            this.showToast('Network Error', errorMessage, 'error');
        } finally {
            this.hideLoading();
        }
    }
    
    showLoading() {
        document.getElementById('loadingContainer').style.display = 'block';
        document.getElementById('resultsContainer').style.display = 'none';
    }
    
    hideLoading() {
        document.getElementById('loadingContainer').style.display = 'none';
    }
    
    simulateProgress() {
        const progressFill = document.getElementById('progressFill');
        const progressText = document.getElementById('progressText');
        let progress = 0;
        
        const interval = setInterval(() => {
            progress += Math.random() * 15;
            if (progress > 90) progress = 90;
            
            progressFill.style.width = progress + '%';
            progressText.textContent = Math.round(progress) + '%';
            
            if (progress >= 90) {
                clearInterval(interval);
            }
        }, 200);
        
        // Complete progress after request finishes
        setTimeout(() => {
            clearInterval(interval);
            progressFill.style.width = '100%';
            progressText.textContent = '100%';
        }, 3000);
    }
    
    showResults(data) {
        // Hide loading and show results
        this.hideLoading();
        document.getElementById('resultsContainer').style.display = 'block';
        
        // Update metadata with enhanced information
        document.getElementById('analysisTime').textContent = new Date(data.timestamp).toLocaleString();
        const modelInfo = `${data.model_type}${data.summary ? ` | ${data.summary.total_diseases_checked} diseases analyzed` : ''}`;
        document.getElementById('modelType').textContent = modelInfo;
        
        // Show enhanced summary if available
        if (data.summary) {
            this.displayEnhancedSummary(data.summary);
        }
        
        // Show key findings
        this.displayFindings(data.positive_findings);
        
        // Show detailed results
        this.displayDetailedResults(data.predictions);
        
        // Create summary chart
        this.createSummaryChart(data.predictions);
        
        // Scroll to results
        document.getElementById('resultsContainer').scrollIntoView({ 
            behavior: 'smooth',
            block: 'start'
        });
    }
    
    displayEnhancedSummary(summary) {
        // Add a summary section if it doesn't exist
        let summaryDiv = document.getElementById('enhancedSummary');
        if (!summaryDiv) {
            summaryDiv = document.createElement('div');
            summaryDiv.id = 'enhancedSummary';
            summaryDiv.className = 'enhanced-summary';
            
            // Insert before results summary
            const resultsSection = document.querySelector('.results-summary');
            if (resultsSection && resultsSection.parentNode) {
                resultsSection.parentNode.insertBefore(summaryDiv, resultsSection);
            } else {
                // Fallback: insert before detailed results
                const detailedResults = document.querySelector('.detailed-results');
                if (detailedResults && detailedResults.parentNode) {
                    detailedResults.parentNode.insertBefore(summaryDiv, detailedResults);
                } else {
                    // Last resort: append to results container
                    const resultsContainer = document.getElementById('resultsContainer');
                    if (resultsContainer) {
                        resultsContainer.appendChild(summaryDiv);
                    }
                }
            }
        }
        
        const confidenceColor = summary.model_confidence === 'High' ? '#22c55e' : 
                                summary.model_confidence === 'Moderate' ? '#f59e0b' : '#ef4444';
        
        summaryDiv.innerHTML = `
            <div class="summary-grid">
                <div class="summary-item">
                    <div class="summary-label">Analysis Confidence</div>
                    <div class="summary-value" style="color: ${confidenceColor}">
                        <i class="fas fa-brain"></i>
                        ${summary.model_confidence}
                    </div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Positive Findings</div>
                    <div class="summary-value">
                        <i class="fas fa-search"></i>
                        ${summary.positive_findings_count} / ${summary.total_diseases_checked}
                    </div>
                </div>
                <div class="summary-item">
                    <div class="summary-label">Optimal Thresholds</div>
                    <div class="summary-value" style="color: #22c55e">
                        <i class="fas fa-check-circle"></i>
                        ${summary.optimal_thresholds_used ? 'Active' : 'Inactive'}
                    </div>
                </div>
            </div>
        `;
    }
    
    displayFindings(findings) {
        const findingsList = document.getElementById('findingsList');
        if (!findingsList) {
            console.error('findingsList element not found');
            return;
        }
        
        findingsList.innerHTML = '';
        
        if (findings.length === 0 || (findings.length === 1 && findings[0] === 'No Finding')) {
            const item = document.createElement('div');
            item.className = 'finding-item negative';
            item.innerHTML = `
                <i class="fas fa-check-circle finding-icon" style="color: var(--success-color);"></i>
                <span class="finding-text">No significant findings detected</span>
                <span class="finding-confidence">Normal</span>
            `;
            findingsList.appendChild(item);
            return;
        }
        
        findings.forEach(finding => {
            if (finding === 'No Finding') return;
            
            const diseaseInfo = this.diseaseInfo[finding] || {};
            const severity = diseaseInfo.severity || 'Moderate';
            
            const item = document.createElement('div');
            item.className = `finding-item ${severity === 'High' ? 'high-risk' : ''}`;
            item.innerHTML = `
                <i class="fas fa-exclamation-triangle finding-icon" style="color: ${diseaseInfo.color || '#ef4444'};"></i>
                <span class="finding-text">${finding.replace('_', ' ')}</span>
                <span class="finding-confidence">${severity} Risk</span>
            `;
            
            item.addEventListener('click', () => {
                this.showDiseaseInfo(finding);
            });
            
            findingsList.appendChild(item);
        });
    }
    
    displayDetailedResults(predictions) {
        const resultsGrid = document.getElementById('resultsGrid');
        if (!resultsGrid) {
            console.error('resultsGrid element not found');
            return;
        }
        
        resultsGrid.innerHTML = '';
        
        Object.entries(predictions).forEach(([disease, data]) => {
            const card = document.createElement('div');
            card.className = 'result-card';
            
            const probability = data.probability * 100;
            const isPositive = data.prediction === 1;
            const confidence = data.confidence || 'Medium';
            const threshold = data.threshold * 100;
            const modelUsed = data.model_used || 'ensemble';
            
            let probabilityClass = 'low';
            if (probability > 70) probabilityClass = 'high';
            else if (probability > 50) probabilityClass = 'medium';
            
            // Enhanced card with individual model probabilities if available
            let individualModelsInfo = '';
            if (modelUsed === 'arnoweng_only') {
                // For Arnoweng-only diseases, show only Model B
                individualModelsInfo = `
                    <div class="model-breakdown arnoweng-only">
                        <small><strong>⭐ Model B:</strong> ${(data.arnoweng_prob * 100).toFixed(1)}%</small>
                        <small class="model-note">Predicted by Model B only</small>
                    </div>
                `;
            } else if (data.champion_prob && data.arnoweng_prob) {
                // For ensemble diseases, show both models with generic names
                individualModelsInfo = `
                    <div class="model-breakdown">
                        <small>Model A: ${(data.champion_prob * 100).toFixed(1)}% | Model B: ${(data.arnoweng_prob * 100).toFixed(1)}%</small>
                    </div>
                `;
            }
            
            card.innerHTML = `
                <div class="result-header">
                    <div class="result-disease">${disease.replace('_', ' ')}</div>
                    <div class="result-status ${isPositive ? 'positive' : 'negative'}">
                        <i class="fas ${isPositive ? 'fa-exclamation-triangle' : 'fa-check-circle'}"></i>
                        ${isPositive ? 'Detected' : 'Normal'}
                    </div>
                </div>
                <div class="result-probability ${probabilityClass}">
                    ${probability.toFixed(1)}%
                </div>
                <div class="result-details">
                    <div class="confidence-info">
                        <span class="confidence-label">Confidence:</span>
                        <span class="confidence-value ${confidence.toLowerCase().replace(' ', '-')}">${confidence}</span>
                    </div>
                    <div class="threshold-info">
                        <span class="threshold-label">Threshold:</span>
                        <span class="threshold-value">${threshold.toFixed(1)}%</span>
                    </div>
                </div>
                ${individualModelsInfo}
                <div class="confidence-bar">
                    <div class="threshold-marker" style="left: ${threshold}%"></div>
                    <div class="confidence-fill" style="width: ${probability}%; background: ${this.getConfidenceColor(probability)}"></div>
                </div>
            `;
            
            card.addEventListener('click', () => {
                this.showDiseaseInfo(disease);
            });
            
            resultsGrid.appendChild(card);
        });
    }
    
    getConfidenceColor(percentage) {
        if (percentage > 70) return 'var(--error-color)';
        if (percentage > 30) return 'var(--warning-color)';
        return 'var(--success-color)';
    }
    
    createSummaryChart(predictions) {
        const canvas = document.getElementById('summaryChart');
        if (!canvas) {
            console.error('summaryChart canvas element not found');
            return;
        }
        
        try {
            const ctx = canvas.getContext('2d');
            
            // Prepare data for doughnut chart
            const positiveCount = Object.values(predictions).filter(p => p.prediction === 1).length;
            const negativeCount = Object.values(predictions).length - positiveCount;
            
            // Destroy existing chart if it exists
            if (this.currentChart) {
                this.currentChart.destroy();
            }
            
            this.currentChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: ['Normal', 'Detected'],
                    datasets: [{
                        data: [negativeCount, positiveCount],
                        backgroundColor: ['#10b981', '#ef4444'],
                        borderWidth: 0
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            position: 'bottom'
                        },
                        title: {
                            display: true,
                            text: 'Findings Summary'
                        }
                    }
                }
            });
        } catch (error) {
            console.error('Error creating chart:', error);
            // Fallback: show text summary
            const chartContainer = canvas.parentNode;
            if (chartContainer) {
                const positiveCount = Object.values(predictions).filter(p => p.prediction === 1).length;
                const negativeCount = Object.values(predictions).length - positiveCount;
                
                chartContainer.innerHTML = `
                    <div style="text-align: center; padding: 20px;">
                        <h4>Findings Summary</h4>
                        <p>Normal: ${negativeCount} | Detected: ${positiveCount}</p>
                    </div>
                `;
            }
        }
    }
    
    showDiseaseInfo(disease) {
        const info = this.diseaseInfo[disease];
        if (!info) return;
        
        const modal = document.getElementById('diseaseModal');
        const title = document.getElementById('diseaseModalTitle');
        const body = document.getElementById('diseaseModalBody');
        
        title.textContent = disease.replace('_', ' ');
        body.innerHTML = `
            <div style="margin-bottom: 1rem;">
                <h4>Description</h4>
                <p>${info.description}</p>
            </div>
            <div style="margin-bottom: 1rem;">
                <h4>Severity Level</h4>
                <span style="background: ${info.color}; color: white; padding: 0.25rem 0.75rem; border-radius: 1rem; font-size: 0.875rem;">
                    ${info.severity}
                </span>
            </div>
            <div class="disclaimer" style="background: #fef3c7; padding: 1rem; border-radius: 0.5rem; border-left: 4px solid #f59e0b;">
                <strong>Medical Disclaimer:</strong> This information is for educational purposes only. 
                Always consult with qualified healthcare professionals for medical advice.
            </div>
        `;
        
        this.showModal('diseaseModal');
    }
    
    showModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.add('active');
        
        // Close modal when clicking outside
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                this.closeModal(modalId);
            }
        });
    }
    
    closeModal(modalId) {
        const modal = document.getElementById(modalId);
        modal.classList.remove('active');
    }
    
    downloadReport() {
        if (!this.currentResults) {
            this.showToast('No Results', 'No analysis results to download', 'error');
            return;
        }
        
        const reportData = {
            timestamp: new Date().toISOString(),
            filename: this.currentFile?.name || 'Unknown',
            analysis: this.currentResults,
            generated_by: 'Chest X-ray AI v1.0'
        };
        
        const blob = new Blob([JSON.stringify(reportData, null, 2)], {
            type: 'application/json'
        });
        
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `xray_analysis_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        this.showToast('Download Complete', 'Analysis report downloaded successfully', 'success');
    }
    
    shareResults() {
        if (!this.currentResults) {
            this.showToast('No Results', 'No analysis results to share', 'error');
            return;
        }
        
        if (navigator.share) {
            navigator.share({
                title: 'Chest X-ray Analysis Results',
                text: `Analysis results for ${this.currentFile?.name || 'X-ray image'}`,
                url: window.location.href
            });
        } else {
            // Fallback to copying URL
            navigator.clipboard.writeText(window.location.href).then(() => {
                this.showToast('Link Copied', 'Share link copied to clipboard', 'success');
            });
        }
    }
    
    analyzeAnother() {
        this.resetUpload();
        this.showSection('analyze');
    }
    
    showToast(title, message, type = 'info') {
        const toastContainer = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        toast.innerHTML = `
            <div class="toast-header">
                <div class="toast-title">${title}</div>
                <button class="toast-close">&times;</button>
            </div>
            <div class="toast-message">${message}</div>
        `;
        
        toastContainer.appendChild(toast);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            toast.remove();
        }, 5000);
        
        // Manual close
        toast.querySelector('.toast-close').addEventListener('click', () => {
            toast.remove();
        });
    }
}

// Global functions for HTML onclick handlers
window.showSection = function(sectionId) {
    app.showSection(sectionId);
    app.updateNavigation(sectionId);
};

window.resetUpload = function() {
    app.resetUpload();
};

window.analyzeImage = function() {
    app.analyzeImage();
};

window.downloadReport = function() {
    app.downloadReport();
};

window.shareResults = function() {
    app.shareResults();
};

window.analyzeAnother = function() {
    app.analyzeAnother();
};

window.closeModal = function(modalId) {
    app.closeModal(modalId);
};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.app = new ChestXrayApp();
});

// Service Worker for offline functionality (optional)
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/static/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}