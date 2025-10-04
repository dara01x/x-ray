#!/usr/bin/env python3
"""
Simplified Web App for Testing
Works without PyTorch to demonstrate the interface
"""

import os
import sys
import json
import uuid
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import numpy as np
from PIL import Image

from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for API endpoints
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Disease information for educational purposes
DISEASE_INFO = {
    'Atelectasis': {
        'description': 'Collapse of part or all of a lung',
        'severity': 'Moderate',
        'color': '#FFA726'
    },
    'Cardiomegaly': {
        'description': 'Enlarged heart',
        'severity': 'High',
        'color': '#EF5350'
    },
    'Effusion': {
        'description': 'Fluid around the lung',
        'severity': 'Moderate',
        'color': '#42A5F5'
    },
    'Infiltration': {
        'description': 'Abnormal substance in the lungs',
        'severity': 'Moderate',
        'color': '#AB47BC'
    },
    'Mass': {
        'description': 'Abnormal tissue growth',
        'severity': 'High',
        'color': '#EF5350'
    },
    'Nodule': {
        'description': 'Small round growth in the lung',
        'severity': 'Moderate',
        'color': '#FF7043'
    },
    'Pneumonia': {
        'description': 'Infection causing inflammation',
        'severity': 'High',
        'color': '#EF5350'
    },
    'Pneumothorax': {
        'description': 'Collapsed lung due to air leak',
        'severity': 'High',
        'color': '#EF5350'
    },
    'Consolidation': {
        'description': 'Lung tissue filled with liquid',
        'severity': 'Moderate',
        'color': '#5C6BC0'
    },
    'Edema': {
        'description': 'Fluid in lung tissue',
        'severity': 'High',
        'color': '#EF5350'
    },
    'Emphysema': {
        'description': 'Damaged air sacs in lungs',
        'severity': 'High',
        'color': '#EF5350'
    },
    'Fibrosis': {
        'description': 'Lung scarring',
        'severity': 'High',
        'color': '#EF5350'
    },
    'Pleural_Thickening': {
        'description': 'Thickened lung lining',
        'severity': 'Moderate',
        'color': '#66BB6A'
    },
    'Hernia': {
        'description': 'Organ displacement',
        'severity': 'Moderate',
        'color': '#26A69A'
    }
}


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_model_files():
    """Check if model files are available."""
    kaggle_files = [
        'kaggle_outputs/best_model_all_out_v1.pth',
        'kaggle_outputs/model.pth.tar',
        'kaggle_outputs/optimal_thresholds_ensemble_final_v1.json'
    ]
    
    available_files = []
    for file_path in kaggle_files:
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            available_files.append({
                'name': os.path.basename(file_path),
                'path': file_path,
                'size_mb': round(size_mb, 1)
            })
    
    return available_files


def generate_realistic_predictions():
    """Generate realistic demo predictions based on actual medical statistics."""
    diseases = [
        'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
        'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
        'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia'
    ]
    
    # More realistic probability distributions based on medical literature
    realistic_probs = {
        'Atelectasis': np.random.beta(2, 8),      # Common but usually mild
        'Cardiomegaly': np.random.beta(3, 7),    # Moderately common
        'Effusion': np.random.beta(2, 6),        # Fairly common
        'Infiltration': np.random.beta(2, 5),    # Common
        'Mass': np.random.beta(1, 20),           # Rare but serious
        'Nodule': np.random.beta(2, 10),         # Uncommon
        'Pneumonia': np.random.beta(3, 7),       # Moderately common
        'Pneumothorax': np.random.beta(1, 15),   # Uncommon but serious
        'Consolidation': np.random.beta(2, 8),   # Uncommon
        'Edema': np.random.beta(2, 12),          # Uncommon
        'Emphysema': np.random.beta(2, 10),      # Age-related
        'Fibrosis': np.random.beta(1, 15),       # Rare
        'Pleural_Thickening': np.random.beta(2, 12), # Uncommon
        'Hernia': np.random.beta(1, 20)          # Rare
    }
    
    # Load actual thresholds if available
    thresholds = {}
    threshold_file = 'kaggle_outputs/optimal_thresholds_ensemble_final_v1.json'
    if os.path.exists(threshold_file):
        try:
            with open(threshold_file, 'r') as f:
                thresholds = json.load(f)
        except:
            pass
    
    predictions = {}
    for disease in diseases:
        prob = realistic_probs.get(disease, np.random.beta(2, 8))
        threshold = thresholds.get(disease, 0.5)
        
        predictions[disease] = {
            'probability': float(prob),
            'prediction': 1 if prob > threshold else 0,
            'threshold': threshold,
            'confidence': 'High' if prob > 0.8 or prob < 0.2 else 'Medium'
        }
    
    return predictions


def process_image_demo(image_path: str) -> Dict:
    """Process uploaded image and return demo predictions."""
    try:
        # Validate image
        with Image.open(image_path) as img:
            width, height = img.size
            
        # Generate realistic predictions
        predictions = generate_realistic_predictions()
        
        # Determine positive findings
        positive_findings = [disease for disease, data in predictions.items() if data['prediction'] == 1]
        if not positive_findings:
            positive_findings = ['No Finding']
        
        # Check if actual model files are available
        model_files = check_model_files()
        model_status = f"Demo Mode - {len(model_files)} model files available"
        
        return {
            'success': True,
            'model_type': 'demo',
            'model_status': model_status,
            'predictions': predictions,
            'positive_findings': positive_findings,
            'image_info': {
                'width': width,
                'height': height,
                'size_mb': round(os.path.getsize(image_path) / (1024 * 1024), 2)
            },
            'available_models': model_files
        }
            
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


@app.route('/')
def index():
    """Main page."""
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'success': False, 'error': 'Invalid file type. Please upload PNG, JPG, JPEG, or DCM files.'}), 400
        
        # Generate unique filename
        filename = secure_filename(file.filename or 'unknown.jpg')
        unique_id = str(uuid.uuid4())
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_extension}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        # Save file
        file.save(file_path)
        
        # Validate image
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            os.remove(file_path)
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        
        # Process image (demo mode)
        results = process_image_demo(file_path)
        
        if results['success']:
            # Add file info to results for display
            results['unique_id'] = unique_id
            results['filename'] = filename
            results['timestamp'] = datetime.now().isoformat()
            
            # Keep uploaded file for display
            # os.remove(file_path)  # Comment out to keep files for display
            
            return jsonify(results)
        else:
            # Clean up file on error
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove uploaded file {file_path}: {e}")
            return jsonify(results), 500
            
    except RequestEntityTooLarge:
        return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413
    except Exception as e:
        print(f"Upload error: {e}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': 'Internal server error'}), 500


@app.route('/api/disease-info')
def get_disease_info():
    """Get disease information."""
    return jsonify(DISEASE_INFO)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/api/status')
def get_status():
    """Get application status."""
    model_files = check_model_files()
    
    return jsonify({
        'status': 'running',
        'model_status': f'Demo mode - {len(model_files)} model files detected',
        'available_models': model_files,
        'device': 'CPU (Demo)',
        'version': '1.0.0-demo',
        'features': [
            'File upload and validation',
            'Image preprocessing',
            'Realistic disease predictions',
            'Interactive web interface',
            'Educational disease information'
        ]
    })


@app.route('/api/model-info')
def get_model_info():
    """Get detailed model information."""
    model_files = check_model_files()
    
    total_size = sum(f['size_mb'] for f in model_files)
    
    return jsonify({
        'ensemble_ready': len(model_files) >= 3,
        'model_files': model_files,
        'total_size_mb': round(total_size, 1),
        'pytorch_status': 'Not loaded (Demo mode)',
        'next_steps': [
            'Fix PyTorch installation',
            'Load ensemble models',
            'Enable GPU acceleration',
            'Add Grad-CAM visualization'
        ]
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors."""
    return render_template('index.html')


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500


if __name__ == '__main__':
    print("🔬 Initializing Chest X-ray AI Web Application (Demo Mode)")
    print("=" * 60)
    
    # Check model files
    model_files = check_model_files()
    print(f"📁 Model files detected: {len(model_files)}")
    for file_info in model_files:
        print(f"   ✅ {file_info['name']} ({file_info['size_mb']} MB)")
    
    if len(model_files) >= 3:
        print("🎉 All ensemble model files present!")
        print("💡 PyTorch installation needed to enable full AI functionality")
    else:
        print("⚠️ Some model files missing - running in demo mode")
    
    print("\n🚀 Starting web server...")
    print("📱 Access the application at: http://localhost:5000")
    print("🔍 Upload chest X-ray images to see demo predictions")
    print("=" * 60)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )