#!/usr/bin/env python3
"""
Advanced Chest X-ray AI Web Application
Professional-grade Flask web application for chest X-ray disease classification
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
import cv2
import numpy as np
from PIL import Image
import torch

from flask import Flask, render_template, request, jsonify, send_from_directory, flash, redirect, url_for
from flask_cors import CORS

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import custom modules
try:
    from utils import load_config, get_device
    from models.ensemble_model import load_ensemble_model
    from scripts.inference import ChestXrayPredictor
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available")

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Enable CORS for API endpoints
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'dcm'}

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for models
predictor = None
ensemble_model = None
config = None

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


def load_optimal_thresholds():
    """Load optimal thresholds from CSV or JSON file."""
    # Check multiple possible locations
    threshold_paths = [
        'models/optimal_thresholds_ensemble_final.json',
        'models/optimal_thresholds_ensemble_final.csv',
        'kaggle_outputs/optimal_thresholds_ensemble_final.csv',
        'outputs/optimal_thresholds.json'
    ]
    
    thresholds = {}
    
    for thresholds_path in threshold_paths:
        if os.path.exists(thresholds_path):
            try:
                if thresholds_path.endswith('.json'):
                    with open(thresholds_path, 'r') as f:
                        thresholds = json.load(f)
                    print(f"✅ Loaded thresholds from {thresholds_path}")
                    break
                elif thresholds_path.endswith('.csv'):
                    import pandas as pd
                    df = pd.read_csv(thresholds_path)
                    for _, row in df.iterrows():
                        thresholds[row['Disease']] = float(row['Optimal_Threshold'])
                    print(f"✅ Loaded thresholds from {thresholds_path}")
                    break
            except Exception as e:
                print(f"Error loading thresholds from {thresholds_path}: {e}")
                continue
    
    if not thresholds:
        # Use default thresholds
        diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
                   'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
                   'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
        thresholds = {disease: 0.5 for disease in diseases}
        print("⚠️ Using default thresholds (0.5 for all diseases)")
    
    return thresholds


def initialize_models():
    """Initialize the AI models."""
    global predictor, ensemble_model, config
    
    try:
        # Load configuration
        config_path = 'configs/config.yaml'
        if os.path.exists(config_path):
            config = load_config(config_path)
            print("✅ Configuration loaded")
        
        # Check for model files in the models directory
        champion_checkpoint = 'models/best_model_all_out_v1.pth'
        arnoweng_checkpoint = 'models/model.pth.tar'
        
        # Also check alternative locations
        if not os.path.exists(champion_checkpoint):
            champion_checkpoint = 'outputs/models/best_model.pth'
        if not os.path.exists(arnoweng_checkpoint):
            arnoweng_checkpoint = 'kaggle_outputs/model.pth.tar'
        
        print(f"🔍 Looking for models:")
        print(f"   Champion model: {champion_checkpoint} - {'✅ Found' if os.path.exists(champion_checkpoint) else '❌ Missing'}")
        print(f"   Arnoweng model: {arnoweng_checkpoint} - {'✅ Found' if os.path.exists(arnoweng_checkpoint) else '❌ Missing'}")
        
        # Try to load ensemble model first (best performance)
        if os.path.exists(arnoweng_checkpoint):
            try:
                thresholds = load_optimal_thresholds()
                
                # Save thresholds as JSON for ensemble model
                thresholds_json = 'outputs/optimal_thresholds.json'
                os.makedirs(os.path.dirname(thresholds_json), exist_ok=True)
                with open(thresholds_json, 'w') as f:
                    json.dump(thresholds, f)
                
                if os.path.exists(champion_checkpoint):
                    # Load ensemble model
                    ensemble_model = load_ensemble_model(
                        champion_checkpoint=champion_checkpoint,
                        arnoweng_checkpoint=arnoweng_checkpoint,
                        ensemble_thresholds=thresholds_json
                    )
                    print("🎉 Ensemble model loaded successfully!")
                    print(f"📊 Model info: {ensemble_model.get_model_info()}")
                    return True
                else:
                    print("⚠️ Champion model not found, trying single Arnoweng model...")
            except Exception as e:
                print(f"❌ Failed to load ensemble model: {e}")
                traceback.print_exc()
        
        # Fallback to single champion model
        if config and os.path.exists(champion_checkpoint):
            try:
                thresholds_json = 'outputs/optimal_thresholds.json'
                thresholds_path = thresholds_json if os.path.exists(thresholds_json) else None
                predictor = ChestXrayPredictor(
                    config_path=config_path,
                    checkpoint_path=champion_checkpoint,
                    thresholds_path=thresholds_path
                )
                print("✅ Single champion model loaded successfully!")
                return True
            except Exception as e:
                print(f"❌ Failed to load champion model: {e}")
                traceback.print_exc()
        
        print("⚠️ No models available. Running in demo mode.")
        print("📋 To use real models, ensure these files exist:")
        print("   - models/best_model_all_out_v1.pth (Champion model)")
        print("   - models/model.pth.tar (Arnoweng model)")
        print("   - models/optimal_thresholds_ensemble_final.csv or .json")
        return False
        
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        traceback.print_exc()
        return False


def process_image(image_path: str) -> Dict:
    """Process uploaded image and return predictions."""
    try:
        # Use ensemble model if available
        if ensemble_model:
            result = ensemble_model.predict_single_image(image_path)
            if result:
                # Format results for web display
                predictions = {}
                for disease, pred_data in result['predictions'].items():
                    predictions[disease] = {
                        'probability': pred_data['ensemble_prob'],
                        'prediction': pred_data['prediction'],
                        'threshold': pred_data['threshold_used'],
                        'confidence': 'High' if pred_data['ensemble_prob'] > 0.8 or pred_data['ensemble_prob'] < 0.2 else 'Medium'
                    }
                
                return {
                    'success': True,
                    'model_type': 'ensemble',
                    'predictions': predictions,
                    'positive_findings': ensemble_model.get_positive_predictions(result)
                }
        
        # Use single model if available
        elif predictor:
            result = predictor.predict_single_image(image_path)
            if result:
                predictions = {}
                for disease, pred_data in result.items():
                    predictions[disease] = {
                        'probability': pred_data['probability'],
                        'prediction': pred_data['prediction'],
                        'threshold': pred_data['threshold'],
                        'confidence': 'High' if pred_data['probability'] > 0.8 or pred_data['probability'] < 0.2 else 'Medium'
                    }
                
                positive_findings = [disease for disease, data in result.items() if data['prediction'] == 1]
                if not positive_findings:
                    positive_findings = ['No Finding']
                
                return {
                    'success': True,
                    'model_type': 'single',
                    'predictions': predictions,
                    'positive_findings': positive_findings
                }
        
        # Demo mode - return mock results
        return {
            'success': True,
            'model_type': 'demo',
            'predictions': generate_demo_predictions(),
            'positive_findings': ['Demo Mode - No real analysis']
        }
            
    except Exception as e:
        print(f"Error processing image: {e}")
        traceback.print_exc()
        return {
            'success': False,
            'error': str(e)
        }


def generate_demo_predictions():
    """Generate demo predictions for testing."""
    diseases = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
               'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
               'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']
    
    predictions = {}
    for disease in diseases:
        prob = np.random.random()
        predictions[disease] = {
            'probability': prob,
            'prediction': 1 if prob > 0.5 else 0,
            'threshold': 0.5,
            'confidence': 'Demo'
        }
    
    return predictions


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
        
        # Process image
        results = process_image(file_path)
        
        if results['success']:
            # Add file info to results for display
            results['unique_id'] = unique_id
            results['filename'] = filename
            results['timestamp'] = datetime.now().isoformat()
            
            # Clean up uploaded file after processing
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove uploaded file {file_path}: {e}")
            
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
    model_status = 'No models loaded'
    if ensemble_model:
        model_status = 'Ensemble model active'
    elif predictor:
        model_status = 'Single model active'
    else:
        model_status = 'Demo mode'
    
    return jsonify({
        'status': 'running',
        'model_status': model_status,
        'device': str(get_device()) if 'get_device' in globals() else 'Unknown',
        'version': '1.0.0'
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
    print("🔬 Initializing Chest X-ray AI Web Application...")
    print("-" * 50)
    
    # Initialize models
    model_loaded = initialize_models()
    
    if model_loaded:
        print("✅ Models initialized successfully")
    else:
        print("⚠️ Running in demo mode - no trained models available")
    
    print("\n🚀 Starting web server...")
    print("📱 Access the application at: http://localhost:5000")
    print("-" * 50)
    
    # Run the app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )