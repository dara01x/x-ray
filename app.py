#!/usr/bin/env python3
"""
Fixed X-Ray AI Web Application
- Stable server configuration
- Enhanced error handling
- Better connection management
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

# Add src to path for imports
import sys
from pathlib import Path
src_path = Path(__file__).parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Import custom modules
try:
    from utils import load_config, get_device  # type: ignore
    from models.ensemble_model import load_ensemble_model  # type: ignore
    from scripts.inference import ChestXrayPredictor  # type: ignore
    print("✅ All modules imported successfully")
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all required modules are available")

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
    threshold_paths = [
        'models/optimal_thresholds_ensemble_final.json',
        'models/optimal_thresholds_ensemble_final.csv',
        'models/optimal_thresholds_ensemble_final.csv',
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
        config_path = 'configs/config.yaml'
        if os.path.exists(config_path):
            config = load_config(config_path)
            print("✅ Configuration loaded")
        
        # Define model paths (simplified)
        model_paths = {
            'champion': 'models/best_model_all_out_v1.pth',
            'arnoweng': 'models/model.pth.tar'
        }
        
        # Check which models exist
        champion_path = model_paths['champion'] if os.path.exists(model_paths['champion']) else None
        arnoweng_path = model_paths['arnoweng'] if os.path.exists(model_paths['arnoweng']) else None
        
        print(f"🔍 Model availability check:")
        print(f"   Model A: {'✅ Found' if champion_path else '❌ Missing'}")
        print(f"   Model B: {'✅ Found' if arnoweng_path else '❌ Missing'}")
        
        if champion_path and arnoweng_path:
            try:
                thresholds = load_optimal_thresholds()
                
                thresholds_json = 'outputs/optimal_thresholds.json'
                os.makedirs(os.path.dirname(thresholds_json), exist_ok=True)
                with open(thresholds_json, 'w') as f:
                    json.dump(thresholds, f)
                
                ensemble_model = load_ensemble_model(
                    champion_checkpoint=champion_path,
                    arnoweng_checkpoint=arnoweng_path,
                    ensemble_thresholds=thresholds_json
                )
                print("🎉 Ensemble model loaded successfully!")
                print(f"📊 Model A: {champion_path}")
                print(f"📊 Model B: {arnoweng_path}")
                print(f"📊 Model info: {ensemble_model.get_model_info()}")
                return 'ensemble'
            except Exception as e:
                print(f"❌ Failed to load ensemble model: {e}")
                traceback.print_exc()
        
        # Try single Model A
        if config and champion_path:
            try:
                thresholds_json = 'outputs/optimal_thresholds.json'
                thresholds_path = thresholds_json if os.path.exists(thresholds_json) else None
                predictor = ChestXrayPredictor(
                    config_path=config_path,
                    checkpoint_path=champion_path,
                    thresholds_path=thresholds_path
                )
                print("✅ Single Model A loaded successfully!")
                print(f"📊 Model path: {champion_path}")
                return 'single'
            except Exception as e:
                print(f"❌ Failed to load Model A: {e}")
                traceback.print_exc()
        
        # Show detailed error information for demo mode
        print("\n" + "="*60)
        print("⚠️ RUNNING IN DEMO MODE - LIMITED FUNCTIONALITY")
        print("="*60)
        print("🚫 NO TRAINED MODELS AVAILABLE")
        print("\nWhat this means:")
        print("✅ Web interface will work")
        print("✅ File uploads will work") 
        print("❌ AI predictions will NOT work")
        print("❌ Disease classification will NOT work")
        print("❌ Medical insights will NOT be generated")
        
        print("\n📥 TO ENABLE FULL FUNCTIONALITY:")
        print("1. Obtain trained model files from original training environment")
        print("2. Place them in the correct directories:")
        print("   - Model A → models/best_model_all_out_v1.pth")
        print("   - Model B → models/model.pth.tar")
        print("   - Or in kaggle_outputs/ directory")
        print("3. Run: python verify_setup.py")
        print("4. Look for success message: '🎉 Setup complete!'")
        print("="*60)
        return 'demo'
        
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        traceback.print_exc()
        return 'error'

def process_image(image_path: str) -> Dict:
    """Process uploaded image and return predictions."""
    try:
        print(f"🔍 Processing image: {image_path}")
        
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            return {
                'success': False,
                'error': 'Image file not found'
            }
        
        if ensemble_model:
            print("📊 Using ensemble model for prediction...")
            try:
                result = ensemble_model.predict_single_image(image_path)
                print(f"📈 Ensemble prediction result: {result is not None}")
                
                if result:
                    # Model B-only diseases
                    arnoweng_only_diseases = ['Pneumothorax', 'Pneumonia', 'Consolidation']
                    
                    predictions = {}
                    positive_count = 0
                    for disease, pred_data in result['predictions'].items():
                        ensemble_prob = pred_data['ensemble_prob']
                        threshold_used = pred_data['threshold_used']
                        is_positive = pred_data['prediction']
                        model_used = pred_data.get('model_used', 'ensemble')
                        
                        if is_positive:
                            positive_count += 1
                        
                        prob_distance = abs(ensemble_prob - threshold_used)
                        if prob_distance > 0.3:
                            confidence = 'Very High'
                        elif prob_distance > 0.2:
                            confidence = 'High'
                        elif prob_distance > 0.1:
                            confidence = 'Moderate'
                        else:
                            confidence = 'Low'
                        
                        # For Model B-only diseases, only show Model B probability
                        if disease in arnoweng_only_diseases:
                            predictions[disease] = {
                                'probability': ensemble_prob,
                                'prediction': is_positive,
                                'threshold': threshold_used,
                                'confidence': confidence,
                                'arnoweng_prob': pred_data['arnoweng_prob'],
                                'model_used': 'arnoweng_only',
                                'note': 'Predicted by Model B only'
                            }
                        else:
                            # For ensemble diseases, show both model probabilities
                            predictions[disease] = {
                                'probability': ensemble_prob,
                                'prediction': is_positive,
                                'threshold': threshold_used,
                                'confidence': confidence,
                                'champion_prob': pred_data['champion_prob'],
                                'arnoweng_prob': pred_data['arnoweng_prob'],
                                'model_used': 'ensemble',
                                'note': 'Ensemble prediction (Model A + Model B)'
                            }
                    
                    positive_findings = ensemble_model.get_positive_predictions(result)
                    if not positive_findings:
                        positive_findings = ['No Finding']
                    
                    print(f"✅ Ensemble prediction successful: {len(predictions)} diseases, {positive_count} positive")
                    
                    return {
                        'success': True,
                        'model_type': 'Ensemble (Model A + Model B)',
                        'predictions': predictions,
                        'positive_findings': positive_findings,
                        'summary': {
                            'total_diseases_checked': len(predictions),
                            'positive_findings_count': positive_count,
                            'model_confidence': 'High' if positive_count <= 2 else 'Moderate',
                            'optimal_thresholds_used': True
                        },
                        'arnoweng_only_diseases': arnoweng_only_diseases
                    }
                else:
                    print("❌ Ensemble model returned None result")
                    return {
                        'success': False,
                        'error': 'Ensemble model failed to process image'
                    }
                    
            except Exception as e:
                print(f"❌ Ensemble model error: {e}")
                import traceback
                traceback.print_exc()
        
        elif predictor:
            print("📊 Using single Model A for prediction...")
            try:
                result = predictor.predict_single_image(image_path)
                if result:
                    predictions = {}
                    positive_count = 0
                    for disease, pred_data in result.items():
                        is_positive = pred_data['prediction']
                        if is_positive:
                            positive_count += 1
                        
                        probability = pred_data['probability']
                        threshold = pred_data['threshold']
                        
                        prob_distance = abs(probability - threshold)
                        if prob_distance > 0.3:
                            confidence = 'Very High'
                        elif prob_distance > 0.2:
                            confidence = 'High'
                        elif prob_distance > 0.1:
                            confidence = 'Moderate'
                        else:
                            confidence = 'Low'
                        
                        predictions[disease] = {
                            'probability': probability,
                            'prediction': is_positive,
                            'threshold': threshold,
                            'confidence': confidence
                        }
                    
                    positive_findings = [disease for disease, data in result.items() if data['prediction'] == 1]
                    if not positive_findings:
                        positive_findings = ['No Finding']
                    
                    print(f"✅ Single model prediction successful: {len(predictions)} diseases, {positive_count} positive")
                    
                    return {
                        'success': True,
                        'model_type': 'Single Model A',
                        'predictions': predictions,
                        'positive_findings': positive_findings,
                        'summary': {
                            'total_diseases_checked': len(predictions),
                            'positive_findings_count': positive_count,
                            'model_confidence': 'High' if positive_count <= 2 else 'Moderate',
                            'optimal_thresholds_used': True
                        }
                    }
                else:
                    print("❌ Single model returned None result")
                    return {
                        'success': False,
                        'error': 'Single model failed to process image'
                    }
            except Exception as e:
                print(f"❌ Single model error: {e}")
                import traceback
                traceback.print_exc()
        
        print("⚠️ Running in demo mode - no real models available")
        return {
            'success': True,
            'model_type': 'Demo Mode',
            'predictions': generate_demo_predictions(),
            'positive_findings': ['Demo Mode - No real analysis'],
            'summary': {
                'total_diseases_checked': 14,
                'positive_findings_count': 0,
                'model_confidence': 'Demo',
                'optimal_thresholds_used': False
            },
            'demo_mode': True,
            'warning': 'This is DEMO MODE - no actual AI analysis performed. To enable real predictions, install trained model files.'
        }
            
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'error': f'Internal error: {str(e)}'
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
    """Serve the main application page."""
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
        
        filename = secure_filename(file.filename or 'unknown.jpg')
        unique_id = str(uuid.uuid4())
        file_extension = filename.rsplit('.', 1)[1].lower()
        unique_filename = f"{unique_id}.{file_extension}"
        file_path = os.path.join(UPLOAD_FOLDER, unique_filename)
        
        file.save(file_path)
        
        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            os.remove(file_path)
            return jsonify({'success': False, 'error': 'Invalid image file'}), 400
        
        results = process_image(file_path)
        
        if results['success']:
            results['unique_id'] = unique_id
            results['filename'] = filename
            results['timestamp'] = datetime.now().isoformat()
            
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Warning: Could not remove uploaded file {file_path}: {e}")
            
            return jsonify(results)
        else:
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

@app.route('/api/status')
def get_status():
    """Get application status."""
    global ensemble_model, predictor
    
    # Determine model status
    if ensemble_model:
        model_status = 'Ensemble model active'
        model_type = 'ensemble'
        functionality = 'full'
    elif predictor:
        model_status = 'Single model active'
        model_type = 'single'
        functionality = 'full'
    else:
        model_status = 'Demo mode - no trained models'
        model_type = 'demo'
        functionality = 'limited'
    
    # Check for missing model files
    missing_files = []
    model_paths = {
        'Model A (primary)': 'models/best_model_all_out_v1.pth',
        'Model A (backup)': 'outputs/models/best_model.pth',
        'Model A (kaggle)': 'kaggle_outputs/best_model_all_out_v1.pth',
        'Model B (primary)': 'models/model.pth.tar',
        'Model B (kaggle)': 'kaggle_outputs/model.pth.tar'
    }
    
    for name, path in model_paths.items():
        if not os.path.exists(path):
            missing_files.append({'name': name, 'path': path})
    
    return jsonify({
        'status': 'running',
        'model_status': model_status,
        'model_type': model_type,
        'functionality': functionality,
        'device': str(get_device()) if 'get_device' in globals() else 'Unknown',
        'version': '1.0.0',
        'missing_files': missing_files,
        'demo_mode': functionality == 'limited',
        'capabilities': {
            'web_interface': True,
            'file_upload': True,
            'ai_predictions': functionality == 'full',
            'disease_classification': functionality == 'full',
            'confidence_scoring': functionality == 'full',
            'medical_insights': functionality == 'full'
        }
    })

@app.errorhandler(413)
def too_large(e):
    """Handle file too large error."""
    return jsonify({'success': False, 'error': 'File too large. Maximum size is 16MB.'}), 413

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🔬 Initializing Fixed Chest X-ray AI Web Application...")
    print("-" * 50)
    
    # Initialize models
    model_status = initialize_models()
    
    if model_status == 'ensemble':
        print("✅ FULL FUNCTIONALITY: Ensemble models loaded successfully")
        print("🎯 AI predictions, disease classification, and medical insights available")
    elif model_status == 'single':
        print("✅ FULL FUNCTIONALITY: Single model loaded successfully")  
        print("🎯 AI predictions, disease classification, and medical insights available")
    elif model_status == 'demo':
        print("⚠️ LIMITED FUNCTIONALITY: Running in demo mode")
        print("❌ AI predictions and medical analysis NOT available")
        print("📋 See startup messages above for instructions to enable full functionality")
    else:
        print("❌ ERROR: Failed to initialize application")
        print("🔧 Check error messages above and ensure dependencies are installed")
    
    print("\n🚀 Starting stable web server...")
    print("📱 Access the application at: http://localhost:5000")
    print("-" * 50)
    
    # Run with stable configuration
    try:
        app.run(
            host='127.0.0.1',  # More specific binding
            port=5000,
            debug=False,
            threaded=True,
            use_reloader=False  # Disable auto-reload for stability
        )
    except Exception as e:
        print(f"❌ Server failed to start: {e}")
        print("Try a different port or check if another service is using port 5000")