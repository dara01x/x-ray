"""
Ensemble Model Implementation
Based on the ab-ensemble.ipynb notebook - combines two DenseNet121 models for improved performance.
"""

import torch
import torch.nn as nn
import torchxrayvision as xrv
import torchvision
import os
import json
import numpy as np
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from PIL import Image


class ChampionModel(nn.Module):
    """Your champion model architecture - DenseNet121 with TorchXRayVision backbone."""
    
    def __init__(self, num_classes: int = 14):
        super(ChampionModel, self).__init__()
        # Use the same architecture as in the notebook
        self.backbone = xrv.models.DenseNet(weights="densenet121-res224-all")
        self.backbone.op_threshs = None
        
        num_features = self.backbone.classifier.in_features
        self.backbone.classifier = nn.Identity()
        
        # Deeper head architecture from the notebook
        self.head = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.6),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)


class ArnowengDenseNet121(nn.Module):
    """Arnoweng's CheXNet model architecture."""
    
    def __init__(self, out_size: int = 14):
        super(ArnowengDenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class EnsembleModel:
    """
    Ensemble model that combines ChampionModel and ArnowengDenseNet121.
    Implements the ensemble logic from the ab-ensemble notebook.
    """
    
    def __init__(self, 
                 champion_checkpoint_path: str,
                 arnoweng_checkpoint_path: str,
                 champion_thresholds_path: Optional[str] = None,
                 ensemble_thresholds_path: Optional[str] = None,
                 num_classes: int = 14,
                 device: str = "auto"):
        
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        
        # Disease labels (same as in notebook)
        self.disease_labels = [
            'Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass', 'Nodule', 
            'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema', 'Emphysema', 
            'Fibrosis', 'Pleural_Thickening', 'Hernia'
        ]
        
        # Load models
        self.champion_model = self._load_champion_model(champion_checkpoint_path)
        self.arnoweng_model = self._load_arnoweng_model(arnoweng_checkpoint_path)
        
        # Load thresholds
        self.champion_thresholds = self._load_thresholds(champion_thresholds_path) if champion_thresholds_path else None
        self.ensemble_thresholds = self._load_thresholds(ensemble_thresholds_path) if ensemble_thresholds_path else None
        
        # Setup transforms (same as notebook)
        self._setup_transforms()
        
    def _load_champion_model(self, checkpoint_path: str) -> ChampionModel:
        """Load the champion model from checkpoint."""
        model = ChampionModel(num_classes=self.num_classes)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ Champion model loaded from epoch {checkpoint.get('epoch', 'N/A')}")
        else:
            print(f"⚠️ Champion model checkpoint not found at {checkpoint_path}")
            
        model.to(self.device)
        model.eval()
        return model
        
    def _load_arnoweng_model(self, checkpoint_path: str) -> ArnowengDenseNet121:
        """Load the Arnoweng model from checkpoint."""
        model = ArnowengDenseNet121(out_size=self.num_classes)
        
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            state_dict_from_file = checkpoint['state_dict']
            
            # Apply the complex key re-mapping logic from the notebook
            final_state_dict = OrderedDict()
            for key, value in state_dict_from_file.items():
                new_key = key.replace("module.", "", 1).replace(".norm.1.", ".norm1.").replace(".norm.2.", ".norm2.").replace(".conv.1.", ".conv1.").replace(".conv.2.", ".conv2.")
                if not new_key.startswith('densenet121.'):
                    if 'classifier' in new_key or 'features' in new_key:
                        new_key = 'densenet121.' + new_key
                final_state_dict[new_key] = value
                
            model.load_state_dict(final_state_dict, strict=False)
            print("✅ Arnoweng model loaded successfully")
        else:
            print(f"⚠️ Arnoweng model checkpoint not found at {checkpoint_path}")
            
        model.to(self.device)
        model.eval()
        return model
        
    def _load_thresholds(self, thresholds_path: str) -> Dict[str, float]:
        """Load optimal thresholds from JSON file."""
        if os.path.exists(thresholds_path):
            with open(thresholds_path, 'r') as f:
                return json.load(f)
        return {}
        
    def _setup_transforms(self):
        """Setup transforms for both models (same as notebook)."""
        # Transform for champion model (Albumentations)
        self.champion_transform = A.Compose([
            A.Resize(height=224, width=224),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.ToFloat(max_value=255.0),
            ToTensorV2()
        ])
        
        # Transform for Arnoweng model (torchvision)
        self.arnoweng_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict_single_image(self, image_path: str, use_optimal_thresholds: bool = True) -> Dict:
        """
        Make ensemble prediction on a single image.
        
        Args:
            image_path: Path to the image file
            use_optimal_thresholds: Whether to use optimal thresholds for binary predictions
            
        Returns:
            Dictionary with predictions from both models and ensemble
        """
        try:
            # Get prediction from champion model
            champion_probs = self._predict_champion(image_path)
            
            # Get prediction from Arnoweng model
            arnoweng_probs = self._predict_arnoweng(image_path)
            
            # Calculate ensemble probabilities (simple average)
            ensemble_probs = (champion_probs + arnoweng_probs) / 2.0
            
            # Prepare results
            results = {
                'image_path': image_path,
                'champion_probabilities': champion_probs.tolist(),
                'arnoweng_probabilities': arnoweng_probs.tolist(),
                'ensemble_probabilities': ensemble_probs.tolist(),
                'predictions': {}
            }
            
            # Generate predictions for each disease
            thresholds = self.ensemble_thresholds if use_optimal_thresholds and self.ensemble_thresholds else {}
            
            for i, disease in enumerate(self.disease_labels):
                threshold = thresholds.get(disease, 0.5)
                results['predictions'][disease] = {
                    'champion_prob': float(champion_probs[i]),
                    'arnoweng_prob': float(arnoweng_probs[i]),
                    'ensemble_prob': float(ensemble_probs[i]),
                    'prediction': int(ensemble_probs[i] >= threshold),
                    'threshold_used': threshold
                }
                
            return results
            
        except Exception as e:
            print(f"Error during prediction for {image_path}: {e}")
            return None
            
    def _predict_champion(self, image_path: str) -> np.ndarray:
        """Get prediction from champion model."""
        # Load and preprocess image
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise IOError(f"Could not load image: {image_path}")
            
        augmented = self.champion_transform(image=image)
        tensor = augmented['image'].unsqueeze(0).to(self.device)
        
        # Get prediction
        with torch.no_grad():
            logits = self.champion_model(tensor)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            
        return probs
        
    def _predict_arnoweng(self, image_path: str) -> np.ndarray:
        """Get prediction from Arnoweng model."""
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        tensor = self.arnoweng_transform(image).unsqueeze(0).to(self.device)
        
        # Get prediction (Arnoweng model outputs probabilities directly)
        with torch.no_grad():
            probs = self.arnoweng_model(tensor).cpu().numpy().flatten()
            
        return probs
        
    def predict_batch(self, image_paths: List[str], batch_size: int = 32) -> List[Dict]:
        """Make ensemble predictions on a batch of images."""
        results = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            
            for image_path in batch_paths:
                result = self.predict_single_image(image_path)
                if result:
                    results.append(result)
                    
        return results
        
    def get_positive_predictions(self, predictions: Dict, threshold_override: Optional[float] = None) -> List[str]:
        """Get list of diseases predicted as positive."""
        positive_diseases = []
        
        for disease, pred_data in predictions['predictions'].items():
            threshold = threshold_override if threshold_override else pred_data['threshold_used']
            if pred_data['ensemble_prob'] >= threshold:
                positive_diseases.append(disease)
                
        return positive_diseases if positive_diseases else ["No Finding"]
        
    def save_thresholds(self, thresholds: Dict[str, float], output_path: str):
        """Save optimal thresholds to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(thresholds, f, indent=4)
        print(f"Thresholds saved to {output_path}")
        
    def get_model_info(self) -> Dict:
        """Get information about the ensemble model."""
        champion_params = sum(p.numel() for p in self.champion_model.parameters())
        arnoweng_params = sum(p.numel() for p in self.arnoweng_model.parameters())
        
        return {
            'ensemble_type': 'simple_averaging',
            'num_classes': self.num_classes,
            'device': str(self.device),
            'champion_model_params': champion_params,
            'arnoweng_model_params': arnoweng_params,
            'total_params': champion_params + arnoweng_params,
            'disease_labels': self.disease_labels,
            'has_champion_thresholds': self.champion_thresholds is not None,
            'has_ensemble_thresholds': self.ensemble_thresholds is not None
        }


def load_ensemble_model(champion_checkpoint: str, 
                       arnoweng_checkpoint: str,
                       champion_thresholds: Optional[str] = None,
                       ensemble_thresholds: Optional[str] = None,
                       device: str = "auto") -> EnsembleModel:
    """
    Convenience function to load the ensemble model.
    
    Args:
        champion_checkpoint: Path to champion model checkpoint
        arnoweng_checkpoint: Path to Arnoweng model checkpoint
        champion_thresholds: Path to champion model thresholds (optional)
        ensemble_thresholds: Path to ensemble thresholds (optional)
        device: Device to load models on
        
    Returns:
        Loaded EnsembleModel instance
    """
    return EnsembleModel(
        champion_checkpoint_path=champion_checkpoint,
        arnoweng_checkpoint_path=arnoweng_checkpoint,
        champion_thresholds_path=champion_thresholds,
        ensemble_thresholds_path=ensemble_thresholds,
        device=device
    )


# Example usage
if __name__ == "__main__":
    # Example of how to use the ensemble model
    ensemble = load_ensemble_model(
        champion_checkpoint="path/to/best_model_all_out_v1.pth",
        arnoweng_checkpoint="path/to/model.pth.tar",
        ensemble_thresholds="path/to/optimal_thresholds_ensemble_final_v1.json"
    )
    
    # Make prediction
    result = ensemble.predict_single_image("path/to/chest_xray.png")
    if result:
        positive_findings = ensemble.get_positive_predictions(result)
        print(f"Predicted findings: {', '.join(positive_findings)}")