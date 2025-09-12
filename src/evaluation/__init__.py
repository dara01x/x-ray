"""
Evaluation utilities for model performance assessment.
"""

import torch
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns


class Evaluator:
    """Evaluator class for multi-label chest X-ray classification."""
    
    def __init__(self, disease_labels: List[str]):
        """
        Initialize evaluator.
        
        Args:
            disease_labels: List of disease label names
        """
        self.disease_labels = disease_labels
        self.num_classes = len(disease_labels)
    
    def evaluate(self, model: torch.nn.Module, criterion: torch.nn.Module,
                data_loader, device: torch.device) -> Tuple[float, Dict[str, float]]:
        """
        Evaluate model on given data loader.
        
        Args:
            model: Model to evaluate
            criterion: Loss function
            data_loader: Data loader for evaluation
            device: Device to run evaluation on
            
        Returns:
            Tuple of (validation_loss, metrics_dict)
        """
        model.eval()
        running_loss = 0.0
        all_true_labels = []
        all_pred_probs = []
        
        progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
        
        with torch.no_grad():
            for batch_data in progress_bar:
                if batch_data is None:
                    continue
                
                images, labels = batch_data
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item() * images.size(0)
                
                pred_probs = torch.sigmoid(outputs)
                
                all_true_labels.append(labels.cpu())
                all_pred_probs.append(pred_probs.cpu())
        
        progress_bar.close()
        
        if not all_true_labels:
            return float('inf'), {}
        
        # Concatenate all predictions
        all_true_labels = torch.cat(all_true_labels, dim=0).numpy()
        all_pred_probs = torch.cat(all_pred_probs, dim=0).numpy()
        
        val_loss = running_loss / len(data_loader.dataset)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_true_labels, all_pred_probs)
        
        return val_loss, metrics
    
    def _calculate_metrics(self, true_labels: np.ndarray, pred_probs: np.ndarray) -> Dict[str, float]:
        """
        Calculate various metrics for multi-label classification.
        
        Args:
            true_labels: Ground truth labels
            pred_probs: Predicted probabilities
            
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Calculate per-class AUC
        per_class_auc = {}
        valid_aucs = []
        
        for i in range(self.num_classes):
            if len(np.unique(true_labels[:, i])) > 1:
                auc = roc_auc_score(true_labels[:, i], pred_probs[:, i])
                per_class_auc[self.disease_labels[i]] = auc
                valid_aucs.append(auc)
            else:
                per_class_auc[self.disease_labels[i]] = np.nan
        
        # Macro AUC
        metrics['macro_auc'] = np.nanmean(valid_aucs) if valid_aucs else 0.0
        
        # F1-Score at threshold 0.5
        pred_binary = (pred_probs >= 0.5).astype(int)
        metrics['macro_f1'] = f1_score(true_labels, pred_binary, average='macro', zero_division=0)
        metrics['micro_f1'] = f1_score(true_labels, pred_binary, average='micro', zero_division=0)
        
        # Add per-class AUCs to metrics
        metrics.update(per_class_auc)
        
        return metrics
    
    def find_optimal_thresholds(self, model: torch.nn.Module, data_loader,
                               device: torch.device) -> Dict[str, float]:
        """
        Find optimal thresholds for each class to maximize F1-score.
        
        Args:
            model: Trained model
            data_loader: Data loader for threshold optimization
            device: Device to run on
            
        Returns:
            Dictionary of optimal thresholds for each disease
        """
        model.eval()
        all_true_labels = []
        all_pred_probs = []
        
        print("Collecting predictions for threshold optimization...")
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Collecting predictions"):
                if batch_data is None:
                    continue
                
                images, labels = batch_data
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                outputs = model(images)
                pred_probs = torch.sigmoid(outputs)
                
                all_true_labels.append(labels.cpu())
                all_pred_probs.append(pred_probs.cpu())
        
        all_true_labels = torch.cat(all_true_labels, dim=0).numpy()
        all_pred_probs = torch.cat(all_pred_probs, dim=0).numpy()
        
        optimal_thresholds = {}
        threshold_range = np.arange(0.01, 0.99, 0.01)
        
        print("Finding optimal thresholds...")
        for i, disease_name in enumerate(self.disease_labels):
            true_labels_class = all_true_labels[:, i]
            pred_probs_class = all_pred_probs[:, i]
            
            best_f1 = 0
            best_thresh = 0.5
            
            for threshold in threshold_range:
                pred_binary = (pred_probs_class >= threshold).astype(int)
                current_f1 = f1_score(true_labels_class, pred_binary, zero_division=0)
                
                if current_f1 > best_f1:
                    best_f1 = current_f1
                    best_thresh = threshold
            
            optimal_thresholds[disease_name] = best_thresh
            print(f"{disease_name}: {best_thresh:.2f} (F1: {best_f1:.4f})")
        
        return optimal_thresholds
    
    def generate_classification_report(self, model: torch.nn.Module, data_loader,
                                     device: torch.device, thresholds: Dict[str, float]) -> str:
        """
        Generate detailed classification report.
        
        Args:
            model: Trained model
            data_loader: Data loader for evaluation
            device: Device to run on
            thresholds: Optimal thresholds for each class
            
        Returns:
            Classification report string
        """
        model.eval()
        all_true_labels = []
        all_pred_probs = []
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Generating report"):
                if batch_data is None:
                    continue
                
                images, labels = batch_data
                images = images.to(device, non_blocking=True)
                
                outputs = model(images)
                pred_probs = torch.sigmoid(outputs)
                
                all_true_labels.append(labels.cpu())
                all_pred_probs.append(pred_probs.cpu())
        
        all_true_labels = torch.cat(all_true_labels, dim=0).numpy()
        all_pred_probs = torch.cat(all_pred_probs, dim=0).numpy()
        
        # Apply optimal thresholds
        final_pred_binary = np.zeros_like(all_pred_probs)
        for i, disease_name in enumerate(self.disease_labels):
            threshold = thresholds.get(disease_name, 0.5)
            final_pred_binary[:, i] = (all_pred_probs[:, i] >= threshold).astype(int)
        
        # Generate classification report
        report = classification_report(
            all_true_labels,
            final_pred_binary,
            target_names=self.disease_labels,
            zero_division=0
        )
        
        return report
    
    def plot_confusion_matrices(self, model: torch.nn.Module, data_loader,
                               device: torch.device, thresholds: Dict[str, float],
                               save_path: Optional[str] = None):
        """
        Plot confusion matrices for all classes.
        
        Args:
            model: Trained model
            data_loader: Data loader for evaluation
            device: Device to run on
            thresholds: Optimal thresholds for each class
            save_path: Optional path to save the plot
        """
        from sklearn.metrics import multilabel_confusion_matrix
        
        model.eval()
        all_true_labels = []
        all_pred_probs = []
        
        with torch.no_grad():
            for batch_data in tqdm(data_loader, desc="Computing confusion matrices"):
                if batch_data is None:
                    continue
                
                images, labels = batch_data
                images = images.to(device, non_blocking=True)
                
                outputs = model(images)
                pred_probs = torch.sigmoid(outputs)
                
                all_true_labels.append(labels.cpu())
                all_pred_probs.append(pred_probs.cpu())
        
        all_true_labels = torch.cat(all_true_labels, dim=0).numpy()
        all_pred_probs = torch.cat(all_pred_probs, dim=0).numpy()
        
        # Apply optimal thresholds
        final_pred_binary = np.zeros_like(all_pred_probs)
        for i, disease_name in enumerate(self.disease_labels):
            threshold = thresholds.get(disease_name, 0.5)
            final_pred_binary[:, i] = (all_pred_probs[:, i] >= threshold).astype(int)
        
        # Calculate multilabel confusion matrix
        mcm = multilabel_confusion_matrix(all_true_labels, final_pred_binary)
        
        # Plot confusion matrices
        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()
        
        for i, (matrix, label) in enumerate(zip(mcm, self.disease_labels)):
            if i >= len(axes):
                break
                
            tn, fp, fn, tp = matrix.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            ax = axes[i]
            sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_title(f"{label}\nPrecision: {precision:.2f}, Recall: {recall:.2f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_xticklabels(['Absent', 'Present'])
            ax.set_yticklabels(['Absent', 'Present'])
        
        # Hide unused subplots
        for j in range(len(self.disease_labels), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        plt.suptitle("Confusion Matrices for All Disease Classes", fontsize=16, y=1.02)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrices saved to {save_path}")
        
        plt.show()


def create_evaluator(disease_labels: List[str]) -> Evaluator:
    """
    Create an evaluator instance.
    
    Args:
        disease_labels: List of disease label names
        
    Returns:
        Evaluator instance
    """
    return Evaluator(disease_labels)
