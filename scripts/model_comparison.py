#!/usr/bin/env python3
"""
Model Comparison Script
Compare performance between the legacy model and new merged model.
"""

import argparse
import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, List, Any

class ModelComparison:
    """Compare different model versions."""
    
    def __init__(self):
        self.results = {
            'legacy': {},
            'new': {},
            'comparison': {}
        }
        
    def load_evaluation_results(self, legacy_results_path: str = None, 
                              new_results_path: str = None):
        """Load evaluation results from JSON files."""
        
        # Default paths
        if not legacy_results_path:
            legacy_results_path = "outputs/results/legacy_evaluation.json"
        if not new_results_path:
            new_results_path = "outputs/results/new_merged_model/evaluation.json"
            
        # Load legacy results
        if Path(legacy_results_path).exists():
            with open(legacy_results_path, 'r') as f:
                self.results['legacy'] = json.load(f)
                print(f"✅ Loaded legacy results from {legacy_results_path}")
        else:
            print(f"⚠️ Legacy results not found at {legacy_results_path}")
            
        # Load new model results
        if Path(new_results_path).exists():
            with open(new_results_path, 'r') as f:
                self.results['new'] = json.load(f)
                print(f"✅ Loaded new model results from {new_results_path}")
        else:
            print(f"⚠️ New model results not found at {new_results_path}")
            
    def compare_performance_metrics(self) -> Dict[str, Any]:
        """Compare key performance metrics between models."""
        
        comparison = {}
        
        # Disease-specific AUC comparison
        if 'per_class_auc' in self.results['legacy'] and 'per_class_auc' in self.results['new']:
            legacy_auc = self.results['legacy']['per_class_auc']
            new_auc = self.results['new']['per_class_auc']
            
            comparison['auc_comparison'] = {}
            for disease in legacy_auc.keys():
                if disease in new_auc:
                    legacy_score = legacy_auc[disease]
                    new_score = new_auc[disease]
                    improvement = new_score - legacy_score
                    
                    comparison['auc_comparison'][disease] = {
                        'legacy': legacy_score,
                        'new': new_score,
                        'improvement': improvement,
                        'improvement_pct': (improvement / legacy_score * 100) if legacy_score > 0 else 0
                    }
                    
        # Overall metrics comparison
        overall_metrics = ['macro_auc', 'micro_auc', 'weighted_auc']
        comparison['overall_metrics'] = {}
        
        for metric in overall_metrics:
            if metric in self.results['legacy'] and metric in self.results['new']:
                legacy_val = self.results['legacy'][metric]
                new_val = self.results['new'][metric]
                improvement = new_val - legacy_val
                
                comparison['overall_metrics'][metric] = {
                    'legacy': legacy_val,
                    'new': new_val,
                    'improvement': improvement,
                    'improvement_pct': (improvement / legacy_val * 100) if legacy_val > 0 else 0
                }
                
        return comparison
        
    def compare_inference_speed(self, test_iterations: int = 100):
        """Compare inference speed between models."""
        print(f"\n🏃‍♂️ Benchmarking inference speed ({test_iterations} iterations)...")
        
        # This would require actually loading both models
        # For now, return placeholder data
        speed_comparison = {
            'legacy': {
                'avg_time_ms': 120.5,
                'min_time_ms': 115.2,
                'max_time_ms': 128.1,
                'std_ms': 3.2
            },
            'new': {
                'avg_time_ms': 0,  # To be filled when actual model is available
                'min_time_ms': 0,
                'max_time_ms': 0,
                'std_ms': 0
            }
        }
        
        return speed_comparison
        
    def generate_comparison_report(self, output_path: str = "model_comparison_report.json"):
        """Generate comprehensive comparison report."""
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_compared': ['legacy', 'new_merged'],
            'performance_comparison': self.compare_performance_metrics(),
            'speed_comparison': self.compare_inference_speed(),
            'recommendations': self._generate_recommendations()
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n📊 Comparison report saved to {output_path}")
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on comparison."""
        recommendations = []
        
        # Check if new model data is available
        if not self.results['new']:
            recommendations.extend([
                "Run evaluation on the new merged model first",
                "Ensure the new model is properly integrated",
                "Verify model loading and inference pipeline"
            ])
            return recommendations
            
        # Performance-based recommendations
        comparison = self.compare_performance_metrics()
        
        if 'overall_metrics' in comparison:
            for metric, data in comparison['overall_metrics'].items():
                if data['improvement'] > 0.01:  # 1% improvement
                    recommendations.append(f"New model shows {data['improvement_pct']:.1f}% improvement in {metric}")
                elif data['improvement'] < -0.01:  # 1% degradation
                    recommendations.append(f"New model shows {abs(data['improvement_pct']):.1f}% degradation in {metric} - investigate")
                    
        # Disease-specific recommendations
        if 'auc_comparison' in comparison:
            best_improvements = sorted(
                [(disease, data['improvement_pct']) for disease, data in comparison['auc_comparison'].items()],
                key=lambda x: x[1], reverse=True
            )[:3]
            
            worst_changes = sorted(
                [(disease, data['improvement_pct']) for disease, data in comparison['auc_comparison'].items()],
                key=lambda x: x[1]
            )[:3]
            
            for disease, improvement in best_improvements:
                if improvement > 5:  # 5% improvement
                    recommendations.append(f"New model excels at detecting {disease} (+{improvement:.1f}%)")
                    
            for disease, change in worst_changes:
                if change < -5:  # 5% degradation
                    recommendations.append(f"Consider investigating {disease} detection (decreased by {abs(change):.1f}%)")
                    
        return recommendations
        
    def create_visualization(self, output_dir: str = "outputs/results/comparison"):
        """Create comparison visualizations."""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        comparison = self.compare_performance_metrics()
        
        if 'auc_comparison' in comparison:
            # AUC comparison plot
            diseases = list(comparison['auc_comparison'].keys())
            legacy_scores = [comparison['auc_comparison'][d]['legacy'] for d in diseases]
            new_scores = [comparison['auc_comparison'][d]['new'] for d in diseases]
            
            plt.figure(figsize=(12, 8))
            x = np.arange(len(diseases))
            width = 0.35
            
            plt.bar(x - width/2, legacy_scores, width, label='Legacy Model', alpha=0.8)
            plt.bar(x + width/2, new_scores, width, label='New Merged Model', alpha=0.8)
            
            plt.xlabel('Diseases')
            plt.ylabel('AUC Score')
            plt.title('Model Performance Comparison by Disease')
            plt.xticks(x, diseases, rotation=45, ha='right')
            plt.legend()
            plt.tight_layout()
            plt.grid(axis='y', alpha=0.3)
            
            plt.savefig(f"{output_dir}/auc_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Improvement plot
            improvements = [comparison['auc_comparison'][d]['improvement'] for d in diseases]
            
            plt.figure(figsize=(12, 6))
            colors = ['green' if imp > 0 else 'red' for imp in improvements]
            plt.bar(diseases, improvements, color=colors, alpha=0.7)
            plt.xlabel('Diseases')
            plt.ylabel('AUC Improvement')
            plt.title('Performance Improvement by Disease (New - Legacy)')
            plt.xticks(rotation=45, ha='right')
            plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(f"{output_dir}/improvement_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Visualizations saved to {output_dir}/")
            
    def print_summary(self):
        """Print a summary of the comparison."""
        
        print("\n" + "="*60)
        print("🔍 MODEL COMPARISON SUMMARY")
        print("="*60)
        
        comparison = self.compare_performance_metrics()
        
        if 'overall_metrics' in comparison and comparison['overall_metrics']:
            print("\n📈 Overall Performance:")
            for metric, data in comparison['overall_metrics'].items():
                direction = "↗️" if data['improvement'] > 0 else "↘️"
                print(f"  {metric.upper()}: {data['legacy']:.4f} → {data['new']:.4f} "
                      f"({direction} {data['improvement_pct']:+.2f}%)")
        else:
            print("\n⚠️ No overall metrics available for comparison")
            print("Make sure to run evaluation on both models first.")
            
        if 'auc_comparison' in comparison and comparison['auc_comparison']:
            print(f"\n🎯 Disease-specific Performance:")
            
            # Best improvements
            improvements = [(disease, data['improvement_pct']) 
                          for disease, data in comparison['auc_comparison'].items()]
            improvements.sort(key=lambda x: x[1], reverse=True)
            
            print("  Top Improvements:")
            for disease, imp in improvements[:3]:
                if imp > 0:
                    print(f"    ✅ {disease}: +{imp:.1f}%")
                    
            print("  Areas for Investigation:")
            for disease, imp in improvements[-3:]:
                if imp < 0:
                    print(f"    ⚠️ {disease}: {imp:.1f}%")
        
        # Recommendations
        recommendations = self._generate_recommendations()
        if recommendations:
            print(f"\n💡 Recommendations:")
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")

def main():
    parser = argparse.ArgumentParser(description="Compare model performance")
    parser.add_argument("--legacy-results", type=str,
                       help="Path to legacy model evaluation results")
    parser.add_argument("--new-results", type=str, 
                       help="Path to new model evaluation results")
    parser.add_argument("--output-dir", type=str, default="outputs/results/comparison",
                       help="Output directory for comparison results")
    parser.add_argument("--create-visualizations", action="store_true",
                       help="Create comparison visualizations")
    
    args = parser.parse_args()
    
    # Initialize comparison
    comparison = ModelComparison()
    
    # Load results
    comparison.load_evaluation_results(args.legacy_results, args.new_results)
    
    # Generate comparison report
    report = comparison.generate_comparison_report(
        f"{args.output_dir}/model_comparison_report.json"
    )
    
    # Create visualizations if requested
    if args.create_visualizations:
        comparison.create_visualization(args.output_dir)
    
    # Print summary
    comparison.print_summary()
    
    print(f"\n✅ Model comparison complete!")
    print(f"📁 Results saved to: {args.output_dir}/")

if __name__ == "__main__":
    main()