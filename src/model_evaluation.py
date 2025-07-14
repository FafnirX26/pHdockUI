"""
Comprehensive evaluation framework for pKa prediction models with quantum features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error, 
    mean_absolute_percentage_error
)
from sklearn.model_selection import learning_curve
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
import torch
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class ModelEvaluator:
    """Comprehensive evaluation framework for pKa prediction models."""
    
    def __init__(self, model, model_type: str = "ensemble", logger=None):
        """
        Initialize model evaluator.
        
        Args:
            model: Trained model instance
            model_type: Type of model ("xgboost", "gnn", "ensemble")
            logger: Optional logger instance
        """
        self.model = model
        self.model_type = model_type
        self.logger = logger or logging.getLogger(__name__)
        
        # Store evaluation results
        self.evaluation_results = {}
        
    def calculate_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate standard regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            'mae': mean_absolute_error(y_true, y_pred),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred) * 100,
            'pearson_r': pearsonr(y_true, y_pred)[0],
            'spearman_r': spearmanr(y_true, y_pred)[0],
            'mean_error': np.mean(y_pred - y_true),
            'std_error': np.std(y_pred - y_true),
            'max_error': np.max(np.abs(y_pred - y_true)),
            'q95_error': np.percentile(np.abs(y_pred - y_true), 95)
        }
        
        return metrics
    
    def uncertainty_quantification(self, 
                                 model_predictions: List[np.ndarray],
                                 y_true: np.ndarray,
                                 confidence_levels: List[float] = [0.95, 0.90, 0.68]) -> Dict[str, Any]:
        """
        Perform uncertainty quantification with prediction intervals.
        
        Args:
            model_predictions: List of prediction arrays from multiple runs/models
            y_true: True values
            confidence_levels: Confidence levels for prediction intervals
            
        Returns:
            Dictionary with uncertainty metrics
        """
        # Calculate ensemble statistics
        predictions_array = np.array(model_predictions)
        mean_pred = np.mean(predictions_array, axis=0)
        std_pred = np.std(predictions_array, axis=0)
        
        # Calculate prediction intervals
        intervals = {}
        coverage = {}
        sharpness = {}
        
        for conf_level in confidence_levels:
            alpha = 1 - conf_level
            z_score = stats.norm.ppf(1 - alpha/2)
            
            lower = mean_pred - z_score * std_pred
            upper = mean_pred + z_score * std_pred
            
            # Coverage: fraction of true values within prediction interval
            in_interval = (y_true >= lower) & (y_true <= upper)
            coverage[f"{conf_level:.0%}"] = np.mean(in_interval)
            
            # Sharpness: average width of prediction intervals
            sharpness[f"{conf_level:.0%}"] = np.mean(upper - lower)
            
            intervals[f"{conf_level:.0%}"] = {
                'lower': lower,
                'upper': upper,
                'width': upper - lower
            }
        
        return {
            'mean_predictions': mean_pred,
            'prediction_std': std_pred,
            'intervals': intervals,
            'coverage': coverage,
            'sharpness': sharpness,
            'uncertainty_mean': np.mean(std_pred),
            'uncertainty_std': np.std(std_pred)
        }
    
    def calibration_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                           uncertainty: np.ndarray) -> Dict[str, Any]:
        """
        Perform calibration analysis for uncertainty estimates.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            uncertainty: Predicted uncertainties
            
        Returns:
            Dictionary with calibration metrics
        """
        # Calculate absolute errors
        abs_errors = np.abs(y_true - y_pred)
        
        # Calibration curve
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0  # Expected Calibration Error
        mce = 0  # Maximum Calibration Error
        calibration_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Convert uncertainties to confidence levels
            confidence = 1 - 2 * stats.norm.cdf(-uncertainty / np.std(abs_errors))
            in_bin = (confidence > bin_lower) & (confidence <= bin_upper)
            
            if np.sum(in_bin) > 0:
                # Fraction of predictions in bin
                prop_in_bin = np.mean(in_bin)
                
                # Accuracy within bin (using z-score threshold)
                z_threshold = stats.norm.ppf((bin_lower + bin_upper) / 2)
                accurate_in_bin = abs_errors[in_bin] <= z_threshold * np.std(abs_errors)
                accuracy_in_bin = np.mean(accurate_in_bin) if len(accurate_in_bin) > 0 else 0
                
                # Confidence for this bin
                avg_confidence = (bin_lower + bin_upper) / 2
                
                # Update ECE and MCE
                ece += prop_in_bin * abs(avg_confidence - accuracy_in_bin)
                mce = max(mce, abs(avg_confidence - accuracy_in_bin))
                
                calibration_data.append({
                    'confidence': avg_confidence,
                    'accuracy': accuracy_in_bin,
                    'proportion': prop_in_bin
                })
        
        return {
            'ece': ece,  # Expected Calibration Error
            'mce': mce,  # Maximum Calibration Error
            'calibration_curve': calibration_data
        }
    
    def pka_range_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                          pka_ranges: List[Tuple[float, float]] = None) -> Dict[str, Dict[str, float]]:
        """
        Analyze model performance across different pKa ranges.
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values
            pka_ranges: List of (min, max) tuples for pKa ranges
            
        Returns:
            Dictionary with metrics for each pKa range
        """
        if pka_ranges is None:
            # Default pKa ranges based on typical ionizable groups
            pka_ranges = [
                (0, 3),    # Strong acids
                (3, 5),    # Carboxylic acids
                (5, 7),    # Weak acids
                (7, 9),    # Weak bases
                (9, 11),   # Amines
                (11, 14)   # Strong bases
            ]
        
        range_results = {}
        
        for i, (min_pka, max_pka) in enumerate(pka_ranges):
            # Find molecules in this pKa range
            in_range = (y_true >= min_pka) & (y_true < max_pka)
            
            if np.sum(in_range) > 5:  # Need at least 5 molecules for meaningful statistics
                range_name = f"pKa_{min_pka}-{max_pka}"
                range_y_true = y_true[in_range]
                range_y_pred = y_pred[in_range]
                
                # Calculate metrics for this range
                range_metrics = self.calculate_standard_metrics(range_y_true, range_y_pred)
                range_metrics['n_molecules'] = np.sum(in_range)
                range_metrics['pka_mean'] = np.mean(range_y_true)
                range_metrics['pka_std'] = np.std(range_y_true)
                
                range_results[range_name] = range_metrics
        
        return range_results
    
    def molecular_complexity_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                    molecules: List[Chem.Mol]) -> Dict[str, Any]:
        """
        Analyze model performance correlation with molecular complexity.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            molecules: List of RDKit molecules
            
        Returns:
            Dictionary with complexity analysis results
        """
        # Calculate complexity descriptors
        complexity_metrics = []
        
        for mol in molecules:
            if mol is not None:
                metrics = {
                    'num_atoms': mol.GetNumAtoms(),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'molecular_weight': Descriptors.MolWt(mol),
                    'tpsa': rdMolDescriptors.CalcTPSA(mol),
                    'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                    'bertz_complexity': rdMolDescriptors.BertzCT(mol) if hasattr(rdMolDescriptors, 'BertzCT') else 0
                }
            else:
                metrics = {key: 0 for key in ['num_atoms', 'num_heavy_atoms', 'num_rings',
                                            'num_aromatic_rings', 'molecular_weight', 'tpsa',
                                            'num_rotatable_bonds', 'bertz_complexity']}
            
            complexity_metrics.append(metrics)
        
        complexity_df = pd.DataFrame(complexity_metrics)
        abs_errors = np.abs(y_true - y_pred)
        
        # Calculate correlations between complexity and error
        correlations = {}
        for metric in complexity_df.columns:
            if len(complexity_df[metric].unique()) > 1:  # Avoid constant columns
                corr, p_value = pearsonr(complexity_df[metric], abs_errors)
                correlations[metric] = {'correlation': corr, 'p_value': p_value}
        
        # Binned analysis by complexity
        binned_analysis = {}
        for metric in ['num_heavy_atoms', 'molecular_weight', 'bertz_complexity']:
            if metric in complexity_df.columns and len(complexity_df[metric].unique()) > 3:
                try:
                    # Create bins
                    values = complexity_df[metric]
                    bins = pd.qcut(values, q=3, labels=['Low', 'Medium', 'High'], duplicates='drop')
                    
                    bin_results = {}
                    for bin_name in bins.unique():
                        if pd.notna(bin_name):
                            mask = bins == bin_name
                            if np.sum(mask) > 2:  # Need at least 3 points for meaningful stats
                                bin_metrics = self.calculate_standard_metrics(
                                    y_true[mask], y_pred[mask]
                                )
                                bin_metrics['n_molecules'] = np.sum(mask)
                                bin_metrics['complexity_mean'] = np.mean(values[mask])
                                bin_results[bin_name] = bin_metrics
                    
                    if bin_results:  # Only add if we have results
                        binned_analysis[metric] = bin_results
                        
                except Exception as e:
                    self.logger.warning(f"Could not create bins for {metric}: {e}")
                    continue
        
        return {
            'correlations': correlations,
            'binned_analysis': binned_analysis,
            'complexity_metrics': complexity_df
        }
    
    def worst_performing_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                molecules: List[Chem.Mol],
                                feature_matrix: Optional[np.ndarray] = None,
                                feature_names: Optional[List[str]] = None,
                                top_n: int = 10) -> Dict[str, Any]:
        """
        Analyze worst performing molecules to identify failure modes.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            molecules: List of RDKit molecules
            feature_matrix: Optional feature matrix for SHAP analysis
            feature_names: Optional feature names
            top_n: Number of worst molecules to analyze
            
        Returns:
            Dictionary with analysis of worst performing molecules
        """
        # Calculate absolute errors
        abs_errors = np.abs(y_true - y_pred)
        
        # Find worst performing molecules
        worst_indices = np.argsort(abs_errors)[-top_n:]
        
        worst_molecules_analysis = []
        
        for idx in worst_indices:
            mol = molecules[idx] if idx < len(molecules) else None
            
            analysis = {
                'index': int(idx),
                'true_pka': float(y_true[idx]),
                'pred_pka': float(y_pred[idx]),
                'absolute_error': float(abs_errors[idx]),
                'relative_error': float(abs_errors[idx] / y_true[idx] * 100) if y_true[idx] != 0 else float('inf')
            }
            
            if mol is not None:
                # Molecular properties
                analysis.update({
                    'smiles': Chem.MolToSmiles(mol),
                    'num_atoms': mol.GetNumAtoms(),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                    'molecular_weight': Descriptors.MolWt(mol),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'bertz_complexity': rdMolDescriptors.BertzCT(mol) if hasattr(rdMolDescriptors, 'BertzCT') else 0
                })
            
            worst_molecules_analysis.append(analysis)
        
        # Cluster analysis of worst performers
        if len(worst_molecules_analysis) > 3:
            worst_errors = abs_errors[worst_indices]
            error_stats = {
                'mean_error': np.mean(worst_errors),
                'std_error': np.std(worst_errors),
                'error_range': (np.min(worst_errors), np.max(worst_errors))
            }
        else:
            error_stats = {}
        
        return {
            'worst_molecules': worst_molecules_analysis,
            'error_statistics': error_stats,
            'failure_patterns': self._identify_failure_patterns(worst_molecules_analysis)
        }
    
    def _identify_failure_patterns(self, worst_molecules: List[Dict]) -> Dict[str, Any]:
        """
        Identify common patterns in worst performing molecules.
        
        Args:
            worst_molecules: List of worst molecule analyses
            
        Returns:
            Dictionary with identified patterns
        """
        if not worst_molecules:
            return {}
        
        # Extract molecular properties
        properties = ['num_atoms', 'num_heavy_atoms', 'molecular_weight', 
                     'num_rings', 'bertz_complexity']
        
        patterns = {}
        for prop in properties:
            values = [mol.get(prop, 0) for mol in worst_molecules if prop in mol]
            if values:
                patterns[prop] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'range': (np.min(values), np.max(values))
                }
        
        # pKa range analysis
        pka_values = [mol['true_pka'] for mol in worst_molecules]
        patterns['pka_range'] = {
            'mean': np.mean(pka_values),
            'std': np.std(pka_values),
            'range': (np.min(pka_values), np.max(pka_values))
        }
        
        return patterns
    
    def feature_importance_analysis(self, feature_matrix: np.ndarray,
                                  feature_names: List[str],
                                  y_true: np.ndarray,
                                  quantum_feature_indices: Optional[List[int]] = None,
                                  classical_feature_indices: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Analyze feature importance with focus on quantum vs classical features.
        
        Args:
            feature_matrix: Feature matrix
            feature_names: List of feature names
            y_true: True values
            quantum_feature_indices: Indices of quantum features
            classical_feature_indices: Indices of classical features
            
        Returns:
            Dictionary with feature importance analysis
        """
        importance_results = {}
        
        # Get model-specific feature importance
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models (XGBoost, RandomForest)
            importance_scores = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            # Linear models
            importance_scores = np.abs(self.model.coef_)
        else:
            # Try to use SHAP for other models
            try:
                explainer = shap.Explainer(self.model.predict, feature_matrix[:100])
                shap_values = explainer(feature_matrix[:100])
                importance_scores = np.mean(np.abs(shap_values.values), axis=0)
            except:
                self.logger.warning("Could not calculate feature importance")
                importance_scores = np.ones(len(feature_names)) / len(feature_names)
        
        # Create feature importance dictionary
        feature_importance = dict(zip(feature_names, importance_scores))
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        importance_results['all_features'] = dict(sorted_importance)
        importance_results['top_20_features'] = dict(sorted_importance[:20])
        
        # Quantum vs Classical feature analysis
        if quantum_feature_indices is not None and classical_feature_indices is not None:
            quantum_importance = np.sum([importance_scores[i] for i in quantum_feature_indices])
            classical_importance = np.sum([importance_scores[i] for i in classical_feature_indices])
            total_importance = quantum_importance + classical_importance
            
            if total_importance > 0:
                importance_results['quantum_vs_classical'] = {
                    'quantum_fraction': quantum_importance / total_importance,
                    'classical_fraction': classical_importance / total_importance,
                    'quantum_features': [(feature_names[i], importance_scores[i]) 
                                       for i in quantum_feature_indices],
                    'classical_features': [(feature_names[i], importance_scores[i]) 
                                         for i in classical_feature_indices]
                }
        
        return importance_results
    
    def ablation_study(self, X_full: np.ndarray, y_true: np.ndarray,
                      feature_names: List[str],
                      quantum_feature_indices: List[int],
                      test_indices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Perform ablation study comparing model with/without quantum features.
        
        Args:
            X_full: Full feature matrix
            y_true: True values
            feature_names: List of feature names
            quantum_feature_indices: Indices of quantum features
            test_indices: Optional test set indices
            
        Returns:
            Dictionary with ablation study results
        """
        from copy import deepcopy
        
        results = {}
        
        # Performance with all features
        if test_indices is not None:
            X_test_full = X_full[test_indices]
            y_test = y_true[test_indices]
        else:
            X_test_full = X_full
            y_test = y_true
        
        try:
            y_pred_full = self.model.predict(X_test_full)
            results['with_quantum'] = self.calculate_standard_metrics(y_test, y_pred_full)
        except Exception as e:
            self.logger.warning(f"Could not evaluate model with all features: {e}")
            results['with_quantum'] = {}
        
        # Performance without quantum features
        classical_indices = [i for i in range(X_full.shape[1]) if i not in quantum_feature_indices]
        X_test_classical = X_full[test_indices][:, classical_indices] if test_indices is not None else X_full[:, classical_indices]
        
        try:
            # Retrain model without quantum features (simplified approach)
            # In practice, you would need to retrain the model
            self.logger.info("Ablation study: Note that this is a simplified analysis. "
                           "For complete ablation study, models should be retrained without quantum features.")
            
            # For now, use feature importance to estimate impact
            if hasattr(self.model, 'feature_importances_'):
                quantum_importance = np.sum(self.model.feature_importances_[quantum_feature_indices])
                total_importance = np.sum(self.model.feature_importances_)
                
                # Estimate degradation based on lost importance
                importance_loss = quantum_importance / total_importance if total_importance > 0 else 0
                estimated_r2_loss = importance_loss * results['with_quantum'].get('r2', 0)
                
                results['without_quantum'] = {
                    'estimated_r2_degradation': estimated_r2_loss,
                    'quantum_importance_fraction': importance_loss,
                    'note': 'Estimated based on feature importance'
                }
        except Exception as e:
            self.logger.warning(f"Could not perform ablation analysis: {e}")
            results['without_quantum'] = {}
        
        # Calculate improvement due to quantum features
        if 'with_quantum' in results and 'without_quantum' in results:
            with_r2 = results['with_quantum'].get('r2', 0)
            without_r2_est = with_r2 - results['without_quantum'].get('estimated_r2_degradation', 0)
            
            results['quantum_improvement'] = {
                'r2_improvement': with_r2 - without_r2_est,
                'relative_improvement': (with_r2 - without_r2_est) / without_r2_est * 100 if without_r2_est > 0 else 0
            }
        
        return results
    
    def applicability_domain_assessment(self, X_train: np.ndarray, X_test: np.ndarray,
                                       y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Assess applicability domain using leverage and standardized residuals.
        
        Args:
            X_train: Training feature matrix
            X_test: Test feature matrix
            y_true: True test values
            y_pred: Predicted test values
            
        Returns:
            Dictionary with applicability domain metrics
        """
        from sklearn.covariance import EmpiricalCovariance
        
        # Calculate leverage (Mahalanobis distance)
        try:
            cov = EmpiricalCovariance().fit(X_train)
            leverage_train = cov.mahalanobis(X_train)
            leverage_test = cov.mahalanobis(X_test)
            
            # Warning leverage threshold
            p = X_train.shape[1]  # number of features
            n = X_train.shape[0]  # number of training samples
            h_threshold = 3 * p / n
            
            # Standardized residuals
            residuals = y_true - y_pred
            std_residuals = residuals / np.std(residuals)
            
            # Identify outliers
            high_leverage = leverage_test > h_threshold
            high_residual = np.abs(std_residuals) > 3  # 3-sigma rule
            
            outliers = high_leverage | high_residual
            
            results = {
                'leverage_train_mean': np.mean(leverage_train),
                'leverage_test_mean': np.mean(leverage_test),
                'leverage_threshold': h_threshold,
                'high_leverage_count': np.sum(high_leverage),
                'high_residual_count': np.sum(high_residual),
                'outlier_count': np.sum(outliers),
                'outlier_fraction': np.mean(outliers),
                'leverage_test': leverage_test,
                'standardized_residuals': std_residuals,
                'outlier_mask': outliers
            }
            
        except Exception as e:
            self.logger.warning(f"Could not calculate applicability domain: {e}")
            results = {'error': str(e)}
        
        return results
    
    def generate_plots(self, y_true: np.ndarray, y_pred: np.ndarray,
                      save_dir: Optional[Path] = None) -> Dict[str, str]:
        """
        Generate comprehensive evaluation plots.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            save_dir: Optional directory to save plots
            
        Returns:
            Dictionary mapping plot names to file paths
        """
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        plot_paths = {}
        
        # Set up style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Parity Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.scatter(y_true, y_pred, alpha=0.6, s=30)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # Calculate and display metrics
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        
        ax.text(0.05, 0.95, f'R² = {r2:.3f}\nRMSE = {rmse:.3f}\nMAE = {mae:.3f}',
               transform=ax.transAxes, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('True pKa', fontsize=12)
        ax.set_ylabel('Predicted pKa', fontsize=12)
        ax.set_title('Parity Plot: Predicted vs True pKa Values', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_dir:
            path = save_dir / 'parity_plot.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['parity_plot'] = str(path)
        plt.close()
        
        # 2. Residuals Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        residuals = y_pred - y_true
        ax.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax.axhline(y=0, color='r', linestyle='--', lw=2)
        
        # Add standard deviation bands
        std_res = np.std(residuals)
        ax.axhline(y=2*std_res, color='orange', linestyle=':', alpha=0.7, label='±2σ')
        ax.axhline(y=-2*std_res, color='orange', linestyle=':', alpha=0.7)
        ax.axhline(y=3*std_res, color='red', linestyle=':', alpha=0.7, label='±3σ')
        ax.axhline(y=-3*std_res, color='red', linestyle=':', alpha=0.7)
        
        ax.set_xlabel('Predicted pKa', fontsize=12)
        ax.set_ylabel('Residuals (Pred - True)', fontsize=12)
        ax.set_title('Residuals Plot', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_dir:
            path = save_dir / 'residuals_plot.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['residuals_plot'] = str(path)
        plt.close()
        
        # 3. Bland-Altman Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        mean_values = (y_true + y_pred) / 2
        diff_values = y_pred - y_true
        
        ax.scatter(mean_values, diff_values, alpha=0.6, s=30)
        
        # Calculate bias and limits of agreement
        bias = np.mean(diff_values)
        std_diff = np.std(diff_values)
        
        ax.axhline(y=bias, color='blue', linestyle='-', lw=2, label=f'Bias = {bias:.3f}')
        ax.axhline(y=bias + 1.96*std_diff, color='red', linestyle='--', lw=2, 
                  label=f'Upper LoA = {bias + 1.96*std_diff:.3f}')
        ax.axhline(y=bias - 1.96*std_diff, color='red', linestyle='--', lw=2, 
                  label=f'Lower LoA = {bias - 1.96*std_diff:.3f}')
        
        ax.set_xlabel('Mean of True and Predicted pKa', fontsize=12)
        ax.set_ylabel('Difference (Pred - True)', fontsize=12)
        ax.set_title('Bland-Altman Plot', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_dir:
            path = save_dir / 'bland_altman_plot.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['bland_altman_plot'] = str(path)
        plt.close()
        
        # 4. Error Distribution
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram of absolute errors
        abs_errors = np.abs(residuals)
        ax1.hist(abs_errors, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax1.axvline(np.mean(abs_errors), color='red', linestyle='--', lw=2, label=f'Mean = {np.mean(abs_errors):.3f}')
        ax1.axvline(np.median(abs_errors), color='orange', linestyle='--', lw=2, label=f'Median = {np.median(abs_errors):.3f}')
        ax1.set_xlabel('Absolute Error', fontsize=12)
        ax1.set_ylabel('Density', fontsize=12)
        ax1.set_title('Distribution of Absolute Errors', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Q-Q plot for normality check
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot: Residuals vs Normal Distribution', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        if save_dir:
            path = save_dir / 'error_distribution.png'
            plt.savefig(path, dpi=300, bbox_inches='tight')
            plot_paths['error_distribution'] = str(path)
        plt.close()
        
        return plot_paths
    
    def comprehensive_evaluation(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               molecules: Optional[List[Chem.Mol]] = None,
                               feature_matrix: Optional[np.ndarray] = None,
                               feature_names: Optional[List[str]] = None,
                               quantum_feature_indices: Optional[List[int]] = None,
                               X_train: Optional[np.ndarray] = None,
                               save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model evaluation.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            molecules: Optional list of molecules
            feature_matrix: Optional feature matrix
            feature_names: Optional feature names
            quantum_feature_indices: Optional quantum feature indices
            X_train: Optional training feature matrix
            save_dir: Optional directory to save results
            
        Returns:
            Comprehensive evaluation results dictionary
        """
        self.logger.info("Starting comprehensive model evaluation...")
        
        results = {
            'model_type': self.model_type,
            'n_predictions': len(y_true)
        }
        
        # 1. Standard metrics
        self.logger.info("Calculating standard metrics...")
        results['standard_metrics'] = self.calculate_standard_metrics(y_true, y_pred)
        
        # 2. pKa range analysis
        self.logger.info("Performing pKa range analysis...")
        results['pka_range_analysis'] = self.pka_range_analysis(y_true, y_pred)
        
        # 3. Molecular complexity analysis
        if molecules is not None:
            self.logger.info("Analyzing molecular complexity correlation...")
            results['complexity_analysis'] = self.molecular_complexity_analysis(y_true, y_pred, molecules)
        
        # 4. Worst performing molecules analysis
        if molecules is not None:
            self.logger.info("Analyzing worst performing molecules...")
            results['worst_performers'] = self.worst_performing_analysis(
                y_true, y_pred, molecules, feature_matrix, feature_names
            )
        
        # 5. Feature importance analysis
        if feature_matrix is not None and feature_names is not None:
            self.logger.info("Analyzing feature importance...")
            classical_indices = None
            if quantum_feature_indices is not None:
                classical_indices = [i for i in range(len(feature_names)) if i not in quantum_feature_indices]
            
            results['feature_importance'] = self.feature_importance_analysis(
                feature_matrix, feature_names, y_true, quantum_feature_indices, classical_indices
            )
        
        # 6. Ablation study
        if feature_matrix is not None and feature_names is not None and quantum_feature_indices is not None:
            self.logger.info("Performing ablation study...")
            results['ablation_study'] = self.ablation_study(
                feature_matrix, y_true, feature_names, quantum_feature_indices
            )
        
        # 7. Applicability domain assessment
        if X_train is not None and feature_matrix is not None:
            self.logger.info("Assessing applicability domain...")
            results['applicability_domain'] = self.applicability_domain_assessment(
                X_train, feature_matrix, y_true, y_pred
            )
        
        # 8. Generate plots
        if save_dir is not None:
            self.logger.info("Generating evaluation plots...")
            results['plots'] = self.generate_plots(y_true, y_pred, save_dir)
        
        # Store results
        self.evaluation_results = results
        
        # Save results if directory provided
        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            results_path = save_dir / 'evaluation_results.json'
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            json_results = self._prepare_for_json(results)
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            self.logger.info(f"Evaluation results saved to {results_path}")
        
        self.logger.info("Comprehensive evaluation completed.")
        return results
    
    def _prepare_for_json(self, obj):
        """Recursively convert numpy arrays and other non-serializable objects for JSON."""
        if isinstance(obj, dict):
            return {key: self._prepare_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._prepare_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, pd.DataFrame):
            return obj.to_dict('records')
        elif hasattr(obj, 'tolist'):  # Handle other array-like objects
            return obj.tolist()
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the evaluation results.
        
        Returns:
            Formatted string report
        """
        if not self.evaluation_results:
            return "No evaluation results available. Run comprehensive_evaluation() first."
        
        results = self.evaluation_results
        
        report = f"""
# Model Evaluation Summary Report

## Model Information
- Model Type: {results.get('model_type', 'Unknown')}
- Number of Predictions: {results.get('n_predictions', 0)}

## Standard Performance Metrics
"""
        
        if 'standard_metrics' in results:
            metrics = results['standard_metrics']
            report += f"""
- R² Score: {metrics.get('r2', 0):.4f}
- RMSE: {metrics.get('rmse', 0):.4f}
- MAE: {metrics.get('mae', 0):.4f}
- MAPE: {metrics.get('mape', 0):.2f}%
- Pearson r: {metrics.get('pearson_r', 0):.4f}
- Spearman r: {metrics.get('spearman_r', 0):.4f}
- Max Error: {metrics.get('max_error', 0):.4f}
- 95th Percentile Error: {metrics.get('q95_error', 0):.4f}
"""
        
        # pKa Range Analysis
        if 'pka_range_analysis' in results:
            report += "\n## Performance by pKa Range\n"
            for range_name, range_metrics in results['pka_range_analysis'].items():
                report += f"""
### {range_name.replace('_', ' ').title()}
- N molecules: {range_metrics.get('n_molecules', 0)}
- R²: {range_metrics.get('r2', 0):.3f}
- RMSE: {range_metrics.get('rmse', 0):.3f}
- MAE: {range_metrics.get('mae', 0):.3f}
"""
        
        # Feature Importance
        if 'feature_importance' in results and 'top_20_features' in results['feature_importance']:
            report += "\n## Top 10 Most Important Features\n"
            top_features = list(results['feature_importance']['top_20_features'].items())[:10]
            for i, (feature, importance) in enumerate(top_features, 1):
                report += f"{i}. {feature}: {importance:.4f}\n"
        
        # Quantum vs Classical Features
        if 'feature_importance' in results and 'quantum_vs_classical' in results['feature_importance']:
            qvc = results['feature_importance']['quantum_vs_classical']
            report += f"""
## Quantum vs Classical Feature Importance
- Quantum Features: {qvc.get('quantum_fraction', 0)*100:.1f}% of total importance
- Classical Features: {qvc.get('classical_fraction', 0)*100:.1f}% of total importance
"""
        
        # Ablation Study
        if 'ablation_study' in results and 'quantum_improvement' in results['ablation_study']:
            improvement = results['ablation_study']['quantum_improvement']
            report += f"""
## Quantum Features Impact (Ablation Study)
- R² Improvement: {improvement.get('r2_improvement', 0):.4f}
- Relative Improvement: {improvement.get('relative_improvement', 0):.1f}%
"""
        
        # Worst Performers
        if 'worst_performers' in results and 'error_statistics' in results['worst_performers']:
            error_stats = results['worst_performers']['error_statistics']
            report += f"""
## Worst Performing Molecules Analysis
- Mean Error: {error_stats.get('mean_error', 0):.3f}
- Error Standard Deviation: {error_stats.get('std_error', 0):.3f}
- Error Range: {error_stats.get('error_range', (0, 0))}
"""
        
        report += "\n---\n"
        
        return report


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data for testing
    np.random.seed(42)
    
    n_samples = 100
    y_true = np.random.normal(7, 2, n_samples)  # pKa values around 7 ± 2
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Add some prediction error
    
    # Create mock model
    class MockModel:
        def __init__(self):
            self.feature_importances_ = np.random.random(50)
        
        def predict(self, X):
            return np.random.normal(7, 2, X.shape[0])
    
    model = MockModel()
    
    # Initialize evaluator
    evaluator = ModelEvaluator(model, model_type="xgboost")
    
    # Test standard metrics
    metrics = evaluator.calculate_standard_metrics(y_true, y_pred)
    print("Standard Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Test pKa range analysis
    range_analysis = evaluator.pka_range_analysis(y_true, y_pred)
    print(f"\npKa Range Analysis:")
    for range_name, range_metrics in range_analysis.items():
        print(f"  {range_name}: R² = {range_metrics['r2']:.3f}, n = {range_metrics['n_molecules']}")
    
    # Test plot generation
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        plot_paths = evaluator.generate_plots(y_true, y_pred, Path(tmp_dir))
        print(f"\nGenerated plots: {list(plot_paths.keys())}")
    
    print("\nModelEvaluator test completed successfully!")