"""
Specialized metrics for pKa prediction that account for the logarithmic scale.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns


class pKaMetrics:
    """Specialized metrics for pKa prediction evaluation."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_concentration_ratio_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                           ph_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate errors in terms of concentration ratios, which is more chemically meaningful.
        
        For acids: Error in [A-]/[HA] ratio
        For bases: Error in [B]/[BH+] ratio
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values  
            ph_values: pH values to evaluate at (default: common physiological pH values)
            
        Returns:
            Dictionary with concentration ratio analysis
        """
        if ph_values is None:
            ph_values = np.array([1.0, 2.0, 6.0, 7.0, 7.4, 8.0, 10.0, 12.0, 14.0])
        
        results = {}
        
        for ph in ph_values:
            # Henderson-Hasselbalch equation: pH = pKa + log([A-]/[HA])
            # Therefore: [A-]/[HA] = 10^(pH - pKa)
            
            # True concentration ratios
            true_ratios = 10**(ph - y_true)
            
            # Predicted concentration ratios
            pred_ratios = 10**(ph - y_pred)
            
            # Ratio of predicted to true ratios (fold error)
            fold_errors = pred_ratios / true_ratios
            
            # Statistics
            results[f'pH_{ph}'] = {
                'mean_fold_error': np.mean(fold_errors),
                'median_fold_error': np.median(fold_errors),
                'fold_error_std': np.std(fold_errors),
                'max_fold_error': np.max(fold_errors),
                'min_fold_error': np.min(fold_errors),
                'percent_within_2fold': np.mean((fold_errors >= 0.5) & (fold_errors <= 2.0)) * 100,
                'percent_within_5fold': np.mean((fold_errors >= 0.2) & (fold_errors <= 5.0)) * 100,
                'percent_within_10fold': np.mean((fold_errors >= 0.1) & (fold_errors <= 10.0)) * 100,
                'true_ratios': true_ratios,
                'pred_ratios': pred_ratios,
                'fold_errors': fold_errors
            }
        
        return results
    
    def calculate_ionization_state_errors(self, y_true: np.ndarray, y_pred: np.ndarray,
                                        ph_values: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Calculate errors in ionization state fractions.
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values
            ph_values: pH values to evaluate at
            
        Returns:
            Dictionary with ionization state analysis
        """
        if ph_values is None:
            ph_values = np.array([1.0, 2.0, 6.0, 7.0, 7.4, 8.0, 10.0, 12.0, 14.0])
        
        results = {}
        
        for ph in ph_values:
            # Fraction ionized (for acids, this is [A-]/([HA] + [A-]))
            # Alpha = 1 / (1 + 10^(pKa - pH))
            
            true_alpha = 1 / (1 + 10**(y_true - ph))
            pred_alpha = 1 / (1 + 10**(y_pred - ph))
            
            # Absolute errors in ionization fraction
            alpha_errors = np.abs(pred_alpha - true_alpha)
            
            results[f'pH_{ph}'] = {
                'mean_alpha_error': np.mean(alpha_errors),
                'median_alpha_error': np.median(alpha_errors),
                'max_alpha_error': np.max(alpha_errors),
                'rmse_alpha': np.sqrt(np.mean(alpha_errors**2)),
                'percent_within_0.1': np.mean(alpha_errors <= 0.1) * 100,
                'percent_within_0.05': np.mean(alpha_errors <= 0.05) * 100,
                'percent_within_0.01': np.mean(alpha_errors <= 0.01) * 100,
                'true_alpha': true_alpha,
                'pred_alpha': pred_alpha,
                'alpha_errors': alpha_errors
            }
        
        return results
    
    def calculate_pka_range_specific_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate metrics specific to different pKa ranges with chemical interpretation.
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values
            
        Returns:
            Dictionary with range-specific metrics
        """
        # Define pKa ranges with chemical meaning
        ranges = {
            'strong_acids': (0, 2),        # Very strong acids (mineral acids)
            'weak_acids': (2, 5),          # Carboxylic acids, etc.
            'very_weak_acids': (5, 7),     # Very weak acids
            'near_neutral': (7, 9),        # Near neutral compounds
            'weak_bases': (9, 11),         # Amines, etc.
            'strong_bases': (11, 14)       # Very strong bases
        }
        
        results = {}
        
        for range_name, (min_pka, max_pka) in ranges.items():
            mask = (y_true >= min_pka) & (y_true < max_pka)
            
            if np.sum(mask) > 0:
                range_true = y_true[mask]
                range_pred = y_pred[mask]
                
                # Standard metrics
                mae = mean_absolute_error(range_true, range_pred)
                rmse = np.sqrt(mean_squared_error(range_true, range_pred))
                r2 = r2_score(range_true, range_pred) if len(range_true) > 1 else 0
                
                # Chemical interpretation
                # For this range, what does the error mean?
                avg_true_pka = np.mean(range_true)
                avg_error = np.mean(range_pred - range_true)
                
                # At pH 7, what's the typical error in ionization fraction?
                ph_7_true_alpha = 1 / (1 + 10**(range_true - 7))
                ph_7_pred_alpha = 1 / (1 + 10**(range_pred - 7))
                avg_alpha_error_ph7 = np.mean(np.abs(ph_7_pred_alpha - ph_7_true_alpha))
                
                # At physiological pH (7.4)
                ph_74_true_alpha = 1 / (1 + 10**(range_true - 7.4))
                ph_74_pred_alpha = 1 / (1 + 10**(range_pred - 7.4))
                avg_alpha_error_ph74 = np.mean(np.abs(ph_74_pred_alpha - ph_74_true_alpha))
                
                results[range_name] = {
                    'n_molecules': np.sum(mask),
                    'mae': mae,
                    'rmse': rmse,
                    'r2': r2,
                    'mean_true_pka': avg_true_pka,
                    'mean_error': avg_error,
                    'std_error': np.std(range_pred - range_true),
                    'max_error': np.max(np.abs(range_pred - range_true)),
                    'chemical_interpretation': {
                        'avg_alpha_error_ph7': avg_alpha_error_ph7,
                        'avg_alpha_error_ph74': avg_alpha_error_ph74,
                        'typical_fold_error_ph7': np.mean(10**(np.abs(range_pred - range_true))),
                        'chemical_meaning': self._get_chemical_meaning(range_name, mae, avg_alpha_error_ph7)
                    }
                }
        
        return results
    
    def _get_chemical_meaning(self, range_name: str, mae: float, alpha_error: float) -> str:
        """Get chemical interpretation of errors."""
        if mae < 0.3:
            error_level = "excellent"
        elif mae < 0.5:
            error_level = "good"
        elif mae < 1.0:
            error_level = "moderate"
        elif mae < 2.0:
            error_level = "poor"
        else:
            error_level = "very poor"
        
        if alpha_error < 0.05:
            alpha_level = "excellent"
        elif alpha_error < 0.1:
            alpha_level = "good"
        elif alpha_error < 0.2:
            alpha_level = "moderate"
        else:
            alpha_level = "poor"
        
        return f"{error_level} pKa prediction ({alpha_level} ionization state prediction at pH 7)"
    
    def calculate_logarithmic_scale_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics that account for the logarithmic nature of pKa.
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values
            
        Returns:
            Dictionary with logarithmic scale metrics
        """
        # Standard metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        # Log-scale specific metrics
        # Mean absolute error in log concentration units
        log_conc_mae = np.mean(np.abs(y_pred - y_true))  # This is the same as MAE but emphasizes the log scale
        
        # Mean fold error (geometric mean of 10^|error|)
        fold_errors = 10**(np.abs(y_pred - y_true))
        mean_fold_error = np.mean(fold_errors)
        median_fold_error = np.median(fold_errors)
        
        # Percentage within certain fold errors
        pct_within_2fold = np.mean(fold_errors <= 2.0) * 100
        pct_within_5fold = np.mean(fold_errors <= 5.0) * 100
        pct_within_10fold = np.mean(fold_errors <= 10.0) * 100
        
        # Percentage within certain pKa units
        abs_errors = np.abs(y_pred - y_true)
        pct_within_0_3 = np.mean(abs_errors <= 0.3) * 100
        pct_within_0_5 = np.mean(abs_errors <= 0.5) * 100
        pct_within_1_0 = np.mean(abs_errors <= 1.0) * 100
        
        # Relative error in terms of Ka (acid dissociation constant)
        # Ka = 10^(-pKa), so relative error = |Ka_pred - Ka_true| / Ka_true
        ka_true = 10**(-y_true)
        ka_pred = 10**(-y_pred)
        relative_ka_errors = np.abs(ka_pred - ka_true) / ka_true
        mean_relative_ka_error = np.mean(relative_ka_errors)
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'log_conc_mae': log_conc_mae,
            'mean_fold_error': mean_fold_error,
            'median_fold_error': median_fold_error,
            'pct_within_2fold': pct_within_2fold,
            'pct_within_5fold': pct_within_5fold,
            'pct_within_10fold': pct_within_10fold,
            'pct_within_0_3_pka': pct_within_0_3,
            'pct_within_0_5_pka': pct_within_0_5,
            'pct_within_1_0_pka': pct_within_1_0,
            'mean_relative_ka_error': mean_relative_ka_error
        }
    
    def calculate_chemical_accuracy_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Calculate metrics relevant to chemical accuracy and practical applications.
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values
            
        Returns:
            Dictionary with chemical accuracy metrics
        """
        results = {}
        
        # 1. Accuracy for drug design (physiological pH relevance)
        physiological_ph = 7.4
        
        # Ionization state accuracy at physiological pH
        true_alpha_phys = 1 / (1 + 10**(y_true - physiological_ph))
        pred_alpha_phys = 1 / (1 + 10**(y_pred - physiological_ph))
        alpha_error_phys = np.abs(pred_alpha_phys - true_alpha_phys)
        
        results['physiological_relevance'] = {
            'mean_alpha_error_ph74': np.mean(alpha_error_phys),
            'median_alpha_error_ph74': np.median(alpha_error_phys),
            'pct_within_0_05_alpha': np.mean(alpha_error_phys <= 0.05) * 100,
            'pct_within_0_1_alpha': np.mean(alpha_error_phys <= 0.1) * 100,
            'pct_within_0_2_alpha': np.mean(alpha_error_phys <= 0.2) * 100
        }
        
        # 2. Accuracy for environmental applications (pH range 6-8)
        env_ph_range = np.array([6.0, 6.5, 7.0, 7.5, 8.0])
        env_errors = []
        
        for ph in env_ph_range:
            true_alpha = 1 / (1 + 10**(y_true - ph))
            pred_alpha = 1 / (1 + 10**(y_pred - ph))
            env_errors.append(np.mean(np.abs(pred_alpha - true_alpha)))
        
        results['environmental_relevance'] = {
            'mean_alpha_error_env_range': np.mean(env_errors),
            'max_alpha_error_env_range': np.max(env_errors)
        }
        
        # 3. Accuracy for analytical chemistry (buffer capacity)
        # Buffer capacity is maximum when pH = pKa
        buffer_capacity_errors = []
        
        for true_pka, pred_pka in zip(y_true, y_pred):
            # At true pKa, buffer capacity is maximum
            # Error is how far off we are from predicting the optimal buffer pH
            buffer_capacity_errors.append(abs(pred_pka - true_pka))
        
        results['buffer_capacity'] = {
            'mean_buffer_ph_error': np.mean(buffer_capacity_errors),
            'pct_within_0_5_buffer': np.mean(np.array(buffer_capacity_errors) <= 0.5) * 100,
            'pct_within_1_0_buffer': np.mean(np.array(buffer_capacity_errors) <= 1.0) * 100
        }
        
        # 4. Solubility prediction accuracy
        # Solubility depends on ionization state, especially for weak acids/bases
        # This is a simplified analysis
        solubility_relevant_errors = []
        
        for true_pka, pred_pka in zip(y_true, y_pred):
            # For compounds with pKa in range 2-12, ionization significantly affects solubility
            if 2 <= true_pka <= 12:
                # At pH 7, error in ionization fraction
                true_alpha = 1 / (1 + 10**(true_pka - 7))
                pred_alpha = 1 / (1 + 10**(pred_pka - 7))
                solubility_relevant_errors.append(abs(pred_alpha - true_alpha))
        
        if solubility_relevant_errors:
            results['solubility_prediction'] = {
                'mean_ionization_error': np.mean(solubility_relevant_errors),
                'pct_within_0_1_ionization': np.mean(np.array(solubility_relevant_errors) <= 0.1) * 100,
                'pct_within_0_2_ionization': np.mean(np.array(solubility_relevant_errors) <= 0.2) * 100
            }
        
        return results
    
    def generate_comprehensive_report(self, y_true: np.ndarray, y_pred: np.ndarray) -> str:
        """
        Generate a comprehensive report on pKa prediction accuracy.
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values
            
        Returns:
            Formatted report string
        """
        # Calculate all metrics
        log_metrics = self.calculate_logarithmic_scale_metrics(y_true, y_pred)
        range_metrics = self.calculate_pka_range_specific_metrics(y_true, y_pred)
        chem_metrics = self.calculate_chemical_accuracy_metrics(y_true, y_pred)
        conc_metrics = self.calculate_concentration_ratio_errors(y_true, y_pred)
        
        report = """
# Comprehensive pKa Prediction Accuracy Report

## Overall Performance Summary
"""
        
        report += f"""
### Standard Metrics
- **MAE**: {log_metrics['mae']:.3f} pKa units
- **RMSE**: {log_metrics['rmse']:.3f} pKa units  
- **R²**: {log_metrics['r2']:.3f}

### Logarithmic Scale Metrics
- **Mean Fold Error**: {log_metrics['mean_fold_error']:.2f}x
- **Median Fold Error**: {log_metrics['median_fold_error']:.2f}x
- **Within 2-fold**: {log_metrics['pct_within_2fold']:.1f}%
- **Within 5-fold**: {log_metrics['pct_within_5fold']:.1f}%
- **Within 10-fold**: {log_metrics['pct_within_10fold']:.1f}%

### Chemical Accuracy Thresholds
- **Within 0.3 pKa units**: {log_metrics['pct_within_0_3_pka']:.1f}% (excellent)
- **Within 0.5 pKa units**: {log_metrics['pct_within_0_5_pka']:.1f}% (good)
- **Within 1.0 pKa units**: {log_metrics['pct_within_1_0_pka']:.1f}% (acceptable)
"""
        
        report += "\n## Performance by pKa Range\n"
        
        for range_name, metrics in range_metrics.items():
            if metrics['n_molecules'] > 0:
                report += f"""
### {range_name.replace('_', ' ').title()} ({metrics['n_molecules']} molecules)
- **MAE**: {metrics['mae']:.3f} pKa units
- **RMSE**: {metrics['rmse']:.3f} pKa units
- **R²**: {metrics['r2']:.3f}
- **Chemical Impact**: {metrics['chemical_interpretation']['chemical_meaning']}
- **Ionization Error at pH 7**: {metrics['chemical_interpretation']['avg_alpha_error_ph7']:.3f}
- **Typical Fold Error**: {metrics['chemical_interpretation']['typical_fold_error_ph7']:.2f}x
"""
        
        report += "\n## Practical Application Accuracy\n"
        
        if 'physiological_relevance' in chem_metrics:
            phys = chem_metrics['physiological_relevance']
            report += f"""
### Drug Design (pH 7.4)
- **Mean Ionization Error**: {phys['mean_alpha_error_ph74']:.3f}
- **Within 5% ionization**: {phys['pct_within_0_05_alpha']:.1f}%
- **Within 10% ionization**: {phys['pct_within_0_1_alpha']:.1f}%
- **Within 20% ionization**: {phys['pct_within_0_2_alpha']:.1f}%
"""
        
        if 'environmental_relevance' in chem_metrics:
            env = chem_metrics['environmental_relevance']
            report += f"""
### Environmental Applications (pH 6-8)
- **Mean Ionization Error**: {env['mean_alpha_error_env_range']:.3f}
- **Max Ionization Error**: {env['max_alpha_error_env_range']:.3f}
"""
        
        if 'buffer_capacity' in chem_metrics:
            buffer = chem_metrics['buffer_capacity']
            report += f"""
### Buffer Design
- **Mean Buffer pH Error**: {buffer['mean_buffer_ph_error']:.3f}
- **Within 0.5 pH units**: {buffer['pct_within_0_5_buffer']:.1f}%
- **Within 1.0 pH units**: {buffer['pct_within_1_0_buffer']:.1f}%
"""
        
        report += "\n## Concentration Ratio Errors (Most Chemically Relevant)\n"
        
        key_ph_values = [7.0, 7.4]  # Focus on most relevant pH values
        for ph in key_ph_values:
            if f'pH_{ph}' in conc_metrics:
                ph_data = conc_metrics[f'pH_{ph}']
                report += f"""
### pH {ph}
- **Mean Fold Error**: {ph_data['mean_fold_error']:.2f}x
- **Median Fold Error**: {ph_data['median_fold_error']:.2f}x
- **Within 2-fold**: {ph_data['percent_within_2fold']:.1f}%
- **Within 5-fold**: {ph_data['percent_within_5fold']:.1f}%
- **Within 10-fold**: {ph_data['percent_within_10fold']:.1f}%
"""
        
        report += "\n## Performance Interpretation\n"
        
        # Overall assessment
        mae = log_metrics['mae']
        fold_2_pct = log_metrics['pct_within_2fold']
        
        if mae < 0.3 and fold_2_pct > 80:
            assessment = "EXCELLENT - Suitable for all applications"
        elif mae < 0.5 and fold_2_pct > 60:
            assessment = "GOOD - Suitable for most applications"
        elif mae < 1.0 and fold_2_pct > 40:
            assessment = "ACCEPTABLE - Suitable for screening applications"
        elif mae < 2.0:
            assessment = "POOR - Limited practical utility"
        else:
            assessment = "VERY POOR - Not suitable for practical applications"
        
        report += f"""
**Overall Assessment**: {assessment}

### Chemical Significance of Errors:
- A 0.3 pKa unit error corresponds to ~2x error in concentration ratios
- A 0.5 pKa unit error corresponds to ~3x error in concentration ratios  
- A 1.0 pKa unit error corresponds to ~10x error in concentration ratios
- A 2.0 pKa unit error corresponds to ~100x error in concentration ratios

### Recommendations:
- For drug design: Aim for <0.5 pKa units MAE
- For environmental modeling: Aim for <1.0 pKa units MAE
- For screening applications: <2.0 pKa units MAE may be acceptable
- For buffer design: <0.3 pKa units MAE is preferred
"""
        
        return report
    
    def plot_comprehensive_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  save_path: Optional[str] = None) -> None:
        """
        Create comprehensive visualization of pKa prediction accuracy.
        
        Args:
            y_true: True pKa values
            y_pred: Predicted pKa values
            save_path: Optional path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Parity plot with fold error shading
        ax1 = axes[0, 0]
        ax1.scatter(y_true, y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        # Fold error bands
        x_line = np.linspace(min_val, max_val, 100)
        ax1.fill_between(x_line, x_line - 0.30, x_line + 0.30, alpha=0.2, color='green', label='2-fold error')
        ax1.fill_between(x_line, x_line - 0.70, x_line + 0.70, alpha=0.1, color='orange', label='5-fold error')
        
        ax1.set_xlabel('True pKa')
        ax1.set_ylabel('Predicted pKa')
        ax1.set_title('Parity Plot with Fold Error Bands')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Fold error distribution
        ax2 = axes[0, 1]
        fold_errors = 10**(np.abs(y_pred - y_true))
        ax2.hist(fold_errors, bins=30, alpha=0.7, density=True, color='skyblue', edgecolor='black')
        ax2.axvline(2, color='green', linestyle='--', lw=2, label='2-fold')
        ax2.axvline(5, color='orange', linestyle='--', lw=2, label='5-fold')
        ax2.axvline(10, color='red', linestyle='--', lw=2, label='10-fold')
        ax2.set_xlabel('Fold Error')
        ax2.set_ylabel('Density')
        ax2.set_title('Distribution of Fold Errors')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Ionization state errors at pH 7.4
        ax3 = axes[0, 2]
        true_alpha = 1 / (1 + 10**(y_true - 7.4))
        pred_alpha = 1 / (1 + 10**(y_pred - 7.4))
        alpha_errors = np.abs(pred_alpha - true_alpha)
        
        ax3.scatter(true_alpha, alpha_errors, alpha=0.6, s=50, c=y_true, cmap='viridis')
        ax3.axhline(0.05, color='green', linestyle='--', label='5% error')
        ax3.axhline(0.1, color='orange', linestyle='--', label='10% error')
        ax3.axhline(0.2, color='red', linestyle='--', label='20% error')
        ax3.set_xlabel('True Ionization Fraction (pH 7.4)')
        ax3.set_ylabel('Absolute Error in Ionization Fraction')
        ax3.set_title('Ionization State Prediction Errors')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Error vs pKa value
        ax4 = axes[1, 0]
        abs_errors = np.abs(y_pred - y_true)
        ax4.scatter(y_true, abs_errors, alpha=0.6, s=50, c=fold_errors, cmap='Reds')
        ax4.axhline(0.3, color='green', linestyle='--', label='0.3 pKa (2-fold)')
        ax4.axhline(0.5, color='orange', linestyle='--', label='0.5 pKa (3-fold)')
        ax4.axhline(1.0, color='red', linestyle='--', label='1.0 pKa (10-fold)')
        ax4.set_xlabel('True pKa')
        ax4.set_ylabel('Absolute Error (pKa units)')
        ax4.set_title('Error vs pKa Value')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Concentration ratio errors at different pH
        ax5 = axes[1, 1]
        ph_values = np.array([6.0, 7.0, 7.4, 8.0])
        mean_fold_errors = []
        
        for ph in ph_values:
            true_ratios = 10**(ph - y_true)
            pred_ratios = 10**(ph - y_pred)
            fold_errs = pred_ratios / true_ratios
            mean_fold_errors.append(np.mean(fold_errs))
        
        ax5.plot(ph_values, mean_fold_errors, 'o-', markersize=8, linewidth=2)
        ax5.axhline(2, color='green', linestyle='--', label='2-fold error')
        ax5.axhline(5, color='orange', linestyle='--', label='5-fold error')
        ax5.set_xlabel('pH')
        ax5.set_ylabel('Mean Fold Error in Concentration Ratios')
        ax5.set_title('Concentration Ratio Errors vs pH')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Performance by pKa range
        ax6 = axes[1, 2]
        ranges = [(0, 2), (2, 5), (5, 7), (7, 9), (9, 11), (11, 14)]
        range_names = ['Strong\nAcids', 'Weak\nAcids', 'Very Weak\nAcids', 
                      'Near\nNeutral', 'Weak\nBases', 'Strong\nBases']
        range_maes = []
        range_counts = []
        
        for min_pka, max_pka in ranges:
            mask = (y_true >= min_pka) & (y_true < max_pka)
            if np.sum(mask) > 0:
                range_mae = mean_absolute_error(y_true[mask], y_pred[mask])
                range_maes.append(range_mae)
                range_counts.append(np.sum(mask))
            else:
                range_maes.append(0)
                range_counts.append(0)
        
        bars = ax6.bar(range_names, range_maes, alpha=0.7, color='lightblue', edgecolor='black')
        
        # Add count labels on bars
        for bar, count in zip(bars, range_counts):
            if count > 0:
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        f'n={count}', ha='center', va='bottom', fontsize=10)
        
        ax6.axhline(0.3, color='green', linestyle='--', label='0.3 pKa (excellent)')
        ax6.axhline(0.5, color='orange', linestyle='--', label='0.5 pKa (good)')
        ax6.axhline(1.0, color='red', linestyle='--', label='1.0 pKa (acceptable)')
        ax6.set_ylabel('MAE (pKa units)')
        ax6.set_title('Performance by pKa Range')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


if __name__ == "__main__":
    # Test the pKa metrics
    np.random.seed(42)
    
    # Generate synthetic data for testing
    n_samples = 50
    y_true = np.random.uniform(2, 12, n_samples)
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Add some error
    
    # Initialize metrics calculator
    pka_metrics = pKaMetrics()
    
    # Test logarithmic scale metrics
    log_metrics = pka_metrics.calculate_logarithmic_scale_metrics(y_true, y_pred)
    print("Logarithmic Scale Metrics:")
    for key, value in log_metrics.items():
        print(f"  {key}: {value:.3f}")
    
    # Test comprehensive report
    report = pka_metrics.generate_comprehensive_report(y_true, y_pred)
    print("\n" + "="*50)
    print("COMPREHENSIVE REPORT")
    print("="*50)
    print(report)
    
    # Test visualization
    pka_metrics.plot_comprehensive_analysis(y_true, y_pred)
    
    print("\npKa metrics testing completed!")