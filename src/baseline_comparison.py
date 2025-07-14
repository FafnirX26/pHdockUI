"""
Baseline comparison framework for pKa prediction models.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Callable
import logging
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class BaselineComparison:
    """Framework for comparing models against various baselines."""
    
    def __init__(self, logger=None):
        """
        Initialize baseline comparison framework.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.baselines = {}
        self.comparison_results = {}
        
    def create_simple_baselines(self, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Create simple statistical baselines.
        
        Args:
            y_train: Training target values
            
        Returns:
            Dictionary of baseline models
        """
        baselines = {}
        
        # Mean baseline
        baselines['mean'] = DummyRegressor(strategy='mean')
        baselines['mean'].fit(np.zeros((len(y_train), 1)), y_train)
        
        # Median baseline
        baselines['median'] = DummyRegressor(strategy='median')
        baselines['median'].fit(np.zeros((len(y_train), 1)), y_train)
        
        # Constant baseline (pKa = 7, neutral pH)
        baselines['neutral_ph'] = DummyRegressor(strategy='constant', constant=7.0)
        baselines['neutral_ph'].fit(np.zeros((len(y_train), 1)), y_train)
        
        return baselines
    
    def create_simple_feature_baselines(self, X_train: np.ndarray, y_train: np.ndarray,
                                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Create baselines using simple features and classical ML models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            feature_names: Optional feature names
            
        Returns:
            Dictionary of baseline models
        """
        baselines = {}
        
        # Linear regression baseline
        baselines['linear'] = LinearRegression()
        baselines['linear'].fit(X_train, y_train)
        
        # Ridge regression baseline
        baselines['ridge'] = Ridge(alpha=1.0, random_state=42)
        baselines['ridge'].fit(X_train, y_train)
        
        # Random Forest baseline
        baselines['random_forest'] = RandomForestRegressor(
            n_estimators=50, max_depth=10, random_state=42, n_jobs=-1
        )
        baselines['random_forest'].fit(X_train, y_train)
        
        # XGBoost baseline (simple configuration)
        baselines['xgboost_simple'] = xgb.XGBRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1, 
            random_state=42, n_jobs=-1
        )
        baselines['xgboost_simple'].fit(X_train, y_train)
        
        return baselines
    
    def create_molecular_descriptor_baselines(self, molecules: List[Chem.Mol],
                                            y_train: np.ndarray) -> Dict[str, Any]:
        """
        Create baselines using only basic molecular descriptors.
        
        Args:
            molecules: List of RDKit molecules
            y_train: Training targets
            
        Returns:
            Dictionary of baseline models
        """
        # Calculate basic molecular descriptors
        basic_descriptors = []
        
        for mol in molecules:
            if mol is not None:
                desc = {
                    'mol_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': rdMolDescriptors.CalcTPSA(mol),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                    'num_hba': rdMolDescriptors.CalcNumHBA(mol),
                    'num_hbd': rdMolDescriptors.CalcNumHBD(mol)
                }
            else:
                desc = {key: 0 for key in ['mol_weight', 'logp', 'tpsa', 'num_heavy_atoms',
                                         'num_rings', 'num_aromatic_rings', 'num_rotatable_bonds',
                                         'num_hba', 'num_hbd']}
            
            basic_descriptors.append(desc)
        
        X_basic = pd.DataFrame(basic_descriptors).values
        
        # Handle NaN values
        X_basic = np.nan_to_num(X_basic, nan=0.0)
        
        baselines = {}
        
        # Basic descriptors + Linear regression
        baselines['basic_linear'] = LinearRegression()
        baselines['basic_linear'].fit(X_basic, y_train)
        
        # Basic descriptors + Random Forest
        baselines['basic_rf'] = RandomForestRegressor(
            n_estimators=50, max_depth=8, random_state=42, n_jobs=-1
        )
        baselines['basic_rf'].fit(X_basic, y_train)
        
        # Basic descriptors + XGBoost
        baselines['basic_xgb'] = xgb.XGBRegressor(
            n_estimators=50, max_depth=4, learning_rate=0.1,
            random_state=42, n_jobs=-1
        )
        baselines['basic_xgb'].fit(X_basic, y_train)
        
        # Store feature matrix for later use
        self._basic_features = X_basic
        
        return baselines
    
    def create_literature_baselines(self) -> Dict[str, Callable]:
        """
        Create baselines based on literature pKa prediction methods.
        
        Returns:
            Dictionary of baseline prediction functions
        """
        baselines = {}
        
        # Simple group contribution method
        def group_contribution_pka(mol):
            """Simple group contribution estimate."""
            if mol is None:
                return 7.0
            
            # Basic group contributions (simplified)
            pka = 7.0  # Start with neutral
            
            # Count functional groups
            carboxyl_pattern = Chem.MolFromSmarts('C(=O)O')
            phenol_pattern = Chem.MolFromSmarts('c1ccccc1O')
            amine_pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
            
            if mol.HasSubstructMatch(carboxyl_pattern):
                pka = 4.2  # Typical carboxylic acid pKa
            elif mol.HasSubstructMatch(phenol_pattern):
                pka = 9.8  # Typical phenol pKa
            elif mol.HasSubstructMatch(amine_pattern):
                pka = 10.2  # Typical amine pKa
            
            return pka
        
        baselines['group_contribution'] = group_contribution_pka
        
        # Hammett equation approximation
        def hammett_approximation(mol):
            """Simple Hammett equation approximation."""
            if mol is None:
                return 7.0
            
            base_pka = 4.2  # Benzoic acid as reference
            
            # Count electron-withdrawing and donating groups (simplified)
            ewg_patterns = [
                'C(=O)O', 'C#N', '[N+](=O)[O-]', 'C(F)(F)F'
            ]
            edg_patterns = [
                'N(C)C', 'OC', 'C'
            ]
            
            sigma = 0  # Hammett constant sum
            
            for pattern_smarts in ewg_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    sigma += 0.4  # Approximate sigma value
            
            for pattern_smarts in edg_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern and mol.HasSubstructMatch(pattern):
                    sigma -= 0.2  # Approximate sigma value
            
            # Hammett equation: pKa = pKa0 - rho * sigma
            rho = 1.0  # Reaction constant
            pka = base_pka - rho * sigma
            
            return max(0, min(14, pka))  # Clamp to pH range
        
        baselines['hammett'] = hammett_approximation
        
        return baselines
    
    def evaluate_baselines(self, X_test: np.ndarray, y_test: np.ndarray,
                          molecules_test: Optional[List[Chem.Mol]] = None) -> Dict[str, Dict[str, float]]:
        """
        Evaluate all baseline models.
        
        Args:
            X_test: Test features
            y_test: Test targets
            molecules_test: Optional test molecules for structure-based baselines
            
        Returns:
            Dictionary with evaluation results for each baseline
        """
        results = {}
        
        # Evaluate sklearn-based baselines
        for name, model in self.baselines.items():
            try:
                if name in ['group_contribution', 'hammett']:
                    # Structure-based baselines
                    if molecules_test is not None:
                        y_pred = np.array([model(mol) for mol in molecules_test])
                    else:
                        continue
                elif name.startswith('basic_'):
                    # Basic descriptor baselines
                    if hasattr(self, '_basic_features'):
                        # Use basic features for prediction
                        test_basic = self._calculate_basic_features(molecules_test) if molecules_test else X_test
                        y_pred = model.predict(test_basic)
                    else:
                        continue
                else:
                    # Regular feature-based baselines
                    y_pred = model.predict(X_test)
                
                # Calculate metrics
                metrics = {
                    'r2': r2_score(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'mae': mean_absolute_error(y_test, y_pred),
                    'mean_error': np.mean(y_pred - y_test),
                    'std_error': np.std(y_pred - y_test)
                }
                
                results[name] = metrics
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate baseline {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results
    
    def _calculate_basic_features(self, molecules: List[Chem.Mol]) -> np.ndarray:
        """Calculate basic molecular descriptors for molecules."""
        basic_descriptors = []
        
        for mol in molecules:
            if mol is not None:
                desc = {
                    'mol_weight': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'tpsa': rdMolDescriptors.CalcTPSA(mol),
                    'num_heavy_atoms': mol.GetNumHeavyAtoms(),
                    'num_rings': rdMolDescriptors.CalcNumRings(mol),
                    'num_aromatic_rings': rdMolDescriptors.CalcNumAromaticRings(mol),
                    'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
                    'num_hba': rdMolDescriptors.CalcNumHBA(mol),
                    'num_hbd': rdMolDescriptors.CalcNumHBD(mol)
                }
            else:
                desc = {key: 0 for key in ['mol_weight', 'logp', 'tpsa', 'num_heavy_atoms',
                                         'num_rings', 'num_aromatic_rings', 'num_rotatable_bonds',
                                         'num_hba', 'num_hbd']}
            
            basic_descriptors.append(desc)
        
        X_basic = pd.DataFrame(basic_descriptors).values
        return np.nan_to_num(X_basic, nan=0.0)
    
    def compare_with_model(self, model_predictions: np.ndarray, y_test: np.ndarray,
                          model_name: str = "Target Model") -> Dict[str, Any]:
        """
        Compare target model performance against all baselines.
        
        Args:
            model_predictions: Predictions from the target model
            y_test: Test targets
            model_name: Name of the target model
            
        Returns:
            Comprehensive comparison results
        """
        # Calculate target model metrics
        target_metrics = {
            'r2': r2_score(y_test, model_predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, model_predictions)),
            'mae': mean_absolute_error(y_test, model_predictions),
            'mean_error': np.mean(model_predictions - y_test),
            'std_error': np.std(model_predictions - y_test)
        }
        
        # Get baseline results
        baseline_results = self.comparison_results
        
        # Calculate improvements over baselines
        improvements = {}
        
        for baseline_name, baseline_metrics in baseline_results.items():
            if 'error' not in baseline_metrics:
                improvements[baseline_name] = {}
                
                for metric in ['r2', 'rmse', 'mae']:
                    if metric in baseline_metrics and metric in target_metrics:
                        baseline_val = baseline_metrics[metric]
                        target_val = target_metrics[metric]
                        
                        if metric == 'r2':
                            # For R², improvement is target - baseline
                            improvement = target_val - baseline_val
                            relative_improvement = (improvement / baseline_val * 100) if baseline_val != 0 else 0
                        else:
                            # For RMSE and MAE, improvement is baseline - target (lower is better)
                            improvement = baseline_val - target_val
                            relative_improvement = (improvement / baseline_val * 100) if baseline_val != 0 else 0
                        
                        improvements[baseline_name][metric] = {
                            'absolute': improvement,
                            'relative': relative_improvement
                        }
        
        comparison_results = {
            'target_model': {
                'name': model_name,
                'metrics': target_metrics
            },
            'baselines': baseline_results,
            'improvements': improvements,
            'ranking': self._rank_models(target_metrics, baseline_results, model_name)
        }
        
        return comparison_results
    
    def _rank_models(self, target_metrics: Dict[str, float], 
                    baseline_results: Dict[str, Dict[str, float]],
                    model_name: str) -> List[Dict[str, Any]]:
        """
        Rank all models by performance.
        
        Args:
            target_metrics: Target model metrics
            baseline_results: Baseline model results
            model_name: Name of target model
            
        Returns:
            List of models ranked by R² score
        """
        all_models = [(model_name, target_metrics)]
        
        for baseline_name, metrics in baseline_results.items():
            if 'error' not in metrics and 'r2' in metrics:
                all_models.append((baseline_name, metrics))
        
        # Sort by R² score (descending)
        ranked = sorted(all_models, key=lambda x: x[1].get('r2', -999), reverse=True)
        
        ranking = []
        for i, (name, metrics) in enumerate(ranked, 1):
            ranking.append({
                'rank': i,
                'model': name,
                'r2': metrics.get('r2', 0),
                'rmse': metrics.get('rmse', float('inf')),
                'mae': metrics.get('mae', float('inf'))
            })
        
        return ranking
    
    def plot_comparison(self, comparison_results: Dict[str, Any],
                       save_path: Optional[Path] = None) -> None:
        """
        Create comparison plots.
        
        Args:
            comparison_results: Results from compare_with_model
            save_path: Optional path to save plot
        """
        # Extract data for plotting
        models = []
        r2_scores = []
        rmse_scores = []
        mae_scores = []
        
        # Add target model
        target = comparison_results['target_model']
        models.append(target['name'])
        r2_scores.append(target['metrics']['r2'])
        rmse_scores.append(target['metrics']['rmse'])
        mae_scores.append(target['metrics']['mae'])
        
        # Add baselines
        for name, metrics in comparison_results['baselines'].items():
            if 'error' not in metrics:
                models.append(name.replace('_', ' ').title())
                r2_scores.append(metrics.get('r2', 0))
                rmse_scores.append(metrics.get('rmse', 0))
                mae_scores.append(metrics.get('mae', 0))
        
        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # R² comparison
        colors = ['red' if model == target['name'] else 'skyblue' for model in models]
        bars1 = axes[0].bar(range(len(models)), r2_scores, color=colors, alpha=0.7)
        axes[0].set_xlabel('Models')
        axes[0].set_ylabel('R² Score')
        axes[0].set_title('R² Score Comparison')
        axes[0].set_xticks(range(len(models)))
        axes[0].set_xticklabels(models, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, score) in enumerate(zip(bars1, r2_scores)):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # RMSE comparison
        bars2 = axes[1].bar(range(len(models)), rmse_scores, color=colors, alpha=0.7)
        axes[1].set_xlabel('Models')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('RMSE Comparison')
        axes[1].set_xticks(range(len(models)))
        axes[1].set_xticklabels(models, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars2, rmse_scores)):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        # MAE comparison
        bars3 = axes[2].bar(range(len(models)), mae_scores, color=colors, alpha=0.7)
        axes[2].set_xlabel('Models')
        axes[2].set_ylabel('MAE')
        axes[2].set_title('MAE Comparison')
        axes[2].set_xticks(range(len(models)))
        axes[2].set_xticklabels(models, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3)
        
        for i, (bar, score) in enumerate(zip(bars3, mae_scores)):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_comparison_report(self, comparison_results: Dict[str, Any]) -> str:
        """
        Generate a text report of the comparison results.
        
        Args:
            comparison_results: Results from compare_with_model
            
        Returns:
            Formatted comparison report
        """
        target = comparison_results['target_model']
        
        report = f"""
# Model Performance Comparison Report

## Target Model: {target['name']}
- R² Score: {target['metrics']['r2']:.4f}
- RMSE: {target['metrics']['rmse']:.4f}
- MAE: {target['metrics']['mae']:.4f}

## Performance Ranking
"""
        
        for rank_info in comparison_results['ranking']:
            report += f"{rank_info['rank']}. {rank_info['model']}: "
            report += f"R² = {rank_info['r2']:.3f}, "
            report += f"RMSE = {rank_info['rmse']:.3f}, "
            report += f"MAE = {rank_info['mae']:.3f}\n"
        
        report += "\n## Improvements Over Baselines\n"
        
        for baseline_name, improvements in comparison_results['improvements'].items():
            if improvements:  # Skip empty improvements
                report += f"\n### vs. {baseline_name.replace('_', ' ').title()}\n"
                for metric, improvement in improvements.items():
                    report += f"- {metric.upper()}: {improvement['absolute']:+.4f} "
                    report += f"({improvement['relative']:+.1f}%)\n"
        
        return report
    
    def setup_all_baselines(self, X_train: np.ndarray, y_train: np.ndarray,
                           molecules_train: Optional[List[Chem.Mol]] = None,
                           feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Set up all baseline models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            molecules_train: Optional training molecules
            feature_names: Optional feature names
            
        Returns:
            Dictionary of all baseline models
        """
        self.logger.info("Setting up baseline models...")
        
        # Simple baselines
        simple_baselines = self.create_simple_baselines(y_train)
        self.baselines.update(simple_baselines)
        
        # Feature-based baselines
        feature_baselines = self.create_simple_feature_baselines(X_train, y_train, feature_names)
        self.baselines.update(feature_baselines)
        
        # Molecular descriptor baselines
        if molecules_train is not None:
            mol_baselines = self.create_molecular_descriptor_baselines(molecules_train, y_train)
            self.baselines.update(mol_baselines)
        
        # Literature-based baselines
        lit_baselines = self.create_literature_baselines()
        self.baselines.update(lit_baselines)
        
        self.logger.info(f"Created {len(self.baselines)} baseline models")
        
        return self.baselines
    
    def full_comparison(self, X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       model_predictions: np.ndarray,
                       molecules_train: Optional[List[Chem.Mol]] = None,
                       molecules_test: Optional[List[Chem.Mol]] = None,
                       feature_names: Optional[List[str]] = None,
                       model_name: str = "Target Model",
                       save_dir: Optional[Path] = None) -> Dict[str, Any]:
        """
        Perform full baseline comparison.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            model_predictions: Target model predictions
            molecules_train: Optional training molecules
            molecules_test: Optional test molecules
            feature_names: Optional feature names
            model_name: Name of target model
            save_dir: Optional directory to save results
            
        Returns:
            Complete comparison results
        """
        # Set up baselines
        self.setup_all_baselines(X_train, y_train, molecules_train, feature_names)
        
        # Evaluate baselines
        self.comparison_results = self.evaluate_baselines(X_test, y_test, molecules_test)
        
        # Compare with target model
        comparison_results = self.compare_with_model(model_predictions, y_test, model_name)
        
        # Generate plots and report
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            
            # Save comparison plot
            plot_path = save_dir / 'baseline_comparison.png'
            self.plot_comparison(comparison_results, plot_path)
            
            # Save comparison report
            report = self.generate_comparison_report(comparison_results)
            report_path = save_dir / 'baseline_comparison_report.txt'
            with open(report_path, 'w') as f:
                f.write(report)
            
            self.logger.info(f"Baseline comparison results saved to {save_dir}")
        
        return comparison_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data
    np.random.seed(42)
    
    n_train, n_test = 100, 30
    n_features = 20
    
    X_train = np.random.randn(n_train, n_features)
    X_test = np.random.randn(n_test, n_features)
    
    # Create correlated targets
    y_train = X_train[:, 0] * 2 + X_train[:, 1] * 1.5 + np.random.normal(0, 0.5, n_train) + 7
    y_test = X_test[:, 0] * 2 + X_test[:, 1] * 1.5 + np.random.normal(0, 0.5, n_test) + 7
    
    # Simulate target model predictions (with some improvement over simple baselines)
    model_predictions = y_test + np.random.normal(0, 0.3, n_test)
    
    # Initialize comparison framework
    comparison = BaselineComparison()
    
    # Run full comparison
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        results = comparison.full_comparison(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            model_predictions=model_predictions,
            model_name="Example Model",
            save_dir=Path(tmp_dir)
        )
        
        print("Baseline Comparison Results:")
        print(f"Target model ranking: {results['ranking'][0]['rank']}")
        print(f"Number of baselines: {len(results['baselines'])}")
        
        # Print top 5 models
        print("\nTop 5 Models:")
        for i, model_info in enumerate(results['ranking'][:5]):
            print(f"{model_info['rank']}. {model_info['model']}: R² = {model_info['r2']:.3f}")
    
    print("\nBaseline comparison test completed successfully!")