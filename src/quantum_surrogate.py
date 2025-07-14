"""
Quantum descriptor surrogate model for fast QM property prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class QuantumSurrogateModel:
    """Surrogate model to predict quantum mechanical descriptors from fast molecular descriptors."""
    
    def __init__(self, 
                 model_type: str = "xgboost",
                 random_state: int = 42,
                 n_estimators: int = 100,
                 normalize_features: bool = True):
        """
        Initialize the quantum surrogate model.
        
        Args:
            model_type: Type of model to use ("xgboost", "random_forest")
            random_state: Random state for reproducibility
            n_estimators: Number of estimators for ensemble methods
            normalize_features: Whether to normalize input features
        """
        self.model_type = model_type
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.normalize_features = normalize_features
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize model
        if model_type == "xgboost":
            self.model = xgb.XGBRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                tree_method='hist',
                n_jobs=-1
            )
        elif model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Initialize feature scaler
        self.scaler = StandardScaler() if normalize_features else None
        
        # Track target quantum descriptors
        self.target_descriptors = [
            'homo_energy',
            'lumo_energy',
            'gap_energy',
            'dipole_moment',
            'polarizability',
            'ionization_potential',
            'electron_affinity',
            'electronegativity',
            'chemical_hardness',
            'chemical_softness'
        ]
        
        # Model training status
        self.is_trained = False
        self.feature_names = None
        self.target_names = None
        
    def generate_synthetic_qm_data(self, molecules: List[Chem.Mol], 
                                  molecular_features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate synthetic quantum mechanical data for training.
        This is a placeholder - in practice, you would use real QM calculations.
        
        Args:
            molecules: List of RDKit molecules
            molecular_features: DataFrame with molecular features
            
        Returns:
            DataFrame with synthetic QM descriptors
        """
        np.random.seed(self.random_state)
        
        qm_data = []
        
        for i, mol in enumerate(molecules):
            try:
                # Get basic molecular properties for correlation
                mol_wt = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                num_atoms = mol.GetNumAtoms()
                
                # Generate synthetic QM descriptors with some correlation to molecular properties
                # In practice, these would come from DFT calculations
                
                # HOMO energy (eV) - correlated with molecular size and electronics
                homo_energy = -8.0 - 0.01 * mol_wt + 0.1 * logp + np.random.normal(0, 0.5)
                
                # LUMO energy (eV) - typically higher than HOMO
                lumo_energy = homo_energy + 3.0 + 0.005 * tpsa + np.random.normal(0, 0.3)
                
                # HOMO-LUMO gap
                gap_energy = lumo_energy - homo_energy
                
                # Dipole moment (Debye) - correlated with polarity
                dipole_moment = 0.5 + 0.1 * tpsa / mol_wt + np.random.normal(0, 0.5)
                dipole_moment = max(0, dipole_moment)
                
                # Polarizability (Bohr^3) - correlated with molecular size
                polarizability = 5.0 + 0.5 * num_atoms + np.random.normal(0, 2.0)
                polarizability = max(0, polarizability)
                
                # Ionization potential (eV) - related to HOMO
                ionization_potential = abs(homo_energy) + np.random.normal(0, 0.2)
                
                # Electron affinity (eV) - related to LUMO
                electron_affinity = abs(lumo_energy) - 3.0 + np.random.normal(0, 0.2)
                
                # Electronegativity (eV) - average of IP and EA
                electronegativity = (ionization_potential + electron_affinity) / 2
                
                # Chemical hardness (eV) - related to HOMO-LUMO gap
                chemical_hardness = gap_energy / 2 + np.random.normal(0, 0.1)
                
                # Chemical softness (eV^-1) - inverse of hardness
                chemical_softness = 1.0 / max(0.1, chemical_hardness)
                
                qm_data.append({
                    'molecule_id': i,
                    'homo_energy': homo_energy,
                    'lumo_energy': lumo_energy,
                    'gap_energy': gap_energy,
                    'dipole_moment': dipole_moment,
                    'polarizability': polarizability,
                    'ionization_potential': ionization_potential,
                    'electron_affinity': electron_affinity,
                    'electronegativity': electronegativity,
                    'chemical_hardness': chemical_hardness,
                    'chemical_softness': chemical_softness
                })
                
            except Exception as e:
                self.logger.warning(f"Failed to generate QM data for molecule {i}: {e}")
                continue
        
        return pd.DataFrame(qm_data)
    
    def prepare_training_data(self, molecular_features: pd.DataFrame, 
                            qm_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare training data for the surrogate model.
        
        Args:
            molecular_features: DataFrame with molecular descriptors
            qm_data: DataFrame with quantum mechanical descriptors
            
        Returns:
            Tuple of (X_features, y_targets)
        """
        # Merge data on molecule_id
        if 'molecule_id' in molecular_features.columns and 'molecule_id' in qm_data.columns:
            merged_data = pd.merge(molecular_features, qm_data, on='molecule_id', how='inner')
        else:
            # Assume same order if no ID columns
            merged_data = pd.concat([molecular_features, qm_data], axis=1)
        
        # Extract features (exclude non-numeric columns and target columns)
        exclude_cols = ['molecule_id', 'smiles'] + self.target_descriptors
        feature_cols = [col for col in merged_data.columns 
                       if col not in exclude_cols and merged_data[col].dtype in ['float64', 'int64']]
        
        X = merged_data[feature_cols].values
        y = merged_data[self.target_descriptors].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)
        
        self.feature_names = feature_cols
        self.target_names = self.target_descriptors
        
        return X, y
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              test_size: float = 0.2, 
              validate: bool = True) -> Dict[str, Any]:
        """
        Train the surrogate model.
        
        Args:
            X: Feature matrix
            y: Target matrix
            test_size: Fraction of data to use for testing
            validate: Whether to perform validation
            
        Returns:
            Dictionary with training results
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features if required
        if self.normalize_features:
            X_train = self.scaler.fit_transform(X_train)
            X_test = self.scaler.transform(X_test)
        
        # Train model
        self.logger.info(f"Training {self.model_type} model with {X_train.shape[0]} samples")
        
        if self.model_type == "xgboost":
            # XGBoost can handle multi-output regression
            self.model.fit(X_train, y_train)
        else:
            # For other models, we might need to wrap for multi-output
            from sklearn.multioutput import MultiOutputRegressor
            self.model = MultiOutputRegressor(self.model)
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluate model
        results = {'train_scores': {}, 'test_scores': {}}
        
        if validate:
            # Predictions
            y_train_pred = self.model.predict(X_train)
            y_test_pred = self.model.predict(X_test)
            
            # Calculate metrics for each target
            for i, target_name in enumerate(self.target_names):
                # Training metrics
                train_r2 = r2_score(y_train[:, i], y_train_pred[:, i])
                train_mse = mean_squared_error(y_train[:, i], y_train_pred[:, i])
                train_mae = mean_absolute_error(y_train[:, i], y_train_pred[:, i])
                
                # Test metrics
                test_r2 = r2_score(y_test[:, i], y_test_pred[:, i])
                test_mse = mean_squared_error(y_test[:, i], y_test_pred[:, i])
                test_mae = mean_absolute_error(y_test[:, i], y_test_pred[:, i])
                
                results['train_scores'][target_name] = {
                    'r2': train_r2, 'mse': train_mse, 'mae': train_mae
                }
                results['test_scores'][target_name] = {
                    'r2': test_r2, 'mse': test_mse, 'mae': test_mae
                }
            
            # Overall metrics
            train_r2_mean = np.mean([scores['r2'] for scores in results['train_scores'].values()])
            test_r2_mean = np.mean([scores['r2'] for scores in results['test_scores'].values()])
            
            results['overall_train_r2'] = train_r2_mean
            results['overall_test_r2'] = test_r2_mean
            
            self.logger.info(f"Training completed. Overall R² - Train: {train_r2_mean:.3f}, Test: {test_r2_mean:.3f}")
        
        return results
    
    def predict(self, molecular_features: pd.DataFrame) -> pd.DataFrame:
        """
        Predict quantum descriptors from molecular features.
        
        Args:
            molecular_features: DataFrame with molecular descriptors
            
        Returns:
            DataFrame with predicted quantum descriptors
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        # Prepare features
        feature_cols = [col for col in self.feature_names if col in molecular_features.columns]
        X = molecular_features[feature_cols].values
        X = np.nan_to_num(X, nan=0.0)
        
        # Scale features if required
        if self.normalize_features:
            X = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Create results DataFrame
        results = pd.DataFrame(y_pred, columns=self.target_names)
        
        # Add molecule ID if available
        if 'molecule_id' in molecular_features.columns:
            results['molecule_id'] = molecular_features['molecule_id'].values
        
        return results
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        importance_dict = {}
        
        if self.model_type == "xgboost":
            # XGBoost feature importance
            importance_scores = self.model.feature_importances_
            for i, feature_name in enumerate(self.feature_names):
                importance_dict[feature_name] = importance_scores[i]
        
        elif hasattr(self.model, 'feature_importances_'):
            # For RandomForest and other tree-based models
            importance_scores = self.model.feature_importances_
            for i, feature_name in enumerate(self.feature_names):
                importance_dict[feature_name] = importance_scores[i]
        
        else:
            self.logger.warning("Feature importance not available for this model type")
        
        return importance_dict
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """
        Save the trained model to disk.
        
        Args:
            model_path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'target_names': self.target_names,
            'model_type': self.model_type,
            'normalize_features': self.normalize_features
        }
        
        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load a trained model from disk.
        
        Args:
            model_path: Path to the saved model
        """
        model_data = joblib.load(model_path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.target_names = model_data['target_names']
        self.model_type = model_data['model_type']
        self.normalize_features = model_data['normalize_features']
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {model_path}")


def train_quantum_surrogate_model(molecules: List[Chem.Mol],
                                 molecular_features: pd.DataFrame,
                                 model_type: str = "xgboost",
                                 save_path: Optional[Union[str, Path]] = None,
                                 **kwargs) -> QuantumSurrogateModel:
    """
    Convenience function to train a quantum surrogate model.
    
    Args:
        molecules: List of RDKit molecules
        molecular_features: DataFrame with molecular features
        model_type: Type of model to use
        save_path: Optional path to save the trained model
        **kwargs: Additional arguments for QuantumSurrogateModel
        
    Returns:
        Trained QuantumSurrogateModel instance
    """
    # Initialize model
    surrogate_model = QuantumSurrogateModel(model_type=model_type, **kwargs)
    
    # Generate synthetic QM data (in practice, this would be real QM calculations)
    qm_data = surrogate_model.generate_synthetic_qm_data(molecules, molecular_features)
    
    # Prepare training data
    X, y = surrogate_model.prepare_training_data(molecular_features, qm_data)
    
    # Train model
    results = surrogate_model.train(X, y)
    
    # Save model if path provided
    if save_path:
        surrogate_model.save_model(save_path)
    
    return surrogate_model


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample molecules
    from src.input_processing import MoleculeProcessor
    from src.feature_engineering import FeatureEngineering
    
    smiles_list = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC(C)(C)C1=CC=C(C=C1)C(C)C(=O)O",  # Naproxen-like
        "CC1=CC=C(C=C1)C(=O)O",  # Para-toluic acid
        "CC(C)C1=CC=C(C=C1)O",  # Carvacrol
        "CC(=O)OCC1=CC=CC=C1",  # Benzyl acetate
        "CC(C)CCCCCCCCCC(=O)O",  # Branched fatty acid
        "C1=CC=C(C=C1)N"  # Aniline
    ]
    
    # Process molecules
    processor = MoleculeProcessor()
    results = processor.process_smiles(smiles_list)
    molecules = results['molecules']
    
    # Calculate molecular features
    feature_eng = FeatureEngineering()
    molecular_features = feature_eng.features_to_dataframe(molecules, smiles_list=smiles_list)
    
    # Train surrogate model
    surrogate_model = train_quantum_surrogate_model(
        molecules, 
        molecular_features, 
        model_type="xgboost",
        n_estimators=50
    )
    
    # Test predictions
    qm_predictions = surrogate_model.predict(molecular_features)
    
    print(f"Trained surrogate model for {len(molecules)} molecules")
    print(f"Predicted QM descriptors: {list(qm_predictions.columns)}")
    print(f"Sample predictions:\n{qm_predictions.head()}")
    
    # Show feature importance
    importance = surrogate_model.get_feature_importance()
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    print(f"\nTop 10 most important features:")
    for feature, score in top_features:
        print(f"  {feature}: {score:.4f}")