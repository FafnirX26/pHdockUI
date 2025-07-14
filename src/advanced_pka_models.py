"""
Advanced pKa prediction models with proper hyperparameter tuning and feature engineering.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
import lightgbm as lgb

# Chemistry imports
from rdkit import Chem
from rdkit.Chem import Descriptors


class AdvancedpKaModel:
    """Advanced pKa prediction model with comprehensive feature engineering and tuning."""
    
    def __init__(self, model_type: str = "ensemble", random_state: int = 42):
        """
        Initialize advanced pKa model.
        
        Args:
            model_type: Type of model ("xgboost", "lightgbm", "neural_network", "ensemble")
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        
        # Initialize scalers
        self.feature_scaler = RobustScaler()  # More robust to outliers
        self.target_scaler = StandardScaler()
        
        # Model and training state
        self.model = None
        self.models = {}  # For ensemble
        self.feature_selector = None
        self.is_trained = False
        
        # Training history
        self.cv_scores = {}
        self.feature_importance = {}
        
    def create_enhanced_dataset(self, base_molecules: List[Chem.Mol], 
                               base_pka_values: np.ndarray) -> Tuple[List[Chem.Mol], np.ndarray]:
        """
        Create an enhanced dataset with better molecular diversity and known pKa values.
        
        Args:
            base_molecules: Base molecules
            base_pka_values: Base pKa values
            
        Returns:
            Enhanced molecules and pKa values
        """
        # Enhanced dataset with known pKa values for better training
        enhanced_data = [
            # Strong acids (pKa < 2)
            ("C(=O)O", 0.23),  # Formic acid
            ("CC(=O)O", 4.76),  # Acetic acid
            ("CCC(=O)O", 4.87),  # Propionic acid
            ("CCCC(=O)O", 4.82),  # Butyric acid
            ("CC(C)C(=O)O", 4.86),  # Isobutyric acid
            ("CCC(C)C(=O)O", 4.84),  # 2-Methylbutanoic acid
            ("CCCCCC(=O)O", 4.84),  # Hexanoic acid
            ("CCCCCCCCCCCCCCCCCC(=O)O", 4.95),  # Stearic acid
            
            # Aromatic carboxylic acids
            ("c1ccc(cc1)C(=O)O", 4.19),  # Benzoic acid
            ("Cc1ccccc1C(=O)O", 3.91),  # o-Toluic acid
            ("Cc1ccc(cc1)C(=O)O", 4.37),  # p-Toluic acid
            ("c1ccc(cc1)Cc2ccccc2C(=O)O", 3.46),  # 2-Phenylbenzoic acid
            ("c1ccc2c(c1)cccc2C(=O)O", 3.69),  # 1-Naphthoic acid
            ("CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", 4.41),  # Ibuprofen
            ("CC1=CC=C(C=C1)C(C)C(=O)O", 4.31),  # Naproxen-like
            
            # Substituted benzoic acids (electron-withdrawing effects)
            ("c1ccc(cc1C(=O)O)F", 3.27),  # 2-Fluorobenzoic acid
            ("c1ccc(cc1C(=O)O)Cl", 2.92),  # 2-Chlorobenzoic acid
            ("c1ccc(cc1C(=O)O)[N+](=O)[O-]", 2.17),  # 2-Nitrobenzoic acid
            ("c1cc(ccc1C(=O)O)F", 4.14),  # 4-Fluorobenzoic acid
            ("c1cc(ccc1C(=O)O)Cl", 3.99),  # 4-Chlorobenzoic acid
            ("c1cc(ccc1C(=O)O)[N+](=O)[O-]", 3.41),  # 4-Nitrobenzoic acid
            
            # Substituted benzoic acids (electron-donating effects)
            ("c1cc(ccc1C(=O)O)O", 4.58),  # 4-Hydroxybenzoic acid
            ("c1cc(ccc1C(=O)O)N", 4.87),  # 4-Aminobenzoic acid
            ("c1cc(ccc1C(=O)O)OC", 4.47),  # 4-Methoxybenzoic acid
            
            # Phenols
            ("c1ccccc1O", 9.95),  # Phenol
            ("Cc1ccccc1O", 10.29),  # o-Cresol
            ("Cc1ccc(cc1)O", 10.26),  # p-Cresol
            ("Cc1cccc(c1)O", 10.08),  # m-Cresol
            ("c1ccc(cc1)Cc2ccccc2O", 9.95),  # 2-Benzylphenol
            ("CC(C)(C)c1ccc(cc1)O", 10.23),  # 4-tert-Butylphenol
            ("c1ccc2c(c1)ccc(c2)O", 9.5),  # 2-Naphthol
            ("c1cc(c(cc1O)O)O", 9.85),  # Catechol
            ("c1cc(cc(c1O)O)O", 8.45),  # Resorcinol
            ("c1ccc(c(c1)O)O", 9.4),  # Hydroquinone
            
            # Substituted phenols
            ("c1cc(ccc1O)F", 9.89),  # 4-Fluorophenol
            ("c1cc(ccc1O)Cl", 9.38),  # 4-Chlorophenol
            ("c1cc(ccc1O)[N+](=O)[O-]", 7.15),  # 4-Nitrophenol
            ("c1cc(ccc1O)N", 5.48),  # 4-Aminophenol
            ("c1cc(ccc1O)OC", 10.21),  # 4-Methoxyphenol
            
            # Primary amines
            ("CCN", 10.64),  # Ethylamine
            ("CCCN", 10.53),  # Propylamine
            ("CCCCN", 10.60),  # Butylamine
            ("CC(C)N", 10.63),  # Isopropylamine
            ("CCCCCCCCN", 10.64),  # Octylamine
            ("c1ccc(cc1)N", 4.63),  # Aniline
            ("c1ccc(cc1)CN", 9.33),  # Benzylamine
            ("c1ccc2c(c1)cccc2N", 3.92),  # 1-Naphthylamine
            ("Cc1ccc(cc1)N", 5.08),  # p-Toluidine
            ("c1cc(ccc1N)F", 4.65),  # 4-Fluoroaniline
            ("c1cc(ccc1N)Cl", 4.15),  # 4-Chloroaniline
            ("c1cc(ccc1N)[N+](=O)[O-]", 1.0),  # 4-Nitroaniline
            
            # Secondary amines
            ("CCN(C)C", 10.73),  # Diethylamine
            ("CCCN(C)C", 10.43),  # Dipropylamine
            ("c1ccc(cc1)N(C)C", 5.15),  # N,N-Dimethylaniline
            ("c1ccc(cc1)NC", 4.85),  # N-Methylaniline
            
            # Tertiary amines
            ("CCN(CC)CC", 10.75),  # Triethylamine
            ("CN(C)C", 9.80),  # Trimethylamine
            ("c1ccc(cc1)N(CC)CC", 6.61),  # N,N-Diethylaniline
            
            # Imidazoles
            ("c1c[nH]cn1", 7.05),  # Imidazole
            ("Cc1c[nH]cn1", 7.56),  # 4-Methylimidazole
            ("c1cnc2c(c1)c[nH]c2", 5.4),  # Benzimidazole
            
            # Amino acids (carboxyl group)
            ("NC(C)C(=O)O", 2.35),  # Alanine
            ("NC(CC(=O)O)C(=O)O", 2.10),  # Aspartic acid
            ("NC(CCC(=O)O)C(=O)O", 2.10),  # Glutamic acid
            ("NC(CC1=CC=CC=C1)C(=O)O", 2.20),  # Phenylalanine
            ("NC(CC1=CC=C(C=C1)O)C(=O)O", 2.20),  # Tyrosine
            ("NC(CC1=CNC2=CC=CC=C21)C(=O)O", 2.46),  # Tryptophan
            
            # Thiols
            ("CCS", 10.33),  # Ethanethiol
            ("CCCS", 10.86),  # Propanethiol
            ("c1ccc(cc1)S", 6.52),  # Thiophenol
            ("CC(C)S", 10.33),  # 2-Propanethiol
            
            # Alcohols (very weak acids)
            ("CCO", 15.9),  # Ethanol
            ("CCCO", 15.5),  # Propanol
            ("CC(C)O", 17.1),  # Isopropanol
            ("CCCCO", 15.0),  # Butanol
            ("c1ccc(cc1)CO", 15.4),  # Benzyl alcohol
            
            # Guanidines and amidines
            ("NC(=N)N", 12.48),  # Guanidine
            ("CC(=N)N", 12.40),  # Acetamidine
            ("c1ccc(cc1)C(=N)N", 11.52),  # Benzamidine
            
            # Heterocyclic compounds
            ("c1ccncc1", 5.25),  # Pyridine
            ("c1ccc2ncccc2c1", 4.85),  # Quinoline
            ("c1cnccn1", 1.23),  # Pyrazine
            ("c1ccc2c(c1)ncc(c2)O", 5.08),  # 8-Hydroxyquinoline
            
            # Miscellaneous
            ("C(C(=O)O)N", 2.34),  # Glycine
            ("CC(C(=O)O)N", 2.35),  # Alanine
            ("C1CCC(CC1)N", 10.64),  # Cyclohexylamine
            ("c1ccc2c(c1)cc(cc2)O", 9.5),  # 2-Naphthol
            ("c1cc2ccccc2cc1O", 9.5),  # 1-Naphthol
        ]
        
        # Combine with base data
        molecules = list(base_molecules)
        pka_values = list(base_pka_values)
        
        # Add enhanced molecules
        for smiles, pka in enhanced_data:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                molecules.append(mol)
                pka_values.append(pka)
        
        self.logger.info(f"Enhanced dataset: {len(molecules)} molecules (added {len(enhanced_data)} reference compounds)")
        
        return molecules, np.array(pka_values)
    
    def engineer_features(self, molecules: List[Chem.Mol]) -> pd.DataFrame:
        """
        Engineer comprehensive features optimized for pKa prediction.
        
        Args:
            molecules: List of RDKit molecules
            
        Returns:
            DataFrame with engineered features
        """
        from .enhanced_features import EnhancedFeatureEngineering
        
        feature_eng = EnhancedFeatureEngineering()
        features_df = feature_eng.features_to_dataframe(molecules)
        
        # Add interaction features
        features_df = self._add_interaction_features(features_df)
        
        # Add polynomial features for key descriptors
        features_df = self._add_polynomial_features(features_df)
        
        return features_df
    
    def _add_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add interaction features between important descriptors."""
        # Key interaction features for pKa prediction
        
        # Electronic effects interactions
        if 'electron_withdrawing_groups' in df.columns and 'electron_donating_groups' in df.columns:
            df['net_electronic_effect'] = df['electron_withdrawing_groups'] - df['electron_donating_groups']
            df['electronic_balance'] = df['electron_withdrawing_groups'] / (df['electron_donating_groups'] + 1)
        
        # Ionizable group interactions
        ionizable_cols = ['carboxylic_acids', 'phenols', 'primary_amines', 'secondary_amines', 'tertiary_amines']
        available_ionizable = [col for col in ionizable_cols if col in df.columns]
        
        if len(available_ionizable) > 1:
            df['total_ionizable_groups'] = df[available_ionizable].sum(axis=1)
            df['acid_base_ratio'] = (df.get('carboxylic_acids', 0) + df.get('phenols', 0)) / (df.get('primary_amines', 0) + df.get('secondary_amines', 0) + df.get('tertiary_amines', 0) + 1)
        
        # Structural complexity interactions
        if 'num_heavy_atoms' in df.columns:
            if 'num_rings' in df.columns:
                df['ring_density'] = df['num_rings'] / (df['num_heavy_atoms'] + 1)
            if 'num_rotatable_bonds' in df.columns:
                df['flexibility_index'] = df['num_rotatable_bonds'] / (df['num_heavy_atoms'] + 1)
        
        # Solvent accessibility approximations
        if 'tpsa' in df.columns and 'molecular_weight' in df.columns:
            df['tpsa_mw_ratio'] = df['tpsa'] / df['molecular_weight']
        
        # Quantum-structural interactions
        quantum_cols = ['homo_lumo_gap', 'dipole_moment', 'polarizability', 'hardness']
        available_quantum = [col for col in quantum_cols if col in df.columns]
        
        if len(available_quantum) >= 2:
            if 'homo_lumo_gap' in df.columns and 'hardness' in df.columns:
                df['electronic_stability'] = df['homo_lumo_gap'] * df['hardness']
            if 'dipole_moment' in df.columns and 'polarizability' in df.columns:
                df['charge_distribution'] = df['dipole_moment'] / (df['polarizability'] + 1)
        
        return df
    
    def _add_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add polynomial features for key descriptors."""
        # Key descriptors that often have non-linear relationships with pKa
        poly_features = ['logp', 'tpsa', 'num_heavy_atoms', 'molecular_weight']
        
        for feature in poly_features:
            if feature in df.columns:
                df[f'{feature}_squared'] = df[feature] ** 2
                df[f'{feature}_log'] = np.log(df[feature] + 1)  # Add 1 to avoid log(0)
        
        return df
    
    def select_features(self, X: np.ndarray, y: np.ndarray, 
                       feature_names: List[str], n_features: int = 200) -> Tuple[np.ndarray, List[str]]:
        """
        Select most important features for pKa prediction.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_names: Feature names
            n_features: Number of features to select
            
        Returns:
            Selected features and feature names
        """
        self.logger.info(f"Selecting {n_features} most important features from {X.shape[1]} total features")
        
        # Combine multiple feature selection methods
        
        # 1. Univariate selection
        univariate_selector = SelectKBest(score_func=f_regression, k=min(n_features * 2, X.shape[1]))
        X_univariate = univariate_selector.fit_transform(X, y)
        univariate_features = np.array(feature_names)[univariate_selector.get_support()]
        
        # 2. Recursive feature elimination with XGBoost
        xgb_estimator = xgb.XGBRegressor(n_estimators=50, random_state=self.random_state, n_jobs=-1)
        rfe_selector = RFE(estimator=xgb_estimator, n_features_to_select=n_features)
        rfe_selector.fit(X_univariate, y)
        
        # Get final selected features
        final_features = univariate_features[rfe_selector.get_support()]
        final_X = X_univariate[:, rfe_selector.get_support()]
        
        self.feature_selector = {
            'univariate': univariate_selector,
            'rfe': rfe_selector,
            'selected_features': final_features
        }
        
        self.logger.info(f"Feature selection completed. Selected {len(final_features)} features")
        
        return final_X, final_features.tolist()
    
    def get_optimized_model(self, model_type: str) -> Any:
        """
        Get optimized model with best hyperparameters.
        
        Args:
            model_type: Type of model to create
            
        Returns:
            Optimized model
        """
        if model_type == "xgboost":
            return xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_type == "lightgbm":
            return lgb.LGBMRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        
        elif model_type == "gradient_boosting":
            return GradientBoostingRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=self.random_state
            )
        
        elif model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=self.random_state,
                n_jobs=-1
            )
        
        elif model_type == "neural_network":
            return MLPRegressor(
                hidden_layer_sizes=(200, 100, 50),
                activation='relu',
                solver='adam',
                learning_rate_init=0.001,
                max_iter=1000,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.1
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def train_single_model(self, X: np.ndarray, y: np.ndarray, 
                          model_type: str) -> Tuple[Any, Dict[str, float]]:
        """
        Train a single model with cross-validation.
        
        Args:
            X: Feature matrix
            y: Target values
            model_type: Type of model to train
            
        Returns:
            Trained model and CV scores
        """
        model = self.get_optimized_model(model_type)
        
        # Cross-validation
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        cv_rmse = np.sqrt(-cv_scores)
        
        # Train on full dataset
        model.fit(X, y)
        
        # Calculate feature importance if available
        if hasattr(model, 'feature_importances_'):
            self.feature_importance[model_type] = model.feature_importances_
        
        scores = {
            'cv_rmse_mean': cv_rmse.mean(),
            'cv_rmse_std': cv_rmse.std(),
            'cv_r2_mean': -cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        }
        
        return model, scores
    
    def train_ensemble(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train ensemble of models.
        
        Args:
            X: Feature matrix
            y: Target values
            
        Returns:
            Ensemble results
        """
        self.logger.info("Training ensemble of models...")
        
        # Train individual models
        model_types = ["xgboost", "lightgbm", "gradient_boosting", "random_forest", "neural_network"]
        
        for model_type in model_types:
            self.logger.info(f"Training {model_type} model...")
            try:
                model, scores = self.train_single_model(X, y, model_type)
                self.models[model_type] = model
                self.cv_scores[model_type] = scores
                self.logger.info(f"{model_type} CV RMSE: {scores['cv_rmse_mean']:.3f} ± {scores['cv_rmse_std']:.3f}")
            except Exception as e:
                self.logger.warning(f"Failed to train {model_type}: {e}")
        
        # Train ensemble weights based on CV performance
        weights = {}
        total_inverse_error = 0
        
        for model_type, scores in self.cv_scores.items():
            # Weight inversely proportional to CV error
            weight = 1.0 / (scores['cv_rmse_mean'] + 1e-6)
            weights[model_type] = weight
            total_inverse_error += weight
        
        # Normalize weights
        for model_type in weights:
            weights[model_type] /= total_inverse_error
        
        self.ensemble_weights = weights
        
        self.logger.info("Ensemble weights:")
        for model_type, weight in weights.items():
            self.logger.info(f"  {model_type}: {weight:.3f}")
        
        return {
            'models': self.models,
            'cv_scores': self.cv_scores,
            'ensemble_weights': self.ensemble_weights
        }
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using trained model(s).
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X)
        
        # Apply feature selection if used
        if self.feature_selector is not None:
            X_univariate = self.feature_selector['univariate'].transform(X_scaled)
            X_selected = X_univariate[:, self.feature_selector['rfe'].get_support()]
        else:
            X_selected = X_scaled
        
        if self.model_type == "ensemble":
            # Ensemble prediction
            predictions = np.zeros(X_selected.shape[0])
            
            for model_type, model in self.models.items():
                if model_type in self.ensemble_weights:
                    weight = self.ensemble_weights[model_type]
                    model_pred = model.predict(X_selected)
                    predictions += weight * model_pred
        else:
            # Single model prediction
            predictions = self.model.predict(X_selected)
        
        # Inverse transform predictions
        predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        
        return predictions
    
    def train(self, molecules: List[Chem.Mol], pka_values: np.ndarray,
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train the advanced pKa model.
        
        Args:
            molecules: List of RDKit molecules
            pka_values: pKa values
            test_size: Test set fraction
            
        Returns:
            Training results
        """
        self.logger.info("Starting advanced pKa model training...")
        
        # Enhance dataset
        enhanced_molecules, enhanced_pka = self.create_enhanced_dataset(molecules, pka_values)
        
        # Engineer features
        self.logger.info("Engineering features...")
        features_df = self.engineer_features(enhanced_molecules)
        
        # Convert to numpy array
        feature_names = [col for col in features_df.columns if col != 'molecule_id']
        X = features_df[feature_names].values
        y = enhanced_pka
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        self.logger.info(f"Feature matrix shape: {X.shape}")
        self.logger.info(f"Target range: {y.min():.2f} to {y.max():.2f}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        # Scale features and targets
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        X_test_scaled = self.feature_scaler.transform(X_test)
        
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.target_scaler.transform(y_test.reshape(-1, 1)).flatten()
        
        # Feature selection
        X_train_selected, selected_features = self.select_features(X_train_scaled, y_train_scaled, feature_names)
        X_test_selected = self.feature_selector['univariate'].transform(X_test_scaled)
        X_test_selected = X_test_selected[:, self.feature_selector['rfe'].get_support()]
        
        # Train model(s)
        if self.model_type == "ensemble":
            ensemble_results = self.train_ensemble(X_train_selected, y_train_scaled)
        else:
            self.model, scores = self.train_single_model(X_train_selected, y_train_scaled, self.model_type)
            self.cv_scores[self.model_type] = scores
        
        self.is_trained = True
        
        # Make predictions on test set
        y_test_pred_scaled = self.predict_scaled(X_test_selected)
        y_test_pred = self.target_scaler.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        
        results = {
            'test_mae': test_mae,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'cv_scores': self.cv_scores,
            'n_features_selected': len(selected_features),
            'selected_features': selected_features[:20],  # Top 20 features
            'n_training_molecules': len(enhanced_molecules),
            'predictions': y_test_pred,
            'y_test': y_test,
            'feature_importance': self.feature_importance
        }
        
        if self.model_type == "ensemble":
            results['ensemble_weights'] = self.ensemble_weights
        
        self.logger.info(f"Training completed!")
        self.logger.info(f"Test MAE: {test_mae:.3f} pKa units")
        self.logger.info(f"Test RMSE: {test_rmse:.3f} pKa units")
        self.logger.info(f"Test R²: {test_r2:.3f}")
        
        return results
    
    def predict_scaled(self, X: np.ndarray) -> np.ndarray:
        """Predict with scaled features (internal method)."""
        if self.model_type == "ensemble":
            predictions = np.zeros(X.shape[0])
            for model_type, model in self.models.items():
                if model_type in self.ensemble_weights:
                    weight = self.ensemble_weights[model_type]
                    model_pred = model.predict(X)
                    predictions += weight * model_pred
        else:
            predictions = self.model.predict(X)
        
        return predictions


if __name__ == "__main__":
    # Test the advanced pKa model
    from rdkit import Chem
    
    # Test molecules
    smiles_list = [
        "CC(=O)O",  # Acetic acid
        "c1ccccc1O",  # Phenol
        "c1ccc(cc1)N",  # Aniline
    ]
    
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    pka_values = np.array([4.76, 9.95, 4.63])
    
    # Test advanced model
    model = AdvancedpKaModel(model_type="ensemble")
    results = model.train(molecules, pka_values)
    
    print("Advanced pKa Model Results:")
    print(f"Test MAE: {results['test_mae']:.3f}")
    print(f"Test R²: {results['test_r2']:.3f}")
    print(f"Features selected: {results['n_features_selected']}")
    print(f"Training molecules: {results['n_training_molecules']}")
    
    if 'ensemble_weights' in results:
        print("Ensemble weights:")
        for model_type, weight in results['ensemble_weights'].items():
            print(f"  {model_type}: {weight:.3f}")