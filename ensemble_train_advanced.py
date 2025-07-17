#!/usr/bin/env python3
"""
Advanced ensemble training script with enhanced features and evaluation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Fragments, Lipinski
import joblib
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngine:
    """Enhanced feature engineering for pKa prediction."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def extract_ionization_features(self, mol):
        """Extract ionization-specific molecular features."""
        features = []
        feature_names = []
        
        # Basic molecular properties
        features.extend([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.TPSA(mol),
            mol.GetNumAtoms(),
            mol.GetNumHeavyAtoms(),
            mol.GetNumBonds(),
        ])
        feature_names.extend([
            'MolWt', 'MolLogP', 'HBD', 'HBA', 'RotBonds', 
            'TPSA', 'NumAtoms', 'NumHeavyAtoms', 'NumBonds'
        ])
        
        # Ring properties (important for aromaticity/conjugation)
        features.extend([
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.RingCount(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumAromaticHeterocycles(mol),
        ])
        feature_names.extend([
            'AromaticRings', 'SaturatedRings', 'AliphaticRings', 
            'RingCount', 'AromaticCarbocycles', 'AromaticHeterocycles'
        ])
        
        # Electronic properties (crucial for pKa)
        try:
            features.extend([
                Descriptors.NumValenceElectrons(mol),
                Descriptors.MaxEStateIndex(mol),
                Descriptors.MinEStateIndex(mol),
                Descriptors.MaxAbsEStateIndex(mol),
                Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol),
                Descriptors.MaxAbsPartialCharge(mol),
            ])
            feature_names.extend([
                'ValenceElectrons', 'MaxEState', 'MinEState', 'MaxAbsEState',
                'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge'
            ])
        except:
            # Fallback if partial charges fail
            features.extend([0] * 7)
            feature_names.extend([
                'ValenceElectrons', 'MaxEState', 'MinEState', 'MaxAbsEState',
                'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge'
            ])
        
        # Functional group counts (ionizable groups)
        features.extend([
            Fragments.fr_COO(mol),      # Carboxylic acids
            Fragments.fr_COO2(mol),     # Carboxylate
            Fragments.fr_phenol(mol),   # Phenols
            Fragments.fr_Ar_OH(mol),    # Aromatic OH
            Fragments.fr_NH0(mol),      # Quaternary nitrogen
            Fragments.fr_NH1(mol),      # Tertiary nitrogen
            Fragments.fr_NH2(mol),      # Secondary nitrogen  
            Fragments.fr_Ndealkylation1(mol), # Primary nitrogen
            Fragments.fr_guanido(mol),  # Guanidine
            Fragments.fr_imidazole(mol), # Imidazole
            Fragments.fr_thiophene(mol), # Thiophene
            Fragments.fr_furan(mol),    # Furan
            Fragments.fr_pyridine(mol), # Pyridine
        ])
        feature_names.extend([
            'COOH', 'COO2', 'Phenol', 'ArOH', 'QuatN', 'TertN', 'SecN', 
            'PrimN', 'Guanidine', 'Imidazole', 'Thiophene', 'Furan', 'Pyridine'
        ])
        
        # Connectivity and shape (affects solvation)
        features.extend([
            Descriptors.Chi0v(mol),     # Connectivity indices
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),    # Shape indices
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
            Descriptors.HallKierAlpha(mol),
            Descriptors.BertzCT(mol),   # Complexity
        ])
        feature_names.extend([
            'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'Kappa3', 
            'HallKierAlpha', 'BertzCT'
        ])
        
        # Additional ionization-relevant features
        try:
            features.extend([
                Descriptors.FractionCsp3(mol) if hasattr(Descriptors, 'FractionCsp3') else 0,
                Descriptors.NumHeteroatoms(mol),
                Descriptors.NumRadicalElectrons(mol),
                Descriptors.NumSaturatedCarbocycles(mol),
                Descriptors.NumSaturatedHeterocycles(mol),
            ])
            feature_names.extend([
                'FractionCsp3', 'NumHeteroatoms', 'RadicalElectrons',
                'SaturatedCarbocycles', 'SaturatedHeterocycles'
            ])
        except:
            features.extend([0] * 5)
            feature_names.extend([
                'FractionCsp3', 'NumHeteroatoms', 'RadicalElectrons',
                'SaturatedCarbocycles', 'SaturatedHeterocycles'
            ])
        
        # Store feature names for interpretability
        if not self.feature_names:
            self.feature_names = feature_names
        
        return np.array(features)
    
    def extract_features_batch(self, smiles_list):
        """Extract features for a batch of SMILES."""
        features = []
        valid_indices = []
        
        print(f"🔧 Extracting enhanced molecular features...")
        for i, smiles in enumerate(smiles_list):
            if i % 1000 == 0:
                print(f"  Processing {i}/{len(smiles_list)}...")
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                try:
                    feat = self.extract_ionization_features(mol)
                    features.append(feat)
                    valid_indices.append(i)
                except Exception as e:
                    continue
        
        features_array = np.array(features)
        
        # Handle NaN values
        print(f"🔧 Handling missing values...")
        nan_mask = np.isnan(features_array)
        if np.any(nan_mask):
            print(f"  Found {np.sum(nan_mask)} NaN values, filling with feature medians...")
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            features_array = imputer.fit_transform(features_array)
        
        print(f"✓ Extracted {len(self.feature_names)} features for {len(features)} molecules")
        return features_array, valid_indices

class LogarithmicMetrics:
    """Evaluation metrics considering logarithmic nature of pKa."""
    
    @staticmethod
    def pka_to_ka(pka_values):
        """Convert pKa to Ka (dissociation constant)."""
        return 10**(-pka_values)
    
    @staticmethod
    def ka_relative_error(pka_true, pka_pred):
        """Calculate relative error in Ka space."""
        ka_true = LogarithmicMetrics.pka_to_ka(pka_true)
        ka_pred = LogarithmicMetrics.pka_to_ka(pka_pred)
        return np.abs((ka_pred - ka_true) / ka_true)
    
    @staticmethod
    def log_rmse(y_true, y_pred):
        """RMSE in logarithmic space (already log scale for pKa)."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def evaluate_comprehensive(y_true, y_pred):
        """Comprehensive evaluation metrics."""
        results = {
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'r2': r2_score(y_true, y_pred),
            'mean_ka_rel_error': np.mean(LogarithmicMetrics.ka_relative_error(y_true, y_pred)),
            'median_ka_rel_error': np.median(LogarithmicMetrics.ka_relative_error(y_true, y_pred)),
        }
        
        # Chemical accuracy metrics (within X pKa units)
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            within_threshold = np.sum(np.abs(y_true - y_pred) <= threshold) / len(y_true)
            results[f'within_{threshold}_pka'] = within_threshold
        
        return results

class AdvancedEnsembleTrainer:
    """Advanced ensemble trainer with XGBoost and Neural Network."""
    
    def __init__(self):
        self.feature_engine = AdvancedFeatureEngine()
        self.models = {}
        self.ensemble_weights = {}
        self.metrics = LogarithmicMetrics()
        
    def create_neural_network(self, input_dim):
        """Create a neural network for pKa prediction."""
        try:
            import torch
            import torch.nn as nn
            
            class pKaNet(nn.Module):
                def __init__(self, input_dim):
                    super(pKaNet, self).__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_dim, 256),
                        nn.ReLU(),
                        nn.Dropout(0.3),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Dropout(0.2),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 1)
                    )
                
                def forward(self, x):
                    return self.layers(x)
            
            return pKaNet(input_dim)
        except ImportError:
            print("⚠️  PyTorch not available, using only tree-based models")
            return None
    
    def train_individual_models(self, X_train, y_train, X_val, y_val):
        """Train individual models for the ensemble."""
        print(f"\n🤖 Training Individual Models")
        print("=" * 50)
        
        model_results = {}
        
        # 1. XGBoost
        print("🚀 Training XGBoost...")
        xgb_model = xgb.XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            early_stopping_rounds=20,
            eval_metric='rmse'
        )
        
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        xgb_pred = xgb_model.predict(X_val)
        xgb_results = self.metrics.evaluate_comprehensive(y_val, xgb_pred)
        
        self.models['xgboost'] = xgb_model
        model_results['xgboost'] = xgb_results
        
        print(f"  XGBoost - R²: {xgb_results['r2']:.3f}, RMSE: {xgb_results['rmse']:.3f}")
        
        # 2. Random Forest (backup/additional diversity)
        print("🌲 Training Random Forest...")
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_val)
        rf_results = self.metrics.evaluate_comprehensive(y_val, rf_pred)
        
        self.models['random_forest'] = rf_model
        model_results['random_forest'] = rf_results
        
        print(f"  Random Forest - R²: {rf_results['r2']:.3f}, RMSE: {rf_results['rmse']:.3f}")
        
        # 3. Neural Network (if available)
        nn_model = self.create_neural_network(X_train.shape[1])
        if nn_model is not None:
            print("🧠 Training Neural Network...")
            # Simplified sklearn-like NN using MLPRegressor
            from sklearn.neural_network import MLPRegressor
            
            nn_model = MLPRegressor(
                hidden_layer_sizes=(256, 128, 64, 32),
                activation='relu',
                solver='adam',
                alpha=0.001,
                batch_size=32,
                learning_rate='adaptive',
                max_iter=200,
                early_stopping=True,
                validation_fraction=0.1,
                random_state=42
            )
            
            nn_model.fit(X_train, y_train)
            nn_pred = nn_model.predict(X_val)
            nn_results = self.metrics.evaluate_comprehensive(y_val, nn_pred)
            
            self.models['neural_network'] = nn_model
            model_results['neural_network'] = nn_results
            
            print(f"  Neural Network - R²: {nn_results['r2']:.3f}, RMSE: {nn_results['rmse']:.3f}")
        
        return model_results
    
    def optimize_ensemble_weights(self, X_val, y_val):
        """Optimize ensemble weights using validation performance."""
        print(f"\n⚖️  Optimizing ensemble weights...")
        
        # Get predictions from all models
        predictions = {}
        for name, model in self.models.items():
            predictions[name] = model.predict(X_val)
        
        # Try different weight combinations
        from scipy.optimize import minimize
        
        def ensemble_rmse(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_pred = np.zeros_like(y_val)
            
            for i, (name, pred) in enumerate(predictions.items()):
                ensemble_pred += weights[i] * pred
            
            return np.sqrt(mean_squared_error(y_val, ensemble_pred))
        
        # Initial equal weights
        initial_weights = np.ones(len(self.models))
        
        # Optimize weights
        result = minimize(
            ensemble_rmse,
            initial_weights,
            method='SLSQP',
            bounds=[(0, 1) for _ in range(len(self.models))],
            constraints={'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        )
        
        optimal_weights = result.x
        
        # Store weights
        for i, name in enumerate(self.models.keys()):
            self.ensemble_weights[name] = optimal_weights[i]
        
        print(f"  Optimal weights:")
        for name, weight in self.ensemble_weights.items():
            print(f"    {name}: {weight:.3f}")
        
        return optimal_weights
    
    def predict_ensemble(self, X):
        """Make ensemble predictions."""
        predictions = np.zeros(len(X))
        
        for name, model in self.models.items():
            weight = self.ensemble_weights.get(name, 1.0 / len(self.models))
            predictions += weight * model.predict(X)
        
        return predictions
    
    def create_feature_importance_plot(self, output_dir="results/"):
        """Create feature importance visualizations."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # XGBoost feature importance
        if 'xgboost' in self.models:
            xgb_importance = self.models['xgboost'].feature_importances_
            
            # Create DataFrame for plotting
            importance_df = pd.DataFrame({
                'feature': self.feature_engine.feature_names,
                'importance': xgb_importance
            }).sort_values('importance', ascending=True).tail(20)
            
            plt.figure(figsize=(10, 8))
            plt.barh(importance_df['feature'], importance_df['importance'])
            plt.xlabel('Feature Importance')
            plt.title('Top 20 XGBoost Feature Importances')
            plt.tight_layout()
            plt.savefig(output_path / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  ✓ Feature importance plot saved: {output_path / 'feature_importance.png'}")
    
    def train_full_pipeline(self, df):
        """Train the complete ensemble pipeline."""
        print("🎯 Advanced Ensemble Training")
        print("=" * 50)
        
        # Extract features
        features, valid_indices = self.feature_engine.extract_features_batch(df['smiles'].tolist())
        
        if len(features) == 0:
            raise ValueError("No valid molecular features extracted")
        
        # Get corresponding pKa values
        y = df.iloc[valid_indices]['pka_value'].values
        
        print(f"✓ Training set: {len(features)} molecules, {features.shape[1]} features")
        print(f"📈 pKa range: {y.min():.2f} to {y.max():.2f}")
        
        # Scale features
        X_scaled = self.feature_engine.scaler.fit_transform(features)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Further split training data for validation
        X_train_sub, X_val, y_train_sub, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )
        
        print(f"✓ Data split - Train: {len(X_train_sub)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Train individual models
        model_results = self.train_individual_models(X_train_sub, y_train_sub, X_val, y_val)
        
        # Optimize ensemble weights
        optimal_weights = self.optimize_ensemble_weights(X_val, y_val)
        
        # Final ensemble evaluation on test set
        print(f"\n📊 Final Ensemble Evaluation")
        print("=" * 50)
        
        test_pred = self.predict_ensemble(X_test)
        test_results = self.metrics.evaluate_comprehensive(y_test, test_pred)
        
        print(f"Test Results:")
        print(f"  R² Score: {test_results['r2']:.3f}")
        print(f"  RMSE: {test_results['rmse']:.3f}")
        print(f"  MAE: {test_results['mae']:.3f}")
        print(f"  Mean Ka Rel Error: {test_results['mean_ka_rel_error']:.3f}")
        
        print(f"\nChemical Accuracy:")
        for threshold in [0.5, 1.0, 1.5, 2.0]:
            accuracy = test_results[f'within_{threshold}_pka']
            print(f"  Within {threshold} pKa units: {accuracy:.1%}")
        
        # Create visualizations
        self.create_feature_importance_plot()
        
        # Save models
        self.save_models()
        
        return {
            'individual_results': model_results,
            'ensemble_results': test_results,
            'ensemble_weights': self.ensemble_weights
        }
    
    def save_models(self, output_dir="models/"):
        """Save trained models."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save individual models
        for name, model in self.models.items():
            model_file = output_path / f"{name}_advanced.pkl"
            joblib.dump(model, model_file)
            print(f"💾 Saved {name}: {model_file}")
        
        # Save feature scaler
        scaler_file = output_path / "feature_scaler.pkl"
        joblib.dump(self.feature_engine.scaler, scaler_file)
        
        # Save ensemble weights and feature names
        ensemble_data = {
            'weights': self.ensemble_weights,
            'feature_names': self.feature_engine.feature_names
        }
        ensemble_file = output_path / "ensemble_config.pkl"
        joblib.dump(ensemble_data, ensemble_file)
        
        print(f"💾 Saved ensemble config: {ensemble_file}")

def main():
    """Main training function."""
    try:
        # Load filtered dataset
        df = pd.read_csv("training_data/filtered_pka_dataset.csv")
        print(f"📊 Loaded filtered dataset: {len(df)} records")
        
        # Train ensemble
        trainer = AdvancedEnsembleTrainer()
        results = trainer.train_full_pipeline(df)
        
        print(f"\n🎉 Advanced ensemble training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())