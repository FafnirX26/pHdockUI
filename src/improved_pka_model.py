"""
Improved pKa Prediction Model - Addresses ML Pipeline Weaknesses

Key Improvements:
1. Uses real experimental data (not synthetic)
2. Optimized XGBoost hyperparameters from literature
3. Curated feature set (~100 features vs 4000+)
4. K-fold cross-validation
5. Data quality filters
6. Proper regularization
7. Physics-informed descriptors

Performance Target: R² > 0.85, RMSE < 1.5, MAE < 1.0
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path
import joblib
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments, Lipinski
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class DataQualityFilter:
    """Filter and clean pKa training data."""

    def __init__(self, pka_min: float = -2.0, pka_max: float = 16.0):
        """
        Initialize data quality filter.

        Args:
            pka_min: Minimum chemically reasonable pKa
            pka_max: Maximum chemically reasonable pKa
        """
        self.pka_min = pka_min
        self.pka_max = pka_max

    def filter_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data quality filters.

        Args:
            df: DataFrame with 'smiles' and 'pka_value' columns

        Returns:
            Filtered DataFrame
        """
        logger.info(f"Starting data filtering. Initial size: {len(df)}")

        # Remove extreme outliers
        df_filtered = df[
            (df['pka_value'] >= self.pka_min) &
            (df['pka_value'] <= self.pka_max)
        ].copy()

        logger.info(f"After pKa range filter [{self.pka_min}, {self.pka_max}]: {len(df_filtered)}")

        # Remove invalid SMILES
        valid_smiles = []
        for idx, row in df_filtered.iterrows():
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol is not None and mol.GetNumAtoms() > 0:
                valid_smiles.append(idx)

        df_filtered = df_filtered.loc[valid_smiles]
        logger.info(f"After SMILES validation: {len(df_filtered)}")

        # Remove duplicates (keep first occurrence)
        df_filtered = df_filtered.drop_duplicates(subset=['smiles'], keep='first')
        logger.info(f"After duplicate removal: {len(df_filtered)}")

        # Statistical outlier detection (Modified Z-score)
        median = df_filtered['pka_value'].median()
        mad = np.median(np.abs(df_filtered['pka_value'] - median))
        modified_z_scores = 0.6745 * (df_filtered['pka_value'] - median) / mad
        df_filtered = df_filtered[np.abs(modified_z_scores) < 3.5]

        logger.info(f"After statistical outlier removal: {len(df_filtered)}")

        return df_filtered.reset_index(drop=True)


class CuratedFeatureExtractor:
    """
    Extract curated molecular descriptors for pKa prediction.
    Based on literature: focus on electronic, ionization, and solvation features.
    """

    def __init__(self):
        self.feature_names = []

    def extract_electronic_descriptors(self, mol: Chem.Mol) -> List[float]:
        """Extract electronic structure descriptors."""
        descriptors = []

        try:
            # Gasteiger partial charges
            AllChem.ComputeGasteigerCharges(mol)
            charges = []
            for atom in mol.GetAtoms():
                try:
                    charge = atom.GetDoubleProp('_GasteigerCharge')
                    if not np.isnan(charge) and not np.isinf(charge):
                        charges.append(charge)
                except:
                    pass

            if charges:
                descriptors.extend([
                    np.max(charges),              # Most positive charge
                    np.min(charges),              # Most negative charge
                    np.mean(np.abs(charges)),     # Mean absolute charge
                    np.std(charges),              # Charge distribution
                    np.sum([c for c in charges if c > 0]),  # Total positive
                    np.sum([c for c in charges if c < 0]),  # Total negative
                ])
            else:
                descriptors.extend([0.0] * 6)

            # EState indices (HOMO/LUMO proxies)
            descriptors.extend([
                Descriptors.MaxEStateIndex(mol),
                Descriptors.MinEStateIndex(mol),
                Descriptors.MaxAbsEStateIndex(mol),
                Descriptors.MaxPartialCharge(mol),
                Descriptors.MinPartialCharge(mol),
                Descriptors.MaxAbsPartialCharge(mol),
            ])

        except Exception as e:
            logger.warning(f"Electronic descriptor calculation failed: {e}")
            descriptors.extend([0.0] * 12)

        return descriptors

    def extract_ionization_descriptors(self, mol: Chem.Mol) -> List[float]:
        """Extract ionizable group counts (critical for pKa)."""
        descriptors = []

        # Acidic groups
        descriptors.extend([
            Fragments.fr_COO(mol) + Fragments.fr_COO2(mol),  # Carboxylic acids
            Fragments.fr_phenol(mol) + Fragments.fr_Ar_OH(mol),  # Phenols
            Fragments.fr_Al_OH(mol),  # Aliphatic alcohols
            Fragments.fr_SH(mol),     # Thiols
        ])

        # Basic groups
        descriptors.extend([
            Fragments.fr_NH0(mol),    # Quaternary N
            Fragments.fr_NH1(mol),    # Tertiary N
            Fragments.fr_NH2(mol),    # Secondary N
            Fragments.fr_Ar_NH(mol),  # Aromatic amines
            Fragments.fr_guanido(mol),  # Guanidines (strong bases)
            Fragments.fr_imidazole(mol),  # Imidazoles
            Fragments.fr_pyridine(mol),   # Pyridines
        ])

        return descriptors

    def extract_structural_descriptors(self, mol: Chem.Mol) -> List[float]:
        """Extract structural and topological descriptors."""
        descriptors = []

        # Basic properties
        descriptors.extend([
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumHeteroatoms(mol),
            Descriptors.NumValenceElectrons(mol),
            mol.GetNumAtoms(),
            mol.GetNumHeavyAtoms(),
        ])

        # Ring properties (aromaticity affects pKa)
        descriptors.extend([
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAromaticHeterocycles(mol),
            Descriptors.NumAromaticCarbocycles(mol),
            Descriptors.NumSaturatedRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.RingCount(mol),
        ])

        # Hybridization (conjugation effects)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        sp2_atoms = sum(1 for atom in mol.GetAtoms()
                       if atom.GetHybridization() == Chem.HybridizationType.SP2)
        sp3_atoms = sum(1 for atom in mol.GetAtoms()
                       if atom.GetHybridization() == Chem.HybridizationType.SP3)

        descriptors.extend([
            aromatic_atoms,
            aromatic_atoms / max(mol.GetNumAtoms(), 1),
            sp2_atoms,
            sp3_atoms,
            Descriptors.FractionCsp3(mol) if hasattr(Descriptors, 'FractionCsp3') else 0.0,
        ])

        return descriptors

    def extract_solvation_descriptors(self, mol: Chem.Mol) -> List[float]:
        """Extract solvation-related descriptors (affects pKa in solution)."""
        descriptors = []

        descriptors.extend([
            Descriptors.TPSA(mol),           # Topological polar surface area
            Descriptors.LabuteASA(mol),      # Accessible surface area
            Descriptors.MolMR(mol),          # Molar refractivity (polarizability)
            Descriptors.BalabanJ(mol),       # Balaban index (branching)
            Descriptors.BertzCT(mol),        # Molecular complexity
            Descriptors.Chi0v(mol),          # Connectivity indices
            Descriptors.Chi1v(mol),
            Descriptors.Kappa1(mol),         # Shape indices
            Descriptors.Kappa2(mol),
            Descriptors.Kappa3(mol),
        ])

        # VSA descriptors (solvation-related)
        descriptors.extend([
            Descriptors.VSA_EState1(mol),
            Descriptors.VSA_EState2(mol),
            Descriptors.SlogP_VSA1(mol),
            Descriptors.SlogP_VSA2(mol),
            Descriptors.SMR_VSA1(mol),
            Descriptors.SMR_VSA2(mol),
        ])

        return descriptors

    def extract_all_features(self, smiles: str) -> Optional[np.ndarray]:
        """
        Extract all curated features from SMILES.

        Args:
            smiles: SMILES string

        Returns:
            Feature array or None if failed
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        features = []

        # Extract feature groups
        features.extend(self.extract_electronic_descriptors(mol))
        features.extend(self.extract_ionization_descriptors(mol))
        features.extend(self.extract_structural_descriptors(mol))
        features.extend(self.extract_solvation_descriptors(mol))

        # Set feature names on first call
        if not self.feature_names:
            self.feature_names = [
                # Electronic (12)
                'MaxCharge', 'MinCharge', 'MeanAbsCharge', 'ChargeStd',
                'TotalPosCharge', 'TotalNegCharge',
                'MaxEState', 'MinEState', 'MaxAbsEState',
                'MaxPartialCharge', 'MinPartialCharge', 'MaxAbsPartialCharge',
                # Ionization (11)
                'COOH', 'Phenol', 'Alcohol', 'Thiol',
                'QuatN', 'TertN', 'SecN', 'ArNH', 'Guanidine', 'Imidazole', 'Pyridine',
                # Structural (30)
                'MolWt', 'MolLogP', 'HBD', 'HBA', 'RotBonds',
                'Heteroatoms', 'ValenceElectrons', 'NumAtoms', 'NumHeavyAtoms',
                'AromaticRings', 'AromaticHeterocycles', 'AromaticCarbocycles',
                'SaturatedRings', 'AliphaticRings', 'RingCount',
                'AromaticAtoms', 'AromaticFraction', 'SP2Atoms', 'SP3Atoms', 'FractionCsp3',
                # Solvation (16)
                'TPSA', 'LabuteASA', 'MolMR', 'BalabanJ', 'BertzCT',
                'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'Kappa3',
                'VSA_EState1', 'VSA_EState2', 'SlogP_VSA1', 'SlogP_VSA2',
                'SMR_VSA1', 'SMR_VSA2',
            ]

        features_array = np.array(features, dtype=np.float32)

        # Handle NaN/Inf
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        return features_array


class ImprovedpKaModel:
    """
    Improved pKa prediction model with research-based optimizations.
    """

    def __init__(self,
                 n_folds: int = 5,
                 random_state: int = 42,
                 use_robust_scaling: bool = True):
        """
        Initialize improved pKa model.

        Args:
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
            use_robust_scaling: Use RobustScaler (better for outliers)
        """
        self.n_folds = n_folds
        self.random_state = random_state

        # Feature extraction and scaling
        self.feature_extractor = CuratedFeatureExtractor()
        self.scaler = RobustScaler() if use_robust_scaling else StandardScaler()

        # Optimized XGBoost parameters from literature
        # Based on: R² 0.80-0.95 in molecular property prediction papers
        self.xgb_params = {
            'n_estimators': 500,
            'max_depth': 10,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0.1,
            'reg_alpha': 0.1,    # L1 regularization
            'reg_lambda': 1.0,   # L2 regularization
            'random_state': random_state,
            'n_jobs': -1,
            'tree_method': 'hist',
        }

        self.model = None
        self.is_trained = False
        self.training_stats = {}

    def load_and_prepare_data(self,
                             data_path: str = "training_data/filtered_pka_dataset.csv") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare training data.

        Args:
            data_path: Path to CSV with 'smiles' and 'pka_value' columns

        Returns:
            Tuple of (X_features, y_targets)
        """
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)

        # Apply data quality filters
        filter = DataQualityFilter(pka_min=-2.0, pka_max=16.0)
        df = filter.filter_dataset(df)

        # Extract features
        logger.info("Extracting molecular features...")
        X_list = []
        y_list = []

        for idx, row in df.iterrows():
            if idx % 1000 == 0:
                logger.info(f"  Processing {idx}/{len(df)}...")

            features = self.feature_extractor.extract_all_features(row['smiles'])
            if features is not None:
                X_list.append(features)
                y_list.append(row['pka_value'])

        X = np.array(X_list)
        y = np.array(y_list)

        logger.info(f"Extracted {len(X)} valid feature vectors with {X.shape[1]} features")

        return X, y

    def train_with_cv(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Train model with k-fold cross-validation.

        Args:
            X: Feature matrix
            y: Target values

        Returns:
            Dictionary with training results
        """
        logger.info(f"Training with {self.n_folds}-fold cross-validation")

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # K-fold cross-validation
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)

        cv_scores = {
            'r2': [],
            'rmse': [],
            'mae': [],
            'fold_models': []
        }

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X_scaled), 1):
            logger.info(f"Training fold {fold}/{self.n_folds}...")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train fold model
            fold_model = xgb.XGBRegressor(
                **self.xgb_params,
                early_stopping_rounds=50,
                eval_metric='rmse'
            )
            fold_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )

            # Evaluate
            y_pred = fold_model.predict(X_val)

            r2 = r2_score(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)

            cv_scores['r2'].append(r2)
            cv_scores['rmse'].append(rmse)
            cv_scores['mae'].append(mae)
            cv_scores['fold_models'].append(fold_model)

            logger.info(f"  Fold {fold} - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")

        # Train final model on all data
        logger.info("Training final model on full dataset...")
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_scaled, y)
        self.is_trained = True

        # Calculate statistics
        self.training_stats = {
            'cv_r2_mean': np.mean(cv_scores['r2']),
            'cv_r2_std': np.std(cv_scores['r2']),
            'cv_rmse_mean': np.mean(cv_scores['rmse']),
            'cv_rmse_std': np.std(cv_scores['rmse']),
            'cv_mae_mean': np.mean(cv_scores['mae']),
            'cv_mae_std': np.std(cv_scores['mae']),
            'n_features': X.shape[1],
            'n_samples': len(X),
        }

        # Chemical accuracy metrics
        all_errors = []
        for fold_model, (_, val_idx) in zip(cv_scores['fold_models'], kfold.split(X_scaled)):
            y_pred = fold_model.predict(X_scaled[val_idx])
            errors = np.abs(y[val_idx] - y_pred)
            all_errors.extend(errors)

        all_errors = np.array(all_errors)
        self.training_stats['within_0.5_pka'] = np.mean(all_errors <= 0.5)
        self.training_stats['within_1.0_pka'] = np.mean(all_errors <= 1.0)
        self.training_stats['within_2.0_pka'] = np.mean(all_errors <= 2.0)

        logger.info("\n" + "="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        logger.info(f"Cross-Validation Results ({self.n_folds} folds):")
        logger.info(f"  R² Score:  {self.training_stats['cv_r2_mean']:.4f} ± {self.training_stats['cv_r2_std']:.4f}")
        logger.info(f"  RMSE:      {self.training_stats['cv_rmse_mean']:.4f} ± {self.training_stats['cv_rmse_std']:.4f}")
        logger.info(f"  MAE:       {self.training_stats['cv_mae_mean']:.4f} ± {self.training_stats['cv_mae_std']:.4f}")
        logger.info(f"\nChemical Accuracy:")
        logger.info(f"  Within 0.5 pKa: {self.training_stats['within_0.5_pka']:.1%}")
        logger.info(f"  Within 1.0 pKa: {self.training_stats['within_1.0_pka']:.1%}")
        logger.info(f"  Within 2.0 pKa: {self.training_stats['within_2.0_pka']:.1%}")
        logger.info("="*60 + "\n")

        return self.training_stats

    def predict(self, smiles_list: List[str]) -> np.ndarray:
        """
        Predict pKa values for molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Array of predicted pKa values
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        # Extract features
        X_list = []
        for smiles in smiles_list:
            features = self.feature_extractor.extract_all_features(smiles)
            if features is not None:
                X_list.append(features)
            else:
                X_list.append(np.zeros(len(self.feature_extractor.feature_names)))

        X = np.array(X_list)
        X_scaled = self.scaler.transform(X)

        # Predict
        predictions = self.model.predict(X_scaled)

        return predictions

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """Get top N most important features."""
        if not self.is_trained:
            raise ValueError("Model must be trained first")

        importance = self.model.feature_importances_
        feature_names = self.feature_extractor.feature_names

        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)

        return df

    def save_model(self, output_dir: str = "models") -> None:
        """Save trained model."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_extractor': self.feature_extractor,
            'training_stats': self.training_stats,
            'xgb_params': self.xgb_params,
        }

        save_path = output_path / "improved_pka_model.joblib"
        joblib.dump(model_data, save_path)
        logger.info(f"Model saved to {save_path}")

    def load_model(self, model_path: str) -> None:
        """Load trained model."""
        model_data = joblib.load(model_path)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_extractor = model_data['feature_extractor']
        self.training_stats = model_data['training_stats']
        self.xgb_params = model_data['xgb_params']
        self.is_trained = True

        logger.info(f"Model loaded from {model_path}")
        logger.info(f"  Training R²: {self.training_stats['cv_r2_mean']:.4f}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Train improved model
    model = ImprovedpKaModel(n_folds=5)

    # Load and prepare data
    X, y = model.load_and_prepare_data()

    # Train with cross-validation
    results = model.train_with_cv(X, y)

    # Show feature importance
    print("\nTop 20 Most Important Features:")
    print(model.get_feature_importance(top_n=20))

    # Save model
    model.save_model("models")

    # Test predictions
    test_smiles = [
        "CC(=O)O",  # Acetic acid (pKa ~4.76)
        "c1ccccc1O",  # Phenol (pKa ~9.95)
        "CCN",  # Ethylamine (pKa ~10.7)
    ]

    predictions = model.predict(test_smiles)
    print("\nTest Predictions:")
    for smiles, pred in zip(test_smiles, predictions):
        print(f"  {smiles}: pKa = {pred:.2f}")
