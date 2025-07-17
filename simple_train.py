#!/usr/bin/env python3
"""
Simplified training script that avoids complex feature extraction.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski, Crippen
import warnings
warnings.filterwarnings('ignore')

def load_dataset():
    """Load the large combined dataset."""
    dataset_path = Path("training_data/combined_pka_dataset_large.csv")
    df = pd.read_csv(dataset_path)
    print(f"📊 Loaded {len(df)} records")
    return df

def extract_simple_features(smiles_list):
    """Extract simple molecular features using basic RDKit descriptors."""
    print("🔧 Extracting molecular features...")
    
    features = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            try:
                # Calculate basic descriptors that are widely available
                feat = [
                    Descriptors.MolWt(mol),
                    Descriptors.MolLogP(mol),
                    Descriptors.NumHDonors(mol),
                    Descriptors.NumHAcceptors(mol),
                    Descriptors.NumRotatableBonds(mol),
                    Descriptors.NumAromaticRings(mol),
                    Descriptors.NumSaturatedRings(mol),
                    Descriptors.NumAliphaticRings(mol),
                    Descriptors.RingCount(mol),
                    Descriptors.TPSA(mol),
                    mol.GetNumAtoms(),
                    mol.GetNumBonds(),
                    mol.GetNumHeavyAtoms()
                ]
                
                features.append(feat)
                valid_indices.append(i)
            except:
                continue
    
    print(f"✓ Extracted features for {len(features)} molecules")
    return np.array(features), valid_indices

def train_simple_model(df, save_dir="models/"):
    """Train a simple Random Forest model."""
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print("\n🤖 Training Random Forest Model")
    print("=" * 50)
    
    # Extract features
    features, valid_indices = extract_simple_features(df['smiles'].tolist())
    
    if len(features) == 0:
        print("❌ No valid features extracted")
        return None, None
    
    # Get corresponding pKa values
    pka_values = df.iloc[valid_indices]['pka_value'].values
    
    print(f"✓ Training set: {len(features)} molecules with {features.shape[1]} features")
    print(f"📈 pKa range: {pka_values.min():.1f} to {pka_values.max():.1f}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        features, pka_values, test_size=0.2, random_state=42
    )
    
    # Train model
    print("🚀 Training Random Forest...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    results = {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }
    
    print(f"\n📊 Training Results:")
    print(f"  Training R² Score: {train_r2:.3f}")
    print(f"  Test R² Score: {test_r2:.3f}")
    print(f"  Test RMSE: {test_rmse:.3f}")
    print(f"  Test MAE: {test_mae:.3f}")
    print(f"  Training size: {len(X_train)}")
    print(f"  Test size: {len(X_test)}")
    
    # Save model
    model_path = save_path / "rf_large_dataset.pkl"
    joblib.dump(model, model_path)
    print(f"💾 Model saved to {model_path}")
    
    return model, results

def main():
    """Main training function."""
    print("🎯 Simple Training Script")
    print("=" * 50)
    
    try:
        # Load dataset
        df = load_dataset()
        
        # Train model
        model, results = train_simple_model(df)
        
        if model is not None:
            print("\n🎉 Training completed successfully!")
        else:
            print("❌ Training failed")
            return 1
            
    except Exception as e:
        print(f"❌ Script failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())