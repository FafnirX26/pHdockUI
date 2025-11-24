#!/usr/bin/env python3
"""
Create PyTorch models for feature extraction and protonation state generation.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, Fragments
import sys
import os

# Add src to path
sys.path.append('src')
from feature_engineering import FeatureEngineering
from protonation_engine import ProtonationEngine


class FeatureExtractorNet(nn.Module):
    """PyTorch neural network for molecular feature extraction."""
    
    def __init__(self, input_size=2048, output_size=55):
        """
        Initialize feature extractor network.
        
        Args:
            input_size: Size of molecular fingerprint input
            output_size: Number of quantum-inspired features to output
        """
        super(FeatureExtractorNet, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_size)
        )
        
        # Initialize feature engineering for reference
        self.rdkit_fe = FeatureEngineering(include_3d=False)
        
    def forward(self, x):
        """Forward pass through feature extractor."""
        return self.feature_extractor(x)
    
    def extract_features_from_smiles(self, smiles_list):
        """Extract features directly from SMILES strings."""
        features_list = []
        
        for smiles in smiles_list:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    features_list.append(torch.zeros(55))
                    continue
                    
                # Calculate RDKit features
                features_dict = self.rdkit_fe.calculate_features(mol)
                features_array = np.array(list(features_dict.values()))
                
                # Handle NaN values
                features_array = np.nan_to_num(features_array, nan=0.0)
                features_tensor = torch.tensor(features_array, dtype=torch.float32)
                
                features_list.append(features_tensor)
                
            except Exception as e:
                print(f"Error processing {smiles}: {e}")
                features_list.append(torch.zeros(55))
        
        return torch.stack(features_list)


class ProtonationStateNet(nn.Module):
    """PyTorch neural network for protonation state prediction and generation."""
    
    def __init__(self, feature_size=55, hidden_size=128):
        """
        Initialize protonation state network.
        
        Args:
            feature_size: Size of molecular feature input
            hidden_size: Size of hidden layers
        """
        super(ProtonationStateNet, self).__init__()
        
        # pKa prediction network
        self.pka_predictor = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single pKa value output
        )
        
        # pH-dependent state classifier
        self.state_classifier = nn.Sequential(
            nn.Linear(feature_size + 1, hidden_size),  # features + pH
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 3 states: neutral, protonated, deprotonated
        )
        
        # Initialize protonation engine for reference
        self.protonation_engine = ProtonationEngine()
        
    def forward(self, features, ph=None):
        """
        Forward pass through protonation network.
        
        Args:
            features: Molecular features tensor
            ph: pH value (optional, for state classification)
            
        Returns:
            pka_pred: Predicted pKa values
            state_probs: Protonation state probabilities (if pH provided)
        """
        pka_pred = self.pka_predictor(features)
        
        if ph is not None:
            # Combine features with pH for state classification
            ph_expanded = ph.expand(features.size(0), 1)
            combined_input = torch.cat([features, ph_expanded], dim=1)
            state_logits = self.state_classifier(combined_input)
            state_probs = torch.softmax(state_logits, dim=1)
            return pka_pred, state_probs
        
        return pka_pred
    
    def predict_protonation_states(self, smiles, ph_values):
        """
        Predict protonation states for given SMILES at different pH values.
        
        Args:
            smiles: SMILES string
            ph_values: List or array of pH values
            
        Returns:
            dict: pH values mapped to protonation state probabilities
        """
        results = {}
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return results
            
            # Calculate features (simplified for demonstration)
            mol_wt = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # Create simplified feature vector
            features = torch.tensor([
                mol_wt, logp, hbd, hba,
                *[0.0] * 51  # Pad to 55 features
            ], dtype=torch.float32).unsqueeze(0)
            
            for ph in ph_values:
                ph_tensor = torch.tensor([ph], dtype=torch.float32)
                pka_pred, state_probs = self.forward(features, ph_tensor)
                
                results[ph] = {
                    'predicted_pka': float(pka_pred.item()),
                    'neutral_prob': float(state_probs[0, 0].item()),
                    'protonated_prob': float(state_probs[0, 1].item()),
                    'deprotonated_prob': float(state_probs[0, 2].item())
                }
                
        except Exception as e:
            print(f"Error processing protonation states for {smiles}: {e}")
            
        return results


def load_training_data():
    """Load training data for model training."""
    try:
        # Load the filtered dataset
        df = pd.read_csv('training_data/filtered_pka_dataset.csv')
        # Rename column to match expected format
        if 'pka_value' in df.columns:
            df['pka'] = df['pka_value']
        print(f"Loaded {len(df)} training samples")
        return df
    except FileNotFoundError:
        print("Training data not found, creating dummy data for model structure")
        # Create dummy data for model creation
        dummy_data = {
            'smiles': ['CCO', 'CC(=O)O', 'c1ccccc1O'] * 100,
            'pka': [15.9, 4.76, 9.95] * 100
        }
        return pd.DataFrame(dummy_data)


def train_feature_extractor(model, training_data, epochs=50):
    """Train the feature extractor model."""
    print("Training feature extractor...")
    
    # Initialize feature engineering
    fe = FeatureEngineering(include_3d=False)
    
    # Prepare training data
    X_features = []
    X_fingerprints = []
    
    for smiles in training_data['smiles'][:1000]:  # Use subset for faster training
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Get RDKit features as target
            features_dict = fe.calculate_features(mol)
            features_array = np.array(list(features_dict.values()))
            features_array = np.nan_to_num(features_array, nan=0.0)
            
            # Get Morgan fingerprint as input
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            fp_array = np.array(fp)
            
            X_features.append(features_array)
            X_fingerprints.append(fp_array)
            
        except Exception as e:
            continue
    
    if len(X_features) == 0:
        print("No valid training data found, using dummy weights")
        return model
    
    X_fp = torch.tensor(np.array(X_fingerprints), dtype=torch.float32)
    y_feat = torch.tensor(np.array(X_features), dtype=torch.float32)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_fp)
        loss = criterion(outputs, y_feat)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    print("Feature extractor training completed")
    return model


def train_protonation_model(model, training_data, epochs=50):
    """Train the protonation state model."""
    print("Training protonation model...")
    
    # Prepare training data
    X_features = []
    y_pka = []
    
    fe = FeatureEngineering(include_3d=False)
    
    valid_count = 0
    for idx, row in training_data.head(1000).iterrows():  # Use subset for faster training
        try:
            smiles = row['smiles']
            pka_value = row['pka']
            
            # Skip if pka value is invalid
            if pd.isna(pka_value) or not isinstance(pka_value, (int, float)):
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
                
            # Simple feature calculation for demonstration
            features = [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.NumHDonors(mol),
                Descriptors.NumHAcceptors(mol),
                *[0.0] * 51  # Pad to 55 features
            ]
            
            X_features.append(features)
            y_pka.append(float(pka_value))
            valid_count += 1
            
        except Exception as e:
            continue
    
    print(f"Found {valid_count} valid training samples")
    
    if len(X_features) == 0:
        print("No valid training data found, using dummy weights")
        return model
    
    X = torch.tensor(np.array(X_features), dtype=torch.float32)
    y = torch.tensor(np.array(y_pka), dtype=torch.float32).unsqueeze(1)
    
    print(f"Training data shapes: X={X.shape}, y={y.shape}")
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pka_pred = model(X)
        loss = criterion(pka_pred, y)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")
    
    print("Protonation model training completed")
    return model


def main():
    """Main function to create and save PyTorch models."""
    print("Creating PyTorch models for feature extraction and protonation state generation...")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load training data
    training_data = load_training_data()
    
    # Create and train feature extractor
    print("\n1. Creating Feature Extractor Network...")
    feature_extractor = FeatureExtractorNet(input_size=2048, output_size=55)
    feature_extractor = train_feature_extractor(feature_extractor, training_data)
    feature_extractor.eval()
    
    # Create and train protonation model
    print("\n2. Creating Protonation State Network...")
    protonation_model = ProtonationStateNet(feature_size=55, hidden_size=128)
    protonation_model = train_protonation_model(protonation_model, training_data)
    protonation_model.eval()
    
    # Save models as .pt files
    print("\n3. Saving models...")
    
    # Save feature extractor
    feature_extractor_path = models_dir / "feature_extractor.pt"
    torch.save({
        'model_state_dict': feature_extractor.state_dict(),
        'model_class': 'FeatureExtractorNet',
        'input_size': 2048,
        'output_size': 55,
        'description': 'Neural network for extracting 55 quantum-inspired molecular features from fingerprints'
    }, feature_extractor_path)
    print(f"✅ Feature extractor saved to: {feature_extractor_path}")
    
    # Save protonation model
    protonation_model_path = models_dir / "protonation_model.pt"
    torch.save({
        'model_state_dict': protonation_model.state_dict(),
        'model_class': 'ProtonationStateNet',
        'feature_size': 55,
        'hidden_size': 128,
        'description': 'Neural network for pKa prediction and pH-dependent protonation state classification'
    }, protonation_model_path)
    print(f"✅ Protonation model saved to: {protonation_model_path}")
    
    # Test the models
    print("\n4. Testing models...")
    test_smiles = ["CCO", "CC(=O)O", "c1ccccc1O"]  # ethanol, acetic acid, phenol
    
    for smiles in test_smiles:
        print(f"\nTesting with {smiles}:")
        
        # Test feature extractor
        features = feature_extractor.extract_features_from_smiles([smiles])
        print(f"  Features extracted: {features.shape}")
        
        # Test protonation model
        ph_values = [7.0, 4.0, 10.0]
        states = protonation_model.predict_protonation_states(smiles, ph_values)
        for ph, state_info in states.items():
            print(f"  pH {ph}: pKa={state_info['predicted_pka']:.2f}")
    
    print(f"\n🎉 Successfully created PyTorch models!")
    print(f"📁 Models saved in: {models_dir.absolute()}")
    print(f"   - feature_extractor.pt: Molecular feature extraction")
    print(f"   - protonation_model.pt: pKa prediction and protonation states")


if __name__ == "__main__":
    main()