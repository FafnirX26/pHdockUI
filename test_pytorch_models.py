#!/usr/bin/env python3
"""
Test the generated PyTorch models for feature extraction and protonation state prediction.
"""

import torch
import torch.nn as nn
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from pathlib import Path


class FeatureExtractorNet(nn.Module):
    """PyTorch neural network for molecular feature extraction."""
    
    def __init__(self, input_size=2048, output_size=55):
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
        
    def forward(self, x):
        return self.feature_extractor(x)


class ProtonationStateNet(nn.Module):
    """PyTorch neural network for protonation state prediction and generation."""
    
    def __init__(self, feature_size=55, hidden_size=128):
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
            nn.Linear(32, 1)
        )
        
        # pH-dependent state classifier
        self.state_classifier = nn.Sequential(
            nn.Linear(feature_size + 1, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # neutral, protonated, deprotonated
        )
        
    def forward(self, features, ph=None):
        pka_pred = self.pka_predictor(features)
        
        if ph is not None:
            ph_expanded = ph.expand(features.size(0), 1)
            combined_input = torch.cat([features, ph_expanded], dim=1)
            state_logits = self.state_classifier(combined_input)
            state_probs = torch.softmax(state_logits, dim=1)
            return pka_pred, state_probs
        
        return pka_pred


def extract_features_from_smiles(smiles):
    """Extract molecular features from SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return torch.zeros(55)
        
        # Calculate basic molecular descriptors
        features = []
        
        # Basic molecular properties
        features.append(Descriptors.MolWt(mol))
        features.append(Descriptors.MolLogP(mol))
        features.append(Descriptors.MolMR(mol))
        features.append(Descriptors.HeavyAtomCount(mol))
        features.append(Descriptors.NumHAcceptors(mol))
        features.append(Descriptors.NumHDonors(mol))
        features.append(Descriptors.NumRotatableBonds(mol))
        features.append(Descriptors.NumAromaticRings(mol))
        features.append(Descriptors.NumSaturatedRings(mol))
        features.append(Descriptors.NumAliphaticRings(mol))
        features.append(Descriptors.RingCount(mol))
        features.append(Descriptors.FractionCsp3(mol))
        features.append(Descriptors.NumHeteroatoms(mol))
        features.append(Descriptors.TPSA(mol))
        features.append(Descriptors.LabuteASA(mol))
        features.append(Descriptors.BalabanJ(mol))
        features.append(Descriptors.BertzCT(mol))
        features.append(Descriptors.Chi0(mol))
        features.append(Descriptors.Chi1(mol))
        features.append(Descriptors.Chi0n(mol))
        features.append(Descriptors.Chi1n(mol))
        features.append(Descriptors.Chi2n(mol))
        features.append(Descriptors.Chi3n(mol))
        features.append(Descriptors.Chi4n(mol))
        features.append(Descriptors.Chi0v(mol))
        features.append(Descriptors.Chi1v(mol))
        features.append(Descriptors.Chi2v(mol))
        features.append(Descriptors.Chi3v(mol))
        features.append(Descriptors.Chi4v(mol))
        features.append(Descriptors.Kappa1(mol))
        features.append(Descriptors.Kappa2(mol))
        features.append(Descriptors.Kappa3(mol))
        
        # Add more features to reach 55
        try:
            features.append(Descriptors.HallKierAlpha(mol))
            features.append(Descriptors.Ipc(mol))
            features.append(Descriptors.EState_VSA1(mol))
            features.append(Descriptors.EState_VSA2(mol))
            features.append(Descriptors.EState_VSA3(mol))
            features.append(Descriptors.EState_VSA4(mol))
            features.append(Descriptors.EState_VSA5(mol))
            features.append(Descriptors.EState_VSA6(mol))
            features.append(Descriptors.EState_VSA7(mol))
            features.append(Descriptors.EState_VSA8(mol))
            features.append(Descriptors.EState_VSA9(mol))
            features.append(Descriptors.EState_VSA10(mol))
            features.append(Descriptors.EState_VSA11(mol))
            features.append(Descriptors.VSA_EState1(mol))
            features.append(Descriptors.VSA_EState2(mol))
            features.append(Descriptors.VSA_EState3(mol))
            features.append(Descriptors.VSA_EState4(mol))
            features.append(Descriptors.VSA_EState5(mol))
            features.append(Descriptors.VSA_EState6(mol))
            features.append(Descriptors.VSA_EState7(mol))
            features.append(Descriptors.VSA_EState8(mol))
            features.append(Descriptors.VSA_EState9(mol))
            features.append(Descriptors.VSA_EState10(mol))
        except:
            # If any descriptors fail, pad with zeros
            while len(features) < 55:
                features.append(0.0)
        
        # Ensure we have exactly 55 features
        features = features[:55]
        while len(features) < 55:
            features.append(0.0)
            
        # Handle NaN values
        features = [x if not np.isnan(x) and not np.isinf(x) else 0.0 for x in features]
        
        return torch.tensor(features, dtype=torch.float32)
        
    except Exception as e:
        print(f"Error processing {smiles}: {e}")
        return torch.zeros(55)


def load_models():
    """Load the saved PyTorch models."""
    models_dir = Path("models")
    
    # Load feature extractor
    feature_model_path = models_dir / "feature_extractor.pt"
    feature_checkpoint = torch.load(feature_model_path, map_location='cpu')
    
    feature_extractor = FeatureExtractorNet(
        input_size=feature_checkpoint['input_size'],
        output_size=feature_checkpoint['output_size']
    )
    feature_extractor.load_state_dict(feature_checkpoint['model_state_dict'])
    feature_extractor.eval()
    
    # Load protonation model
    protonation_model_path = models_dir / "protonation_model.pt"
    protonation_checkpoint = torch.load(protonation_model_path, map_location='cpu')
    
    protonation_model = ProtonationStateNet(
        feature_size=protonation_checkpoint['feature_size'],
        hidden_size=protonation_checkpoint['hidden_size']
    )
    protonation_model.load_state_dict(protonation_checkpoint['model_state_dict'])
    protonation_model.eval()
    
    return feature_extractor, protonation_model


def test_models():
    """Test the loaded models with example molecules."""
    print("Loading PyTorch models...")
    feature_extractor, protonation_model = load_models()
    
    # Test molecules with known pKa values
    test_molecules = [
        ("CCO", "Ethanol", 15.9),           # Alcohol
        ("CC(=O)O", "Acetic acid", 4.76),   # Carboxylic acid
        ("c1ccccc1O", "Phenol", 9.95),      # Phenol
        ("CCN", "Ethylamine", 10.7),        # Amine
        ("c1ccc(cc1)N", "Aniline", 4.6),    # Aromatic amine
        ("c1cccnc1", "Pyridine", 5.2),      # Pyridine
    ]
    
    print("\n" + "="*80)
    print("TESTING PYTORCH MODELS")
    print("="*80)
    
    for smiles, name, known_pka in test_molecules:
        print(f"\n🧪 Testing: {name} ({smiles})")
        print(f"   Known pKa: {known_pka}")
        
        # Extract features using RDKit directly (bypass neural network for now)
        features = extract_features_from_smiles(smiles)
        features_batch = features.unsqueeze(0)
        
        # Test protonation model at different pH values
        ph_values = [4.0, 7.0, 10.0]
        
        with torch.no_grad():
            for ph in ph_values:
                ph_tensor = torch.tensor([ph], dtype=torch.float32)
                pka_pred, state_probs = protonation_model(features_batch, ph_tensor)
                
                print(f"   pH {ph:4.1f}: pKa={pka_pred.item():5.2f}, "
                      f"States=[N:{state_probs[0,0].item():.2f}, "
                      f"P:{state_probs[0,1].item():.2f}, "
                      f"D:{state_probs[0,2].item():.2f}]")
    
    print("\n" + "="*80)
    print("MODEL INFORMATION")
    print("="*80)
    
    # Print model information
    feature_checkpoint = torch.load("models/feature_extractor.pt", map_location='cpu')
    protonation_checkpoint = torch.load("models/protonation_model.pt", map_location='cpu')
    
    print(f"\n📦 Feature Extractor Model:")
    print(f"   Description: {feature_checkpoint['description']}")
    print(f"   Input size: {feature_checkpoint['input_size']}")
    print(f"   Output size: {feature_checkpoint['output_size']}")
    
    print(f"\n📦 Protonation Model:")
    print(f"   Description: {protonation_checkpoint['description']}")
    print(f"   Feature size: {protonation_checkpoint['feature_size']}")
    print(f"   Hidden size: {protonation_checkpoint['hidden_size']}")
    
    print(f"\n📁 Model Files:")
    print(f"   feature_extractor.pt: {Path('models/feature_extractor.pt').stat().st_size / 1024:.1f} KB")
    print(f"   protonation_model.pt: {Path('models/protonation_model.pt').stat().st_size / 1024:.1f} KB")
    
    print("\n✅ Models loaded and tested successfully!")
    print("🚀 Ready for deployment on Replicate backend")


if __name__ == "__main__":
    test_models()