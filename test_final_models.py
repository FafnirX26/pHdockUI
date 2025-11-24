#!/usr/bin/env python3
"""
Test the final production PyTorch models.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import importlib.util
from pathlib import Path

# Import the working classes
spec = importlib.util.spec_from_file_location("working_models", "create_working_models.py")
working_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(working_models)

TorchFeatureExtractor = working_models.TorchFeatureExtractor
TorchProtonationModel = working_models.TorchProtonationModel


def load_models():
    """Load the saved production models."""
    models_dir = Path("models")
    
    # Load feature extractor
    print("Loading feature extractor...")
    feature_extractor = TorchFeatureExtractor()
    feature_checkpoint = torch.load(models_dir / "working_feature_extractor.pt", map_location='cpu', weights_only=False)
    feature_extractor.load_state_dict(feature_checkpoint['model_state_dict'])
    feature_extractor.eval()
    
    # Load protonation model
    print("Loading protonation model...")
    protonation_model = TorchProtonationModel()
    protonation_checkpoint = torch.load(models_dir / "working_protonation_model.pt", map_location='cpu', weights_only=False)
    protonation_model.load_state_dict(protonation_checkpoint['model_state_dict'])
    protonation_model.eval()
    
    return feature_extractor, protonation_model


def main():
    """Test the final production models."""
    print("🧪 Testing Final Production PyTorch Models")
    print("="*50)
    
    # Load models
    feature_extractor, protonation_model = load_models()
    
    # Test molecules with known pKa values
    test_cases = [
        ("CCO", "Ethanol", 15.9),
        ("CC(=O)O", "Acetic acid", 4.76),
        ("c1ccccc1O", "Phenol", 9.95),
        ("CCN", "Ethylamine", 10.7),
        ("c1cccnc1", "Pyridine", 5.2),
        ("Nc1ccccc1", "Aniline", 4.6),
        ("CC(C)(C)C(=O)O", "Pivaloic acid", 5.03),
        ("CN", "Methylamine", 10.6),
    ]
    
    print(f"\n📊 pKa Prediction Test Results:")
    print("-" * 60)
    print(f"{'Molecule':<15} {'Known pKa':<10} {'Predicted':<10} {'Error':<8}")
    print("-" * 60)
    
    total_error = 0
    valid_predictions = 0
    
    for smiles, name, known_pka in test_cases:
        try:
            # Test pKa prediction
            pka_pred = protonation_model.forward(smiles)
            pred_value = float(pka_pred.item()) if isinstance(pka_pred, torch.Tensor) else float(pka_pred)
            
            error = abs(pred_value - known_pka)
            total_error += error
            valid_predictions += 1
            
            print(f"{name:<15} {known_pka:<10.2f} {pred_value:<10.2f} {error:<8.2f}")
            
        except Exception as e:
            print(f"{name:<15} {'ERROR':<10} {'ERROR':<10} {str(e):<8}")
    
    if valid_predictions > 0:
        mae = total_error / valid_predictions
        print("-" * 60)
        print(f"📈 Overall MAE: {mae:.2f} pKa units")
        print(f"✅ Performance: {'Good' if mae < 2.0 else 'Acceptable' if mae < 3.0 else 'Needs improvement'}")
    
    # Test protonation state prediction
    print(f"\n🧬 Protonation State Prediction Test:")
    print("-" * 80)
    
    test_molecules = [
        ("CC(=O)O", "Acetic acid (carboxylic acid)"),
        ("c1ccccc1O", "Phenol (phenolic OH)"),
        ("CCN", "Ethylamine (basic amine)"),
    ]
    
    ph_values = [4.0, 7.0, 10.0]
    
    for smiles, description in test_molecules:
        print(f"\n{description} ({smiles}):")
        
        try:
            states = protonation_model.predict_protonation_states(smiles, ph_values)
            
            for ph in ph_values:
                if ph in states:
                    info = states[ph]
                    print(f"  pH {ph:4.1f}: pKa={info['predicted_pka']:5.2f} | "
                          f"Neutral={info['neutral_prob']:5.2f} | "
                          f"Protonated={info['protonated_prob']:5.2f} | "
                          f"Deprotonated={info['deprotonated_prob']:5.2f}")
                          
        except Exception as e:
            print(f"  ERROR: {e}")
    
    # Model information
    print(f"\n📦 Model Information:")
    print("-" * 50)
    
    feature_checkpoint = torch.load("models/working_feature_extractor.pt", map_location='cpu', weights_only=False)
    protonation_checkpoint = torch.load("models/working_protonation_model.pt", map_location='cpu', weights_only=False)
    
    print(f"🔧 Feature Extractor:")
    print(f"   Description: {feature_checkpoint['description']}")
    print(f"   Features: {feature_checkpoint['features']}")
    print(f"   Performance: {feature_checkpoint['performance']}")
    
    print(f"\n⚗️ Protonation Model:")
    print(f"   Description: {protonation_checkpoint['description']}")
    print(f"   Features: {protonation_checkpoint['features']}")
    print(f"   Performance: {protonation_checkpoint['performance']}")
    print(f"   Ensemble Weights: XGB={protonation_checkpoint['ensemble_weights'][0]:.2f}, RF={protonation_checkpoint['ensemble_weights'][1]:.2f}")
    
    # File sizes
    feature_size = Path("models/working_feature_extractor.pt").stat().st_size / 1024
    protonation_size = Path("models/working_protonation_model.pt").stat().st_size / 1024
    
    print(f"\n💾 File Sizes:")
    print(f"   working_feature_extractor.pt: {feature_size:.1f} KB")
    print(f"   working_protonation_model.pt: {protonation_size:.1f} KB")
    print(f"   Total: {(feature_size + protonation_size):.1f} KB")
    
    print(f"\n🎯 Models are ready for Replicate deployment!")
    print(f"📁 Location: {Path('models').absolute()}")


if __name__ == "__main__":
    main()