"""
Ensemble model with attention mechanisms for pKa prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from rdkit import Chem


class ConformerAttention(nn.Module):
    """Attention mechanism for conformer ensemble."""
    
    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        """
        Initialize conformer attention.
        
        Args:
            feature_dim: Dimension of conformer features
            hidden_dim: Hidden dimension for attention
        """
        super(ConformerAttention, self).__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # Attention layers
        self.attention_fc = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Feature transformation
        self.feature_transform = nn.Linear(feature_dim, hidden_dim)
        
    def forward(self, conformer_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for attention mechanism.
        
        Args:
            conformer_features: Tensor of shape (batch_size, num_conformers, feature_dim)
            
        Returns:
            Tuple of (attended_features, attention_weights)
        """
        batch_size, num_conformers, feature_dim = conformer_features.shape
        
        # Calculate attention scores
        attention_scores = self.attention_fc(conformer_features)  # (batch_size, num_conformers, 1)
        attention_weights = F.softmax(attention_scores, dim=1)  # Normalize over conformers
        
        # Transform features
        transformed_features = self.feature_transform(conformer_features)  # (batch_size, num_conformers, hidden_dim)
        
        # Apply attention
        attended_features = torch.sum(attention_weights * transformed_features, dim=1)  # (batch_size, hidden_dim)
        
        return attended_features, attention_weights.squeeze(-1)


class MultiModalFusion(nn.Module):
    """Multi-modal fusion layer for combining different feature types."""
    
    def __init__(self, 
                 molecular_dim: int,
                 conformer_dim: int,
                 quantum_dim: int,
                 fusion_dim: int = 128):
        """
        Initialize multi-modal fusion.
        
        Args:
            molecular_dim: Dimension of molecular features
            conformer_dim: Dimension of conformer features
            quantum_dim: Dimension of quantum features
            fusion_dim: Dimension of fused representation
        """
        super(MultiModalFusion, self).__init__()
        
        # Individual feature projections
        self.molecular_proj = nn.Linear(molecular_dim, fusion_dim)
        self.conformer_proj = nn.Linear(conformer_dim, fusion_dim)
        self.quantum_proj = nn.Linear(quantum_dim, fusion_dim)
        
        # Cross-modal attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Feature normalization
        self.layer_norm = nn.LayerNorm(fusion_dim)
        
        # Fusion gate
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 3, fusion_dim),
            nn.Sigmoid()
        )
        
    def forward(self, molecular_features: torch.Tensor,
                conformer_features: torch.Tensor,
                quantum_features: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for multi-modal fusion.
        
        Args:
            molecular_features: Molecular descriptors
            conformer_features: Conformer features
            quantum_features: Quantum descriptors
            
        Returns:
            Fused feature representation
        """
        # Project to common dimension
        mol_proj = self.molecular_proj(molecular_features)
        conf_proj = self.conformer_proj(conformer_features)
        quant_proj = self.quantum_proj(quantum_features)
        
        # Stack for cross-attention
        stacked_features = torch.stack([mol_proj, conf_proj, quant_proj], dim=1)  # (batch, 3, fusion_dim)
        
        # Apply cross-modal attention
        attended_features, _ = self.cross_attention(
            stacked_features, stacked_features, stacked_features
        )
        
        # Layer normalization
        attended_features = self.layer_norm(attended_features)
        
        # Flatten for gating
        flattened = attended_features.view(attended_features.size(0), -1)  # (batch, 3*fusion_dim)
        
        # Compute gate weights
        gate_weights = self.gate(flattened)  # (batch, fusion_dim)
        
        # Weighted combination
        fused_features = gate_weights * (mol_proj + conf_proj + quant_proj)
        
        return fused_features


class EnsembleAttentionModel(nn.Module):
    """Ensemble model with attention mechanisms for pKa prediction."""
    
    def __init__(self,
                 molecular_dim: int,
                 conformer_dim: int,
                 quantum_dim: int,
                 hidden_dim: int = 128,
                 dropout: float = 0.3,
                 num_conformers: int = 10):
        """
        Initialize ensemble attention model.
        
        Args:
            molecular_dim: Dimension of molecular features
            conformer_dim: Dimension of individual conformer features
            quantum_dim: Dimension of quantum features
            hidden_dim: Hidden layer dimension
            dropout: Dropout rate
            num_conformers: Number of conformers
        """
        super(EnsembleAttentionModel, self).__init__()
        
        self.molecular_dim = molecular_dim
        self.conformer_dim = conformer_dim
        self.quantum_dim = quantum_dim
        self.hidden_dim = hidden_dim
        self.num_conformers = num_conformers
        
        # Conformer attention mechanism
        self.conformer_attention = ConformerAttention(conformer_dim, hidden_dim // 2)
        
        # Multi-modal fusion
        self.fusion = MultiModalFusion(
            molecular_dim=molecular_dim,
            conformer_dim=hidden_dim // 2,  # Output from conformer attention
            quantum_dim=quantum_dim,
            fusion_dim=hidden_dim
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, molecular_features: torch.Tensor,
                conformer_features: torch.Tensor,
                quantum_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            molecular_features: Molecular descriptors (batch_size, molecular_dim)
            conformer_features: Conformer features (batch_size, num_conformers, conformer_dim)
            quantum_features: Quantum descriptors (batch_size, quantum_dim)
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        # Apply conformer attention
        attended_conformers, attention_weights = self.conformer_attention(conformer_features)
        
        # Multi-modal fusion
        fused_features = self.fusion(molecular_features, attended_conformers, quantum_features)
        
        # Final prediction
        predictions = self.predictor(fused_features)
        
        return predictions, attention_weights


class EnsemblePredictor:
    """Complete ensemble prediction system."""
    
    def __init__(self,
                 hidden_dim: int = 128,
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 device: str = "auto"):
        """
        Initialize ensemble predictor.
        
        Args:
            hidden_dim: Hidden layer dimension
            learning_rate: Learning rate
            batch_size: Batch size
            device: Device to use
        """
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        
        # Feature scalers
        self.molecular_scaler = StandardScaler()
        self.conformer_scaler = StandardScaler()
        self.quantum_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, molecular_features: np.ndarray,
                        conformer_features: np.ndarray,
                        quantum_features: np.ndarray,
                        targets: Optional[np.ndarray] = None,
                        fit_scalers: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Prepare and scale features.
        
        Args:
            molecular_features: Molecular descriptors
            conformer_features: Conformer features
            quantum_features: Quantum descriptors
            targets: Target values (optional)
            fit_scalers: Whether to fit scalers
            
        Returns:
            Tuple of scaled tensors
        """
        # Scale features
        if fit_scalers:
            molecular_scaled = self.molecular_scaler.fit_transform(molecular_features)
            quantum_scaled = self.quantum_scaler.fit_transform(quantum_features)
            
            # Handle conformer features (reshape for scaling)
            batch_size, num_conformers, conformer_dim = conformer_features.shape
            conf_reshaped = conformer_features.reshape(-1, conformer_dim)
            conf_scaled = self.conformer_scaler.fit_transform(conf_reshaped)
            conformer_scaled = conf_scaled.reshape(batch_size, num_conformers, conformer_dim)
            
            if targets is not None:
                targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        else:
            molecular_scaled = self.molecular_scaler.transform(molecular_features)
            quantum_scaled = self.quantum_scaler.transform(quantum_features)
            
            batch_size, num_conformers, conformer_dim = conformer_features.shape
            conf_reshaped = conformer_features.reshape(-1, conformer_dim)
            conf_scaled = self.conformer_scaler.transform(conf_reshaped)
            conformer_scaled = conf_scaled.reshape(batch_size, num_conformers, conformer_dim)
            
            if targets is not None:
                targets_scaled = self.target_scaler.transform(targets.reshape(-1, 1)).flatten()
        
        # Convert to tensors
        molecular_tensor = torch.FloatTensor(molecular_scaled).to(self.device)
        conformer_tensor = torch.FloatTensor(conformer_scaled).to(self.device)
        quantum_tensor = torch.FloatTensor(quantum_scaled).to(self.device)
        
        targets_tensor = None
        if targets is not None:
            targets_tensor = torch.FloatTensor(targets_scaled).to(self.device)
        
        return molecular_tensor, conformer_tensor, quantum_tensor, targets_tensor
    
    def train(self, molecular_features: np.ndarray,
              conformer_features: np.ndarray,
              quantum_features: np.ndarray,
              targets: np.ndarray,
              num_epochs: int = 100,
              test_size: float = 0.2,
              patience: int = 15,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the ensemble model.
        
        Args:
            molecular_features: Molecular descriptors
            conformer_features: Conformer features
            quantum_features: Quantum descriptors
            targets: Target pKa values
            num_epochs: Number of training epochs
            test_size: Test set fraction
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training results
        """
        # Split data
        indices = np.arange(len(targets))
        train_idx, test_idx = train_test_split(indices, test_size=test_size, random_state=42)
        
        train_mol = molecular_features[train_idx]
        train_conf = conformer_features[train_idx]
        train_quant = quantum_features[train_idx]
        train_targets = targets[train_idx]
        
        test_mol = molecular_features[test_idx]
        test_conf = conformer_features[test_idx]
        test_quant = quantum_features[test_idx]
        test_targets = targets[test_idx]
        
        # Prepare features
        train_mol_t, train_conf_t, train_quant_t, train_targets_t = self.prepare_features(
            train_mol, train_conf, train_quant, train_targets, fit_scalers=True
        )
        
        test_mol_t, test_conf_t, test_quant_t, test_targets_t = self.prepare_features(
            test_mol, test_conf, test_quant, test_targets, fit_scalers=False
        )
        
        # Initialize model
        self.model = EnsembleAttentionModel(
            molecular_dim=molecular_features.shape[1],
            conformer_dim=conformer_features.shape[2],
            quantum_dim=quantum_features.shape[1],
            hidden_dim=self.hidden_dim,
            num_conformers=conformer_features.shape[1]
        ).to(self.device)
        
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {'train_loss': [], 'val_loss': []}
        
        for epoch in range(num_epochs):
            # Training
            self.model.train()
            train_predictions, _ = self.model(train_mol_t, train_conf_t, train_quant_t)
            train_loss = self.criterion(train_predictions.squeeze(), train_targets_t)
            
            self.optimizer.zero_grad()
            train_loss.backward()
            self.optimizer.step()
            
            history['train_loss'].append(train_loss.item())
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_predictions, attention_weights = self.model(test_mol_t, test_conf_t, test_quant_t)
                val_loss = self.criterion(val_predictions.squeeze(), test_targets_t)
                history['val_loss'].append(val_loss.item())
            
            # Early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 10 == 0:
                # Calculate metrics in original scale
                val_pred_orig = self.target_scaler.inverse_transform(
                    val_predictions.cpu().numpy().reshape(-1, 1)
                ).flatten()
                val_true_orig = self.target_scaler.inverse_transform(
                    test_targets_t.cpu().numpy().reshape(-1, 1)
                ).flatten()
                
                val_r2 = r2_score(val_true_orig, val_pred_orig)
                val_mae = mean_absolute_error(val_true_orig, val_pred_orig)
                
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                               f"Train Loss: {train_loss.item():.4f}, "
                               f"Val Loss: {val_loss.item():.4f}, "
                               f"Val R²: {val_r2:.3f}, "
                               f"Val MAE: {val_mae:.3f}")
            
            if patience_counter >= patience:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if 'best_model_state' in locals():
            self.model.load_state_dict(best_model_state)
        
        self.is_trained = True
        
        # Final evaluation
        self.model.eval()
        with torch.no_grad():
            final_predictions, final_attention = self.model(test_mol_t, test_conf_t, test_quant_t)
            
            # Convert back to original scale
            pred_orig = self.target_scaler.inverse_transform(
                final_predictions.cpu().numpy().reshape(-1, 1)
            ).flatten()
            true_orig = self.target_scaler.inverse_transform(
                test_targets_t.cpu().numpy().reshape(-1, 1)
            ).flatten()
        
        results = {
            'history': history,
            'val_r2': r2_score(true_orig, pred_orig),
            'val_mae': mean_absolute_error(true_orig, pred_orig),
            'val_rmse': np.sqrt(mean_squared_error(true_orig, pred_orig)),
            'predictions': pred_orig,
            'targets': true_orig,
            'attention_weights': final_attention.cpu().numpy()
        }
        
        return results
    
    def predict(self, molecular_features: np.ndarray,
                conformer_features: np.ndarray,
                quantum_features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with attention weights.
        
        Args:
            molecular_features: Molecular descriptors
            conformer_features: Conformer features
            quantum_features: Quantum descriptors
            
        Returns:
            Tuple of (predictions, attention_weights)
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        # Prepare features
        mol_t, conf_t, quant_t, _ = self.prepare_features(
            molecular_features, conformer_features, quantum_features, fit_scalers=False
        )
        
        # Make predictions
        self.model.eval()
        with torch.no_grad():
            predictions, attention_weights = self.model(mol_t, conf_t, quant_t)
        
        # Convert back to original scale
        pred_orig = self.target_scaler.inverse_transform(
            predictions.cpu().numpy().reshape(-1, 1)
        ).flatten()
        
        return pred_orig, attention_weights.cpu().numpy()
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'molecular_scaler': self.molecular_scaler,
            'conformer_scaler': self.conformer_scaler,
            'quantum_scaler': self.quantum_scaler,
            'target_scaler': self.target_scaler,
            'hidden_dim': self.hidden_dim,
            'batch_size': self.batch_size
        }
        
        torch.save(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Union[str, Path],
                   molecular_dim: int,
                   conformer_dim: int,
                   quantum_dim: int,
                   num_conformers: int) -> None:
        """Load a trained model."""
        model_data = torch.load(model_path, map_location=self.device)
        
        # Restore scalers
        self.molecular_scaler = model_data['molecular_scaler']
        self.conformer_scaler = model_data['conformer_scaler']
        self.quantum_scaler = model_data['quantum_scaler']
        self.target_scaler = model_data['target_scaler']
        self.hidden_dim = model_data['hidden_dim']
        self.batch_size = model_data['batch_size']
        
        # Recreate model
        self.model = EnsembleAttentionModel(
            molecular_dim=molecular_dim,
            conformer_dim=conformer_dim,
            quantum_dim=quantum_dim,
            hidden_dim=self.hidden_dim,
            num_conformers=num_conformers
        ).to(self.device)
        
        self.model.load_state_dict(model_data['model_state_dict'])
        self.is_trained = True
        self.logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create synthetic data for testing
    np.random.seed(42)
    
    num_molecules = 100
    molecular_dim = 50
    conformer_dim = 20
    quantum_dim = 10
    num_conformers = 8
    
    # Generate synthetic features
    molecular_features = np.random.randn(num_molecules, molecular_dim)
    conformer_features = np.random.randn(num_molecules, num_conformers, conformer_dim)
    quantum_features = np.random.randn(num_molecules, quantum_dim)
    
    # Generate synthetic targets with some correlation to features
    targets = (
        np.mean(molecular_features[:, :5], axis=1) * 2.0 +
        np.mean(conformer_features[:, :, :3], axis=(1, 2)) * 1.5 +
        np.mean(quantum_features[:, :3], axis=1) * 1.0 +
        np.random.normal(0, 0.5, num_molecules) + 7.0  # Base pKa around 7
    )
    
    # Initialize and train ensemble model
    ensemble = EnsemblePredictor(
        hidden_dim=64,  # Smaller for testing
        learning_rate=0.001,
        batch_size=16
    )
    
    try:
        results = ensemble.train(
            molecular_features=molecular_features,
            conformer_features=conformer_features,
            quantum_features=quantum_features,
            targets=targets,
            num_epochs=50,
            patience=10,
            verbose=True
        )
        
        print(f"\nEnsemble training completed!")
        print(f"Validation R²: {results['val_r2']:.3f}")
        print(f"Validation RMSE: {results['val_rmse']:.3f}")
        print(f"Validation MAE: {results['val_mae']:.3f}")
        
        # Test predictions with attention
        test_predictions, attention_weights = ensemble.predict(
            molecular_features[:5],
            conformer_features[:5],
            quantum_features[:5]
        )
        
        print(f"\nSample predictions: {test_predictions}")
        print(f"Actual values: {targets[:5]}")
        print(f"Attention weights shape: {attention_weights.shape}")
        print(f"Sample attention weights:\n{attention_weights[0]}")  # First molecule's attention
        
    except Exception as e:
        print(f"Ensemble training failed: {e}")
        import traceback
        traceback.print_exc()