"""
Graph Neural Network model for pKa prediction.
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
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool, global_add_pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors


class MolecularDataset(Dataset):
    """Dataset for molecular graphs."""
    
    def __init__(self, molecules: List[Chem.Mol], targets: np.ndarray):
        """
        Initialize molecular dataset.
        
        Args:
            molecules: List of RDKit molecules
            targets: Target values (pKa)
        """
        self.molecules = molecules
        self.targets = torch.FloatTensor(targets)
        
    def __len__(self):
        return len(self.molecules)
    
    def __getitem__(self, idx):
        mol = self.molecules[idx]
        target = self.targets[idx]
        
        # Convert molecule to graph
        graph_data = self.mol_to_graph(mol)
        graph_data.y = target.unsqueeze(0)  # Add dimension for target
        
        return graph_data
    
    def mol_to_graph(self, mol: Chem.Mol) -> Data:
        """
        Convert RDKit molecule to PyTorch Geometric graph.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            PyTorch Geometric Data object
        """
        # Node features (atoms)
        node_features = []
        for atom in mol.GetAtoms():
            features = [
                atom.GetAtomicNum(),
                atom.GetDegree(),
                atom.GetFormalCharge(),
                int(atom.GetIsAromatic()),
                atom.GetNumImplicitHs(),
                int(atom.GetChiralTag()),
                atom.GetTotalNumHs(),
                int(atom.GetHybridization()),
            ]
            node_features.append(features)
        
        # Edge indices and features
        edge_indices = []
        edge_features = []
        
        for bond in mol.GetBonds():
            start_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            
            # Add edge in both directions (undirected graph)
            edge_indices.extend([[start_idx, end_idx], [end_idx, start_idx]])
            
            # Bond features
            bond_features = [
                bond.GetBondTypeAsDouble(),
                int(bond.GetIsAromatic()),
                int(bond.GetIsConjugated()),
                int(bond.IsInRing()),
            ]
            
            # Add features for both directions
            edge_features.extend([bond_features, bond_features])
        
        # Convert to tensors
        x = torch.FloatTensor(node_features)
        edge_index = torch.LongTensor(edge_indices).t().contiguous() if edge_indices else torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.FloatTensor(edge_features) if edge_features else torch.empty((0, 4), dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class GNNModel(nn.Module):
    """Graph Neural Network for pKa prediction."""
    
    def __init__(self, 
                 input_dim: int = 8,
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 output_dim: int = 1,
                 gnn_type: str = "gcn",
                 dropout: float = 0.2,
                 pooling: str = "mean"):
        """
        Initialize GNN model.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            output_dim: Output dimension (1 for pKa prediction)
            gnn_type: Type of GNN ("gcn", "gat")
            dropout: Dropout rate
            pooling: Graph pooling method ("mean", "max", "add")
        """
        super(GNNModel, self).__init__()
        
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.dropout = dropout
        self.pooling = pooling
        
        # Graph neural network layers
        self.gnn_layers = nn.ModuleList()
        
        if gnn_type == "gcn":
            self.gnn_layers.append(GCNConv(input_dim, hidden_dim))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GCNConv(hidden_dim, hidden_dim))
        elif gnn_type == "gat":
            self.gnn_layers.append(GATConv(input_dim, hidden_dim, heads=4, concat=False))
            for _ in range(num_layers - 1):
                self.gnn_layers.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
        else:
            raise ValueError(f"Unsupported GNN type: {gnn_type}")
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Final prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, output_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, data):
        """
        Forward pass.
        
        Args:
            data: PyTorch Geometric batch
            
        Returns:
            Predictions
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph neural network layers
        for i, gnn_layer in enumerate(self.gnn_layers):
            x = gnn_layer(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Graph pooling
        if self.pooling == "mean":
            x = global_mean_pool(x, batch)
        elif self.pooling == "max":
            x = global_max_pool(x, batch)
        elif self.pooling == "add":
            x = global_add_pool(x, batch)
        else:
            raise ValueError(f"Unsupported pooling method: {self.pooling}")
        
        # Final prediction
        x = self.predictor(x)
        
        return x


class GNNTrainer:
    """Trainer for GNN models."""
    
    def __init__(self, 
                 model: GNNModel,
                 learning_rate: float = 0.001,
                 weight_decay: float = 1e-5,
                 device: str = "auto"):
        """
        Initialize GNN trainer.
        
        Args:
            model: GNN model to train
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay for regularization
            device: Device to use ("cpu", "cuda", or "auto")
        """
        self.model = model
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        self.model.to(self.device)
        
        # Optimizer and loss function
        self.optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            batch = batch.to(self.device)
            
            # Forward pass
            predictions = self.model(batch)
            loss = self.criterion(predictions.squeeze(), batch.y.squeeze())
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, predictions, targets)
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                
                predictions = self.model(batch)
                loss = self.criterion(predictions.squeeze(), batch.y.squeeze())
                
                total_loss += loss.item()
                num_batches += 1
                
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_targets.extend(batch.y.cpu().numpy().flatten())
        
        avg_loss = total_loss / num_batches
        return avg_loss, np.array(all_predictions), np.array(all_targets)
    
    def train(self, train_loader: DataLoader, 
              val_loader: Optional[DataLoader] = None,
              num_epochs: int = 100,
              patience: int = 10,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Optional validation data loader
            num_epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Whether to print training progress
            
        Returns:
            Training results dictionary
        """
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_predictions, val_targets = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    val_r2 = r2_score(val_targets, val_predictions)
                    val_mae = mean_absolute_error(val_targets, val_predictions)
                    self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                                   f"Train Loss: {train_loss:.4f}, "
                                   f"Val Loss: {val_loss:.4f}, "
                                   f"Val R²: {val_r2:.3f}, "
                                   f"Val MAE: {val_mae:.3f}")
                
                # Early stopping
                if patience_counter >= patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    self.logger.info(f"Epoch {epoch+1}/{num_epochs}: Train Loss: {train_loss:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        # Final evaluation
        results = {'history': self.history}
        
        if val_loader is not None:
            final_val_loss, final_predictions, final_targets = self.validate(val_loader)
            results.update({
                'val_loss': final_val_loss,
                'val_r2': r2_score(final_targets, final_predictions),
                'val_mae': mean_absolute_error(final_targets, final_predictions),
                'val_rmse': np.sqrt(mean_squared_error(final_targets, final_predictions)),
                'predictions': final_predictions,
                'targets': final_targets
            })
        
        return results
    
    def predict(self, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            data_loader: Data loader for prediction
            
        Returns:
            Predictions array
        """
        self.model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in data_loader:
                batch = batch.to(self.device)
                predictions = self.model(batch)
                all_predictions.extend(predictions.cpu().numpy().flatten())
        
        return np.array(all_predictions)


class GNNpKaPredictor:
    """Complete GNN-based pKa prediction system."""
    
    def __init__(self, 
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 gnn_type: str = "gcn",
                 learning_rate: float = 0.001,
                 batch_size: int = 32,
                 device: str = "auto"):
        """
        Initialize GNN pKa predictor.
        
        Args:
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            gnn_type: Type of GNN ("gcn", "gat")
            learning_rate: Learning rate
            batch_size: Batch size for training
            device: Device to use
        """
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gnn_type = gnn_type
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device
        
        self.model = None
        self.trainer = None
        self.target_scaler = StandardScaler()
        
        self.is_trained = False
        self.logger = logging.getLogger(__name__)
    
    def prepare_data(self, molecules: List[Chem.Mol], 
                    targets: np.ndarray,
                    test_size: float = 0.2) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data for training.
        
        Args:
            molecules: List of molecules
            targets: Target values
            test_size: Fraction for test set
            
        Returns:
            Tuple of (train_loader, test_loader)
        """
        # Split data
        train_mols, test_mols, train_targets, test_targets = train_test_split(
            molecules, targets, test_size=test_size, random_state=42
        )
        
        # Scale targets
        train_targets_scaled = self.target_scaler.fit_transform(train_targets.reshape(-1, 1)).flatten()
        test_targets_scaled = self.target_scaler.transform(test_targets.reshape(-1, 1)).flatten()
        
        # Create datasets
        train_dataset = MolecularDataset(train_mols, train_targets_scaled)
        test_dataset = MolecularDataset(test_mols, test_targets_scaled)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        return train_loader, test_loader
    
    def train(self, molecules: List[Chem.Mol], 
              targets: np.ndarray,
              num_epochs: int = 100,
              patience: int = 10,
              verbose: bool = True) -> Dict[str, Any]:
        """
        Train the GNN model.
        
        Args:
            molecules: List of molecules
            targets: Target pKa values
            num_epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Whether to print progress
            
        Returns:
            Training results
        """
        # Prepare data
        train_loader, test_loader = self.prepare_data(molecules, targets)
        
        # Initialize model
        self.model = GNNModel(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            gnn_type=self.gnn_type
        )
        
        # Initialize trainer
        self.trainer = GNNTrainer(
            self.model,
            learning_rate=self.learning_rate,
            device=self.device
        )
        
        # Train model
        results = self.trainer.train(
            train_loader=train_loader,
            val_loader=test_loader,
            num_epochs=num_epochs,
            patience=patience,
            verbose=verbose
        )
        
        self.is_trained = True
        
        # Convert back to original scale for reporting
        if 'predictions' in results and 'targets' in results:
            results['predictions'] = self.target_scaler.inverse_transform(
                results['predictions'].reshape(-1, 1)
            ).flatten()
            results['targets'] = self.target_scaler.inverse_transform(
                results['targets'].reshape(-1, 1)
            ).flatten()
            
            # Recalculate metrics in original scale
            results['val_r2'] = r2_score(results['targets'], results['predictions'])
            results['val_mae'] = mean_absolute_error(results['targets'], results['predictions'])
            results['val_rmse'] = np.sqrt(mean_squared_error(results['targets'], results['predictions']))
        
        return results
    
    def predict(self, molecules: List[Chem.Mol]) -> np.ndarray:
        """
        Predict pKa values.
        
        Args:
            molecules: List of molecules
            
        Returns:
            Predicted pKa values
        """
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        # Create dataset with dummy targets
        dummy_targets = np.zeros(len(molecules))
        dataset = MolecularDataset(molecules, dummy_targets)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        # Make predictions
        predictions_scaled = self.trainer.predict(data_loader)
        
        # Scale back to original range
        predictions = self.target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()
        
        return predictions
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save the trained model."""
        if not self.is_trained:
            raise ValueError("Model has not been trained yet")
        
        model_data = {
            'model_state_dict': self.model.state_dict(),
            'target_scaler': self.target_scaler,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'gnn_type': self.gnn_type,
            'batch_size': self.batch_size
        }
        
        torch.save(model_data, model_path)
        self.logger.info(f"Model saved to {model_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load a trained model."""
        model_data = torch.load(model_path, map_location='cpu')
        
        self.hidden_dim = model_data['hidden_dim']
        self.num_layers = model_data['num_layers']
        self.gnn_type = model_data['gnn_type']
        self.batch_size = model_data['batch_size']
        self.target_scaler = model_data['target_scaler']
        
        # Recreate model
        self.model = GNNModel(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            gnn_type=self.gnn_type
        )
        self.model.load_state_dict(model_data['model_state_dict'])
        
        # Recreate trainer
        self.trainer = GNNTrainer(self.model, device=self.device)
        
        self.is_trained = True
        self.logger.info(f"Model loaded from {model_path}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample molecules
    from src.pka_prediction import pKaPredictionModel
    
    smiles_list = [
        "CC(=O)O",  # Acetic acid
        "c1ccccc1O",  # Phenol
        "c1ccc(cc1)N",  # Aniline
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
        "CC1=CC=C(C=C1)C(=O)O",  # Para-toluic acid
        "CCN(CC)CC",  # Triethylamine
        "C1=CC=C(C=C1)C(=O)O",  # Benzoic acid
        "NC(=N)N",  # Guanidine
        "CC(C)(C)C1=CC=C(C=C1)O",  # 4-tert-butylphenol
        "CCCCCCCC(=O)O"  # Octanoic acid
    ]
    
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    molecules = [mol for mol in molecules if mol is not None]
    
    # Generate synthetic pKa data
    baseline_model = pKaPredictionModel()
    synthetic_data = baseline_model.generate_synthetic_pka_data(molecules, pd.DataFrame())
    
    if len(synthetic_data) > 0:
        # Get targets
        targets = synthetic_data.groupby('molecule_id')['pka'].first().values
        
        # Filter molecules and targets
        valid_indices = ~np.isnan(targets)
        valid_molecules = [molecules[i] for i in range(len(molecules)) if valid_indices[i]]
        valid_targets = targets[valid_indices]
        
        if len(valid_molecules) > 2:  # Need at least 3 molecules for train/test split
            # Initialize and train GNN
            gnn_predictor = GNNpKaPredictor(
                hidden_dim=32,  # Smaller for this example
                num_layers=2,
                gnn_type="gcn",
                batch_size=8
            )
            
            try:
                results = gnn_predictor.train(
                    molecules=valid_molecules,
                    targets=valid_targets,
                    num_epochs=50,  # Fewer epochs for testing
                    patience=10,
                    verbose=True
                )
                
                print(f"\nGNN training completed!")
                print(f"Validation R²: {results.get('val_r2', 'N/A'):.3f}")
                print(f"Validation RMSE: {results.get('val_rmse', 'N/A'):.3f}")
                print(f"Validation MAE: {results.get('val_mae', 'N/A'):.3f}")
                
                # Test predictions
                predictions = gnn_predictor.predict(valid_molecules[:3])
                print(f"\nSample predictions: {predictions}")
                print(f"Actual values: {valid_targets[:3]}")
                
            except Exception as e:
                print(f"GNN training failed: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Not enough valid molecules for GNN training")
    else:
        print("No synthetic pKa data generated")