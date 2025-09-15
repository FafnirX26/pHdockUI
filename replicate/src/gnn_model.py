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