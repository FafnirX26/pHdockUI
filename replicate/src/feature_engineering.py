# Minimal feature engineering for model loading
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from rdkit import Chem
from rdkit.Chem import Descriptors, Fragments


class FeatureEngineering:
    """Basic feature engineering for molecular descriptors."""
    
    def __init__(self):
        pass
    
    def calculate_basic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """Calculate basic molecular descriptors."""
        return {
            'MolWt': Descriptors.MolWt(mol),
            'MolLogP': Descriptors.MolLogP(mol),
            'NumHAcceptors': Descriptors.NumHAcceptors(mol),
            'NumHDonors': Descriptors.NumHDonors(mol),
            'TPSA': Descriptors.TPSA(mol),
        }
    
    def features_to_dataframe(self, molecules: List[Chem.Mol], smiles_list: List[str] = None) -> pd.DataFrame:
        """Convert molecules to feature dataframe."""
        features = []
        for i, mol in enumerate(molecules):
            if mol is not None:
                desc = self.calculate_basic_descriptors(mol)
                desc['molecule_id'] = i
                if smiles_list:
                    desc['smiles'] = smiles_list[i]
                features.append(desc)
        return pd.DataFrame(features)