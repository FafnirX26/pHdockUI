"""
Enhanced feature engineering specifically designed for pKa prediction.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Descriptors import MolLogP, MolWt, NumHAcceptors, NumHDonors
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.Fragments import *
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Fingerprints import FingerprintMols
import warnings
warnings.filterwarnings('ignore')


class EnhancedFeatureEngineering:
    """Enhanced feature engineering specifically optimized for pKa prediction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize fingerprint generators
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        self.atom_pair_gen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048)
        
    def identify_ionizable_groups(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Identify and count ionizable groups relevant for pKa prediction.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary with counts of ionizable groups
        """
        if mol is None:
            return {key: 0 for key in ['carboxylic_acids', 'phenols', 'primary_amines', 
                                     'secondary_amines', 'tertiary_amines', 'imidazoles', 
                                     'thiols', 'alcohols', 'guanidines', 'amidines']}
        
        groups = {}
        
        # Carboxylic acids (pKa ~3-5)
        carboxyl_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
        groups['carboxylic_acids'] = len(mol.GetSubstructMatches(carboxyl_pattern)) if carboxyl_pattern else 0
        
        # Phenols (pKa ~8-11)
        phenol_pattern = Chem.MolFromSmarts('[OX2H1][c]')
        groups['phenols'] = len(mol.GetSubstructMatches(phenol_pattern)) if phenol_pattern else 0
        
        # Primary amines (pKa ~9-11)
        primary_amine_pattern = Chem.MolFromSmarts('[NX3;H2;!$(NC=O);!$(NS=O)]')
        groups['primary_amines'] = len(mol.GetSubstructMatches(primary_amine_pattern)) if primary_amine_pattern else 0
        
        # Secondary amines (pKa ~9-11)
        secondary_amine_pattern = Chem.MolFromSmarts('[NX3;H1;!$(NC=O);!$(NS=O)]')
        groups['secondary_amines'] = len(mol.GetSubstructMatches(secondary_amine_pattern)) if secondary_amine_pattern else 0
        
        # Tertiary amines (pKa ~9-11)
        tertiary_amine_pattern = Chem.MolFromSmarts('[NX3;H0;!$(NC=O);!$(NS=O)]')
        groups['tertiary_amines'] = len(mol.GetSubstructMatches(tertiary_amine_pattern)) if tertiary_amine_pattern else 0
        
        # Imidazoles (pKa ~6-7)
        imidazole_pattern = Chem.MolFromSmarts('[nH1][c][nH0]')
        groups['imidazoles'] = len(mol.GetSubstructMatches(imidazole_pattern)) if imidazole_pattern else 0
        
        # Thiols (pKa ~8-9)
        thiol_pattern = Chem.MolFromSmarts('[SX2H1]')
        groups['thiols'] = len(mol.GetSubstructMatches(thiol_pattern)) if thiol_pattern else 0
        
        # Alcohols (pKa ~15-16)
        alcohol_pattern = Chem.MolFromSmarts('[OX2H1][CX4]')
        groups['alcohols'] = len(mol.GetSubstructMatches(alcohol_pattern)) if alcohol_pattern else 0
        
        # Guanidines (pKa ~12-13)
        guanidine_pattern = Chem.MolFromSmarts('[NX3][CX3](=[NX2])[NX3]')
        groups['guanidines'] = len(mol.GetSubstructMatches(guanidine_pattern)) if guanidine_pattern else 0
        
        # Amidines (pKa ~11-12)
        amidine_pattern = Chem.MolFromSmarts('[NX3][CX3](=[NX2])[CX4]')
        groups['amidines'] = len(mol.GetSubstructMatches(amidine_pattern)) if amidine_pattern else 0
        
        return groups
    
    def calculate_electronic_effects(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate electronic effects that influence pKa values.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary with electronic effect descriptors
        """
        if mol is None:
            return {key: 0.0 for key in ['electron_withdrawing_groups', 'electron_donating_groups',
                                       'inductive_effects', 'resonance_effects', 'field_effects']}
        
        effects = {}
        
        # Electron withdrawing groups
        ewg_patterns = [
            '[N+](=O)[O-]',  # Nitro
            'C(F)(F)(F)',     # Trifluoromethyl
            'C#N',            # Cyano
            'C=O',            # Carbonyl
            'S(=O)(=O)',      # Sulfonyl
            '[Cl,Br,I,F]'     # Halogens
        ]
        
        ewg_count = 0
        for pattern_smarts in ewg_patterns:
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern:
                    ewg_count += len(mol.GetSubstructMatches(pattern))
            except:
                continue
        
        effects['electron_withdrawing_groups'] = ewg_count
        
        # Electron donating groups
        edg_patterns = [
            '[NX3]',          # Amines
            '[OX2]',          # Ethers/alcohols
            '[CH3]',          # Methyl groups
            '[c]'             # Aromatic carbons
        ]
        
        edg_count = 0
        for pattern_smarts in edg_patterns:
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern:
                    edg_count += len(mol.GetSubstructMatches(pattern))
            except:
                continue
        
        effects['electron_donating_groups'] = edg_count
        
        # Inductive effects (simplified)
        effects['inductive_effects'] = ewg_count - edg_count * 0.5
        
        # Resonance effects (aromatic systems)
        aromatic_rings = rdMolDescriptors.CalcNumAromaticRings(mol)
        effects['resonance_effects'] = aromatic_rings
        
        # Field effects (distance-dependent)
        effects['field_effects'] = ewg_count * 0.5  # Simplified
        
        return effects
    
    def calculate_structural_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate structural features relevant for pKa prediction.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary with structural descriptors
        """
        if mol is None:
            return {key: 0.0 for key in ['molecular_weight', 'logp', 'tpsa', 'num_heavy_atoms',
                                       'num_rings', 'num_aromatic_rings', 'num_rotatable_bonds',
                                       'num_hbd', 'num_hba', 'formal_charge', 'flexibility']}
        
        features = {}
        
        # Basic molecular properties
        features['molecular_weight'] = Descriptors.MolWt(mol)
        features['logp'] = Descriptors.MolLogP(mol)
        features['tpsa'] = rdMolDescriptors.CalcTPSA(mol)
        features['num_heavy_atoms'] = mol.GetNumHeavyAtoms()
        features['num_rings'] = rdMolDescriptors.CalcNumRings(mol)
        features['num_aromatic_rings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
        features['num_rotatable_bonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
        features['num_hbd'] = rdMolDescriptors.CalcNumHBD(mol)
        features['num_hba'] = rdMolDescriptors.CalcNumHBA(mol)
        features['formal_charge'] = Chem.rdmolops.GetFormalCharge(mol)
        
        # Molecular flexibility
        features['flexibility'] = features['num_rotatable_bonds'] / max(features['num_heavy_atoms'], 1)
        
        return features
    
    def calculate_environment_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate environmental features around ionizable groups.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary with environmental descriptors
        """
        if mol is None:
            return {key: 0.0 for key in ['nearest_ewg_distance', 'nearest_edg_distance',
                                       'aromatic_proximity', 'steric_hindrance']}
        
        features = {}
        
        # Simplified environmental features
        # In a full implementation, these would involve graph distances
        features['nearest_ewg_distance'] = 3.0  # Default distance
        features['nearest_edg_distance'] = 3.0  # Default distance
        features['aromatic_proximity'] = rdMolDescriptors.CalcNumAromaticRings(mol) / max(mol.GetNumHeavyAtoms(), 1)
        features['steric_hindrance'] = rdMolDescriptors.CalcNumRotatableBonds(mol) / max(mol.GetNumHeavyAtoms(), 1)
        
        return features
    
    def calculate_fingerprints(self, mol: Chem.Mol) -> Dict[str, np.ndarray]:
        """
        Calculate molecular fingerprints for similarity and ML.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary with fingerprint arrays
        """
        if mol is None:
            return {
                'morgan_fp': np.zeros(512),
                'atom_pair_fp': np.zeros(512)
            }
        
        fingerprints = {}
        
        # Morgan fingerprint (circular)
        morgan_fp = self.morgan_gen.GetFingerprintAsNumPy(mol)
        fingerprints['morgan_fp'] = morgan_fp[:512]  # Truncate to 512 bits
        
        # Atom pair fingerprint
        atom_pair_fp = self.atom_pair_gen.GetFingerprintAsNumPy(mol)
        fingerprints['atom_pair_fp'] = atom_pair_fp[:512]  # Truncate to 512 bits
        
        return fingerprints
    
    def calculate_quantum_inspired_features(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate quantum-inspired features that would normally come from QM calculations.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary with quantum-inspired descriptors
        """
        if mol is None:
            return {key: 0.0 for key in ['homo_lumo_gap', 'dipole_moment', 'polarizability',
                                       'electron_affinity', 'ionization_potential', 'hardness',
                                       'electrophilicity', 'nucleophilicity']}
        
        features = {}
        
        # Estimate quantum properties from structural features
        # These are simplified estimates - in practice, you'd use actual QM calculations
        
        num_electrons = sum(atom.GetAtomicNum() for atom in mol.GetAtoms())
        num_aromatic = rdMolDescriptors.CalcNumAromaticRings(mol)
        num_heteroatoms = mol.GetNumHeavyAtoms() - len([atom for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6])
        
        # HOMO-LUMO gap (estimated)
        features['homo_lumo_gap'] = 8.0 - num_aromatic * 1.2 + num_heteroatoms * 0.3
        
        # Dipole moment (estimated)
        features['dipole_moment'] = abs(Chem.rdmolops.GetFormalCharge(mol)) * 2.0 + num_heteroatoms * 0.5
        
        # Polarizability (estimated)
        features['polarizability'] = mol.GetNumHeavyAtoms() * 1.5 + num_aromatic * 2.0
        
        # Electron affinity (estimated)
        features['electron_affinity'] = num_heteroatoms * 1.2 - num_aromatic * 0.3
        
        # Ionization potential (estimated)
        features['ionization_potential'] = 9.0 + num_heteroatoms * 0.5 - num_aromatic * 0.8
        
        # Chemical hardness (estimated)
        features['hardness'] = features['homo_lumo_gap'] / 2.0
        
        # Electrophilicity index (estimated)
        features['electrophilicity'] = (features['ionization_potential'] + features['electron_affinity'])**2 / (8 * features['hardness'])
        
        # Nucleophilicity (estimated)
        features['nucleophilicity'] = 1.0 / (features['ionization_potential'] + 1.0)
        
        return features
    
    def calculate_comprehensive_features(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate all molecular features for pKa prediction.
        
        Args:
            mol: RDKit molecule
            
        Returns:
            Dictionary with all calculated features
        """
        all_features = {}
        
        # Calculate different feature types
        all_features.update(self.identify_ionizable_groups(mol))
        all_features.update(self.calculate_electronic_effects(mol))
        all_features.update(self.calculate_structural_features(mol))
        all_features.update(self.calculate_environment_features(mol))
        all_features.update(self.calculate_quantum_inspired_features(mol))
        
        # Add fingerprints
        fingerprints = self.calculate_fingerprints(mol)
        all_features.update(fingerprints)
        
        return all_features
    
    def features_to_dataframe(self, molecules: List[Chem.Mol]) -> pd.DataFrame:
        """
        Convert molecules to feature DataFrame.
        
        Args:
            molecules: List of RDKit molecules
            
        Returns:
            DataFrame with molecular features
        """
        all_features = []
        
        for i, mol in enumerate(molecules):
            try:
                features = self.calculate_comprehensive_features(mol)
                features['molecule_id'] = i
                all_features.append(features)
            except Exception as e:
                self.logger.warning(f"Failed to calculate features for molecule {i}: {e}")
                # Create default features
                default_features = {
                    'molecule_id': i,
                    'carboxylic_acids': 0, 'phenols': 0, 'primary_amines': 0,
                    'secondary_amines': 0, 'tertiary_amines': 0, 'imidazoles': 0,
                    'thiols': 0, 'alcohols': 0, 'guanidines': 0, 'amidines': 0,
                    'electron_withdrawing_groups': 0, 'electron_donating_groups': 0,
                    'inductive_effects': 0, 'resonance_effects': 0, 'field_effects': 0,
                    'molecular_weight': 0, 'logp': 0, 'tpsa': 0, 'num_heavy_atoms': 0,
                    'num_rings': 0, 'num_aromatic_rings': 0, 'num_rotatable_bonds': 0,
                    'num_hbd': 0, 'num_hba': 0, 'formal_charge': 0, 'flexibility': 0,
                    'nearest_ewg_distance': 0, 'nearest_edg_distance': 0,
                    'aromatic_proximity': 0, 'steric_hindrance': 0,
                    'homo_lumo_gap': 0, 'dipole_moment': 0, 'polarizability': 0,
                    'electron_affinity': 0, 'ionization_potential': 0, 'hardness': 0,
                    'electrophilicity': 0, 'nucleophilicity': 0,
                    'morgan_fp': np.zeros(512), 'atom_pair_fp': np.zeros(512)
                }
                all_features.append(default_features)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_features)
        
        # Handle fingerprint columns
        if 'morgan_fp' in df.columns:
            morgan_features = np.vstack(df['morgan_fp'].values)
            morgan_df = pd.DataFrame(morgan_features, columns=[f'morgan_{i}' for i in range(morgan_features.shape[1])])
            df = pd.concat([df.drop('morgan_fp', axis=1), morgan_df], axis=1)
        
        if 'atom_pair_fp' in df.columns:
            atom_pair_features = np.vstack(df['atom_pair_fp'].values)
            atom_pair_df = pd.DataFrame(atom_pair_features, columns=[f'atom_pair_{i}' for i in range(atom_pair_features.shape[1])])
            df = pd.concat([df.drop('atom_pair_fp', axis=1), atom_pair_df], axis=1)
        
        return df


if __name__ == "__main__":
    # Test the enhanced feature engineering
    from rdkit import Chem
    
    # Test molecules
    smiles_list = [
        "CC(=O)O",  # Acetic acid
        "c1ccccc1O",  # Phenol
        "c1ccc(cc1)N",  # Aniline
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
    ]
    
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    
    feature_eng = EnhancedFeatureEngineering()
    features_df = feature_eng.features_to_dataframe(molecules)
    
    print(f"Generated {features_df.shape[0]} molecules with {features_df.shape[1]} features")
    print(f"Feature columns: {list(features_df.columns)[:20]}...")  # Show first 20
    
    # Test ionizable groups detection
    for i, mol in enumerate(molecules):
        groups = feature_eng.identify_ionizable_groups(mol)
        print(f"\nMolecule {i} ({smiles_list[i]}):")
        for group, count in groups.items():
            if count > 0:
                print(f"  {group}: {count}")