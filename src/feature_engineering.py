"""
Feature engineering module for calculating molecular descriptors.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, Crippen, Lipinski, Fragments, AllChem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
import pandas as pd


class FeatureEngineering:
    """Generate molecular features for machine learning models."""
    
    def __init__(self, include_3d: bool = True, 
                 fingerprint_radius: int = 2,
                 fingerprint_bits: int = 2048):
        """
        Initialize feature engineering.
        
        Args:
            include_3d: Whether to include 3D descriptors
            fingerprint_radius: Radius for Morgan fingerprints
            fingerprint_bits: Number of bits for fingerprints
        """
        self.include_3d = include_3d
        self.fingerprint_radius = fingerprint_radius
        self.fingerprint_bits = fingerprint_bits
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize fingerprint generators
        self.morgan_gen = rdFingerprintGenerator.GetMorganGenerator(
            radius=fingerprint_radius, fpSize=fingerprint_bits
        )
        
        # Pre-defined descriptor lists
        self.basic_descriptors = [
            'MolWt', 'MolLogP', 'MolMR', 'HeavyAtomCount', 'NumHAcceptors',
            'NumHDonors', 'NumRotatableBonds', 'NumAromaticRings', 'NumSaturatedRings',
            'NumAliphaticRings', 'RingCount', 'FractionCsp3', 'NumHeteroatoms',
            'TPSA', 'LabuteASA', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi1', 'Chi0n',
            'Chi1n', 'Chi0v', 'Chi1v', 'Kappa1', 'Kappa2', 'Kappa3'
        ]
        
        self.lipinski_descriptors = [
            'NumHAcceptors', 'NumHDonors', 'MolWt', 'MolLogP'
        ]
        
        self.fragment_descriptors = [
            'fr_Al_COO', 'fr_Al_OH', 'fr_Al_OH_noTert', 'fr_ArN', 'fr_Ar_COO',
            'fr_Ar_N', 'fr_Ar_NH', 'fr_Ar_OH', 'fr_COO', 'fr_COO2', 'fr_C_O',
            'fr_C_O_noCOO', 'fr_C_S', 'fr_HOCCN', 'fr_Imine', 'fr_NH0', 'fr_NH1',
            'fr_NH2', 'fr_N_O', 'fr_Ndealkylation1', 'fr_Ndealkylation2',
            'fr_Nhpyrrole', 'fr_SH', 'fr_aldehyde', 'fr_alkyl_carbamate',
            'fr_alkyl_halide', 'fr_allylic_oxid', 'fr_amide', 'fr_amidine',
            'fr_aniline', 'fr_aryl_methyl', 'fr_azide', 'fr_azo', 'fr_barbitur',
            'fr_benzene', 'fr_benzodiazepine', 'fr_bicyclic', 'fr_diazo',
            'fr_dihydropyridine', 'fr_epoxide', 'fr_ester', 'fr_ether',
            'fr_furan', 'fr_guanido', 'fr_halogen', 'fr_hdrzine', 'fr_hdrzone',
            'fr_imidazole', 'fr_imide', 'fr_isocyan', 'fr_isothiocyan',
            'fr_ketone', 'fr_ketone_Topliss', 'fr_lactam', 'fr_lactone',
            'fr_methoxy', 'fr_morpholine', 'fr_nitrile', 'fr_nitro',
            'fr_nitro_arom', 'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_oxazole',
            'fr_oxime', 'fr_para_hydroxylation', 'fr_phenol', 'fr_phenol_noOrthoHbond',
            'fr_phos_acid', 'fr_phos_ester', 'fr_piperdine', 'fr_piperzine',
            'fr_priamide', 'fr_prisulfonamd', 'fr_pyridine', 'fr_quatN',
            'fr_sulfide', 'fr_sulfonamd', 'fr_sulfone', 'fr_term_acetylene',
            'fr_tetrazole', 'fr_thiazole', 'fr_thiocyan', 'fr_thiophene',
            'fr_unbrch_alkane', 'fr_urea'
        ]
    
    def calculate_basic_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate basic molecular descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of basic descriptors
        """
        descriptors = {}
        
        try:
            # Basic molecular properties
            descriptors['MolWt'] = Descriptors.MolWt(mol)
            descriptors['MolLogP'] = Descriptors.MolLogP(mol)
            descriptors['MolMR'] = Descriptors.MolMR(mol)
            descriptors['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
            descriptors['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
            descriptors['NumHDonors'] = Descriptors.NumHDonors(mol)
            descriptors['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
            descriptors['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
            descriptors['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
            descriptors['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
            descriptors['RingCount'] = Descriptors.RingCount(mol)
            descriptors['FractionCsp3'] = Descriptors.FractionCsp3(mol)
            descriptors['NumHeteroatoms'] = Descriptors.NumHeteroatoms(mol)
            descriptors['TPSA'] = Descriptors.TPSA(mol)
            descriptors['LabuteASA'] = Descriptors.LabuteASA(mol)
            
            # Topological descriptors
            descriptors['BalabanJ'] = Descriptors.BalabanJ(mol)
            descriptors['BertzCT'] = Descriptors.BertzCT(mol)
            descriptors['Chi0'] = Descriptors.Chi0(mol)
            descriptors['Chi1'] = Descriptors.Chi1(mol)
            descriptors['Chi0n'] = Descriptors.Chi0n(mol)
            descriptors['Chi1n'] = Descriptors.Chi1n(mol)
            descriptors['Chi0v'] = Descriptors.Chi0v(mol)
            descriptors['Chi1v'] = Descriptors.Chi1v(mol)
            descriptors['Kappa1'] = Descriptors.Kappa1(mol)
            descriptors['Kappa2'] = Descriptors.Kappa2(mol)
            descriptors['Kappa3'] = Descriptors.Kappa3(mol)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate basic descriptors: {e}")
            # Fill with NaN for failed calculations
            for desc_name in self.basic_descriptors:
                if desc_name not in descriptors:
                    descriptors[desc_name] = np.nan
        
        return descriptors
    
    def calculate_fragment_descriptors(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Calculate fragment-based descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of fragment descriptors
        """
        descriptors = {}
        
        try:
            # Get all fragment descriptors
            for frag_name in self.fragment_descriptors:
                try:
                    frag_func = getattr(Fragments, frag_name)
                    descriptors[frag_name] = frag_func(mol)
                except AttributeError:
                    self.logger.warning(f"Fragment descriptor {frag_name} not found")
                    descriptors[frag_name] = 0
                except Exception as e:
                    self.logger.warning(f"Failed to calculate {frag_name}: {e}")
                    descriptors[frag_name] = 0
                    
        except Exception as e:
            self.logger.warning(f"Failed to calculate fragment descriptors: {e}")
        
        return descriptors
    
    def calculate_fingerprints(self, mol: Chem.Mol) -> Dict[str, Any]:
        """
        Calculate molecular fingerprints.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary containing different fingerprint types
        """
        fingerprints = {}
        
        try:
            # Morgan fingerprint (ECFP)
            morgan_fp = GetMorganFingerprintAsBitVect(
                mol, self.fingerprint_radius, nBits=self.fingerprint_bits
            )
            fingerprints['morgan'] = np.array(morgan_fp)
            
            # MACCS keys
            maccs_fp = rdMolDescriptors.GetMACCSKeysFingerprint(mol)
            fingerprints['maccs'] = np.array(maccs_fp)
            
            # Topological fingerprint
            topo_fp = FingerprintMols.FingerprintMol(mol)
            # Convert to bit vector
            topo_bits = np.zeros(self.fingerprint_bits)
            for bit in topo_fp.GetOnBits():
                topo_bits[bit % self.fingerprint_bits] = 1
            fingerprints['topological'] = topo_bits
            
            # Atom pair fingerprint
            ap_fp = Pairs.GetAtomPairFingerprint(mol)
            # Convert to bit vector
            ap_bits = np.zeros(self.fingerprint_bits)
            for bit in ap_fp.GetNonzeroElements():
                ap_bits[bit % self.fingerprint_bits] = 1
            fingerprints['atom_pair'] = ap_bits
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate fingerprints: {e}")
            # Return empty fingerprints
            fingerprints['morgan'] = np.zeros(self.fingerprint_bits)
            fingerprints['maccs'] = np.zeros(167)  # MACCS keys are 167 bits
            fingerprints['topological'] = np.zeros(self.fingerprint_bits)
            fingerprints['atom_pair'] = np.zeros(self.fingerprint_bits)
        
        return fingerprints
    
    def calculate_charge_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate charge-related descriptors.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of charge descriptors
        """
        descriptors = {}
        
        try:
            # Calculate partial charges using Gasteiger method
            AllChem.ComputeGasteigerCharges(mol)
            
            charges = []
            for atom in mol.GetAtoms():
                charge = atom.GetDoubleProp('_GasteigerCharge')
                if not np.isnan(charge):
                    charges.append(charge)
            
            if charges:
                descriptors['charge_mean'] = np.mean(charges)
                descriptors['charge_std'] = np.std(charges)
                descriptors['charge_min'] = np.min(charges)
                descriptors['charge_max'] = np.max(charges)
                descriptors['charge_range'] = np.max(charges) - np.min(charges)
                descriptors['positive_charge_sum'] = np.sum([c for c in charges if c > 0])
                descriptors['negative_charge_sum'] = np.sum([c for c in charges if c < 0])
                descriptors['total_charge'] = np.sum(charges)
            else:
                # Fill with zeros if no valid charges
                for key in ['charge_mean', 'charge_std', 'charge_min', 'charge_max',
                           'charge_range', 'positive_charge_sum', 'negative_charge_sum', 'total_charge']:
                    descriptors[key] = 0.0
                    
        except Exception as e:
            self.logger.warning(f"Failed to calculate charge descriptors: {e}")
            # Fill with NaN for failed calculations
            for key in ['charge_mean', 'charge_std', 'charge_min', 'charge_max',
                       'charge_range', 'positive_charge_sum', 'negative_charge_sum', 'total_charge']:
                descriptors[key] = np.nan
        
        return descriptors
    
    def calculate_atom_type_counts(self, mol: Chem.Mol) -> Dict[str, int]:
        """
        Calculate atom type counts.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of atom type counts
        """
        descriptors = {}
        
        try:
            # Initialize common atom types
            atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
            
            for atom_type in atom_types:
                descriptors[f'count_{atom_type}'] = 0
            
            # Count atoms
            for atom in mol.GetAtoms():
                symbol = atom.GetSymbol()
                if symbol in atom_types:
                    descriptors[f'count_{symbol}'] += 1
            
            # Additional atom counts
            descriptors['count_aromatic'] = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            descriptors['count_sp2'] = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2)
            descriptors['count_sp3'] = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate atom type counts: {e}")
            # Fill with zeros
            atom_types = ['C', 'N', 'O', 'S', 'P', 'F', 'Cl', 'Br', 'I']
            for atom_type in atom_types:
                descriptors[f'count_{atom_type}'] = 0
            descriptors['count_aromatic'] = 0
            descriptors['count_sp2'] = 0
            descriptors['count_sp3'] = 0
        
        return descriptors
    
    def calculate_pka_relevant_descriptors(self, mol: Chem.Mol) -> Dict[str, float]:
        """
        Calculate descriptors specifically relevant for pKa prediction.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Dictionary of pKa-relevant descriptors
        """
        descriptors = {}
        
        try:
            # Ionizable groups
            descriptors['acidic_groups'] = 0
            descriptors['basic_groups'] = 0
            descriptors['carboxylic_acids'] = 0
            descriptors['amines'] = 0
            descriptors['phenols'] = 0
            descriptors['thiols'] = 0
            
            # SMARTS patterns for ionizable groups
            carboxylic_acid_pattern = Chem.MolFromSmarts('[CX3](=O)[OX2H1]')
            amine_pattern = Chem.MolFromSmarts('[NX3;H2,H1;!$(NC=O)]')
            phenol_pattern = Chem.MolFromSmarts('[OH1][c]')
            thiol_pattern = Chem.MolFromSmarts('[SH1]')
            
            # Count matches
            if carboxylic_acid_pattern:
                descriptors['carboxylic_acids'] = len(mol.GetSubstructMatches(carboxylic_acid_pattern))
                descriptors['acidic_groups'] += descriptors['carboxylic_acids']
            
            if amine_pattern:
                descriptors['amines'] = len(mol.GetSubstructMatches(amine_pattern))
                descriptors['basic_groups'] += descriptors['amines']
            
            if phenol_pattern:
                descriptors['phenols'] = len(mol.GetSubstructMatches(phenol_pattern))
                descriptors['acidic_groups'] += descriptors['phenols']
            
            if thiol_pattern:
                descriptors['thiols'] = len(mol.GetSubstructMatches(thiol_pattern))
                descriptors['acidic_groups'] += descriptors['thiols']
            
            # Electronic effects
            descriptors['electron_withdrawing_groups'] = 0
            descriptors['electron_donating_groups'] = 0
            
            # SMARTS for electron-withdrawing groups
            ewg_patterns = [
                '[CX3](=O)',  # Carbonyl
                '[NX3+]',     # Nitro
                '[SX6](=O)(=O)', # Sulfonyl
                '[F,Cl,Br,I]'    # Halogens
            ]
            
            for pattern_smarts in ewg_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern:
                    descriptors['electron_withdrawing_groups'] += len(mol.GetSubstructMatches(pattern))
            
            # SMARTS for electron-donating groups
            edg_patterns = [
                '[CH3]',      # Methyl
                '[OH1]',      # Hydroxyl
                '[NH2]'       # Amino
            ]
            
            for pattern_smarts in edg_patterns:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern:
                    descriptors['electron_donating_groups'] += len(mol.GetSubstructMatches(pattern))
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate pKa-relevant descriptors: {e}")
            # Fill with zeros
            pka_keys = ['acidic_groups', 'basic_groups', 'carboxylic_acids', 'amines', 
                       'phenols', 'thiols', 'electron_withdrawing_groups', 'electron_donating_groups']
            for key in pka_keys:
                descriptors[key] = 0
        
        return descriptors
    
    def calculate_all_features(self, mol: Chem.Mol, 
                              ensemble_descriptors: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate all molecular features.
        
        Args:
            mol: RDKit molecule object
            ensemble_descriptors: Optional 3D ensemble descriptors
            
        Returns:
            Dictionary containing all calculated features
        """
        features = {}
        
        try:
            # Basic descriptors
            basic_desc = self.calculate_basic_descriptors(mol)
            features.update(basic_desc)
            
            # Fragment descriptors
            fragment_desc = self.calculate_fragment_descriptors(mol)
            features.update(fragment_desc)
            
            # Fingerprints
            fingerprints = self.calculate_fingerprints(mol)
            features.update(fingerprints)
            
            # Charge descriptors
            charge_desc = self.calculate_charge_descriptors(mol)
            features.update(charge_desc)
            
            # Atom type counts
            atom_counts = self.calculate_atom_type_counts(mol)
            features.update(atom_counts)
            
            # pKa-relevant descriptors
            pka_desc = self.calculate_pka_relevant_descriptors(mol)
            features.update(pka_desc)
            
            # Add 3D ensemble descriptors if provided
            if ensemble_descriptors:
                features.update(ensemble_descriptors)
            
        except Exception as e:
            self.logger.error(f"Failed to calculate molecular features: {e}")
            raise
        
        return features
    
    def features_to_dataframe(self, molecules: List[Chem.Mol], 
                             ensemble_descriptors_list: Optional[List[Dict[str, float]]] = None,
                             smiles_list: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Convert molecular features to pandas DataFrame.
        
        Args:
            molecules: List of RDKit molecules
            ensemble_descriptors_list: Optional list of ensemble descriptors
            smiles_list: Optional list of SMILES strings
            
        Returns:
            DataFrame containing all molecular features
        """
        feature_list = []
        
        for i, mol in enumerate(molecules):
            try:
                # Get ensemble descriptors if provided
                ensemble_desc = None
                if ensemble_descriptors_list and i < len(ensemble_descriptors_list):
                    ensemble_desc = ensemble_descriptors_list[i]
                
                # Calculate features
                features = self.calculate_all_features(mol, ensemble_desc)
                
                # Add metadata
                features['molecule_id'] = i
                if smiles_list and i < len(smiles_list):
                    features['smiles'] = smiles_list[i]
                
                feature_list.append(features)
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate features for molecule {i}: {e}")
                continue
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_list)
        
        # Separate fingerprints from other features
        fingerprint_columns = ['morgan', 'maccs', 'topological', 'atom_pair']
        other_columns = [col for col in df.columns if col not in fingerprint_columns]
        
        # Create separate DataFrames for fingerprints
        fingerprint_dfs = {}
        for fp_type in fingerprint_columns:
            if fp_type in df.columns:
                fp_array = np.vstack(df[fp_type].values)
                fp_df = pd.DataFrame(fp_array, columns=[f'{fp_type}_{i}' for i in range(fp_array.shape[1])])
                fingerprint_dfs[fp_type] = fp_df
        
        # Combine other features
        main_df = df[other_columns]
        
        # Combine all features
        final_df = main_df.copy()
        for fp_type, fp_df in fingerprint_dfs.items():
            final_df = pd.concat([final_df, fp_df], axis=1)
        
        return final_df
    
    def save_features(self, df: pd.DataFrame, output_path: Union[str, Path]) -> None:
        """
        Save features to CSV file.
        
        Args:
            df: DataFrame containing features
            output_path: Path to save CSV file
        """
        output_path = Path(output_path)
        df.to_csv(output_path, index=False)
        self.logger.info(f"Saved features to {output_path}")


def calculate_molecular_features(molecules: List[Chem.Mol],
                                ensemble_descriptors_list: Optional[List[Dict[str, float]]] = None,
                                smiles_list: Optional[List[str]] = None,
                                **kwargs) -> pd.DataFrame:
    """
    Convenience function to calculate molecular features.
    
    Args:
        molecules: List of RDKit molecules
        ensemble_descriptors_list: Optional list of ensemble descriptors
        smiles_list: Optional list of SMILES strings
        **kwargs: Additional arguments for FeatureEngineering
        
    Returns:
        DataFrame containing all molecular features
    """
    feature_eng = FeatureEngineering(**kwargs)
    return feature_eng.features_to_dataframe(molecules, ensemble_descriptors_list, smiles_list)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample molecules
    smiles_list = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"  # Ibuprofen
    ]
    
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    molecules = [mol for mol in molecules if mol is not None]
    
    # Calculate features
    features_df = calculate_molecular_features(molecules, smiles_list=smiles_list)
    
    print(f"Calculated features for {len(features_df)} molecules")
    print(f"Feature columns: {len(features_df.columns)}")
    print(f"Sample feature names: {list(features_df.columns)[:10]}")
    
    # Show basic statistics
    print("\nBasic feature statistics:")
    print(features_df.describe())