"""
Input processing module for SMILES and SDF standardization.
"""

import os
from typing import List, Union, Optional, Dict, Any
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
import pandas as pd


class MoleculeProcessor:
    """Process and standardize molecular inputs from SMILES or SDF files."""
    
    def __init__(self, remove_salts: bool = True, neutralize: bool = True):
        """
        Initialize the molecule processor.
        
        Args:
            remove_salts: Whether to remove salts from molecules
            neutralize: Whether to neutralize molecules
        """
        self.remove_salts = remove_salts
        self.neutralize = neutralize
        
        # Initialize standardization components
        self.normalizer = rdMolStandardize.Normalizer()
        self.largest_frag_chooser = rdMolStandardize.LargestFragmentChooser()
        self.uncharger = rdMolStandardize.Uncharger()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def standardize_molecule(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Standardize a molecule using RDKit standardization tools.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Standardized molecule or None if standardization fails
        """
        if mol is None:
            return None
            
        try:
            # Normalize the molecule
            mol = self.normalizer.normalize(mol)
            
            # Remove salts by choosing largest fragment
            if self.remove_salts:
                mol = self.largest_frag_chooser.choose(mol)
            
            # Neutralize the molecule
            if self.neutralize:
                mol = self.uncharger.uncharge(mol)
            
            # Sanitize the molecule
            Chem.SanitizeMol(mol)
            
            return mol
            
        except Exception as e:
            self.logger.warning(f"Failed to standardize molecule: {e}")
            return None
    
    def process_smiles(self, smiles: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Process SMILES string(s) and return standardized molecules.
        
        Args:
            smiles: Single SMILES string or list of SMILES strings
            
        Returns:
            Dictionary containing processed molecules and metadata
        """
        if isinstance(smiles, str):
            smiles = [smiles]
        
        results = {
            'molecules': [],
            'smiles': [],
            'failed_indices': [],
            'processing_stats': {}
        }
        
        for i, smi in enumerate(smiles):
            try:
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    results['failed_indices'].append(i)
                    continue
                
                # Standardize the molecule
                standardized_mol = self.standardize_molecule(mol)
                
                if standardized_mol is None:
                    results['failed_indices'].append(i)
                    continue
                
                # Store results
                results['molecules'].append(standardized_mol)
                results['smiles'].append(Chem.MolToSmiles(standardized_mol))
                
            except Exception as e:
                self.logger.warning(f"Failed to process SMILES {i}: {smi}, error: {e}")
                results['failed_indices'].append(i)
        
        # Calculate processing stats
        total_input = len(smiles)
        successful = len(results['molecules'])
        failed = len(results['failed_indices'])
        
        results['processing_stats'] = {
            'total_input': total_input,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_input if total_input > 0 else 0.0
        }
        
        return results
    
    def process_sdf(self, sdf_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process SDF file and return standardized molecules.
        
        Args:
            sdf_path: Path to SDF file
            
        Returns:
            Dictionary containing processed molecules and metadata
        """
        sdf_path = Path(sdf_path)
        
        if not sdf_path.exists():
            raise FileNotFoundError(f"SDF file not found: {sdf_path}")
        
        results = {
            'molecules': [],
            'smiles': [],
            'failed_indices': [],
            'processing_stats': {},
            'mol_properties': []
        }
        
        try:
            supplier = Chem.SDMolSupplier(str(sdf_path))
            
            for i, mol in enumerate(supplier):
                if mol is None:
                    results['failed_indices'].append(i)
                    continue
                
                # Standardize the molecule
                standardized_mol = self.standardize_molecule(mol)
                
                if standardized_mol is None:
                    results['failed_indices'].append(i)
                    continue
                
                # Store results
                results['molecules'].append(standardized_mol)
                results['smiles'].append(Chem.MolToSmiles(standardized_mol))
                
                # Extract properties from original molecule
                props = mol.GetPropsAsDict()
                results['mol_properties'].append(props)
                
        except Exception as e:
            self.logger.error(f"Failed to process SDF file {sdf_path}: {e}")
            raise
        
        # Calculate processing stats
        total_input = len(results['molecules']) + len(results['failed_indices'])
        successful = len(results['molecules'])
        failed = len(results['failed_indices'])
        
        results['processing_stats'] = {
            'total_input': total_input,
            'successful': successful,
            'failed': failed,
            'success_rate': successful / total_input if total_input > 0 else 0.0
        }
        
        return results
    
    def validate_molecules(self, molecules: List[Chem.Mol]) -> Dict[str, Any]:
        """
        Validate a list of molecules and return validation results.
        
        Args:
            molecules: List of RDKit molecule objects
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid_molecules': [],
            'invalid_indices': [],
            'validation_stats': {}
        }
        
        for i, mol in enumerate(molecules):
            try:
                # Basic validation checks
                if mol is None:
                    validation_results['invalid_indices'].append(i)
                    continue
                
                # Check if molecule has atoms
                if mol.GetNumAtoms() == 0:
                    validation_results['invalid_indices'].append(i)
                    continue
                
                # Check if molecule can be sanitized
                Chem.SanitizeMol(mol)
                
                # Check for reasonable molecular weight
                mw = Descriptors.MolWt(mol)
                if mw < 50 or mw > 2000:  # Reasonable range for drug-like molecules
                    self.logger.warning(f"Molecule {i} has unusual molecular weight: {mw}")
                
                validation_results['valid_molecules'].append(mol)
                
            except Exception as e:
                self.logger.warning(f"Molecule {i} failed validation: {e}")
                validation_results['invalid_indices'].append(i)
        
        # Calculate validation stats
        total_input = len(molecules)
        valid = len(validation_results['valid_molecules'])
        invalid = len(validation_results['invalid_indices'])
        
        validation_results['validation_stats'] = {
            'total_input': total_input,
            'valid': valid,
            'invalid': invalid,
            'validity_rate': valid / total_input if total_input > 0 else 0.0
        }
        
        return validation_results
    
    def save_processed_molecules(self, molecules: List[Chem.Mol], 
                                output_path: Union[str, Path],
                                format: str = 'sdf') -> None:
        """
        Save processed molecules to file.
        
        Args:
            molecules: List of processed molecules
            output_path: Path to save file
            format: Output format ('sdf' or 'smiles')
        """
        output_path = Path(output_path)
        
        if format.lower() == 'sdf':
            writer = Chem.SDWriter(str(output_path))
            for mol in molecules:
                if mol is not None:
                    writer.write(mol)
            writer.close()
            
        elif format.lower() == 'smiles':
            with open(output_path, 'w') as f:
                for mol in molecules:
                    if mol is not None:
                        f.write(f"{Chem.MolToSmiles(mol)}\n")
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Saved {len(molecules)} molecules to {output_path}")


def process_input_file(input_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None,
                      remove_salts: bool = True,
                      neutralize: bool = True) -> Dict[str, Any]:
    """
    Convenience function to process input files.
    
    Args:
        input_path: Path to input file (SMILES or SDF)
        output_path: Optional path to save processed molecules
        remove_salts: Whether to remove salts
        neutralize: Whether to neutralize molecules
        
    Returns:
        Dictionary containing processing results
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    processor = MoleculeProcessor(remove_salts=remove_salts, neutralize=neutralize)
    
    # Determine file type and process accordingly
    if input_path.suffix.lower() == '.sdf':
        results = processor.process_sdf(input_path)
    elif input_path.suffix.lower() in ['.smi', '.smiles', '.txt']:
        # Read SMILES from file
        with open(input_path, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        results = processor.process_smiles(smiles_list)
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Save processed molecules if output path provided
    if output_path and results['molecules']:
        processor.save_processed_molecules(results['molecules'], output_path)
    
    return results


# Wrapper function for backend compatibility
def process_input(input_data: Union[str, Path], input_type: str = "smiles") -> Dict[str, Any]:
    """
    Wrapper function to process molecular input for backend compatibility.
    
    Args:
        input_data: SMILES string or file path
        input_type: "smiles" or "sdf"
        
    Returns:
        Dictionary with molecule data
    """
    processor = MoleculeProcessor()
    
    if input_type == "smiles":
        # Process single SMILES
        mol = Chem.MolFromSmiles(str(input_data))
        if mol is None:
            raise ValueError(f"Invalid SMILES: {input_data}")
        
        mol = processor.standardize_molecule(mol)
        if mol is None:
            raise ValueError(f"Failed to standardize molecule: {input_data}")
        
        return {
            'mol': mol,
            'smiles': Chem.MolToSmiles(mol),
            'name': 'Unknown',
            'molecular_weight': Descriptors.MolWt(mol)
        }
    
    elif input_type == "sdf":
        # Process SDF file
        results = processor.process_sdf_file(Path(input_data))
        if not results['molecules'] or len(results['molecules']) == 0:
            raise ValueError(f"No valid molecules found in file: {input_data}")
        
        # Return first molecule
        mol = results['molecules'][0]
        return {
            'mol': mol,
            'smiles': Chem.MolToSmiles(mol),
            'name': results.get('mol_names', ['Unknown'])[0],
            'molecular_weight': Descriptors.MolWt(mol)
        }
    
    else:
        raise ValueError(f"Unsupported input type: {input_type}")


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample SMILES
    test_smiles = [
        "CCO",  # Ethanol
        "CC(=O)O",  # Acetic acid
        "c1ccccc1",  # Benzene
        "invalid_smiles"  # This should fail
    ]
    
    processor = MoleculeProcessor()
    results = processor.process_smiles(test_smiles)
    
    print(f"Processing results: {results['processing_stats']}")
    print(f"Processed SMILES: {results['smiles']}")
    print(f"Failed indices: {results['failed_indices']}")