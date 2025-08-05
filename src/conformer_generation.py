"""
Conformer generation module using RDKit ETKDG algorithm.
"""

import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import logging
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, Descriptors3D
from rdkit.Chem.rdMolAlign import AlignMol
from rdkit.Chem import rdDistGeom
import pandas as pd


class ConformerGenerator:
    """Generate and manage molecular conformers using RDKit ETKDG."""
    
    def __init__(self, 
                 num_conformers: int = 100,
                 max_attempts: int = 1000,
                 random_seed: int = 42,
                 prune_rms_threshold: float = 0.5,
                 use_random_coords: bool = True,
                 force_field: str = "MMFF94s"):
        """
        Initialize the conformer generator.
        
        Args:
            num_conformers: Number of conformers to generate
            max_attempts: Maximum attempts for conformer generation
            random_seed: Random seed for reproducibility
            prune_rms_threshold: RMS threshold for pruning similar conformers
            use_random_coords: Whether to use random coordinates as starting point
            force_field: Force field for optimization ("MMFF94s" or "UFF")
        """
        self.num_conformers = num_conformers
        self.max_attempts = max_attempts
        self.random_seed = random_seed
        self.prune_rms_threshold = prune_rms_threshold
        self.use_random_coords = use_random_coords
        self.force_field = force_field
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Validate force field
        if force_field not in ["MMFF94s", "UFF"]:
            raise ValueError(f"Unsupported force field: {force_field}")
    
    def generate_conformers(self, mol: Chem.Mol) -> Optional[Chem.Mol]:
        """
        Generate conformers for a molecule using ETKDG.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            Molecule with generated conformers or None if generation fails
        """
        if mol is None:
            return None
        
        try:
            # Add hydrogens
            mol = Chem.AddHs(mol)
            
            # Setup ETKDG parameters
            params = rdDistGeom.ETKDGv3()
            params.randomSeed = self.random_seed
            params.maxAttempts = self.max_attempts
            params.pruneRmsThresh = self.prune_rms_threshold
            params.useRandomCoords = self.use_random_coords
            params.numThreads = 0  # Use all available threads
            
            # Generate conformers
            conformer_ids = rdDistGeom.EmbedMultipleConfs(
                mol, 
                numConfs=self.num_conformers, 
                params=params
            )
            
            if len(conformer_ids) == 0:
                self.logger.warning("No conformers generated")
                return None
            
            # Optimize conformers with force field
            if self.force_field == "MMFF94s":
                # MMFF94s optimization
                mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
                if mmff_props is None:
                    self.logger.warning("MMFF94s properties not available, falling back to UFF")
                    self._optimize_with_uff(mol, conformer_ids)
                else:
                    self._optimize_with_mmff(mol, conformer_ids, mmff_props)
            else:
                # UFF optimization
                self._optimize_with_uff(mol, conformer_ids)
            
            self.logger.info(f"Generated {len(conformer_ids)} conformers")
            return mol
            
        except Exception as e:
            self.logger.error(f"Conformer generation failed: {e}")
            return None
    
    def _optimize_with_mmff(self, mol: Chem.Mol, conformer_ids: List[int], 
                           mmff_props: Any) -> None:
        """Optimize conformers using MMFF94s force field."""
        for conf_id in conformer_ids:
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                if ff is not None:
                    ff.Minimize()
            except Exception as e:
                self.logger.warning(f"MMFF optimization failed for conformer {conf_id}: {e}")
    
    def _optimize_with_uff(self, mol: Chem.Mol, conformer_ids: List[int]) -> None:
        """Optimize conformers using UFF force field."""
        for conf_id in conformer_ids:
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                if ff is not None:
                    ff.Minimize()
            except Exception as e:
                self.logger.warning(f"UFF optimization failed for conformer {conf_id}: {e}")
    
    def calculate_conformer_energies(self, mol: Chem.Mol) -> Dict[int, float]:
        """
        Calculate energies for all conformers.
        
        Args:
            mol: Molecule with conformers
            
        Returns:
            Dictionary mapping conformer ID to energy
        """
        energies = {}
        
        if self.force_field == "MMFF94s":
            mmff_props = AllChem.MMFFGetMoleculeProperties(mol)
            if mmff_props is not None:
                for conf in mol.GetConformers():
                    try:
                        ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf.GetId())
                        if ff is not None:
                            energies[conf.GetId()] = ff.CalcEnergy()
                    except Exception as e:
                        self.logger.warning(f"Energy calculation failed for conformer {conf.GetId()}: {e}")
            else:
                self.logger.warning("MMFF94s not available, using UFF for energy calculation")
                self._calculate_uff_energies(mol, energies)
        else:
            self._calculate_uff_energies(mol, energies)
        
        return energies
    
    def _calculate_uff_energies(self, mol: Chem.Mol, energies: Dict[int, float]) -> None:
        """Calculate UFF energies for conformers."""
        for conf in mol.GetConformers():
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf.GetId())
                if ff is not None:
                    energies[conf.GetId()] = ff.CalcEnergy()
            except Exception as e:
                self.logger.warning(f"UFF energy calculation failed for conformer {conf.GetId()}: {e}")
    
    def get_boltzmann_weights(self, energies: Dict[int, float], 
                             temperature: float = 298.15) -> Dict[int, float]:
        """
        Calculate Boltzmann weights for conformers.
        
        Args:
            energies: Dictionary of conformer energies
            temperature: Temperature in Kelvin
            
        Returns:
            Dictionary of Boltzmann weights
        """
        if not energies:
            return {}
        
        # Convert to numpy array for easier manipulation
        conf_ids = list(energies.keys())
        energy_values = np.array(list(energies.values()))
        
        # Convert kcal/mol to J/mol and calculate weights
        RT = 8.314 * temperature  # J/(mol·K)
        energy_values_j = energy_values * 4184  # kcal/mol to J/mol
        
        # Shift energies to prevent overflow
        min_energy = np.min(energy_values_j)
        exp_values = np.exp(-(energy_values_j - min_energy) / RT)
        
        # Normalize weights
        weights = exp_values / np.sum(exp_values)
        
        return dict(zip(conf_ids, weights))
    
    def select_diverse_conformers(self, mol: Chem.Mol, 
                                 num_select: int = 10,
                                 energy_weight: float = 0.5) -> List[int]:
        """
        Select diverse conformers based on energy and structural diversity.
        
        Args:
            mol: Molecule with conformers
            num_select: Number of conformers to select
            energy_weight: Weight for energy in selection (0-1)
            
        Returns:
            List of selected conformer IDs
        """
        if mol.GetNumConformers() == 0:
            return []
        
        # Calculate energies
        energies = self.calculate_conformer_energies(mol)
        
        if not energies:
            # If no energies available, select first few conformers
            return [conf.GetId() for conf in mol.GetConformers()][:num_select]
        
        # Sort by energy and select low-energy conformers
        sorted_conf_ids = sorted(energies.keys(), key=lambda x: energies[x])
        
        if len(sorted_conf_ids) <= num_select:
            return sorted_conf_ids
        
        # Select diverse conformers
        selected_ids = [sorted_conf_ids[0]]  # Always include lowest energy
        
        # Calculate RMS distances between conformers
        for i in range(1, len(sorted_conf_ids)):
            conf_id = sorted_conf_ids[i]
            
            # Check diversity with already selected conformers
            diverse = True
            for selected_id in selected_ids:
                try:
                    rms = AllChem.GetConformerRMS(mol, conf_id, selected_id)
                    if rms < self.prune_rms_threshold * 2:  # More lenient for selection
                        diverse = False
                        break
                except Exception:
                    continue
            
            if diverse:
                selected_ids.append(conf_id)
                
                if len(selected_ids) >= num_select:
                    break
        
        return selected_ids
    
    def calculate_3d_descriptors(self, mol: Chem.Mol, 
                                conformer_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Calculate 3D descriptors for conformers.
        
        Args:
            mol: Molecule with conformers
            conformer_ids: List of conformer IDs to calculate descriptors for
            
        Returns:
            Dictionary containing 3D descriptors
        """
        if conformer_ids is None:
            conformer_ids = [conf.GetId() for conf in mol.GetConformers()]
        
        descriptors = {
            'PMI1': [],
            'PMI2': [],
            'PMI3': [],
            'NPR1': [],
            'NPR2': [],
            'RadiusOfGyration': [],
            'InertialShapeFactor': [],
            'Eccentricity': [],
            'Asphericity': [],
            'SpherocityIndex': [],
            'conformer_ids': conformer_ids
        }
        
        for conf_id in conformer_ids:
            try:
                # Principal moments of inertia
                pmi1 = Descriptors3D.PMI1(mol, confId=conf_id)
                pmi2 = Descriptors3D.PMI2(mol, confId=conf_id)
                pmi3 = Descriptors3D.PMI3(mol, confId=conf_id)
                
                # Normalized principal moments ratios
                npr1 = Descriptors3D.NPR1(mol, confId=conf_id)
                npr2 = Descriptors3D.NPR2(mol, confId=conf_id)
                
                # Other 3D descriptors
                rog = Descriptors3D.RadiusOfGyration(mol, confId=conf_id)
                isf = Descriptors3D.InertialShapeFactor(mol, confId=conf_id)
                ecc = Descriptors3D.Eccentricity(mol, confId=conf_id)
                asp = Descriptors3D.Asphericity(mol, confId=conf_id)
                sph = Descriptors3D.SpherocityIndex(mol, confId=conf_id)
                
                # Store descriptors
                descriptors['PMI1'].append(pmi1)
                descriptors['PMI2'].append(pmi2)
                descriptors['PMI3'].append(pmi3)
                descriptors['NPR1'].append(npr1)
                descriptors['NPR2'].append(npr2)
                descriptors['RadiusOfGyration'].append(rog)
                descriptors['InertialShapeFactor'].append(isf)
                descriptors['Eccentricity'].append(ecc)
                descriptors['Asphericity'].append(asp)
                descriptors['SpherocityIndex'].append(sph)
                
            except Exception as e:
                self.logger.warning(f"3D descriptor calculation failed for conformer {conf_id}: {e}")
                # Add NaN values for failed calculations
                for key in descriptors:
                    if key != 'conformer_ids':
                        descriptors[key].append(np.nan)
        
        return descriptors
    
    def get_conformer_ensemble_descriptors(self, mol: Chem.Mol, 
                                          use_boltzmann_weights: bool = True,
                                          temperature: float = 298.15) -> Dict[str, float]:
        """
        Calculate ensemble-averaged 3D descriptors.
        
        Args:
            mol: Molecule with conformers
            use_boltzmann_weights: Whether to use Boltzmann weighting
            temperature: Temperature for Boltzmann weighting
            
        Returns:
            Dictionary of ensemble-averaged descriptors
        """
        if mol.GetNumConformers() == 0:
            return {}
        
        # Calculate 3D descriptors for all conformers
        descriptors = self.calculate_3d_descriptors(mol)
        
        if use_boltzmann_weights:
            # Calculate Boltzmann weights
            energies = self.calculate_conformer_energies(mol)
            weights = self.get_boltzmann_weights(energies, temperature)
            
            # Convert to numpy array aligned with conformer IDs
            weight_array = np.array([weights.get(conf_id, 1.0) for conf_id in descriptors['conformer_ids']])
            weight_array = weight_array / np.sum(weight_array)  # Normalize
        else:
            # Use equal weights
            weight_array = np.ones(len(descriptors['conformer_ids'])) / len(descriptors['conformer_ids'])
        
        # Calculate weighted averages
        ensemble_descriptors = {}
        for key, values in descriptors.items():
            if key != 'conformer_ids':
                values_array = np.array(values)
                # Handle NaN values
                valid_mask = ~np.isnan(values_array)
                if np.any(valid_mask):
                    weighted_values = values_array[valid_mask] * weight_array[valid_mask]
                    ensemble_descriptors[f'{key}_mean'] = np.sum(weighted_values) / np.sum(weight_array[valid_mask])
                    ensemble_descriptors[f'{key}_std'] = np.std(values_array[valid_mask])
                else:
                    ensemble_descriptors[f'{key}_mean'] = np.nan
                    ensemble_descriptors[f'{key}_std'] = np.nan
        
        return ensemble_descriptors
    
    def save_conformers(self, mol: Chem.Mol, output_path: Path, 
                       conformer_ids: Optional[List[int]] = None) -> None:
        """
        Save conformers to SDF file.
        
        Args:
            mol: Molecule with conformers
            output_path: Output file path
            conformer_ids: List of conformer IDs to save
        """
        if conformer_ids is None:
            conformer_ids = [conf.GetId() for conf in mol.GetConformers()]
        
        writer = Chem.SDWriter(str(output_path))
        
        for conf_id in conformer_ids:
            try:
                writer.write(mol, confId=conf_id)
            except Exception as e:
                self.logger.warning(f"Failed to write conformer {conf_id}: {e}")
        
        writer.close()
        self.logger.info(f"Saved {len(conformer_ids)} conformers to {output_path}")


def generate_conformer_ensemble(molecules: List[Chem.Mol], 
                               num_conformers: int = 100,
                               select_diverse: int = 10,
                               **kwargs) -> List[Tuple[Chem.Mol, Dict[str, float]]]:
    """
    Generate conformer ensembles for a list of molecules.
    
    Args:
        molecules: List of RDKit molecules
        num_conformers: Number of conformers to generate per molecule
        select_diverse: Number of diverse conformers to select
        **kwargs: Additional arguments for ConformerGenerator
        
    Returns:
        List of tuples containing (molecule_with_conformers, ensemble_descriptors)
    """
    generator = ConformerGenerator(num_conformers=num_conformers, **kwargs)
    results = []
    
    for i, mol in enumerate(molecules):
        try:
            # Generate conformers
            mol_with_conformers = generator.generate_conformers(mol)
            
            if mol_with_conformers is None:
                logging.warning(f"No conformers generated for molecule {i}")
                continue
            
            # Select diverse conformers
            if select_diverse > 0:
                selected_ids = generator.select_diverse_conformers(mol_with_conformers, select_diverse)
                
                # Create new molecule with only selected conformers
                new_mol = Chem.Mol(mol_with_conformers)
                new_mol.RemoveAllConformers()
                
                for conf_id in selected_ids:
                    new_mol.AddConformer(mol_with_conformers.GetConformer(conf_id), assignId=True)
                
                mol_with_conformers = new_mol
            
            # Calculate ensemble descriptors
            ensemble_descriptors = generator.get_conformer_ensemble_descriptors(mol_with_conformers)
            
            results.append((mol_with_conformers, ensemble_descriptors))
            
        except Exception as e:
            logging.error(f"Conformer generation failed for molecule {i}: {e}")
            continue
    
    return results


# Wrapper function for backend compatibility
def generate_conformers(mol: Chem.Mol, n_conformers: int = 10) -> List[str]:
    """
    Wrapper function to generate conformers for backend compatibility.
    
    Args:
        mol: RDKit molecule object
        n_conformers: Number of conformers to generate
        
    Returns:
        List of conformer identifiers (simplified for backend)
    """
    generator = ConformerGenerator()
    
    try:
        # Generate conformers using the class method
        mol_with_conformers = generator.generate_conformers(mol)
        
        if mol_with_conformers is None:
            return []
        
        # Return simple conformer identifiers
        num_conformers = mol_with_conformers.GetNumConformers()
        return [f"conformer_{i}" for i in range(min(num_conformers, n_conformers))]
    
    except Exception as e:
        logging.error(f"Failed to generate conformers: {e}")
        return []


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
    
    # Generate conformer ensembles
    results = generate_conformer_ensemble(molecules, num_conformers=50, select_diverse=5)
    
    print(f"Generated conformer ensembles for {len(results)} molecules")
    for i, (mol, descriptors) in enumerate(results):
        print(f"Molecule {i}: {mol.GetNumConformers()} conformers")
        print(f"  Sample descriptors: {list(descriptors.keys())[:3]}")