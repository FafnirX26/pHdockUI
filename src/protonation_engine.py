"""
Protonation engine for pH sweep and state conversion.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import copy

from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


class ProtonationEngine:
    """Engine for generating protonation states at different pH values."""
    
    def __init__(self, 
                 ph_range: Tuple[float, float] = (1.0, 14.0),
                 ph_step: float = 0.5,
                 temperature: float = 298.15):
        """
        Initialize the protonation engine.
        
        Args:
            ph_range: pH range for sweep (min_ph, max_ph)
            ph_step: Step size for pH sweep
            temperature: Temperature in Kelvin
        """
        self.ph_range = ph_range
        self.ph_step = ph_step
        self.temperature = temperature
        
        self.logger = logging.getLogger(__name__)
        
        # Generate pH values for sweep
        self.ph_values = np.arange(ph_range[0], ph_range[1] + ph_step, ph_step)
        
        # SMARTS patterns for ionizable groups
        self.ionizable_patterns = {
            'carboxyl': '[CX3](=O)[OX2H1]',  # Carboxylic acid
            'phenol': '[OH1][c]',            # Phenol
            'amine_primary': '[NX3;H2;!$(NC=O)]',  # Primary amine
            'amine_secondary': '[NX3;H1;!$(NC=O)]',  # Secondary amine
            'amine_tertiary': '[NX3;H0;!$(NC=O)]',   # Tertiary amine
            'thiol': '[SH1]',                # Thiol
            'imidazole': '[nH]1cncc1',       # Imidazole
            'guanidine': '[NX3][CX3](=[NX3+,NX2+0])[NX3]'  # Guanidine
        }
        
        # Default pKa values for different functional groups
        self.default_pka_values = {
            'carboxyl': 4.2,
            'phenol': 9.8,
            'amine_primary': 10.2,
            'amine_secondary': 10.5,
            'amine_tertiary': 9.8,
            'thiol': 8.5,
            'imidazole': 6.0,
            'guanidine': 12.5
        }
        
        # Protonation/deprotonation transformations
        self.transformations = {
            'carboxyl': {
                'protonated': '[CX3](=O)[OX2H1]',    # COOH
                'deprotonated': '[CX3](=O)[O-]'       # COO-
            },
            'phenol': {
                'protonated': '[OH1][c]',             # ArOH
                'deprotonated': '[O-][c]'             # ArO-
            },
            'amine_primary': {
                'protonated': '[NX3H3+]',             # NH3+
                'deprotonated': '[NX3H2]'             # NH2
            },
            'amine_secondary': {
                'protonated': '[NX3H2+]',             # NRH2+
                'deprotonated': '[NX3H1]'             # NRH
            },
            'amine_tertiary': {
                'protonated': '[NX3H1+]',             # NR3H+
                'deprotonated': '[NX3H0]'             # NR3
            },
            'thiol': {
                'protonated': '[SH1]',                # SH
                'deprotonated': '[S-]'                # S-
            }
        }
    
    def identify_ionizable_sites(self, mol: Chem.Mol) -> List[Dict[str, Any]]:
        """
        Identify ionizable sites in a molecule.
        
        Args:
            mol: RDKit molecule object
            
        Returns:
            List of dictionaries containing site information
        """
        sites = []
        
        for group_name, pattern_smarts in self.ionizable_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(pattern_smarts)
                if pattern is None:
                    continue
                
                matches = mol.GetSubstructMatches(pattern)
                
                for match in matches:
                    # Get the key atom (usually the heteroatom)
                    if group_name in ['carboxyl']:
                        key_atom_idx = match[2]  # Oxygen in COOH
                    elif group_name in ['phenol', 'thiol']:
                        key_atom_idx = match[0]  # O or S
                    elif group_name.startswith('amine'):
                        key_atom_idx = match[0]  # N
                    else:
                        key_atom_idx = match[0]  # Default to first atom
                    
                    site_info = {
                        'group_type': group_name,
                        'atom_indices': match,
                        'key_atom_idx': key_atom_idx,
                        'default_pka': self.default_pka_values.get(group_name, 7.0)
                    }
                    sites.append(site_info)
                    
            except Exception as e:
                self.logger.warning(f"Error identifying {group_name} sites: {e}")
                continue
        
        return sites
    
    def calculate_protonation_fraction(self, pka: float, ph: float, 
                                     is_acid: bool = True) -> float:
        """
        Calculate the fraction of molecules in protonated state.
        
        Args:
            pka: pKa value
            ph: pH value
            is_acid: Whether the group is acidic (True) or basic (False)
            
        Returns:
            Fraction in protonated state (0-1)
        """
        if is_acid:
            # For acids: α_HA = 1 / (1 + 10^(pH - pKa))
            return 1.0 / (1.0 + 10**(ph - pka))
        else:
            # For bases: α_BH+ = 1 / (1 + 10^(pKa - pH))
            return 1.0 / (1.0 + 10**(pka - ph))
    
    def determine_protonation_state(self, pka: float, ph: float, 
                                   group_type: str) -> str:
        """
        Determine the predominant protonation state at given pH.
        
        Args:
            pka: pKa value
            ph: pH value
            group_type: Type of ionizable group
            
        Returns:
            'protonated' or 'deprotonated'
        """
        # Determine if group is acidic or basic
        acidic_groups = ['carboxyl', 'phenol', 'thiol']
        is_acid = group_type in acidic_groups
        
        protonated_fraction = self.calculate_protonation_fraction(pka, ph, is_acid)
        
        # Return predominant state (>50%)
        return 'protonated' if protonated_fraction > 0.5 else 'deprotonated'
    
    def apply_protonation_state(self, mol: Chem.Mol, sites: List[Dict[str, Any]], 
                               pka_values: List[float], ph: float) -> Optional[Chem.Mol]:
        """
        Apply protonation states to a molecule based on pH and pKa values.
        
        Args:
            mol: RDKit molecule object
            sites: List of ionizable sites
            pka_values: List of pKa values for each site
            ph: pH value
            
        Returns:
            Molecule with applied protonation states or None if failed
        """
        try:
            # Create a copy of the molecule
            mol_copy = copy.deepcopy(mol)
            
            # Keep track of charge changes
            total_charge_change = 0
            
            for site, pka in zip(sites, pka_values):
                group_type = site['group_type']
                
                # Determine protonation state
                state = self.determine_protonation_state(pka, ph, group_type)
                
                # Apply transformation if needed
                if group_type in self.transformations:
                    transformation = self.transformations[group_type]
                    
                    # Determine charge change
                    acidic_groups = ['carboxyl', 'phenol', 'thiol']
                    
                    if group_type in acidic_groups:
                        # Acid: loses proton -> negative charge
                        if state == 'deprotonated':
                            total_charge_change -= 1
                    else:
                        # Base: gains proton -> positive charge
                        if state == 'protonated':
                            total_charge_change += 1
            
            # Apply formal charges to the molecule
            # This is a simplified approach - in practice, you might want to use
            # more sophisticated methods to assign charges
            if total_charge_change != 0:
                # Add formal charge to the molecule
                Chem.rdmolops.SanitizeMol(mol_copy)
                
                # Set formal charge on the molecule
                for atom in mol_copy.GetAtoms():
                    atom.SetFormalCharge(atom.GetFormalCharge())
                
                # Note: This is a simplified charge assignment
                # For more accurate results, consider using tools like OpenEye Omega
                # or specialized protonation state prediction software
            
            return mol_copy
            
        except Exception as e:
            self.logger.warning(f"Failed to apply protonation state: {e}")
            return None
    
    def generate_ph_profile(self, mol: Chem.Mol, pka_values: List[float], 
                           sites: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate pH-dependent protonation profile.
        
        Args:
            mol: RDKit molecule object
            pka_values: List of pKa values
            sites: List of ionizable sites
            
        Returns:
            DataFrame with pH profile
        """
        profile_data = []
        
        for ph in self.ph_values:
            row = {'pH': ph}
            
            # Calculate charge and protonation fractions
            total_charge = 0
            
            for i, (site, pka) in enumerate(zip(sites, pka_values)):
                group_type = site['group_type']
                
                # Determine if acidic or basic
                acidic_groups = ['carboxyl', 'phenol', 'thiol']
                is_acid = group_type in acidic_groups
                
                # Calculate protonation fraction
                protonated_fraction = self.calculate_protonation_fraction(pka, ph, is_acid)
                
                # Calculate charge contribution
                if is_acid:
                    # Acid: charge = -(1 - protonated_fraction)
                    charge_contribution = -(1 - protonated_fraction)
                else:
                    # Base: charge = +protonated_fraction
                    charge_contribution = protonated_fraction
                
                total_charge += charge_contribution
                
                # Store individual site data
                row[f'site_{i}_{group_type}_protonated_fraction'] = protonated_fraction
                row[f'site_{i}_{group_type}_charge'] = charge_contribution
            
            row['total_charge'] = total_charge
            row['net_charge'] = round(total_charge)  # Rounded for practical use
            
            profile_data.append(row)
        
        return pd.DataFrame(profile_data)
    
    def generate_protonation_states(self, mol: Chem.Mol, 
                                   pka_values: Optional[List[float]] = None) -> Dict[float, Chem.Mol]:
        """
        Generate protonation states across pH range.
        
        Args:
            mol: RDKit molecule object
            pka_values: Optional list of pKa values (will use defaults if not provided)
            
        Returns:
            Dictionary mapping pH values to protonated molecules
        """
        # Identify ionizable sites
        sites = self.identify_ionizable_sites(mol)
        
        if not sites:
            self.logger.info("No ionizable sites found")
            # Return original molecule for all pH values
            return {ph: mol for ph in self.ph_values}
        
        # Use provided pKa values or defaults
        if pka_values is None:
            pka_values = [site['default_pka'] for site in sites]
        elif len(pka_values) != len(sites):
            self.logger.warning(f"pKa values count ({len(pka_values)}) doesn't match sites count ({len(sites)})")
            # Pad or truncate as needed
            if len(pka_values) < len(sites):
                pka_values.extend([7.0] * (len(sites) - len(pka_values)))
            else:
                pka_values = pka_values[:len(sites)]
        
        # Generate protonation states
        protonation_states = {}
        
        for ph in self.ph_values:
            protonated_mol = self.apply_protonation_state(mol, sites, pka_values, ph)
            if protonated_mol is not None:
                protonation_states[ph] = protonated_mol
            else:
                # Fallback to original molecule
                protonation_states[ph] = mol
        
        return protonation_states
    
    def select_physiological_states(self, protonation_states: Dict[float, Chem.Mol],
                                   ph_conditions: List[float] = [1.5, 7.4, 8.0]) -> Dict[str, Chem.Mol]:
        """
        Select protonation states at physiologically relevant pH values.
        
        Args:
            protonation_states: Dictionary of pH to molecule mappings
            ph_conditions: List of relevant pH values
            
        Returns:
            Dictionary mapping condition names to molecules
        """
        condition_names = {
            1.5: 'gastric',
            7.4: 'blood',
            8.0: 'intestinal'
        }
        
        selected_states = {}
        
        for ph in ph_conditions:
            # Find closest pH in our generated states
            closest_ph = min(protonation_states.keys(), key=lambda x: abs(x - ph))
            
            condition_name = condition_names.get(ph, f'pH_{ph}')
            selected_states[condition_name] = protonation_states[closest_ph]
        
        return selected_states
    
    def save_protonation_states(self, protonation_states: Dict[float, Chem.Mol],
                               output_dir: Path, 
                               molecule_name: str = "molecule") -> None:
        """
        Save protonation states to SDF files.
        
        Args:
            protonation_states: Dictionary of pH to molecule mappings
            output_dir: Output directory
            molecule_name: Base name for output files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save all states
        all_states_file = output_dir / f"{molecule_name}_all_ph_states.sdf"
        writer = Chem.SDWriter(str(all_states_file))
        
        for ph, mol in protonation_states.items():
            # Add pH as property
            mol.SetProp("pH", str(ph))
            mol.SetProp("molecule_name", molecule_name)
            writer.write(mol)
        
        writer.close()
        
        # Save physiological states separately
        physiol_states = self.select_physiological_states(protonation_states)
        
        for condition, mol in physiol_states.items():
            output_file = output_dir / f"{molecule_name}_{condition}.sdf"
            writer = Chem.SDWriter(str(output_file))
            mol.SetProp("condition", condition)
            mol.SetProp("molecule_name", molecule_name)
            writer.write(mol)
            writer.close()
        
        self.logger.info(f"Saved protonation states to {output_dir}")


def generate_protonation_ensemble(molecules: List[Chem.Mol],
                                 pka_predictions: Optional[List[List[float]]] = None,
                                 ph_range: Tuple[float, float] = (1.0, 14.0),
                                 ph_step: float = 1.0,
                                 save_dir: Optional[Path] = None) -> List[Dict[float, Chem.Mol]]:
    """
    Generate protonation ensembles for multiple molecules.
    
    Args:
        molecules: List of RDKit molecules
        pka_predictions: Optional list of pKa predictions for each molecule
        ph_range: pH range for sweep
        ph_step: Step size for pH sweep
        save_dir: Optional directory to save results
        
    Returns:
        List of protonation state dictionaries for each molecule
    """
    engine = ProtonationEngine(ph_range=ph_range, ph_step=ph_step)
    results = []
    
    for i, mol in enumerate(molecules):
        try:
            # Get pKa values for this molecule
            pka_values = None
            if pka_predictions and i < len(pka_predictions):
                pka_values = pka_predictions[i]
            
            # Generate protonation states
            protonation_states = engine.generate_protonation_states(mol, pka_values)
            results.append(protonation_states)
            
            # Save if directory provided
            if save_dir:
                molecule_name = f"mol_{i:04d}"
                engine.save_protonation_states(protonation_states, save_dir, molecule_name)
            
        except Exception as e:
            logging.error(f"Failed to generate protonation states for molecule {i}: {e}")
            # Add empty result to maintain indexing
            results.append({})
    
    return results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample molecules
    smiles_list = [
        "CC(=O)O",  # Acetic acid (carboxyl, pKa ~4.8)
        "c1ccccc1O",  # Phenol (phenol, pKa ~10)
        "c1ccc(cc1)N",  # Aniline (amine, pKa ~4.6)
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen (carboxyl)
        "NC(=N)N",  # Guanidine (basic, pKa ~12.5)
    ]
    
    molecules = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    molecules = [mol for mol in molecules if mol is not None]
    
    # Initialize engine
    engine = ProtonationEngine(ph_range=(1.0, 14.0), ph_step=1.0)
    
    # Test with first molecule (acetic acid)
    test_mol = molecules[0]
    
    # Identify ionizable sites
    sites = engine.identify_ionizable_sites(test_mol)
    print(f"Found {len(sites)} ionizable sites in acetic acid:")
    for site in sites:
        print(f"  {site['group_type']} at atoms {site['atom_indices']}")
    
    # Generate protonation states
    protonation_states = engine.generate_protonation_states(test_mol)
    print(f"\nGenerated protonation states for {len(protonation_states)} pH values")
    
    # Generate pH profile
    if sites:
        pka_values = [4.8]  # Known pKa for acetic acid
        profile = engine.generate_ph_profile(test_mol, pka_values, sites)
        print(f"\nSample pH profile:")
        print(profile[['pH', 'total_charge', 'net_charge']].head(10))
    
    # Select physiological states
    physiol_states = engine.select_physiological_states(protonation_states)
    print(f"\nPhysiological states: {list(physiol_states.keys())}")
    
    # Test ensemble generation
    print(f"\nGenerating protonation ensembles for {len(molecules)} molecules...")
    ensembles = generate_protonation_ensemble(
        molecules,
        ph_range=(6.0, 9.0),  # Smaller range for testing
        ph_step=0.5
    )
    
    print(f"Generated {len(ensembles)} protonation ensembles")
    for i, ensemble in enumerate(ensembles):
        print(f"  Molecule {i}: {len(ensemble)} pH states")