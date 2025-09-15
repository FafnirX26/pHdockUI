# Minimal protonation engine for Replicate
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from rdkit import Chem


class ProtonationEngine:
    """Basic protonation engine."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def generate_protonation_states(self, mol: Chem.Mol, pka_values: Optional[List[float]] = None) -> Dict[float, Chem.Mol]:
        """Generate basic protonation states."""
        # Simple implementation - return original molecule at different pH
        return {7.4: mol}


def protonate_ligand(mol: Chem.Mol, ph: float = 7.4, pka_values: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Wrapper function to generate protonation states for backend compatibility.
    
    Args:
        mol: RDKit molecule object
        ph: Target pH value
        pka_values: List of pKa site dictionaries
        
    Returns:
        List of protonation state dictionaries
    """
    import random
    
    try:
        engine = ProtonationEngine()
        
        # Extract pKa values if provided
        if pka_values:
            pka_list = [site.get('pka', 7.0) for site in pka_values]
        else:
            pka_list = None
        
        # Generate protonation states
        protonation_states = engine.generate_protonation_states(mol, pka_list)
        
        # Convert to backend-expected format
        states = []
        for i, (ph_val, mol_state) in enumerate(protonation_states.items()):
            if abs(ph_val - ph) <= 2.0:  # Only include states near target pH
                # Calculate probability based on Henderson-Hasselbalch
                if pka_values and len(pka_values) > 0:
                    # Use actual pKa for probability calculation
                    pka = pka_values[0].get('pka', 7.0)
                    if pka < ph:  # Acid behavior
                        probability = 1.0 / (1.0 + 10**(pka - ph))
                    else:  # Base behavior
                        probability = 1.0 / (1.0 + 10**(ph - pka))
                else:
                    probability = random.uniform(0.1, 0.9)
                
                # Estimate charge based on pH and pKa
                charge = 0
                if pka_values:
                    for site in pka_values:
                        site_pka = site.get('pka', 7.0)
                        if site_pka < ph:  # Likely deprotonated
                            charge -= 1
                        elif site_pka > ph + 2:  # Likely protonated
                            charge += 1
                
                states.append({
                    'smiles': Chem.MolToSmiles(mol_state),
                    'probability': probability,
                    'charge': charge
                })
        
        # If no states generated, return original molecule
        if not states:
            states.append({
                'smiles': Chem.MolToSmiles(mol),
                'probability': 1.0,
                'charge': 0
            })
        
        return states
        
    except Exception as e:
        logging.error(f"Protonation state generation failed: {e}")
        # Return original molecule as fallback
        return [{
            'smiles': Chem.MolToSmiles(mol),
            'probability': 1.0,
            'charge': 0
        }]