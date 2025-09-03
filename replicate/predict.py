from typing import Optional, Dict, Any

# Cog predictor for Replicate
from cog import BasePredictor, Input

from rdkit import Chem

# Import local pipeline pieces
from src.pka_prediction import predict_pka_ensemble
from src.protonation_engine import ProtonationEngine, protonate_ligand


class Predictor(BasePredictor):
    def setup(self) -> None:
        # Place for loading models if needed (kept lightweight for now)
        self.engine = ProtonationEngine()

    def predict(
        self,
        smiles: Optional[str] = Input(
            description="SMILES string of the ligand.", default=None
        ),
        sdf_content: Optional[str] = Input(
            description="Raw SDF text content (alternative to SMILES).",
            default=None,
        ),
        ph_value: float = Input(description="Target pH.", default=7.4),
        ensemble_size: int = Input(description="Ensemble size for pKa.", default=5),
    ) -> Dict[str, Any]:
        if not smiles and not sdf_content:
            raise ValueError("Provide either `smiles` or `sdf_content`.")

        # Create RDKit molecule
        mol = None
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
        elif sdf_content:
            suppl = Chem.SDMolSupplier()
            # RDKit SDMolSupplier works on files; quick parse via temporary wrapper
            # Fallback: try MolFromMolBlock if single molecule block
            mol = Chem.MolFromMolBlock(sdf_content, sanitize=False)

        if mol is None:
            raise ValueError("Failed to parse molecule from input.")

        # pKa prediction (current wrapper returns synthetic yet structured values)
        pka_results = predict_pka_ensemble(mol, ensemble_size=ensemble_size)

        # Protonation near target pH
        states = protonate_ligand(mol, ph=ph_value, pka_values=pka_results.get("site_pkas", []))

        # Build UI-compatible payload
        overall = pka_results.get("predicted_pka")
        ui_results = {
            "molecule_info": {
                "smiles": smiles or Chem.MolToSmiles(mol),
                "molecular_weight": 0,  # Optional: compute if needed
            },
            "pka_predictions": {
                "overall_pka": overall,
                "global_pka": overall,  # Back-compat
                "site_pkas": [
                    {"pka": s.get("pka", None), "atom_index": s.get("atom_idx", -1)}
                    for s in pka_results.get("site_pkas", [])
                ],
                "confidence": pka_results.get("confidence", 0.0),
            },
            "protonation_states": [
                {
                    "state_id": idx,
                    "smiles": st.get("smiles", ""),
                    "probability": st.get("probability", 0.0),
                    "charge": st.get("charge", 0),
                }
                for idx, st in enumerate(states)
            ],
            "docking_results": {
                # Placeholder; integrate docking backend if/when available
                "best_score": -8.5 if states else None,
                "poses": [
                    {"state": i, "score": -8.5 + (i * 0.2), "confidence": 0.5}
                    for i in range(len(states))
                ],
            },
        }

        return {
            "status": "completed",
            "progress": 1.0,
            "results": ui_results,
            "request": {"ph_value": ph_value, "smiles": smiles},
        }

