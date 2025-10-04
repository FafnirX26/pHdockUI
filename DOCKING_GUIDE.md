# pH-Aware Molecular Docking Guide

This guide explains how to use the pH-aware molecular docking features in pHdock.

## Overview

pHdock integrates advanced pKa prediction with molecular docking to account for pH-dependent protonation states. This provides more accurate binding affinity predictions compared to traditional docking approaches that ignore pH effects.

## Architecture

```
User Input (SMILES/SDF)
  ↓
Input Processing & Validation
  ↓
Conformer Generation (3D structures)
  ↓
pKa Prediction (Quantum-Enhanced Model)
  ↓
Protonation State Generation (pH 1-14)
  ↓
Molecular Docking (GNINA/AutoDock)
  ↓
pH-Dependent Analysis & Results
```

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Install GNINA (see GNINA_INSTALL.md)
conda install -c conda-forge gnina

# Download example receptors
cd receptors/example
wget https://files.rcsb.org/download/1ATP.pdb
```

### 2. Run Full Pipeline

```bash
# Command-line interface
python main.py \
  --input aspirin.smi \
  --receptor receptors/example/1ATP.pdb \
  --mode full_pipeline \
  --docking_tool gnina \
  --ph_min 6.0 \
  --ph_max 8.0 \
  --ph_step 0.5

# Web interface
cd website/backend
python main.py
# Navigate to http://localhost:8000
```

### 3. Analyze Results

Results include:
- **Best pH**: Optimal protonation state for binding
- **Docking Scores**: Binding affinity at each pH value
- **Poses**: Top-ranked binding conformations
- **pH Profile**: Score vs. pH curve

## Pipeline Components

### 1. Input Processing

Supported formats:
- **SMILES**: Single-line text representation
- **SDF**: Structure-Data File with 3D coordinates
- **MOL2**: Tripos MOL2 format

Example:
```python
from src.input_processing import MoleculeProcessor

processor = MoleculeProcessor()
results = processor.process_smiles(["CC(=O)Oc1ccccc1C(=O)O"])  # Aspirin
molecules = results['molecules']
```

### 2. pKa Prediction

Uses quantum-enhanced ensemble model (R² = 0.874):

```python
from src.pka_prediction import pKaPredictionModel

model = pKaPredictionModel()
model.load_model("models/fast_quantum_xgb.pkl")
pka_values = model.predict(molecular_features)
```

### 3. Protonation State Generation

Generates pH-dependent molecular states:

```python
from src.protonation_engine import ProtonationEngine

engine = ProtonationEngine(ph_range=(6.0, 8.0), ph_step=0.5)
protonation_states = engine.generate_protonation_states(molecule)

# Returns: {6.0: mol_pH6, 6.5: mol_pH6.5, 7.0: mol_pH7, ...}
```

### 4. Molecular Docking

GNINA-based docking with CNN scoring:

```python
from src.docking_integration import DockingIntegration

docking = DockingIntegration(docking_tool="gnina")
results = docking.dock_protonation_states(
    receptor_file="receptor.pdb",
    protonation_states=protonation_states,
    molecule_name="aspirin"
)
```

### 5. Result Analysis

Analyze pH-dependent binding:

```python
# Get docking analysis
analysis_df = docking.analyze_docking_results(results)

# Select optimal state
optimal_ph, info = docking.select_optimal_protonation_state(
    analysis_df,
    selection_criteria="consensus"  # or "best_score", "physiological"
)

print(f"Optimal pH: {optimal_ph}")
print(f"Best score: {info['score']}")
```

## Web Interface

### Starting the API Server

```bash
cd website/backend
python main.py

# Server runs on http://localhost:8000
# API docs at http://localhost:8000/docs
```

### API Endpoints

#### List Receptors
```bash
GET /api/receptors

Response:
[
  {
    "id": "1ATP",
    "name": "Cyclooxygenase-2 (COX-2)",
    "pdb_id": "1ATP",
    "description": "Aspirin and NSAID target",
    "available": true
  }
]
```

#### Submit Docking Job
```bash
POST /api/jobs

Body:
{
  "smiles": "CC(=O)Oc1ccccc1C(=O)O",
  "ph_value": 7.4,
  "conformer_count": 10,
  "ensemble_size": 5,
  "receptor_id": "1ATP",
  "docking_backend": "gnina"
}

Response:
{
  "job_id": "uuid-here",
  "status": "pending",
  "created_at": "2025-10-04T..."
}
```

#### Get Job Results
```bash
GET /api/jobs/{job_id}

Response:
{
  "job_id": "uuid",
  "status": "completed",
  "progress": 1.0,
  "results": {
    "molecule_info": {...},
    "pka_predictions": {...},
    "protonation_states": [...],
    "docking_results": {
      "best_score": -8.5,
      "poses": [...]
    }
  }
}
```

## Advanced Features

### Custom Receptors

Add your own receptor:
```bash
# 1. Prepare PDB file
# 2. Place in receptors/example/
cp my_receptor.pdb receptors/example/MYRECEPTOR.pdb

# 3. Add metadata to backend/main.py
# Edit receptor_metadata dict in get_receptors()

# 4. Restart API
```

### pH Range Optimization

Fine-tune pH sampling:
```python
# Coarse sampling
engine = ProtonationEngine(ph_range=(1.0, 14.0), ph_step=1.0)

# Fine sampling around physiological pH
engine = ProtonationEngine(ph_range=(6.0, 8.0), ph_step=0.1)

# Variable step sizes (optimized for biology)
# See src/protonation_engine.py for variable_ph_steps
```

### Docking Parameters

Customize GNINA settings:
```python
docking = DockingIntegration(docking_tool="gnina")
docking.docking_params = {
    'exhaustiveness': 16,  # Higher = more thorough (default: 8)
    'num_modes': 20,       # More poses (default: 9)
    'energy_range': 5.0,   # Wider range (default: 3.0)
    'cpu': 8               # More cores (default: 4)
}
```

## Troubleshooting

### GNINA Not Found
```bash
# Check installation
which gnina

# Install via conda
conda install -c conda-forge gnina

# Or use Docker
docker pull gnina/gnina
```

### Docking Failures
- **Receptor issues**: Check PDB format, missing atoms
- **Ligand issues**: Ensure valid SMILES, 3D coordinates
- **Timeout**: Increase timeout in docking_integration.py
- **Memory**: Reduce num_conformers, ensemble_size

### Poor Docking Scores
- **Receptor preparation**: Add hydrogens, optimize side chains
- **Ligand preparation**: Generate better conformers
- **pH range**: Ensure pH range covers pKa values
- **Box size**: Adjust search space in GNINA

## Performance Tips

### Speed Optimization
```python
# Reduce conformer generation
conformer_gen = ConformerGenerator(num_conformers=20)  # Default: 50

# Reduce protonation states
engine = ProtonationEngine(ph_step=1.0)  # Default: 0.5

# Parallel docking
docking.docking_params['cpu'] = 8  # Use all cores
```

### Accuracy vs. Speed

| Setting | Speed | Accuracy |
|---------|-------|----------|
| **Quick** | Fast | Moderate |
| - num_conformers: 10 | | |
| - ph_step: 1.0 | | |
| - exhaustiveness: 4 | | |
| **Balanced** | Medium | Good |
| - num_conformers: 50 | | |
| - ph_step: 0.5 | | |
| - exhaustiveness: 8 | | |
| **Thorough** | Slow | Best |
| - num_conformers: 100 | | |
| - ph_step: 0.1 | | |
| - exhaustiveness: 16 | | |

## Example Workflows

### Aspirin + COX-2 (Anti-inflammatory)

```bash
# 1. Download receptor
cd receptors/example
wget https://files.rcsb.org/download/1ATP.pdb

# 2. Run docking
python main.py \
  --input <(echo "CC(=O)Oc1ccccc1C(=O)O") \
  --receptor receptors/example/1ATP.pdb \
  --mode full_pipeline \
  --ph_min 6.5 \
  --ph_max 8.5

# 3. Check results
ls docking_results/docking_summary.csv
```

### Drug Screening Batch

```python
# Screen multiple compounds
compounds = pd.read_csv("compounds.csv")  # columns: name, smiles

for idx, row in compounds.iterrows():
    results = run_full_docking_pipeline(
        molecules=[Chem.MolFromSmiles(row['smiles'])],
        protonation_states_list=[...],
        receptor_file="receptor.pdb"
    )
    # Save results
    ...
```

## References

- [GNINA Documentation](https://gnina.github.io/gnina/)
- [RDKit](https://www.rdkit.org/)
- [Protein Data Bank](https://www.rcsb.org/)
- [pHdock Repository](https://github.com/FafnirX26/pHdock)

## Support

- Issues: [GitHub Issues](https://github.com/DBD808/pHdockUI/issues)
- Discussions: [GitHub Discussions](https://github.com/DBD808/pHdockUI/discussions)
- Email: Contact via website form
