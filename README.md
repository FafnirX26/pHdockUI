# phDock: pH-Dependent Molecular Docking Pipeline

A comprehensive machine learning system for predicting molecular protonation states and performing pH-aware molecular docking.

## Overview

phDock addresses a critical challenge in computational chemistry: molecules change their protonation state at different pH values, which dramatically affects their binding to proteins. Traditional docking methods ignore this pH dependence, leading to inaccurate predictions.

This system implements a 7-stage pipeline that:
1. Standardizes molecular inputs
2. Generates 3D conformers
3. Calculates molecular features
4. Predicts quantum properties
5. Predicts pKa values using machine learning
6. Generates pH-dependent protonation states
7. Performs molecular docking with optimal states

## Features

- **Multiple ML Models**: XGBoost, Graph Neural Networks, and Ensemble models
- **Real Data Integration**: ChEMBL experimental pKa values
- **pH-Aware Chemistry**: Protonation states from pH 1-14
- **Docking Integration**: GNINA support with extensible framework
- **Comprehensive Features**: 2D/3D descriptors, fingerprints, and quantum surrogates

## Installation

### Prerequisites
- Python 3.8+
- RDKit
- PyTorch
- GNINA (for docking)

### Setup
1. Clone the repository:
```bash
git clone https://github.com/FafnirX26/pHdock.git
cd phDock
```

2. Create and activate virtual environment:
```bash
python -m venv phd
source phd/bin/activate  # On Windows: phd\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install GNINA (optional, for docking):
```bash
# Follow GNINA installation instructions
```

## Usage

### Basic pKa Prediction
```bash
python main.py --input molecules.smi --mode pka_prediction --output results/
```

### Generate Protonation States
```bash
python main.py --input molecules.sdf --mode protonation_states --ph_min 1 --ph_max 14
```

**Note:** The protonation engine now uses intelligent variable pH step sizes optimized for biological systems:
- Step 1.0 from pH_min to pH 6
- Step 0.1 from pH 6 to pH 7 
- Step 0.05 from pH 7 to pH 7.6
- Step 0.1 from pH 7.6 to pH 8
- Step 1.0 from pH 8 to pH_max

This provides fine-grained sampling around physiologically relevant pH values (6-8) where most pKa transitions occur, while using coarser steps outside this range for efficiency. The `--ph_step` parameter is maintained for backward compatibility but is ignored.

### Full Pipeline with Docking
```bash
python main.py --input ligands.sdf --receptor protein.pdb --mode full_pipeline --docking_tool gnina
```

### Train Models
```bash
# Train with synthetic data
python main.py --input training_data.sdf --mode train_models --save_models models/ --model_type ensemble

# Train with real ChEMBL data
python main.py --input molecules.smi --data_source chembl --data_limit 1000 --save_models models/
```

## Command Line Options

- `--input`: Input file (SMILES or SDF format)
- `--mode`: Operating mode (pka_prediction, protonation_states, full_pipeline, train_models)
- `--output`: Output directory
- `--receptor`: Receptor PDB file (for docking)
- `--docking_tool`: Docking tool (gnina, diffdock, equibind)
- `--data_source`: Training data source (synthetic, chembl, combined)
- `--model_type`: Model type (xgboost, gnn, ensemble)
- `--save_models`: Directory to save trained models
- `--load_models`: Directory to load pre-trained models
- `--ph_min`, `--ph_max`, `--ph_step`: pH range parameters

## Architecture

### Pipeline Stages
1. **Input Processing**: Standardizes SMILES/SDF inputs using RDKit
2. **Conformer Generation**: Generates 3D conformers with energy minimization
3. **Feature Engineering**: Calculates molecular descriptors and fingerprints
4. **Quantum Surrogate**: Predicts quantum mechanical properties
5. **pKa Prediction**: ML models for pKa prediction
6. **Protonation Engine**: Generates pH-dependent states
7. **Docking Integration**: Interfaces with molecular docking tools

### Model Types
- **XGBoost**: Tree-based baseline model
- **Graph Neural Networks**: GCN/GAT models for molecular graphs
- **Ensemble**: Multi-modal fusion with attention mechanisms

## Performance

- Input processing: ~1-5 seconds per molecule
- Conformer generation: ~10-60 seconds per molecule
- pKa prediction: <1 second per molecule (after training)
- Protonation state generation: ~1-5 seconds per molecule
- Docking: 1-10 minutes per molecule per pH state

## Data Sources

- **Synthetic**: Generated using empirical rules
- **ChEMBL**: Real experimental pKa values via API
- **Combined**: Multiple sources merged

## Testing

Test individual modules:
```bash
# Test input processing
python -c "from src.input_processing import MoleculeProcessor; print('Input processing module loaded')"

# Test pKa prediction
python src/pka_prediction.py

# Test conformer generation
python src/conformer_generation.py
```
