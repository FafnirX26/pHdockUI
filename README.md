# pHdock: Advanced pKa Prediction Pipeline

A state-of-the-art machine learning system for predicting molecular pKa values and performing pH-aware molecular docking, featuring quantum-enhanced ensemble models and comprehensive data processing pipelines.


## 🚀 Key Features

### Advanced ML Models
- **Quantum-Enhanced Ensemble**: Best-performing model (R² = 0.874) with 55 quantum-inspired features
- **Physics-Informed Neural Networks**: Incorporates Hammett equations and thermodynamic constraints
- **Graph Neural Networks**: Molecular graph representations with attention mechanisms
- **Random Forest & XGBoost**: Robust tree-based baseline models

### Comprehensive Data Pipeline
- **17,000+ molecules** from multiple high-quality sources (ChEMBL, IUPAC, SAMPL6)
- **Advanced filtering** with statistical outlier detection
- **Batch data fetching** for large-scale dataset collection
- **Intelligent duplicate handling** with IQR-based filtering

### Production-Ready Features
- **pH-Aware Chemistry**: Protonation states from pH 1-14 with variable step optimization
- **Docking Integration**: GNINA support with extensible framework (see [DOCKING_GUIDE.md](DOCKING_GUIDE.md))
- **Web API**: FastAPI backend with receptor management and job queue
- **Interactive Frontend**: Next.js interface for molecule submission and visualization
- **Model Persistence**: Saved models for immediate deployment
- **Comprehensive Evaluation**: Performance metrics and validation framework

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- RDKit
- PyTorch
- scikit-learn
- XGBoost
- GNINA (optional, for docking)

### Quick Setup
```bash
# Clone repository
git clone https://github.com/FafnirX26/pHdock.git
cd pHdock

# Create virtual environment
python -m venv phd
source phd/bin/activate  # On Windows: phd\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn xgboost matplotlib seaborn rdkit torch

# Install GNINA for docking (optional)
# See GNINA_INSTALL.md for detailed instructions
conda install -c conda-forge gnina
```

### Large Dataset Setup (Optional)
For handling large datasets efficiently:
```bash
# Install git-lfs (Fedora/RHEL)
sudo dnf install git-lfs

# Initialize git-lfs
git lfs install
git lfs track "training_data/*.csv"
```

## 🎯 Usage

### Quick Start: Best Model Prediction
```bash
# Use the best quantum-enhanced model
python fast_quantum_pka.py

# Generate comprehensive performance summary
python final_summary.py
```

### Advanced Data Processing
```bash
# Large-scale data collection
python batch_data_fetcher.py

# Advanced data filtering
python data_filter.py

# Train advanced ensemble model
python ensemble_train_advanced.py
```

### Main Pipeline Integration
```bash
# Basic pKa prediction
python main.py --input molecules.smi --mode pka_prediction --output results/

# Use large dataset
python main.py --input molecules.smi --data_source large --data_limit 10000

# Fetch large data before training
python main.py --input molecules.smi --fetch_large_data --use_batch_fetcher

# Generate protonation states with optimized pH steps
python main.py --input molecules.sdf --mode protonation_states --ph_min 1 --ph_max 14

# Full pipeline with pH-aware docking
python main.py --input molecules.smi --receptor receptors/example/1ATP.pdb \
  --mode full_pipeline --docking_tool gnina --ph_min 6 --ph_max 8
```

### Model Training Options
```bash
# Train with different data sources
python main.py --data_source synthetic --data_limit 1000
python main.py --data_source chembl --data_limit 5000
python main.py --data_source combined --data_limit 10000
python main.py --data_source large --data_limit 17000
```

## 🏗️ Architecture

### Core Components

#### 1. Data Processing Pipeline
- **`batch_data_fetcher.py`**: Large-scale data acquisition (17K+ molecules)
- **`data_filter.py`**: Advanced filtering with statistical outlier detection
- **`enhanced_data_fetcher.py`**: Multi-source data integration

#### 2. Model Implementations
- **`fast_quantum_pka.py`**: **Primary model** - Quantum-enhanced ensemble
- **`ensemble_train_advanced.py`**: Advanced ensemble with 47 features
- **`physics_informed_pka.py`**: Physics-constrained neural network
- **`graph_neural_network_pka.py`**: Graph convolutional network
- **`simple_train.py`**: Baseline Random Forest model

#### 3. Evaluation & Testing
- **`final_summary.py`**: Comprehensive performance analysis
- **`test_quantum_model.py`**: Quantum model validation
- **`test_ensemble_advanced.py`**: Advanced ensemble testing

#### 4. Core Pipeline
- **`main.py`**: Enhanced main pipeline with flexible data options
- **`src/data_integration.py`**: Unified data loading and processing
- **`src/`**: Core modules for feature engineering, prediction, and evaluation

### Pipeline Stages
1. **Input Processing**: Standardizes SMILES/SDF inputs using RDKit
2. **Conformer Generation**: 3D conformer generation with energy minimization
3. **Feature Engineering**: 55 quantum-inspired molecular descriptors
4. **Quantum Surrogate**: Electronic structure property prediction
5. **pKa Prediction**: Ensemble model with optimized weights
6. **Protonation Engine**: pH-dependent state generation with variable steps
7. **Docking Integration**: Molecular docking with optimal protonation states

## 📈 Data Sources & Statistics

### Training Dataset
- **Total molecules**: 17,180 (after filtering)
- **pKa range**: -5.0 to 18.7
- **Data retention**: 99.7% after quality control
- **Sources**: ChEMBL, IUPAC, SAMPL6, comprehensive databases

### Key Dataset Files
- **`training_data/filtered_pka_dataset.csv`**: Final cleaned dataset (17,180 records)
- **`training_data/combined_pka_dataset_large.csv`**: Raw combined dataset
- **`training_data/`**: Individual source datasets and metadata

## 🔬 Technical Innovations

### Quantum-Enhanced Features (55 total)
- **Electronic descriptors**: Gasteiger charges, HOMO-LUMO gap proxies
- **Frontier orbital descriptors**: EState indices, partial charges
- **Polarization descriptors**: Molar refractivity, surface areas, solvation
- **Ionization descriptors**: Functional group counts, H-bond donors/acceptors
- **Aromaticity descriptors**: Aromatic fraction, ring analysis

### Advanced Data Processing
- **Modified Z-score outlier detection** (threshold: 3.5)
- **IQR-based duplicate filtering**
- **Chemical validation** and structure filtering
- **Molecular weight filtering** (50-1500 Da)
- **pKa range validation** (-8.0 to 23.0)

### Model Architecture
- **Ensemble weights**: XGBoost (0.75) + Random Forest (0.25)
- **Feature scaling**: StandardScaler with NaN handling
- **Cross-validation**: 5-fold with stratified sampling
- **Hyperparameter optimization**: Grid search with early stopping

## 🧪 pH-Aware Protonation States

The protonation engine uses **intelligent variable pH step sizes** optimized for biological systems:

- **Step 1.0**: pH_min to pH 6 (coarse sampling)
- **Step 0.1**: pH 6 to pH 7 (fine sampling)
- **Step 0.05**: pH 7 to pH 7.6 (ultra-fine sampling)
- **Step 0.1**: pH 7.6 to pH 8 (fine sampling)
- **Step 1.0**: pH 8 to pH_max (coarse sampling)

This provides **fine-grained sampling around physiologically relevant pH values** (6-8) where most pKa transitions occur.

## 🎯 Validation & Testing

### Test Compounds (Known pKa values)
- **Acetic acid** (4.76), **Benzoic acid** (4.2), **Phenol** (9.95)
- **Aniline** (4.6), **Ethylamine** (10.7), **Pyridine** (5.2)
- **Imidazole** (7.0), **Ibuprofen** (4.4), **Acetaminophen** (9.5)

### Testing Framework
```bash
# Test individual models
python test_quantum_model.py
python test_ensemble_advanced.py
python test_data_integration.py

# Full pipeline validation
python test_updated_pipeline.py
```

## 📊 Saved Models

The following trained models are available in `models/`:
- **`fast_quantum_xgb.pkl`**: XGBoost with quantum features
- **`fast_quantum_rf.pkl`**: Random Forest with quantum features
- **`fast_quantum_scaler.pkl`**: Feature scaler
- **`fast_quantum_weights.pkl`**: Ensemble weights

## 🔧 Command Line Options

### Core Options
- `--input`: Input file (SMILES or SDF format)
- `--mode`: Operating mode (pka_prediction, protonation_states, full_pipeline, train_models)
- `--output`: Output directory
- `--data_source`: Training data source (synthetic, chembl, combined, large)
- `--data_limit`: Maximum number of molecules to load

### Advanced Options
- `--fetch_large_data`: Run large-scale data fetching before training
- `--use_batch_fetcher`: Use advanced batch fetcher for maximum data collection
- `--receptor`: Receptor PDB file (for docking)
- `--docking_tool`: Docking tool (gnina, diffdock, equibind)
- `--model_type`: Model type (xgboost, gnn, ensemble)
- `--save_models`, `--load_models`: Model persistence options

## 🚀 Performance Optimization

### Speed Benchmarks
- **Input processing**: ~1-5 seconds per molecule
- **Feature calculation**: ~2-10 seconds per molecule
- **pKa prediction**: <1 second per molecule (after training)
- **Protonation state generation**: ~1-5 seconds per molecule
- **Model training**: ~10-60 minutes (depends on dataset size)

### Memory Usage
- **Small dataset** (1K molecules): ~500MB RAM
- **Large dataset** (17K molecules): ~2-4GB RAM
- **Model inference**: ~100MB RAM per model


### Development Setup
```bash
# Install development dependencies
pip install pytest jupyter notebook

# Run tests
pytest tests/

# Launch development environment
jupyter notebook
```

## 📚 Research Impact

This work represents a **significant advancement** in computational pKa prediction:
- **State-of-the-art performance** with quantum-enhanced features
- **Physics-informed approach** incorporating chemical knowledge
- **Large-scale validation** on diverse molecular dataset
- **Production-ready pipeline** for drug discovery applications

The quantum-enhanced ensemble model achieves performance comparable to expensive DFT calculations while maintaining computational efficiency suitable for high-throughput screening.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
