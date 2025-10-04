# GNINA Installation Guide

GNINA is a molecular docking program with integrated support for scoring and optimizing ligands using convolutional neural networks.

## Installation Options

### Option 1: Conda (Recommended)
```bash
# Create conda environment
conda create -n phdock python=3.10
conda activate phdock

# Install GNINA via conda-forge
conda install -c conda-forge gnina

# Verify installation
gnina --version
```

### Option 2: Docker (Best for Production)
```bash
# Pull GNINA Docker image
docker pull gnina/gnina

# Create alias for easy use
alias gnina='docker run -v $(pwd):/data gnina/gnina'

# Test
gnina --version
```

### Option 3: Build from Source (Advanced)
```bash
# Install dependencies
sudo apt-get install build-essential cmake libboost-all-dev \
  libopenbabel-dev libeigen3-dev librdkit-dev

# Clone repository
git clone https://github.com/gnina/gnina.git
cd gnina

# Build
mkdir build && cd build
cmake ..
make -j$(nproc)
sudo make install

# Verify
gnina --version
```

### Option 4: Pre-compiled Binary (Quick Start)
```bash
# Download latest release
wget https://github.com/gnina/gnina/releases/download/v1.0.3/gnina
chmod +x gnina
sudo mv gnina /usr/local/bin/

# Verify
gnina --version
```

## Arch Linux Specific

### AUR Installation
```bash
# Using yay
yay -S gnina-git

# Using paru
paru -S gnina-git
```

### Manual Build on Arch
```bash
# Install dependencies
sudo pacman -S base-devel cmake boost openbabel eigen rdkit

# Clone and build as above
```

## Verifying Installation

Test GNINA with a simple example:

```bash
# Download test files
wget https://github.com/gnina/gnina/raw/master/test/receptor.pdb
wget https://github.com/gnina/gnina/raw/master/test/ligand.sdf

# Run docking
gnina --receptor receptor.pdb \
      --ligand ligand.sdf \
      --out docked.sdf \
      --autobox_ligand ligand.sdf \
      --exhaustiveness 8

# Check output
ls -lh docked.sdf
```

Expected output: `docked.sdf` file with docked poses.

## Configuration for pHdock

After installation, update your environment:

```bash
# Add to ~/.bashrc or ~/.zshrc
export GNINA_PATH=$(which gnina)
export RECEPTOR_DIR="$HOME/Documents/Projects/pHdockUI/receptors"
```

## Troubleshooting

### GNINA not found
```bash
# Check if in PATH
echo $PATH | grep -o '/usr/local/bin'

# Find GNINA location
which gnina
find /usr -name gnina 2>/dev/null
```

### Permission denied
```bash
chmod +x $(which gnina)
```

### Missing libraries
```bash
# Check dependencies
ldd $(which gnina)

# Install missing libs (Ubuntu/Debian)
sudo apt-get install libboost-system1.71.0 libboost-filesystem1.71.0

# Install missing libs (Arch)
sudo pacman -S boost-libs
```

### CUDA/GPU issues
GNINA can use GPU acceleration but works fine on CPU:
```bash
# Force CPU mode
gnina --cpu_only ...
```

## Alternative: Using AutoDock Vina

If GNINA installation fails, you can use AutoDock Vina as a fallback:

```bash
# Install via conda
conda install -c conda-forge vina

# Or download binary
wget https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64
chmod +x vina_1.2.5_linux_x86_64
sudo mv vina_1.2.5_linux_x86_64 /usr/local/bin/vina
```

Then update `src/docking_integration.py` to use Vina instead of GNINA.

## Next Steps

After installation:
1. Download example receptors (see `receptors/README.md`)
2. Test the docking pipeline with `python main.py --mode full_pipeline`
3. Run the backend API: `cd website/backend && python main.py`

## Resources

- [GNINA Documentation](https://gnina.github.io/gnina/)
- [GNINA GitHub](https://github.com/gnina/gnina)
- [AutoDock Vina](https://vina.scripps.edu/)
- [PDB Database](https://www.rcsb.org/)
