# Receptor Files for Molecular Docking

This directory contains protein receptor structures (PDB format) used for molecular docking.

## Directory Structure

```
receptors/
├── example/          # Curated example receptors
│   ├── 1ATP.pdb     # Aspirin target (COX-2)
│   └── 3CL.pdb      # COVID-19 main protease
└── user_uploads/     # User-uploaded receptors (not tracked in git)
```

## Example Receptors

### 1ATP - Cyclooxygenase-2 (COX-2)
- **PDB ID**: 1ATP
- **Description**: Aspirin and NSAID target
- **Resolution**: 2.8 Å
- **Use Case**: Testing anti-inflammatory compounds

### 3CL - SARS-CoV-2 Main Protease
- **PDB ID**: 3CL (6LU7)
- **Description**: COVID-19 main protease (Mpro)
- **Resolution**: 2.16 Å
- **Use Case**: Antiviral drug screening

## Adding Custom Receptors

### Option 1: Download from PDB
```bash
cd receptors/example
wget https://files.rcsb.org/download/1ATP.pdb
```

### Option 2: Use the API (when implemented)
```bash
curl -X POST http://localhost:8000/api/receptors \
  -F "file=@your_receptor.pdb" \
  -F "name=My Receptor" \
  -F "description=Custom target"
```

### Option 3: Manual Upload via Frontend
- Navigate to the docking interface
- Click "Upload Receptor"
- Select your PDB file
- Add metadata (name, description)

## Receptor Preparation

For best results, receptors should be:
1. **Cleaned** - Remove water molecules, ligands, non-standard residues
2. **Protonated** - Add hydrogen atoms at appropriate pH
3. **Optimized** - Minimize energy, fix missing atoms/residues
4. **Validated** - Check for clashes, unusual geometry

### Recommended Tools
- **UCSF Chimera** - Visual preparation and validation
- **PyMOL** - Structure editing and cleaning
- **PDB2PQR** - Add hydrogens, assign protonation states
- **Reduce** - Add hydrogens to PDB files

## File Requirements

- **Format**: PDB (.pdb)
- **Size**: < 50 MB
- **Validation**: Must contain ATOM records
- **Chains**: Single chain or multi-chain supported
- **Ligands**: Co-crystallized ligands will be removed during docking

## Notes

- User-uploaded receptors are stored in `user_uploads/` and not tracked in git
- Example receptors are provided for testing and demonstration
- For production use, always validate receptor quality before docking
