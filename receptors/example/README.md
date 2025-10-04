# Example Receptors

This directory contains example receptor PDB files for testing the docking pipeline.

## Available Receptors

Currently, no PDB files are included in this repository due to size constraints. You can download them from the Protein Data Bank (PDB).

### 1ATP - Cyclooxygenase-2 (COX-2)
```bash
cd receptors/example
wget https://files.rcsb.org/download/1ATP.pdb
```

### 6LU7 - SARS-CoV-2 Main Protease (rename to 3CL.pdb)
```bash
cd receptors/example
wget https://files.rcsb.org/download/6LU7.pdb
mv 6LU7.pdb 3CL.pdb
```

## Preparing Receptors

Before using receptors for docking, they should be prepared:

1. **Remove Water and Ligands:**
   ```python
   from rdkit import Chem

   # Load PDB
   mol = Chem.MolFromPDBFile("receptor.pdb", removeHs=False)

   # Save cleaned version
   Chem.MolToPDBFile(mol, "receptor_clean.pdb")
   ```

2. **Add Hydrogens (if needed):**
   - Use tools like `reduce` or `pdb2pqr`
   - Or use UCSF Chimera/PyMOL

3. **Validate Structure:**
   - Check for missing atoms/residues
   - Verify protein chains are correct
   - Ensure no severe clashes

## Using Your Own Receptors

1. Place PDB files in this directory
2. Name them with a short ID (e.g., `MYRECEPTOR.pdb`)
3. Add metadata to `website/backend/main.py` in the `receptor_metadata` dict
4. Restart the backend API

## Notes

- The backend will automatically detect PDB files in this directory
- Files must have `.pdb` extension
- Maximum file size: 50MB recommended
