# Quick Setup for Real Docking

## 1. Install GNINA (Pick One)

### Option A: Conda (Recommended)
```bash
conda install -c conda-forge gnina
gnina --version
```

### Option B: Binary Download
```bash
wget https://github.com/gnina/gnina/releases/download/v1.0.3/gnina
chmod +x gnina
sudo mv gnina /usr/local/bin/
gnina --version
```

### Option C: AUR (Arch Linux)
```bash
yay -S gnina-git
```

## 2. Download Receptors

```bash
cd receptors/example

# COX-2 (for aspirin, NSAIDs)
wget https://files.rcsb.org/download/1ATP.pdb

# COVID-19 protease
wget https://files.rcsb.org/download/6LU7.pdb
mv 6LU7.pdb 3CL.pdb
```

## 3. Restart Backend

```bash
cd website/backend
python main.py
```

## 4. Test in Frontend

1. Open http://localhost:3000
2. Enter SMILES: `CC(=O)Oc1ccccc1C(=O)O`
3. Advanced Settings → Select "Cyclooxygenase-2 (COX-2)"
4. Run Analysis
5. Wait ~30-60 seconds for real docking!

## Verification

Real docking will:
- ✅ Take 30-60 seconds (not instant)
- ✅ Show actual binding scores
- ✅ Create files in `docking_results/`

Mock docking:
- ⚡ Instant response
- 🎲 Random scores
- 📭 No files created

## Troubleshooting

### GNINA not found
```bash
which gnina
# If empty, reinstall or add to PATH
```

### Permission denied
```bash
sudo chmod +x /usr/local/bin/gnina
```

### Docking fails
- Check receptor file exists: `ls receptors/example/*.pdb`
- Check backend logs for errors
- Try mock receptor first to verify pipeline works
