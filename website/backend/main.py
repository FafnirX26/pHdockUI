from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uuid
import sys
import os
from pathlib import Path
import tempfile
import json
import time
from datetime import datetime
import asyncio
import smtplib
from email.message import EmailMessage
import requests
import shutil

# Add parent directory to path to import the main project modules
sys.path.append(str(Path(__file__).parent.parent.parent))

# Try to import the actual modules, fall back to mock implementations
try:
    from src.input_processing import process_input
    from src.conformer_generation import generate_conformers
    from src.protonation_engine import protonate_ligand
    from src.pka_prediction import predict_pka_ensemble
    from src.docking_integration import run_docking
    from src.model_evaluation import evaluate_model_performance
    MODULES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Using mock implementations for demonstration")
    MODULES_AVAILABLE = False

# Try to load improved pKa model components
IMPROVED_PKA_MODEL = None
PKA_SCALER = None
PKA_FEATURE_EXTRACTOR = None

try:
    import joblib
    from src.improved_pka_model import CuratedFeatureExtractor

    base_path = Path(__file__).parent.parent.parent / "models"
    xgb_path = base_path / "fast_quantum_xgb.pkl"
    scaler_path = base_path / "fast_quantum_scaler.pkl"

    if xgb_path.exists() and scaler_path.exists():
        IMPROVED_PKA_MODEL = joblib.load(xgb_path)
        PKA_SCALER = joblib.load(scaler_path)
        PKA_FEATURE_EXTRACTOR = CuratedFeatureExtractor()
        print(f"✓ Loaded improved pKa model (XGBoost + RobustScaler)")
        print(f"  Model ready for predictions")
    else:
        print(f"⚠ Model files not found at {base_path}, using mock pKa predictions")
except Exception as e:
    print(f"⚠ Could not load improved pKa model: {e}")
    IMPROVED_PKA_MODEL = None

app = FastAPI(title="pH Docking API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://ph-dock.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (replace with Redis in production)
jobs_db = {}
# Simple in-memory store for contact messages (replace with DB in production)
contacts_db: List[Dict[str, Any]] = []

class JobRequest(BaseModel):
    smiles: Optional[str] = None
    sdf_content: Optional[str] = None
    ph_value: float = 7.4
    conformer_count: int = 10
    ensemble_size: int = 5
    quantum_fallback: bool = False
    docking_backend: str = "gnina"
    receptor_id: Optional[str] = "mock"  # Default to mock for now

class JobResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime

class JobResult(BaseModel):
    job_id: str
    status: str
    progress: float
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

class ContactRequest(BaseModel):
    name: str
    email: str
    institution: Optional[str] = None
    subject: str
    message: str

class ContactResponse(BaseModel):
    id: str
    status: str
    received_at: datetime

# Mock implementations for when modules are not available
def mock_process_input(input_data, input_type="smiles"):
    return {
        'mol': input_data,
        'smiles': input_data if input_type == "smiles" else "CC(=O)Oc1ccccc1C(=O)O",
        'name': 'Mock Molecule',
        'molecular_weight': 180.16
    }

def mock_generate_conformers(mol, n_conformers=10):
    return [f"conformer_{i}" for i in range(n_conformers)]

def mock_predict_pka_ensemble(mol, ensemble_size=5):
    """Predict pKa using improved model if available, else mock."""
    if IMPROVED_PKA_MODEL is not None and PKA_SCALER is not None and PKA_FEATURE_EXTRACTOR is not None:
        try:
            # Convert mol to SMILES if needed
            if hasattr(mol, 'GetProp'):
                from rdkit import Chem
                smiles = Chem.MolToSmiles(mol)
            elif isinstance(mol, str):
                smiles = mol
            else:
                smiles = str(mol)

            # Extract features
            features = PKA_FEATURE_EXTRACTOR.extract_all_features(smiles)
            if features is None:
                raise ValueError("Could not extract features")

            # Scale and predict
            import numpy as np
            features_scaled = PKA_SCALER.transform(features.reshape(1, -1))
            predicted_pka = IMPROVED_PKA_MODEL.predict(features_scaled)[0]

            # Calculate confidence (based on CV RMSE = 1.745)
            confidence = 0.83  # ~1.0 - (1.745/10)

            # Generate site-specific pKas (simplified)
            from rdkit import Chem
            from rdkit.Chem import Fragments
            mol_obj = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else mol

            site_pkas = []
            idx = 0

            # Detect ionizable groups
            if Fragments.fr_COO(mol_obj) > 0:
                site_pkas.append({'pka': predicted_pka, 'atom_idx': idx, 'type': 'carboxylic'})
                idx += 1
            if Fragments.fr_phenol(mol_obj) > 0:
                site_pkas.append({'pka': predicted_pka + 5, 'atom_idx': idx, 'type': 'phenol'})
                idx += 1
            if Fragments.fr_NH2(mol_obj) > 0:
                site_pkas.append({'pka': predicted_pka + 6, 'atom_idx': idx, 'type': 'amine'})
                idx += 1

            if not site_pkas:
                site_pkas.append({'pka': predicted_pka, 'atom_idx': 0, 'type': 'primary'})

            # Convert all numpy types to Python native types for JSON serialization
            return {
                'predicted_pka': float(predicted_pka),
                'site_pkas': [
                    {
                        'pka': float(site['pka']),
                        'atom_idx': int(site['atom_idx']),
                        'type': str(site['type'])
                    }
                    for site in site_pkas
                ],
                'confidence': float(confidence),
                'model': 'improved_xgboost',
                'cv_r2': 0.690,  # From training
                'cv_rmse': 1.745  # From training
            }
        except Exception as e:
            print(f"Error using improved model: {e}, falling back to mock")
            import traceback
            traceback.print_exc()

    # Fallback to random mock
    import random
    return {
        'predicted_pka': 4.2 + random.uniform(-0.5, 0.5),
        'site_pkas': [
            {'pka': 4.2 + random.uniform(-0.3, 0.3), 'atom_idx': 0, 'type': 'mock'},
            {'pka': 9.8 + random.uniform(-0.3, 0.3), 'atom_idx': 1, 'type': 'mock'}
        ],
        'confidence': 0.85 + random.uniform(-0.1, 0.1),
        'model': 'mock'
    }

def mock_protonate_ligand(mol, ph=7.4, pka_values=None):
    import random
    # Use the input SMILES as base, or aspirin as default
    base_smiles = mol if isinstance(mol, str) else "CC(=O)Oc1ccccc1C(=O)O"

    states = []
    # Generate 3 mock protonation states with actual SMILES variations
    # These represent different protonation states of aspirin
    smiles_variants = [
        "CC(=O)Oc1ccccc1C(=O)O",      # Neutral aspirin
        "CC(=O)Oc1ccccc1C(=O)[O-]",   # Deprotonated (anionic)
        "CC(=O)Oc1ccccc1C(=O)O"       # Alternative neutral form
    ]

    for i in range(3):
        states.append({
            'smiles': smiles_variants[i] if i < len(smiles_variants) else base_smiles,
            'probability': random.uniform(0.1, 0.8),
            'charge': [-1, 0, 0][i] if i < 3 else 0
        })
    return states

@app.get("/")
async def root():
    return {"message": "pH Docking API is running", "modules_available": MODULES_AVAILABLE, "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "modules_available": MODULES_AVAILABLE}

@app.get("/api/receptors")
async def get_receptors():
    """Get list of available receptors"""
    receptors_dir = Path(__file__).parent.parent.parent / "receptors" / "example"
    receptors = []

    # Define metadata for example receptors
    receptor_metadata = {
        "1ATP": {
            "id": "1ATP",
            "name": "Cyclooxygenase-2 (COX-2)",
            "pdb_id": "1ATP",
            "description": "Aspirin and NSAID target",
            "resolution": "2.8 Å",
            "use_case": "Anti-inflammatory compound screening"
        },
        "3CL": {
            "id": "3CL",
            "name": "SARS-CoV-2 Main Protease",
            "pdb_id": "6LU7",
            "description": "COVID-19 main protease (Mpro)",
            "resolution": "2.16 Å",
            "use_case": "Antiviral drug screening"
        },
        "mock": {
            "id": "mock",
            "name": "Mock Receptor (No Real Docking)",
            "pdb_id": "MOCK",
            "description": "Returns simulated docking results without GNINA",
            "resolution": "N/A",
            "use_case": "Testing and demonstration"
        }
    }

    # Check for actual PDB files
    if receptors_dir.exists():
        for pdb_file in receptors_dir.glob("*.pdb"):
            receptor_id = pdb_file.stem
            if receptor_id in receptor_metadata:
                receptors.append({
                    **receptor_metadata[receptor_id],
                    "file_path": str(pdb_file),
                    "available": True
                })

    # Always include mock receptor
    receptors.append({
        **receptor_metadata["mock"],
        "file_path": None,
        "available": True
    })

    return receptors

@app.post("/api/contact", response_model=ContactResponse)
async def submit_contact(request: ContactRequest):
    """Receive a contact form submission"""
    contact_id = str(uuid.uuid4())
    record = {
        "id": contact_id,
        "name": request.name,
        "email": request.email,
        "institution": request.institution,
        "subject": request.subject,
        "message": request.message,
        "received_at": datetime.now(),
    }
    contacts_db.append(record)

    # Append to a local file for persistence during demos
    try:
        log_path = Path(__file__).parent / "contact_messages.jsonl"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "id": record["id"],
                "name": record["name"],
                "email": record["email"],
                "institution": record["institution"],
                "subject": record["subject"],
                "message": record["message"],
                "received_at": record["received_at"].isoformat(),
            }) + "\n")
    except Exception as e:
        # Non-fatal in demo mode
        print(f"Warning: failed to persist contact message: {e}")

    # Attempt to send email notification if SMTP environment is configured
    try:
        smtp_host = os.getenv("SMTP_HOST", "smtp.gmail.com")
        smtp_port = int(os.getenv("SMTP_PORT", "587"))
        smtp_user = os.getenv("SMTP_USER")
        smtp_pass = os.getenv("SMTP_PASS")
        smtp_from = os.getenv("SMTP_FROM", os.getenv("SMTP_USER", ""))
        smtp_to = os.getenv("SMTP_TO", "phdockteam@gmail.com")

        if smtp_user and smtp_pass and smtp_from:
            msg = EmailMessage()
            msg["Subject"] = f"[pHdockUI Contact] {request.subject}"
            msg["From"] = smtp_from
            msg["To"] = smtp_to
            # Make replies go to the user who filled the form
            if request.email:
                msg["Reply-To"] = request.email
            body_lines = [
                f"New contact message (ID: {contact_id})",
                "",
                f"Name: {request.name}",
                f"Email: {request.email}",
                f"Institution: {request.institution or '-'}",
                f"Subject: {request.subject}",
                "",
                "Message:",
                request.message,
                "",
                f"Received at: {record['received_at'].isoformat()}",
            ]
            msg.set_content("\n".join(body_lines))

            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.ehlo()
                if smtp_port == 587:
                    server.starttls()
                    server.ehlo()
                server.login(smtp_user, smtp_pass)
                server.send_message(msg)
        else:
            print("SMTP not configured; skipping email send. Set SMTP_USER/SMTP_PASS/SMTP_FROM to enable.")
    except Exception as e:
        print(f"Warning: failed to send contact email: {e}")

    return ContactResponse(id=contact_id, status="received", received_at=record["received_at"])

@app.post("/api/jobs", response_model=JobResponse)
async def create_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Create a new pH docking job"""
    job_id = str(uuid.uuid4())
    
    # Initialize job in database
    jobs_db[job_id] = {
        "id": job_id,
        "status": "pending",
        "progress": 0.0,
        "created_at": datetime.now(),
        "completed_at": None,
        "results": None,
        "error": None,
        "request": request.dict()
    }
    
    # Start background processing
    background_tasks.add_task(process_job, job_id, request)
    
    return JobResponse(
        job_id=job_id,
        status="pending",
        created_at=jobs_db[job_id]["created_at"]
    )

@app.get("/api/jobs/{job_id}", response_model=JobResult)
async def get_job_status(job_id: str):
    """Get the status of a job"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs_db[job_id]
    return JobResult(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        results=job["results"],
        error=job["error"],
        created_at=job["created_at"],
        completed_at=job["completed_at"]
    )

@app.post("/api/upload")
async def upload_molecule(file: UploadFile = File(...)):
    """Upload a molecule file (SDF/MOL2)"""
    if not file.filename.endswith(('.sdf', '.mol2', '.mol')):
        raise HTTPException(status_code=400, detail="Invalid file format")
    
    content = await file.read()
    return {"filename": file.filename, "content": content.decode('utf-8')}

async def process_job(job_id: str, request: JobRequest):
    """Process a pH docking job in the background"""
    try:
        # Update status
        jobs_db[job_id]["status"] = "processing"
        jobs_db[job_id]["progress"] = 0.1
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Process input
        if request.smiles:
            if MODULES_AVAILABLE:
                mol_data = process_input(request.smiles, input_type="smiles")
            else:
                mol_data = mock_process_input(request.smiles, input_type="smiles")
        elif request.sdf_content:
            # Save SDF content to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sdf', delete=False) as f:
                f.write(request.sdf_content)
                temp_path = f.name
            if MODULES_AVAILABLE:
                mol_data = process_input(temp_path, input_type="sdf")
            else:
                mol_data = mock_process_input(temp_path, input_type="sdf")
            os.unlink(temp_path)
        else:
            raise ValueError("No molecule input provided")
        
        jobs_db[job_id]["progress"] = 0.2
        await asyncio.sleep(1)
        
        # Generate conformers
        if MODULES_AVAILABLE:
            conformers = generate_conformers(
                mol_data['mol'],
                n_conformers=request.conformer_count
            )
        else:
            conformers = mock_generate_conformers(
                mol_data['mol'],
                n_conformers=request.conformer_count
            )
        
        jobs_db[job_id]["progress"] = 0.4
        await asyncio.sleep(1)
        
        # Predict pKa values
        if MODULES_AVAILABLE:
            pka_results = predict_pka_ensemble(
                mol_data['mol'],
                ensemble_size=request.ensemble_size
            )
        else:
            pka_results = mock_predict_pka_ensemble(
                mol_data['mol'],
                ensemble_size=request.ensemble_size
            )
        
        jobs_db[job_id]["progress"] = 0.6
        await asyncio.sleep(1)
        
        # Generate protonation states
        if MODULES_AVAILABLE:
            protonation_states = protonate_ligand(
                mol_data['mol'],
                ph=request.ph_value,
                pka_values=pka_results['site_pkas']
            )
        else:
            protonation_states = mock_protonate_ligand(
                mol_data['mol'],
                ph=request.ph_value,
                pka_values=pka_results['site_pkas']
            )
        
        jobs_db[job_id]["progress"] = 0.8
        await asyncio.sleep(1)

        # Run docking (real or mock based on receptor and availability)
        receptor_path = None
        if request.receptor_id and request.receptor_id != "mock":
            receptors_dir = Path(__file__).parent.parent.parent / "receptors" / "example"
            potential_receptor = receptors_dir / f"{request.receptor_id}.pdb"
            if potential_receptor.exists():
                receptor_path = str(potential_receptor)

        # Use real docking if receptor is available and modules are loaded
        if MODULES_AVAILABLE and receptor_path:
            try:
                docking_results = run_docking(
                    protonation_states=protonation_states,
                    receptor_path=receptor_path
                )
            except Exception as e:
                print(f"Real docking failed, using mock: {e}")
                docking_results = {
                    "best_score": -8.5,
                    "poses": [
                        {"state": i, "score": -8.5 + i * 0.2}
                        for i in range(len(protonation_states))
                    ],
                    "error": f"Docking failed: {str(e)}"
                }
        else:
            # Mock docking results
            import random
            docking_results = {
                "best_score": -8.5,
                "poses": [
                    {"state": i, "score": -8.5 + i * 0.2 + random.uniform(-0.5, 0.5)}
                    for i in range(len(protonation_states))
                ],
                "receptor_used": request.receptor_id or "mock"
            }
        
        # Compile results
        results = {
            "molecule_info": {
                "smiles": mol_data.get('smiles', ''),
                "name": mol_data.get('name', 'Unknown'),
                "molecular_weight": mol_data.get('molecular_weight', 0)
            },
            "pka_predictions": {
                "global_pka": pka_results.get('predicted_pka', None),
                "site_pkas": pka_results.get('site_pkas', []),
                "confidence": pka_results.get('confidence', 0)
            },
            "protonation_states": [
                {
                    "state_id": i,
                    "smiles": state.get('smiles', ''),
                    "probability": state.get('probability', 0),
                    "confidence": state.get('probability', 0),  # Use probability as confidence
                    "charge": state.get('charge', 0)
                }
                for i, state in enumerate(protonation_states)
            ],
            "docking_results": docking_results,
            "conformers_generated": len(conformers)
        }
        
        # Update job completion
        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["progress"] = 1.0
        jobs_db[job_id]["results"] = results
        jobs_db[job_id]["completed_at"] = datetime.now()
        
    except Exception as e:
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        jobs_db[job_id]["completed_at"] = datetime.now()

@app.get("/api/example-molecules")
async def get_example_molecules():
    """Get example molecules for testing"""
    return [
        {"name": "Aspirin", "smiles": "CC(=O)Oc1ccccc1C(=O)O"},
        {"name": "Ibuprofen", "smiles": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"},
        {"name": "Caffeine", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
        {"name": "Dopamine", "smiles": "NCCc1ccc(O)c(O)c1"},
        {"name": "Serotonin", "smiles": "NCCc1c[nH]c2ccc(O)cc12"}
    ]

@app.get("/api/receptors/search")
async def search_receptors(query: str, limit: int = 20):
    """Search RCSB PDB database for receptors"""
    if not query or len(query) < 2:
        return []

    try:
        # Search RCSB PDB using their REST API
        search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

        # Build search query - simple text search
        search_payload = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {
                    "value": query
                }
            },
            "return_type": "entry",
            "request_options": {
                "paginate": {
                    "start": 0,
                    "rows": limit
                }
            }
        }

        response = requests.post(search_url, json=search_payload, timeout=10)

        # Debug logging
        if response.status_code != 200:
            print(f"PDB API Error: {response.status_code}")
            print(f"Response: {response.text}")

        response.raise_for_status()
        search_results = response.json()

        # Get PDB IDs from search results
        pdb_ids = [item["identifier"] for item in search_results.get("result_set", [])][:limit]

        if not pdb_ids:
            return []

        # Fetch detailed information for each PDB entry
        results = []
        for pdb_id in pdb_ids:
            try:
                info_url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
                info_response = requests.get(info_url, timeout=5)
                info_response.raise_for_status()
                info_data = info_response.json()

                # Extract relevant metadata
                title = info_data.get("struct", {}).get("title", "Unknown")
                resolution = info_data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0]
                experimental_method = info_data.get("exptl", [{}])[0].get("method", "Unknown")

                results.append({
                    "pdb_id": pdb_id,
                    "title": title,
                    "resolution": f"{resolution:.2f} Å" if resolution else "N/A",
                    "method": experimental_method,
                    "description": f"{title[:100]}..." if len(title) > 100 else title
                })
            except Exception as e:
                print(f"Error fetching info for {pdb_id}: {e}")
                # Include minimal info if detailed fetch fails
                results.append({
                    "pdb_id": pdb_id,
                    "title": pdb_id,
                    "resolution": "N/A",
                    "method": "Unknown",
                    "description": pdb_id
                })

        return results

    except Exception as e:
        print(f"Error searching PDB: {e}")
        return []

@app.get("/api/receptors/{receptor_id}/pdb")
async def get_receptor_pdb(receptor_id: str):
    """Get PDB file content for a receptor - downloads from RCSB if not cached locally"""
    receptors_dir = Path(__file__).parent.parent.parent / "receptors" / "example"
    pdb_file = receptors_dir / f"{receptor_id}.pdb"

    # Check if already cached locally
    if pdb_file.exists():
        with open(pdb_file, 'r') as f:
            pdb_content = f.read()
        return {"receptor_id": receptor_id, "pdb_content": pdb_content, "source": "cache"}

    # Download from RCSB PDB
    try:
        download_url = f"https://files.rcsb.org/download/{receptor_id}.pdb"
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()
        pdb_content = response.text

        # Cache the downloaded file
        receptors_dir.mkdir(parents=True, exist_ok=True)
        with open(pdb_file, 'w') as f:
            f.write(pdb_content)

        return {"receptor_id": receptor_id, "pdb_content": pdb_content, "source": "downloaded"}

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Could not fetch PDB file for {receptor_id}: {str(e)}"
        )

@app.get("/api/jobs/{job_id}/ligand-pdb")
async def get_ligand_pdb(job_id: str, state_id: int = 0):
    """Get PDB representation of ligand for a specific protonation state"""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs_db[job_id]
    if job["status"] != "completed" or not job["results"]:
        raise HTTPException(status_code=400, detail="Job not completed or no results available")

    protonation_states = job["results"].get("protonation_states", [])
    if state_id >= len(protonation_states):
        raise HTTPException(status_code=400, detail="Invalid state_id")

    # Convert SMILES to 3D PDB using RDKit (if available)
    smiles = protonation_states[state_id].get("smiles", "")

    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise HTTPException(status_code=500, detail="Failed to parse SMILES")

        # Add hydrogens and generate 3D coordinates
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)

        # Convert to PDB format
        pdb_content = Chem.MolToPDBBlock(mol)

        return {"job_id": job_id, "state_id": state_id, "pdb_content": pdb_content}
    except ImportError:
        # Fallback if RDKit is not available
        raise HTTPException(status_code=501, detail="RDKit not available for PDB generation")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating PDB: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 