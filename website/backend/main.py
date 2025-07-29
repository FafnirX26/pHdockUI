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

app = FastAPI(title="pH Docking API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job storage (replace with Redis in production)
jobs_db = {}

class JobRequest(BaseModel):
    smiles: Optional[str] = None
    sdf_content: Optional[str] = None
    ph_value: float = 7.4
    conformer_count: int = 10
    ensemble_size: int = 5
    quantum_fallback: bool = False
    docking_backend: str = "autodock"

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
    import random
    return {
        'predicted_pka': 4.2 + random.uniform(-0.5, 0.5),
        'site_pkas': [
            {'pka': 4.2 + random.uniform(-0.3, 0.3), 'atom_idx': 0},
            {'pka': 9.8 + random.uniform(-0.3, 0.3), 'atom_idx': 1}
        ],
        'confidence': 0.85 + random.uniform(-0.1, 0.1)
    }

def mock_protonate_ligand(mol, ph=7.4, pka_values=None):
    import random
    states = []
    for i in range(3):  # Generate 3 mock protonation states
        states.append({
            'smiles': f"Mock_State_{i}_SMILES",
            'probability': random.uniform(0.1, 0.8),
            'charge': random.choice([-1, 0, 1])
        })
    return states

@app.get("/")
async def root():
    return {"message": "pH Docking API is running", "modules_available": MODULES_AVAILABLE}

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
        
        # Mock docking results
        docking_results = {
            "best_score": -8.5,
            "poses": [
                {"state": i, "score": -8.5 + i * 0.2}
                for i in range(len(protonation_states))
            ]
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 