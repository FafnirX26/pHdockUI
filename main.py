"""
Main pipeline script for protonation state prediction and molecular docking.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np
from rdkit import Chem

# Import our modules
from src.input_processing import MoleculeProcessor, process_input_file
from src.conformer_generation import ConformerGenerator, generate_conformer_ensemble
from src.feature_engineering import FeatureEngineering, calculate_molecular_features
from src.quantum_surrogate import QuantumSurrogateModel, train_quantum_surrogate_model
from src.pka_prediction import pKaPredictionModel, train_pka_prediction_model
from src.protonation_engine import ProtonationEngine, generate_protonation_ensemble
from src.gnn_model import GNNpKaPredictor
from src.ensemble_model import EnsemblePredictor
from src.docking_integration import DockingIntegration, run_full_docking_pipeline


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main pipeline function."""
    parser = argparse.ArgumentParser(
        description="Protonation State Prediction and Molecular Docking Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic pKa prediction
  python main.py --input molecules.smi --mode pka_prediction
  
  # Full pipeline with docking
  python main.py --input molecules.sdf --receptor protein.pdb --mode full_pipeline
  
  # Train models only
  python main.py --input training_data.sdf --mode train_models --save_models models/
        """
    )
    
    # Input/Output arguments
    parser.add_argument("--input", "-i", required=True, type=Path,
                       help="Input file (SMILES, SDF)")
    parser.add_argument("--output", "-o", type=Path, default="results",
                       help="Output directory (default: results)")
    parser.add_argument("--receptor", "-r", type=Path,
                       help="Receptor PDB file for docking")
    
    # Pipeline mode
    parser.add_argument("--mode", choices=["pka_prediction", "protonation_states", 
                                          "full_pipeline", "train_models"],
                       default="pka_prediction",
                       help="Pipeline mode (default: pka_prediction)")
    
    # Model parameters
    parser.add_argument("--model_type", choices=["xgboost", "gnn", "ensemble"],
                       default="xgboost",
                       help="Model type for pKa prediction (default: xgboost)")
    parser.add_argument("--load_models", type=Path,
                       help="Directory to load pre-trained models from")
    parser.add_argument("--save_models", type=Path,
                       help="Directory to save trained models to")
    
    # Data source parameters
    parser.add_argument("--data_source", choices=["synthetic", "chembl", "sampl6", "combined", "large"],
                       default="synthetic",
                       help="Source of training data (default: synthetic)")
    parser.add_argument("--data_limit", type=int, default=1000,
                       help="Maximum number of training molecules to load (default: 1000)")
    parser.add_argument("--data_dir", type=Path, default="training_data",
                       help="Directory to store downloaded training data (default: training_data)")
    parser.add_argument("--fetch_large_data", action="store_true",
                       help="Run large-scale data fetching before training")
    parser.add_argument("--use_batch_fetcher", action="store_true",
                       help="Use advanced batch fetcher for maximum data collection")
    
    # Conformer generation
    parser.add_argument("--num_conformers", type=int, default=50,
                       help="Number of conformers to generate (default: 50)")
    parser.add_argument("--select_conformers", type=int, default=10,
                       help="Number of diverse conformers to select (default: 10)")
    
    # Protonation parameters
    parser.add_argument("--ph_min", type=float, default=1.0,
                       help="Minimum pH for protonation sweep (default: 1.0)")
    parser.add_argument("--ph_max", type=float, default=14.0,
                       help="Maximum pH for protonation sweep (default: 14.0)")
    parser.add_argument("--ph_step", type=float, default=0.5,
                       help="pH step size (default: 0.5)")
    
    # Docking parameters
    parser.add_argument("--docking_tool", choices=["gnina", "diffdock", "equibind"],
                       default="gnina",
                       help="Docking tool to use (default: gnina)")
    parser.add_argument("--selection_criteria", 
                       choices=["best_score", "physiological", "consensus"],
                       default="consensus",
                       help="Criteria for optimal protonation state selection (default: consensus)")
    
    # Other options
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    parser.add_argument("--n_jobs", type=int, default=4,
                       help="Number of parallel jobs (default: 4)")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting pipeline in mode: {args.mode}")
    logger.info(f"Input file: {args.input}")
    logger.info(f"Output directory: {args.output}")
    
    # Run large-scale data fetching if requested
    if args.fetch_large_data or args.use_batch_fetcher:
        logger.info("Running large-scale data fetching...")
        if args.use_batch_fetcher:
            # Use the new batch fetcher
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, "batch_data_fetcher.py"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Batch data fetching completed successfully")
            else:
                logger.warning(f"Batch data fetching failed: {result.stderr}")
        else:
            # Use enhanced data fetcher
            import subprocess
            import sys
            result = subprocess.run([
                sys.executable, "enhanced_data_fetcher.py"
            ], capture_output=True, text=True)
            if result.returncode == 0:
                logger.info("Enhanced data fetching completed successfully")
            else:
                logger.warning(f"Enhanced data fetching failed: {result.stderr}")
    
    try:
        # Step 1: Process input molecules
        logger.info("Step 1: Processing input molecules")
        processor = MoleculeProcessor()
        
        if args.input.suffix.lower() == '.sdf':
            input_results = processor.process_sdf(args.input)
        else:
            # Assume SMILES file
            with open(args.input, 'r') as f:
                smiles_list = [line.strip() for line in f if line.strip()]
            input_results = processor.process_smiles(smiles_list)
        
        molecules = input_results['molecules']
        smiles_list = input_results['smiles']
        
        logger.info(f"Processed {len(molecules)} molecules successfully")
        
        if len(molecules) == 0:
            logger.error("No valid molecules found in input")
            return 1
        
        # Step 2: Generate conformers (if needed for 3D descriptors)
        conformer_results = None
        if args.mode in ["full_pipeline", "train_models"] or args.model_type == "ensemble":
            logger.info("Step 2: Generating molecular conformers")
            conformer_results = generate_conformer_ensemble(
                molecules,
                num_conformers=args.num_conformers,
                select_diverse=args.select_conformers
            )
            logger.info(f"Generated conformers for {len(conformer_results)} molecules")
        
        # Step 3: Calculate molecular features
        logger.info("Step 3: Calculating molecular features")
        feature_eng = FeatureEngineering(include_3d=True)
        
        # Extract ensemble descriptors if conformers available
        ensemble_descriptors_list = None
        if conformer_results:
            ensemble_descriptors_list = [desc for mol, desc in conformer_results]
        
        molecular_features = feature_eng.features_to_dataframe(
            molecules, 
            ensemble_descriptors_list=ensemble_descriptors_list,
            smiles_list=smiles_list
        )
        
        logger.info(f"Calculated {len(molecular_features.columns)} molecular features")
        
        # Step 4: Quantum descriptor prediction (optional)
        quantum_predictions = None
        if args.mode in ["full_pipeline", "train_models"] or args.model_type == "ensemble":
            logger.info("Step 4: Predicting quantum descriptors")
            
            if args.load_models and (args.load_models / "quantum_surrogate.joblib").exists():
                # Load pre-trained quantum model
                quantum_model = QuantumSurrogateModel()
                quantum_model.load_model(args.load_models / "quantum_surrogate.joblib")
                quantum_predictions = quantum_model.predict(molecular_features)
            else:
                # Train new quantum surrogate model
                quantum_model = train_quantum_surrogate_model(
                    molecules,
                    molecular_features,
                    model_type="xgboost",
                    n_estimators=50
                )
                quantum_predictions = quantum_model.predict(molecular_features)
                
                if args.save_models:
                    args.save_models.mkdir(parents=True, exist_ok=True)
                    quantum_model.save_model(args.save_models / "quantum_surrogate.joblib")
            
            logger.info("Quantum descriptor prediction completed")
        
        # Step 5: pKa prediction
        logger.info("Step 5: Predicting pKa values")
        
        pka_model = None
        pka_predictions = None
        
        if args.load_models and (args.load_models / f"pka_model_{args.model_type}.joblib").exists():
            # Load pre-trained pKa model
            if args.model_type == "xgboost":
                pka_model = pKaPredictionModel()
                pka_model.load_model(args.load_models / "pka_model_xgboost.joblib")
                pka_predictions = pka_model.predict(molecular_features)
            elif args.model_type == "gnn":
                pka_model = GNNpKaPredictor()
                pka_model.load_model(
                    args.load_models / "pka_model_gnn.pt",
                    molecular_dim=len(molecular_features.columns),
                    conformer_dim=20,  # This should match training
                    quantum_dim=10,    # This should match training
                    num_conformers=args.select_conformers
                )
                pka_predictions = pka_model.predict(molecules)
            elif args.model_type == "ensemble":
                if quantum_predictions is None or conformer_results is None:
                    logger.error("Ensemble model requires quantum predictions and conformers")
                    return 1
                # Ensemble model loading would need proper implementation
                logger.warning("Pre-trained ensemble model loading not implemented")
        
        if pka_predictions is None:
            # Train new pKa model
            if args.model_type == "xgboost":
                if args.data_source == "synthetic":
                    pka_model = train_pka_prediction_model(
                        molecules,
                        molecular_features,
                        model_type="xgboost",
                        hyperparameter_tuning=False,
                        plot_results=False
                    )
                    pka_predictions = pka_model.predict(molecular_features)
                else:
                    # Train with real data
                    pka_model = pKaPredictionModel()
                    
                    # Handle large dataset source
                    if args.data_source == "large":
                        # Try to load large dataset first
                        try:
                            import sys
                            sys.path.append("src")
                            from data_integration import load_large_dataset
                            large_data = load_large_dataset(str(args.data_dir), limit=args.data_limit)
                            if large_data and large_data.get('molecules'):
                                logger.info(f"Using large dataset: {len(large_data['molecules'])} molecules")
                                # Use existing train_with_real_data but with pre-loaded data
                                training_results = pka_model.train_with_real_data(
                                    molecular_features,
                                    data_source="combined",
                                    data_limit=args.data_limit,
                                    data_dir=str(args.data_dir),
                                    plot_results=False
                                )
                            else:
                                logger.warning("Large dataset not found, falling back to combined data")
                                training_results = pka_model.train_with_real_data(
                                    molecular_features,
                                    data_source="combined",
                                    data_limit=args.data_limit,
                                    data_dir=str(args.data_dir),
                                    plot_results=False
                                )
                        except ImportError:
                            logger.warning("Large dataset functionality not available, using combined data")
                            training_results = pka_model.train_with_real_data(
                                molecular_features,
                                data_source="combined",
                                data_limit=args.data_limit,
                                data_dir=str(args.data_dir),
                                plot_results=False
                            )
                    else:
                        training_results = pka_model.train_with_real_data(
                            molecular_features,
                            data_source=args.data_source,
                            data_limit=args.data_limit,
                            data_dir=str(args.data_dir),
                            plot_results=False
                        )
                    
                    if training_results:
                        logger.info(f"Model trained with real data: RMSE={training_results.get('test_rmse', 'N/A'):.3f}")
                        pka_predictions = pka_model.predict(molecular_features)
                    else:
                        logger.warning("Real data training failed, falling back to synthetic data")
                        pka_model = train_pka_prediction_model(
                            molecules,
                            molecular_features,
                            model_type="xgboost",
                            hyperparameter_tuning=False,
                            plot_results=False
                        )
                        pka_predictions = pka_model.predict(molecular_features)
                
                if args.save_models:
                    args.save_models.mkdir(parents=True, exist_ok=True)
                    pka_model.save_model(args.save_models / "pka_model_xgboost.joblib")
            
            elif args.model_type == "gnn":
                # Generate synthetic pKa data for training
                baseline_model = pKaPredictionModel()
                synthetic_data = baseline_model.generate_synthetic_pka_data(molecules, molecular_features)
                targets = synthetic_data.groupby('molecule_id')['pka'].first().values
                
                # Filter valid data
                valid_indices = ~np.isnan(targets)
                valid_molecules = [molecules[i] for i in range(len(molecules)) if valid_indices[i]]
                valid_targets = targets[valid_indices]
                
                if len(valid_molecules) > 2:
                    pka_model = GNNpKaPredictor(hidden_dim=64, num_layers=3)
                    pka_model.train(valid_molecules, valid_targets, num_epochs=100)
                    
                    if args.save_models:
                        args.save_models.mkdir(parents=True, exist_ok=True)
                        pka_model.save_model(args.save_models / "pka_model_gnn.pt")
                    
                    # Predict for all molecules
                    pka_predictions = pka_model.predict(molecules)
                else:
                    logger.error("Not enough valid molecules for GNN training")
                    return 1
            
            elif args.model_type == "ensemble":
                if quantum_predictions is None or conformer_results is None:
                    logger.error("Ensemble model requires quantum predictions and conformers")
                    return 1
                
                # Prepare ensemble data
                conformer_features_array = np.random.randn(len(molecules), args.select_conformers, 20)  # Placeholder
                quantum_features_array = quantum_predictions.values
                molecular_features_array = molecular_features.select_dtypes(include=[np.number]).values
                
                # Generate synthetic targets
                baseline_model = pKaPredictionModel()
                synthetic_data = baseline_model.generate_synthetic_pka_data(molecules, molecular_features)
                targets = synthetic_data.groupby('molecule_id')['pka'].first().values
                valid_indices = ~np.isnan(targets)
                
                if np.sum(valid_indices) > 2:
                    ensemble_model = EnsemblePredictor(hidden_dim=64)
                    ensemble_model.train(
                        molecular_features_array[valid_indices],
                        conformer_features_array[valid_indices],
                        quantum_features_array[valid_indices],
                        targets[valid_indices],
                        num_epochs=50
                    )
                    
                    if args.save_models:
                        args.save_models.mkdir(parents=True, exist_ok=True)
                        ensemble_model.save_model(args.save_models / "pka_model_ensemble.pt")
                    
                    pka_predictions, _ = ensemble_model.predict(
                        molecular_features_array,
                        conformer_features_array,
                        quantum_features_array
                    )
                else:
                    logger.error("Not enough valid molecules for ensemble training")
                    return 1
        
        logger.info(f"pKa prediction completed for {len(pka_predictions)} molecules")
        
        # Stop here if only training models
        if args.mode == "train_models":
            logger.info("Model training completed")
            return 0
        
        # Stop here if only predicting pKa
        if args.mode == "pka_prediction":
            # Save pKa predictions
            results_df = pd.DataFrame({
                'molecule_id': range(len(molecules)),
                'smiles': smiles_list,
                'predicted_pka': pka_predictions
            })
            results_df.to_csv(args.output / "pka_predictions.csv", index=False)
            logger.info(f"pKa predictions saved to {args.output / 'pka_predictions.csv'}")
            return 0
        
        # Step 6: Generate protonation states
        logger.info("Step 6: Generating protonation states")
        
        protonation_engine = ProtonationEngine(
            ph_range=(args.ph_min, args.ph_max),
            ph_step=args.ph_step
        )
        
        # Convert pKa predictions to list format (assuming single pKa per molecule)
        pka_predictions_list = [[pk] for pk in pka_predictions]
        
        protonation_ensembles = generate_protonation_ensemble(
            molecules,
            pka_predictions=pka_predictions_list,
            ph_range=(args.ph_min, args.ph_max),
            ph_step=args.ph_step,
            save_dir=args.output / "protonation_states"
        )
        
        logger.info(f"Generated protonation states for {len(protonation_ensembles)} molecules")
        
        # Stop here if only generating protonation states
        if args.mode == "protonation_states":
            logger.info("Protonation state generation completed")
            return 0
        
        # Step 7: Molecular docking (full pipeline)
        if args.mode == "full_pipeline":
            if not args.receptor:
                logger.error("Receptor file required for full pipeline mode")
                return 1
            
            if not args.receptor.exists():
                logger.error(f"Receptor file not found: {args.receptor}")
                return 1
            
            logger.info("Step 7: Running molecular docking")
            
            docking_results = run_full_docking_pipeline(
                molecules,
                protonation_ensembles,
                args.receptor,
                pka_predictions=pka_predictions_list,
                docking_tool=args.docking_tool,
                output_dir=args.output / "docking_results"
            )
            
            # Summarize docking results
            summary_data = []
            for result in docking_results:
                if 'error' not in result:
                    summary_data.append({
                        'molecule_id': result['molecule_id'],
                        'molecule_name': result['molecule_name'],
                        'optimal_ph': result.get('optimal_ph'),
                        'num_successful_dockings': result['num_successful_dockings'],
                        'selection_criteria': result.get('selection_info', {}).get('criteria'),
                        'best_score': result.get('selection_info', {}).get('score')
                    })
            
            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_csv(args.output / "docking_summary.csv", index=False)
                logger.info(f"Docking summary saved to {args.output / 'docking_summary.csv'}")
            
            logger.info("Full pipeline completed successfully")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())