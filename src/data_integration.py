"""
Data integration module for real pKa datasets.
Provides loaders for SAMPL challenges, ChEMBL, and other experimental datasets.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import requests
import json
from urllib.parse import urljoin
import logging
from rdkit import Chem
from rdkit.Chem import Descriptors
import warnings

logger = logging.getLogger(__name__)


class DataIntegration:
    """Main class for integrating real pKa datasets into the phDock pipeline."""
    
    def __init__(self, data_dir: str = "data"):
        """
        Initialize data integration.
        
        Args:
            data_dir: Directory to store downloaded datasets
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset URLs and configurations
        self.datasets = {
            'sampl6': {
                'url': 'https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa/experimental_data/macroscopic_pKa_experimental_and_calculated.csv',
                'smiles_url': 'https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa/molecule_ID_and_SMILES.csv'
            },
            'chembl': {
                'base_url': 'https://www.ebi.ac.uk/chembl/api/data/',
                'endpoints': {
                    'molecules': 'molecule',
                    'activities': 'activity'
                }
            }
        }
    
    def download_sampl6_data(self) -> Dict[str, pd.DataFrame]:
        """
        Download SAMPL6 pKa challenge dataset.
        
        Returns:
            Dictionary containing experimental pKa values and molecule information
        """
        logger.info("Downloading SAMPL6 pKa dataset...")
        
        try:
            # Download experimental pKa values
            pka_url = self.datasets['sampl6']['url']
            pka_response = requests.get(pka_url)
            pka_response.raise_for_status()
            
            # Save to local file
            pka_file = self.data_dir / "sampl6_pka_experimental.csv"
            with open(pka_file, 'w') as f:
                f.write(pka_response.text)
            
            # Download molecule SMILES
            smiles_url = self.datasets['sampl6']['smiles_url']
            smiles_response = requests.get(smiles_url)
            smiles_response.raise_for_status()
            
            smiles_file = self.data_dir / "sampl6_molecules.csv"
            with open(smiles_file, 'w') as f:
                f.write(smiles_response.text)
            
            # Read and process the data
            pka_df = pd.read_csv(pka_file)
            smiles_df = pd.read_csv(smiles_file)
            
            logger.info(f"Downloaded SAMPL6 dataset: {len(pka_df)} pKa measurements, {len(smiles_df)} molecules")
            
            return {
                'pka_data': pka_df,
                'molecules': smiles_df
            }
            
        except requests.RequestException as e:
            logger.error(f"Failed to download SAMPL6 data: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error processing SAMPL6 data: {e}")
            return {}
    
    def query_chembl_pka_data(self, limit: int = 1000) -> pd.DataFrame:
        """
        Query ChEMBL for molecules with pKa activity data.
        
        Args:
            limit: Maximum number of molecules to retrieve
            
        Returns:
            DataFrame with ChEMBL pKa data
        """
        logger.info(f"Querying ChEMBL for pKa data (limit: {limit})...")
        
        try:
            base_url = self.datasets['chembl']['base_url']
            
            # Query for activities related to pKa
            activity_url = urljoin(base_url, 'activity.json')
            params = {
                'standard_type__in': 'pKa,pKa1,pKa2,pKb',
                'limit': limit,
                'format': 'json'
            }
            
            response = requests.get(activity_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            activities = data.get('activities', [])
            
            if not activities:
                logger.warning("No pKa activities found in ChEMBL")
                return pd.DataFrame()
            
            # Process activities into DataFrame
            records = []
            for activity in activities:
                if activity.get('standard_value') is not None:
                    record = {
                        'chembl_id': activity.get('molecule_chembl_id'),
                        'pka_value': float(activity.get('standard_value')),
                        'pka_type': activity.get('standard_type'),
                        'assay_id': activity.get('assay_chembl_id'),
                        'units': activity.get('standard_units'),
                        'relation': activity.get('standard_relation')
                    }
                    records.append(record)
            
            df = pd.DataFrame(records)
            
            # Get SMILES for these molecules
            if not df.empty:
                df = self._fetch_chembl_smiles(df)
            
            logger.info(f"Retrieved {len(df)} pKa measurements from ChEMBL")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Failed to query ChEMBL: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error processing ChEMBL data: {e}")
            return pd.DataFrame()
    
    def _fetch_chembl_smiles(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fetch SMILES strings for ChEMBL molecules.
        
        Args:
            df: DataFrame with ChEMBL IDs
            
        Returns:
            DataFrame with SMILES added
        """
        logger.info("Fetching SMILES for ChEMBL molecules...")
        
        base_url = self.datasets['chembl']['base_url']
        molecule_url = urljoin(base_url, 'molecule.json')
        
        # Get unique ChEMBL IDs
        chembl_ids = df['chembl_id'].unique()
        smiles_dict = {}
        
        # Batch fetch molecules (ChEMBL allows multiple IDs in one request)
        batch_size = 50
        for i in range(0, len(chembl_ids), batch_size):
            batch_ids = chembl_ids[i:i+batch_size]
            
            try:
                params = {
                    'molecule_chembl_id__in': ','.join(batch_ids),
                    'format': 'json'
                }
                
                response = requests.get(molecule_url, params=params)
                response.raise_for_status()
                
                data = response.json()
                molecules = data.get('molecules', [])
                
                for mol in molecules:
                    chembl_id = mol.get('molecule_chembl_id')
                    smiles = mol.get('molecule_structures', {}).get('canonical_smiles')
                    if smiles:
                        smiles_dict[chembl_id] = smiles
                        
            except requests.RequestException as e:
                logger.warning(f"Failed to fetch SMILES for batch {i}: {e}")
                continue
        
        # Add SMILES to DataFrame
        df['smiles'] = df['chembl_id'].map(smiles_dict)
        
        # Remove rows without SMILES
        initial_count = len(df)
        df = df.dropna(subset=['smiles'])
        final_count = len(df)
        
        if initial_count > final_count:
            logger.warning(f"Removed {initial_count - final_count} entries without SMILES")
        
        return df
    
    def load_local_dataset(self, file_path: str, format: str = 'csv') -> pd.DataFrame:
        """
        Load a local pKa dataset.
        
        Args:
            file_path: Path to the dataset file
            format: File format ('csv', 'sdf', 'json')
            
        Returns:
            DataFrame with pKa data
        """
        logger.info(f"Loading local dataset: {file_path}")
        
        path = Path(file_path)
        if not path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return pd.DataFrame()
        
        try:
            if format == 'csv':
                df = pd.read_csv(path)
            elif format == 'json':
                df = pd.read_json(path)
            elif format == 'sdf':
                # Basic SDF reading - would need more sophisticated parsing
                logger.warning("SDF format not fully implemented")
                return pd.DataFrame()
            else:
                logger.error(f"Unsupported format: {format}")
                return pd.DataFrame()
            
            logger.info(f"Loaded {len(df)} records from {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            return pd.DataFrame()
    
    def validate_and_clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean pKa dataset.
        
        Args:
            df: Raw pKa dataset
            
        Returns:
            Cleaned DataFrame
        """
        logger.info("Validating and cleaning pKa data...")
        
        initial_count = len(df)
        
        # Required columns
        required_cols = ['smiles', 'pka_value']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=required_cols)
        
        # Validate SMILES
        valid_smiles = []
        for idx, smiles in df['smiles'].items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                valid_smiles.append(idx)
            else:
                logger.debug(f"Invalid SMILES at index {idx}: {smiles}")
        
        df = df.loc[valid_smiles]
        
        # Validate pKa values (reasonable range)
        df = df[(df['pka_value'] >= -5) & (df['pka_value'] <= 20)]
        
        # Remove duplicates based on SMILES
        df = df.drop_duplicates(subset=['smiles'])
        
        final_count = len(df)
        logger.info(f"Data cleaning complete: {initial_count} → {final_count} records")
        
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Dict[str, Union[List, np.ndarray]]:
        """
        Prepare cleaned pKa data for model training.
        
        Args:
            df: Cleaned pKa dataset
            
        Returns:
            Dictionary with molecules, SMILES, and pKa values
        """
        logger.info("Preparing training data...")
        
        # Convert SMILES to RDKit molecules
        molecules = []
        valid_indices = []
        
        for idx, smiles in df['smiles'].items():
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                molecules.append(mol)
                valid_indices.append(idx)
        
        if not molecules:
            logger.error("No valid molecules found in dataset")
            return {}
        
        # Filter DataFrame to valid molecules
        df_valid = df.loc[valid_indices]
        
        training_data = {
            'molecules': molecules,
            'smiles': df_valid['smiles'].tolist(),
            'pka_values': df_valid['pka_value'].values,
            'molecule_ids': df_valid.index.tolist()
        }
        
        # Add additional columns if present
        optional_cols = ['chembl_id', 'pka_type', 'assay_id', 'molecule_name']
        for col in optional_cols:
            if col in df_valid.columns:
                training_data[col] = df_valid[col].tolist()
        
        logger.info(f"Prepared training data for {len(molecules)} molecules")
        return training_data
    
    def get_combined_dataset(self, sources: List[str] = None) -> Dict[str, Union[List, np.ndarray]]:
        """
        Get combined pKa dataset from multiple sources.
        
        Args:
            sources: List of data sources to combine ['sampl6', 'chembl', 'local']
            
        Returns:
            Combined training data
        """
        if sources is None:
            sources = ['sampl6']
        
        logger.info(f"Combining datasets from sources: {sources}")
        
        combined_df = pd.DataFrame()
        
        for source in sources:
            if source == 'sampl6':
                sampl_data = self.download_sampl6_data()
                if sampl_data:
                    # Process SAMPL6 data
                    pka_df = sampl_data['pka_data']
                    mol_df = sampl_data['molecules']
                    
                    # Merge pKa values with molecules
                    merged = pd.merge(pka_df, mol_df, on='ID', how='inner')
                    
                    # Rename columns to standard format
                    if 'SMILES' in merged.columns:
                        merged = merged.rename(columns={'SMILES': 'smiles'})
                    elif 'Canonical_SMILES' in merged.columns:
                        merged = merged.rename(columns={'Canonical_SMILES': 'smiles'})
                    
                    if 'pKa' in merged.columns:
                        merged = merged.rename(columns={'pKa': 'pka_value'})
                    elif 'Experimental_pKa' in merged.columns:
                        merged = merged.rename(columns={'Experimental_pKa': 'pka_value'})
                    elif 'macroscopic_pKa' in merged.columns:
                        merged = merged.rename(columns={'macroscopic_pKa': 'pka_value'})
                    
                    combined_df = pd.concat([combined_df, merged], ignore_index=True)
            
            elif source == 'chembl':
                chembl_df = self.query_chembl_pka_data()
                if not chembl_df.empty:
                    combined_df = pd.concat([combined_df, chembl_df], ignore_index=True)
        
        if combined_df.empty:
            logger.warning("No data retrieved from any source")
            return {}
        
        # Clean and validate combined data
        cleaned_df = self.validate_and_clean_data(combined_df)
        
        # Prepare for training
        training_data = self.prepare_training_data(cleaned_df)
        
        return training_data


# Convenience functions for easy integration
def load_sampl6_dataset(data_dir: str = "data") -> Dict[str, Union[List, np.ndarray]]:
    """
    Load SAMPL6 pKa dataset.
    
    Args:
        data_dir: Directory to store downloaded data
        
    Returns:
        Training data dictionary
    """
    integrator = DataIntegration(data_dir)
    return integrator.get_combined_dataset(['sampl6'])


def load_chembl_pka_dataset(data_dir: str = "data", limit: int = 1000) -> Dict[str, Union[List, np.ndarray]]:
    """
    Load ChEMBL pKa dataset.
    
    Args:
        data_dir: Directory to store downloaded data
        limit: Maximum number of molecules to retrieve
        
    Returns:
        Training data dictionary
    """
    integrator = DataIntegration(data_dir)
    return integrator.get_combined_dataset(['chembl'])


def load_combined_dataset(sources: List[str] = None, data_dir: str = "data", limit: int = None) -> Dict[str, Union[List, np.ndarray]]:
    """
    Load combined pKa dataset from multiple sources.
    
    Args:
        sources: List of data sources to combine
        data_dir: Directory to store downloaded data
        limit: Maximum number of molecules to load
        
    Returns:
        Training data dictionary
    """
    integrator = DataIntegration(data_dir)
    result = integrator.get_combined_dataset(sources)
    
    # Apply limit if specified
    if limit and result and 'molecules' in result:
        if len(result['molecules']) > limit:
            for key in result:
                if isinstance(result[key], (list, np.ndarray)) and len(result[key]) == len(result['molecules']):
                    result[key] = result[key][:limit]
    
    return result


def load_large_dataset(data_dir: str = "training_data", limit: int = None) -> Dict[str, Union[List, np.ndarray]]:
    """
    Load the large combined dataset created by batch fetcher.
    
    Args:
        data_dir: Directory containing the large dataset
        limit: Maximum number of molecules to load
        
    Returns:
        Training data dictionary
    """
    data_path = Path(data_dir)
    large_file = data_path / "combined_pka_dataset_large.csv"
    
    if not large_file.exists():
        logger.warning(f"Large dataset not found at {large_file}")
        return {}
    
    try:
        # Load the large dataset
        df = pd.read_csv(large_file)
        logger.info(f"Loading large dataset: {len(df)} records")
        
        # Apply limit if specified
        if limit and len(df) > limit:
            df = df.head(limit)
            logger.info(f"Limited to {limit} records")
        
        # Validate and clean
        integrator = DataIntegration(str(data_path))
        cleaned_df = integrator.validate_and_clean_data(df)
        
        # Prepare for training
        training_data = integrator.prepare_training_data(cleaned_df)
        
        logger.info(f"Large dataset loaded: {len(training_data.get('molecules', []))} molecules")
        return training_data
        
    except Exception as e:
        logger.error(f"Error loading large dataset: {e}")
        return {}