#!/usr/bin/env python3
"""
Batch data fetcher for large-scale pKa dataset collection.
Handles parallel downloads, rate limiting, and data deduplication.
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from dataclasses import dataclass
from rdkit import Chem
import json

@dataclass
class DataSource:
    """Configuration for a data source."""
    name: str
    url: str
    method: str = 'GET'
    params: Dict = None
    headers: Dict = None
    rate_limit: float = 1.0  # seconds between requests
    batch_size: int = 1000
    max_retries: int = 3

class BatchDataFetcher:
    """Advanced batch fetcher for maximizing pKa dataset size."""
    
    def __init__(self, data_dir: str = "training_data", max_workers: int = 4):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.max_workers = max_workers
        self.logger = logging.getLogger(__name__)
        
        # Configure data sources for parallel fetching
        self.data_sources = [
            DataSource(
                name="IUPAC_large",
                url="https://raw.githubusercontent.com/IUPAC/Dissociation-Constants/main/iupac_high-confidence_v2_3.csv",
                batch_size=5000
            ),
            DataSource(
                name="ChEMBL_pKa_batch1",
                url="https://www.ebi.ac.uk/chembl/api/data/activity.json",
                params={'standard_type__in': 'pKa,pKa1,pKa2', 'limit': 2000, 'offset': 0}
            ),
            DataSource(
                name="ChEMBL_pKa_batch2", 
                url="https://www.ebi.ac.uk/chembl/api/data/activity.json",
                params={'standard_type__in': 'pKa,pKa1,pKa2', 'limit': 2000, 'offset': 2000}
            ),
            DataSource(
                name="ChEMBL_pKb_batch",
                url="https://www.ebi.ac.uk/chembl/api/data/activity.json", 
                params={'standard_type__in': 'pKb,pKi', 'limit': 2000}
            ),
            DataSource(
                name="DrugBank_experimental",
                url="https://go.drugbank.com/releases/latest",
                method='HEAD'  # Just check availability
            )
        ]
    
    async def fetch_source_async(self, session: aiohttp.ClientSession, source: DataSource) -> Tuple[str, Optional[pd.DataFrame]]:
        """Asynchronously fetch data from a single source."""
        print(f"🔄 Fetching {source.name}...")
        
        try:
            # Rate limiting
            await asyncio.sleep(source.rate_limit)
            
            # Make request
            if source.method.upper() == 'GET':
                async with session.get(source.url, params=source.params) as response:
                    if response.status == 200:
                        content = await response.text()
                        df = self._process_response_content(content, source)
                        print(f"✓ {source.name}: {len(df) if df is not None else 0} records")
                        return source.name, df
                    else:
                        print(f"✗ {source.name}: HTTP {response.status}")
                        return source.name, None
            else:
                print(f"⚠ {source.name}: Method {source.method} not implemented")
                return source.name, None
                
        except Exception as e:
            print(f"✗ {source.name}: {e}")
            return source.name, None
    
    def _process_response_content(self, content: str, source: DataSource) -> Optional[pd.DataFrame]:
        """Process response content based on source type."""
        try:
            if source.name.startswith('IUPAC'):
                return pd.read_csv(pd.io.common.StringIO(content))
            elif source.name.startswith('ChEMBL'):
                data = json.loads(content)
                if 'activities' in data:
                    return self._process_chembl_activities(data['activities'])
            elif 'csv' in source.url.lower():
                return pd.read_csv(pd.io.common.StringIO(content))
            else:
                # Try JSON first, then CSV
                try:
                    data = json.loads(content)
                    return pd.DataFrame(data)
                except:
                    return pd.read_csv(pd.io.common.StringIO(content))
        except Exception as e:
            self.logger.warning(f"Failed to process {source.name}: {e}")
            return None
    
    def _process_chembl_activities(self, activities: List[Dict]) -> pd.DataFrame:
        """Convert ChEMBL activities to standardized format."""
        records = []
        for activity in activities:
            standard_value = activity.get('standard_value')
            if standard_value is not None:
                # Try to get SMILES from molecule structure
                canonical_smiles = ''
                if 'molecule_structures' in activity:
                    canonical_smiles = activity['molecule_structures'].get('canonical_smiles', '')
                elif 'canonical_smiles' in activity:
                    canonical_smiles = activity.get('canonical_smiles', '')
                
                records.append({
                    'smiles': canonical_smiles,
                    'pka_value': standard_value,  # Keep as-is for now, will be cleaned later
                    'pka_type': activity.get('standard_type', ''),
                    'chembl_id': activity.get('molecule_chembl_id', ''),
                    'units': activity.get('standard_units', ''),
                    'source': 'chembl'
                })
        return pd.DataFrame(records)
    
    async def fetch_all_sources_async(self) -> Dict[str, pd.DataFrame]:
        """Fetch data from all sources asynchronously."""
        print("🚀 Starting parallel data fetching...")
        
        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_source_async(session, source) for source in self.data_sources]
            results = await asyncio.gather(*tasks)
        
        # Organize results
        fetched_data = {}
        for name, df in results:
            if df is not None:
                fetched_data[name] = df
        
        return fetched_data
    
    def fetch_chembl_comprehensive(self, max_records: int = 50000) -> pd.DataFrame:
        """Comprehensively fetch ChEMBL pKa data with pagination."""
        print(f"🔍 Comprehensive ChEMBL fetch (target: {max_records:,} records)")
        
        all_data = []
        batch_size = 2000
        offset = 0
        
        while len(all_data) < max_records:
            try:
                print(f"  Batch {offset//batch_size + 1}: offset {offset:,}")
                
                import requests
                response = requests.get(
                    "https://www.ebi.ac.uk/chembl/api/data/activity.json",
                    params={
                        'standard_type__in': 'pKa,pKa1,pKa2,pKa3,pKb',
                        'limit': batch_size,
                        'offset': offset,
                        'format': 'json'
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    activities = data.get('activities', [])
                    
                    if not activities:
                        print("  ✓ No more data available")
                        break
                    
                    batch_df = self._process_chembl_activities(activities)
                    if not batch_df.empty:
                        all_data.append(batch_df)
                        print(f"    +{len(batch_df)} records")
                    
                    offset += batch_size
                    time.sleep(1)  # Rate limiting
                else:
                    print(f"  ✗ HTTP {response.status_code}")
                    break
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
                break
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print(f"✓ ChEMBL comprehensive: {len(combined_df):,} total records")
            return combined_df
        else:
            return pd.DataFrame()
    
    def deduplicate_and_clean(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Deduplicate and clean combined datasets."""
        print("🧹 Deduplicating and cleaning datasets...")
        
        all_dfs = []
        total_before = 0
        
        for name, df in datasets.items():
            if df is not None and not df.empty:
                total_before += len(df)
                # Standardize column names
                df = self._standardize_columns(df)
                df['data_source'] = name
                all_dfs.append(df)
        
        if not all_dfs:
            print("✗ No valid datasets to combine")
            return pd.DataFrame()
        
        # Combine all datasets
        combined_df = pd.concat(all_dfs, ignore_index=True)
        print(f"📊 Combined: {len(combined_df):,} total records from {len(all_dfs)} sources")
        
        # Clean and deduplicate
        cleaned_df = self._clean_dataset(combined_df)
        
        print(f"✨ Final dataset: {len(cleaned_df):,} records ({len(combined_df) - len(cleaned_df):,} removed)")
        
        return cleaned_df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names across different sources."""
        column_mapping = {
            # SMILES variations
            'SMILES': 'smiles',
            'Canonical_SMILES': 'smiles', 
            'canonical_smiles': 'smiles',
            'CANONICAL_SMILES': 'smiles',
            
            # pKa variations
            'pKa': 'pka_value',
            'pKa_value': 'pka_value',
            'pka': 'pka_value',
            'experimental_pKa': 'pka_value',
            'Experimental_pKa': 'pka_value',
            'macroscopic_pKa': 'pka_value',
            'standard_value': 'pka_value',
            
            # ID variations
            'ID': 'molecule_id',
            'id': 'molecule_id',
            'chembl_id': 'chembl_id',
            'molecule_chembl_id': 'chembl_id'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def _clean_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the combined dataset."""
        initial_count = len(df)
        
        # Ensure required columns exist
        if 'smiles' not in df.columns or 'pka_value' not in df.columns:
            print("⚠ Missing required columns (smiles, pka_value)")
            return pd.DataFrame()
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['smiles', 'pka_value'])
        
        # Clean and convert pKa values to numeric
        print("🔧 Converting pKa values to numeric...")
        def safe_numeric_convert(value):
            try:
                if pd.isna(value) or value in ['', 'NaN', 'null', 'None']:
                    return np.nan
                # Handle string values that might contain non-numeric characters
                if isinstance(value, str):
                    # Remove common non-numeric characters but keep decimal points and minus signs
                    cleaned = ''.join(c for c in value if c.isdigit() or c in '.-')
                    if cleaned and cleaned not in ['.', '-', '.-']:
                        return float(cleaned)
                    else:
                        return np.nan
                return float(value)
            except (ValueError, TypeError):
                return np.nan
        
        df['pka_value'] = df['pka_value'].apply(safe_numeric_convert)
        
        # Remove rows where pKa conversion failed
        df = df.dropna(subset=['pka_value'])
        
        # Validate SMILES
        print("🧪 Validating SMILES strings...")
        valid_smiles_mask = df['smiles'].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None if pd.notna(x) else False)
        df = df[valid_smiles_mask]
        
        # Validate pKa values (reasonable range)
        print("📊 Filtering pKa values to reasonable range...")
        df = df[(df['pka_value'] >= -5) & (df['pka_value'] <= 25)]
        
        # Remove duplicates based on SMILES (keep first occurrence)
        print("🔄 Removing duplicate molecules...")
        df = df.drop_duplicates(subset=['smiles'], keep='first')
        
        # Add molecule hash for tracking
        df['mol_hash'] = df['smiles'].apply(lambda x: hashlib.md5(str(x).encode()).hexdigest()[:8])
        
        print(f"✨ Cleaning complete: {initial_count:,} → {len(df):,} records")
        
        return df
    
    def save_datasets(self, datasets: Dict[str, pd.DataFrame], combined_df: pd.DataFrame) -> None:
        """Save individual and combined datasets."""
        print("💾 Saving datasets...")
        
        # Save individual datasets
        for name, df in datasets.items():
            if df is not None and not df.empty:
                filename = self.data_dir / f"{name.lower()}_data.csv"
                df.to_csv(filename, index=False)
                print(f"  ✓ {name}: {filename}")
        
        # Save combined dataset
        combined_file = self.data_dir / "combined_pka_dataset_large.csv"
        combined_df.to_csv(combined_file, index=False)
        print(f"  ✓ Combined: {combined_file}")
        
        # Generate metadata
        metadata = {
            'total_records': len(combined_df),
            'unique_molecules': combined_df['smiles'].nunique(),
            'sources': list(datasets.keys()),
            'creation_time': pd.Timestamp.now().isoformat(),
            'pka_range': [float(combined_df['pka_value'].min()), float(combined_df['pka_value'].max())],
            'data_types': combined_df['pka_type'].value_counts().to_dict() if 'pka_type' in combined_df.columns else {}
        }
        
        metadata_file = self.data_dir / "dataset_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"  ✓ Metadata: {metadata_file}")

def main():
    """Run comprehensive data fetching."""
    logging.basicConfig(level=logging.INFO)
    
    print("🎯 Batch pKa Data Fetcher - Maximizing Dataset Size")
    print("=" * 60)
    
    fetcher = BatchDataFetcher()
    
    # Method 1: Async fetching of multiple sources
    print("\n1️⃣ Parallel fetching from multiple sources...")
    async_data = asyncio.run(fetcher.fetch_all_sources_async())
    
    # Method 2: Comprehensive ChEMBL fetching
    print("\n2️⃣ Comprehensive ChEMBL data fetching...")
    chembl_large = fetcher.fetch_chembl_comprehensive(max_records=20000)
    if not chembl_large.empty:
        async_data['ChEMBL_comprehensive'] = chembl_large
    
    # Method 3: Combine and clean all data
    print("\n3️⃣ Data processing and deduplication...")
    combined_dataset = fetcher.deduplicate_and_clean(async_data)
    
    # Save everything
    print("\n4️⃣ Saving datasets...")
    fetcher.save_datasets(async_data, combined_dataset)
    
    # Final summary
    print("\n" + "=" * 60)
    print("📈 FINAL SUMMARY")
    print("=" * 60)
    
    if not combined_dataset.empty:
        total_molecules = combined_dataset['smiles'].nunique()
        total_measurements = len(combined_dataset)
        pka_range = [combined_dataset['pka_value'].min(), combined_dataset['pka_value'].max()]
        
        print(f"Total unique molecules: {total_molecules:,}")
        print(f"Total pKa measurements: {total_measurements:,}")
        print(f"pKa value range: {pka_range[0]:.1f} to {pka_range[1]:.1f}")
        
        if 'data_source' in combined_dataset.columns:
            print("\nData source breakdown:")
            source_counts = combined_dataset['data_source'].value_counts()
            for source, count in source_counts.items():
                print(f"  {source}: {count:,} records")
        
        print(f"\n🎉 SUCCESS! Large-scale dataset ready for training.")
        print(f"📁 Saved to: {fetcher.data_dir / 'combined_pka_dataset_large.csv'}")
        
        # Training recommendations
        if total_measurements > 10000:
            print("\n🚀 RECOMMENDED TRAINING COMMANDS:")
            print("python main.py --mode train_models --model_type ensemble --data_source combined --data_limit 50000")
        elif total_measurements > 1000:
            print("\n✅ GOOD DATASET SIZE - RECOMMENDED TRAINING:")
            print("python main.py --mode train_models --model_type xgboost --data_source combined --data_limit 10000")
    else:
        print("❌ No valid data collected. Check network connection and API availability.")

if __name__ == "__main__":
    main()