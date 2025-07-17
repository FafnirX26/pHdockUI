#!/usr/bin/env python3
"""
Enhanced data fetcher to maximize training data from multiple sources.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import requests
import logging
from typing import Dict, List, Optional
from rdkit import Chem
import time

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_integration import DataIntegration

class EnhancedDataFetcher:
    """Fetch maximum amount of pKa data from all available sources."""
    
    def __init__(self, data_dir: str = "training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def fetch_iupac_data(self) -> Optional[pd.DataFrame]:
        """Fetch IUPAC high-confidence pKa dataset (24,211 rows)."""
        print("=" * 60)
        print("Fetching IUPAC High-Confidence pKa Dataset")
        print("=" * 60)
        
        try:
            url = "https://raw.githubusercontent.com/IUPAC/Dissociation-Constants/main/iupac_high-confidence_v2_3.csv"
            
            print(f"Downloading IUPAC dataset from GitHub...")
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Save locally
            iupac_file = self.data_dir / "iupac_pka_data.csv"
            with open(iupac_file, 'w') as f:
                f.write(response.text)
            
            # Read and process
            df = pd.read_csv(iupac_file)
            print(f"✓ IUPAC dataset downloaded: {len(df):,} pKa measurements")
            print(f"✓ Unique molecules: {df.get('inchi', df.iloc[:, 0]).nunique():,}")
            print(f"✓ Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"✗ Failed to fetch IUPAC data: {e}")
            return None
    
    def fetch_sampl6_corrected(self) -> Optional[pd.DataFrame]:
        """Try to fetch SAMPL6 data with corrected URLs."""
        print("\\n" + "=" * 60)
        print("Fetching SAMPL6 pKa Dataset (Corrected URLs)")
        print("=" * 60)
        
        # Try multiple potential URLs for SAMPL6 data
        urls_to_try = [
            # Direct experimental data
            "https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa/experimental_data/pKa-experimental-data.csv",
            "https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa/experimental_data/experimental_pKa_values.csv",
            "https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa/experimental_data/pKa_experimental_values.csv",
            # Molecule information
            "https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa/molecule_ID_and_SMILES.csv",
            "https://raw.githubusercontent.com/samplchallenges/SAMPL6/master/physical_properties/pKa/SMILES.csv",
            # Reference calculations
            "https://raw.githubusercontent.com/choderalab/sampl6-physicochemical-properties/master/data/pKa_challenge/experimental_data.csv"
        ]
        
        data_files = {}
        
        for url in urls_to_try:
            try:
                print(f"Trying: {url.split('/')[-1]}")
                response = requests.get(url, timeout=15)
                
                if response.status_code == 200:
                    filename = url.split('/')[-1]
                    file_path = self.data_dir / f"sampl6_{filename}"
                    
                    with open(file_path, 'w') as f:
                        f.write(response.text)
                    
                    df = pd.read_csv(file_path)
                    data_files[filename] = df
                    print(f"  ✓ Downloaded {filename}: {len(df)} rows")
                else:
                    print(f"  ✗ {response.status_code}")
                    
            except Exception as e:
                print(f"  ✗ Error: {e}")
        
        if data_files:
            print(f"✓ SAMPL6 data found: {len(data_files)} files")
            return data_files
        else:
            print("✗ No SAMPL6 data found")
            return None
    
    def fetch_chembl_maximum(self, max_limit: int = 10000) -> Optional[pd.DataFrame]:
        """Fetch maximum amount of ChEMBL pKa data."""
        print("\\n" + "=" * 60)
        print(f"Fetching Maximum ChEMBL pKa Data (up to {max_limit:,})")
        print("=" * 60)
        
        try:
            integration = DataIntegration(str(self.data_dir))
            
            # Try with larger limits and different query strategies
            all_data = []
            batch_size = 1000
            
            for i in range(0, max_limit, batch_size):
                print(f"Fetching batch {i//batch_size + 1}: records {i:,}-{min(i+batch_size, max_limit):,}")
                
                try:
                    # Use offset parameter if available
                    batch_data = integration.query_chembl_pka_data(limit=batch_size)
                    
                    if not batch_data.empty:
                        all_data.append(batch_data)
                        print(f"  ✓ Got {len(batch_data)} records")
                        
                        # Avoid overwhelming the API
                        time.sleep(1)
                    else:
                        print(f"  ✗ No more data available")
                        break
                        
                except Exception as e:
                    print(f"  ✗ Batch failed: {e}")
                    break
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                # Remove duplicates
                combined_df = combined_df.drop_duplicates(subset=['chembl_id', 'pka_value'])
                
                print(f"✓ Total ChEMBL data: {len(combined_df):,} unique pKa measurements")
                
                # Save for reuse
                combined_df.to_csv(self.data_dir / "chembl_pka_large.csv", index=False)
                
                return combined_df
            else:
                print("✗ No ChEMBL data retrieved")
                return None
                
        except Exception as e:
            print(f"✗ Error fetching ChEMBL data: {e}")
            return None
    
    def fetch_additional_sources(self) -> Dict[str, pd.DataFrame]:
        """Fetch data from additional pKa databases."""
        print("\\n" + "=" * 60)
        print("Fetching Additional pKa Data Sources")
        print("=" * 60)
        
        additional_data = {}
        
        # Extended sources for maximum data coverage
        sources = [
            {
                'name': 'ChEMBL_extended',
                'url': 'https://www.ebi.ac.uk/chembl/api/data/activity.json',
                'params': {
                    'standard_type__in': 'pKa,pKa1,pKa2,pKb,pKi,IC50',  # Extended search
                    'limit': 5000,
                    'format': 'json'
                }
            },
            {
                'name': 'BindingDB',
                'url': 'https://www.bindingdb.org/axis2/services/BDBService/getLigandsByUniprotId',
                'method': 'post',
                'data': {'uniprot': 'P00533'}  # EGFR as example
            },
            {
                'name': 'PubChem_BioAssay',
                'url': 'https://pubchem.ncbi.nlm.nih.gov/rest/pug/assay/target/protein-name/kinase/summary/JSON',
                'params': {'list_return': 'listkey'}
            },
            {
                'name': 'ChEBI_experimental',
                'url': 'https://www.ebi.ac.uk/chebi/searchId.do',
                'params': {
                    'chebiId': '',
                    'searchCategory': 'ALL',
                    'maximumResults': 1000
                }
            }
        ]
        
        for source in sources:
            try:
                print(f"Trying {source['name']}...")
                
                if source.get('method') == 'post':
                    response = requests.post(source['url'], data=source.get('data'), timeout=30)
                else:
                    response = requests.get(source['url'], params=source.get('params'), timeout=30)
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        if 'activities' in data:
                            activities = data['activities']
                            print(f"  ✓ {source['name']}: {len(activities)} activities found")
                            additional_data[source['name']] = self._process_chembl_activities(activities)
                        elif 'InformationList' in data:  # PubChem format
                            info_list = data['InformationList']['Information']
                            print(f"  ✓ {source['name']}: {len(info_list)} assays found")
                            additional_data[source['name']] = self._process_pubchem_data(info_list)
                        else:
                            print(f"  ⚠ {source['name']}: Data found but format unknown")
                    except (ValueError, KeyError) as e:
                        print(f"  ⚠ {source['name']}: Non-JSON response or unexpected format")
                else:
                    print(f"  ✗ {source['name']}: HTTP {response.status_code}")
                    
            except requests.RequestException as e:
                print(f"  ✗ {source['name']}: Network error - {e}")
            except Exception as e:
                print(f"  ✗ {source['name']}: {e}")
        
        return additional_data
    
    def _process_chembl_activities(self, activities: list) -> pd.DataFrame:
        """Process ChEMBL activities into standardized format."""
        records = []
        for activity in activities:
            if activity.get('standard_value'):
                records.append({
                    'smiles': activity.get('canonical_smiles', ''),
                    'pka_value': activity.get('standard_value'),
                    'pka_type': activity.get('standard_type'),
                    'chembl_id': activity.get('molecule_chembl_id'),
                    'source': 'chembl_extended'
                })
        return pd.DataFrame(records)
    
    def _process_pubchem_data(self, info_list: list) -> pd.DataFrame:
        """Process PubChem data into standardized format."""
        records = []
        for info in info_list:
            # This would need more sophisticated parsing based on actual PubChem structure
            records.append({
                'assay_id': info.get('AID'),
                'title': info.get('Title', ''),
                'source': 'pubchem'
            })
        return pd.DataFrame(records)
    
    def generate_summary(self, datasets: Dict[str, pd.DataFrame]) -> None:
        """Generate summary of all fetched data."""
        print("\\n" + "=" * 70)
        print("DATA SUMMARY")
        print("=" * 70)
        
        total_molecules = 0
        total_measurements = 0
        
        for name, df in datasets.items():
            if df is not None and not df.empty:
                n_measurements = len(df)
                n_molecules = df.iloc[:, 0].nunique() if len(df.columns) > 0 else 0
                
                print(f"{name:<20} {n_measurements:>8,} measurements  {n_molecules:>8,} molecules")
                
                total_measurements += n_measurements
                total_molecules += n_molecules
        
        print("-" * 70)
        print(f"{'TOTAL':<20} {total_measurements:>8,} measurements  {total_molecules:>8,} molecules")
        print("=" * 70)
        
        if total_measurements > 1000:
            print("\\n🎉 EXCELLENT! You have substantial training data for high-accuracy models!")
            print("\\nRecommended training commands:")
            print("1. Ensemble model with all data:")
            print("   python main.py --mode train_models --model_type ensemble --data_source combined --data_limit 50000")
            print("\\n2. XGBoost with hyperparameter tuning:")
            print("   python main.py --mode train_models --model_type xgboost --data_source combined --data_limit 20000")
        elif total_measurements > 100:
            print("\\n✅ Good amount of data available for training!")
            print("\\nRecommended training commands:")
            print("1. Start with ChEMBL data:")
            print("   python main.py --mode train_models --model_type ensemble --data_source chembl --data_limit 5000")
        else:
            print("\\n⚠️  Limited data available. Will use synthetic data as primary source.")

def main():
    """Fetch all available pKa training data."""
    logging.basicConfig(level=logging.INFO)
    
    print("Enhanced pKa Data Fetcher")
    print("Maximizing training data from all sources...")
    
    fetcher = EnhancedDataFetcher()
    datasets = {}
    
    # Fetch from all sources
    iupac_data = fetcher.fetch_iupac_data()
    if iupac_data is not None:
        datasets['IUPAC'] = iupac_data
    
    sampl6_data = fetcher.fetch_sampl6_corrected()
    if sampl6_data:
        datasets.update({f'SAMPL6_{k}': v for k, v in sampl6_data.items()})
    
    chembl_data = fetcher.fetch_chembl_maximum()
    if chembl_data is not None:
        datasets['ChEMBL'] = chembl_data
    
    additional_data = fetcher.fetch_additional_sources()
    if additional_data:
        datasets.update(additional_data)
    
    # Generate summary
    fetcher.generate_summary(datasets)
    
    # Save combined dataset info
    summary_data = []
    for name, df in datasets.items():
        if df is not None and not df.empty:
            summary_data.append({
                'source': name,
                'measurements': len(df),
                'molecules': df.iloc[:, 0].nunique() if len(df.columns) > 0 else 0,
                'columns': list(df.columns) if hasattr(df, 'columns') else []
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(fetcher.data_dir / "data_sources_summary.csv", index=False)
    
    print(f"\\n📊 Summary saved to: {fetcher.data_dir / 'data_sources_summary.csv'}")

if __name__ == "__main__":
    main()