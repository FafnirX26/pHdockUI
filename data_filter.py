#!/usr/bin/env python3
"""
Advanced data filtering script for pKa dataset cleaning.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class pKaDataFilter:
    """Advanced pKa dataset filtering and quality control."""
    
    def __init__(self, input_file="training_data/combined_pka_dataset_large.csv"):
        self.input_file = Path(input_file)
        self.df = None
        self.filtered_df = None
        self.filter_stats = {}
        
    def load_data(self):
        """Load the raw dataset."""
        print("📊 Loading dataset...")
        self.df = pd.read_csv(self.input_file)
        print(f"✓ Loaded {len(self.df)} records")
        print(f"📈 pKa range: {self.df['pka_value'].min():.1f} to {self.df['pka_value'].max():.1f}")
        
        # Store initial stats
        self.filter_stats['initial_count'] = len(self.df)
        self.filter_stats['initial_pka_range'] = (self.df['pka_value'].min(), self.df['pka_value'].max())
        
    def analyze_data_quality(self):
        """Analyze the quality of the current dataset."""
        print("\n🔍 Data Quality Analysis")
        print("=" * 50)
        
        # Basic statistics
        print(f"Total records: {len(self.df)}")
        print(f"Unique SMILES: {self.df['smiles'].nunique()}")
        print(f"Duplicate SMILES: {len(self.df) - self.df['smiles'].nunique()}")
        
        # pKa distribution analysis
        pka_stats = self.df['pka_value'].describe()
        print(f"\npKa Statistics:")
        print(f"  Mean: {pka_stats['mean']:.2f}")
        print(f"  Std:  {pka_stats['std']:.2f}")
        print(f"  Min:  {pka_stats['min']:.2f}")
        print(f"  Max:  {pka_stats['max']:.2f}")
        
        # Check for extreme outliers (beyond typical pKa range)
        extreme_low = self.df[self.df['pka_value'] < -10]['pka_value'].count()
        extreme_high = self.df[self.df['pka_value'] > 25]['pka_value'].count()
        print(f"\nExtreme values:")
        print(f"  pKa < -10: {extreme_low} records")
        print(f"  pKa > 25:  {extreme_high} records")
        
        # Check molecular validity
        invalid_smiles = 0
        valid_smiles = 0
        mol_weights = []
        
        print(f"\n🧪 Validating molecular structures...")
        for smiles in self.df['smiles']:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                invalid_smiles += 1
            else:
                valid_smiles += 1
                mol_weights.append(Descriptors.MolWt(mol))
        
        print(f"  Valid SMILES: {valid_smiles}")
        print(f"  Invalid SMILES: {invalid_smiles}")
        
        if mol_weights:
            print(f"  Molecular weight range: {min(mol_weights):.1f} - {max(mol_weights):.1f}")
            print(f"  Mean molecular weight: {np.mean(mol_weights):.1f}")
        
        # Store analysis results
        self.filter_stats['analysis'] = {
            'unique_smiles': self.df['smiles'].nunique(),
            'duplicate_smiles': len(self.df) - self.df['smiles'].nunique(),
            'invalid_smiles': invalid_smiles,
            'extreme_low': extreme_low,
            'extreme_high': extreme_high,
            'mol_weight_range': (min(mol_weights), max(mol_weights)) if mol_weights else None
        }
        
    def filter_extreme_pka_values(self):
        """Filter out extreme pKa values that are likely errors."""
        print(f"\n🔧 Filtering extreme pKa values...")
        
        initial_count = len(self.df)
        
        # Define reasonable pKa ranges based on chemical knowledge
        # Most organic compounds have pKa between -10 and 25
        min_pka = -8.0  # Very strong acids (like mineral acids)
        max_pka = 23.0  # Very weak acids (like alkanes)
        
        print(f"  Keeping pKa values between {min_pka} and {max_pka}")
        
        self.df = self.df[
            (self.df['pka_value'] >= min_pka) & 
            (self.df['pka_value'] <= max_pka)
        ].copy()
        
        removed = initial_count - len(self.df)
        print(f"  ✓ Removed {removed} records with extreme pKa values")
        print(f"  ✓ Remaining: {len(self.df)} records")
        
        self.filter_stats['extreme_pka_removed'] = removed
        
    def filter_invalid_molecules(self):
        """Remove molecules with invalid SMILES."""
        print(f"\n🧪 Filtering invalid molecular structures...")
        
        initial_count = len(self.df)
        valid_indices = []
        
        for idx, smiles in enumerate(self.df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                # Additional validity checks
                try:
                    # Check if we can calculate basic properties
                    _ = Descriptors.MolWt(mol)
                    _ = rdMolDescriptors.CalcNumAtoms(mol)
                    valid_indices.append(idx)
                except:
                    continue
        
        self.df = self.df.iloc[valid_indices].copy()
        
        removed = initial_count - len(self.df)
        print(f"  ✓ Removed {removed} records with invalid structures")
        print(f"  ✓ Remaining: {len(self.df)} records")
        
        self.filter_stats['invalid_structures_removed'] = removed
        
    def filter_molecular_weight_outliers(self):
        """Remove molecules with extreme molecular weights."""
        print(f"\n⚖️  Filtering molecular weight outliers...")
        
        initial_count = len(self.df)
        
        # Calculate molecular weights
        mol_weights = []
        valid_indices = []
        
        for idx, smiles in enumerate(self.df['smiles']):
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mw = Descriptors.MolWt(mol)
                mol_weights.append(mw)
                valid_indices.append((idx, mw))
        
        # Filter by reasonable molecular weight range
        # Most drug-like compounds are 100-800 Da
        # We'll be more generous: 50-1500 Da
        min_mw = 50.0
        max_mw = 1500.0
        
        print(f"  Keeping molecules between {min_mw} and {max_mw} Da")
        
        filtered_indices = [idx for idx, mw in valid_indices if min_mw <= mw <= max_mw]
        self.df = self.df.iloc[filtered_indices].copy()
        
        removed = initial_count - len(self.df)
        print(f"  ✓ Removed {removed} records with extreme molecular weights")
        print(f"  ✓ Remaining: {len(self.df)} records")
        
        self.filter_stats['mw_outliers_removed'] = removed
        
    def handle_duplicates_intelligently(self):
        """Handle duplicate SMILES with statistical analysis."""
        print(f"\n🔄 Handling duplicate molecules...")
        
        initial_count = len(self.df)
        
        # Group by SMILES and analyze pKa value distributions
        grouped = self.df.groupby('smiles')['pka_value']
        
        filtered_data = []
        outlier_removed = 0
        
        for smiles, pka_values in grouped:
            pka_list = pka_values.tolist()
            
            if len(pka_list) == 1:
                # Single measurement - keep it
                filtered_data.append({'smiles': smiles, 'pka_value': pka_list[0]})
            else:
                # Multiple measurements - apply statistical filtering
                pka_array = np.array(pka_list)
                
                if len(pka_list) == 2:
                    # Two measurements - check if they're reasonably close
                    if abs(pka_array[0] - pka_array[1]) <= 2.0:  # Within 2 pKa units
                        # Take the average
                        filtered_data.append({'smiles': smiles, 'pka_value': np.mean(pka_array)})
                    else:
                        # Too different - keep the median
                        filtered_data.append({'smiles': smiles, 'pka_value': np.median(pka_array)})
                        outlier_removed += 1
                else:
                    # Multiple measurements - use robust statistics
                    # Remove outliers using IQR method
                    Q1 = np.percentile(pka_array, 25)
                    Q3 = np.percentile(pka_array, 75)
                    IQR = Q3 - Q1
                    
                    if IQR > 0:
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR
                        
                        # Filter outliers
                        filtered_pka = pka_array[
                            (pka_array >= lower_bound) & 
                            (pka_array <= upper_bound)
                        ]
                        
                        if len(filtered_pka) > 0:
                            # Use median of filtered values
                            filtered_data.append({'smiles': smiles, 'pka_value': np.median(filtered_pka)})
                            outlier_removed += len(pka_list) - len(filtered_pka)
                        else:
                            # All were outliers - use original median
                            filtered_data.append({'smiles': smiles, 'pka_value': np.median(pka_array)})
                    else:
                        # No variation - use mean
                        filtered_data.append({'smiles': smiles, 'pka_value': np.mean(pka_array)})
        
        # Create new dataframe
        self.df = pd.DataFrame(filtered_data)
        
        removed = initial_count - len(self.df)
        print(f"  ✓ Processed {grouped.ngroups} unique molecules")
        print(f"  ✓ Removed {removed} duplicate records")
        print(f"  ✓ Filtered {outlier_removed} statistical outliers")
        print(f"  ✓ Final dataset: {len(self.df)} records")
        
        self.filter_stats['duplicates_removed'] = removed
        self.filter_stats['statistical_outliers_removed'] = outlier_removed
        
    def apply_statistical_outlier_detection(self):
        """Apply statistical outlier detection to pKa values."""
        print(f"\n📊 Statistical outlier detection...")
        
        initial_count = len(self.df)
        
        # Use Modified Z-score method (more robust than standard Z-score)
        pka_values = self.df['pka_value'].values
        median_pka = np.median(pka_values)
        mad = np.median(np.abs(pka_values - median_pka))
        
        # Modified Z-score
        modified_z_scores = 0.6745 * (pka_values - median_pka) / mad
        
        # Threshold for outliers (typically 3.5)
        threshold = 3.5
        outlier_mask = np.abs(modified_z_scores) < threshold
        
        self.df = self.df[outlier_mask].copy()
        
        removed = initial_count - len(self.df)
        print(f"  ✓ Removed {removed} statistical outliers (Modified Z-score > {threshold})")
        print(f"  ✓ Remaining: {len(self.df)} records")
        
        self.filter_stats['statistical_outliers_final'] = removed
        
    def create_visualization(self, output_dir="results/"):
        """Create visualizations of the filtering process."""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n📊 Creating visualizations...")
        
        # pKa distribution
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.hist(self.df['pka_value'], bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('pKa Value')
        plt.ylabel('Frequency')
        plt.title(f'Filtered pKa Distribution (n={len(self.df)})')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.boxplot(self.df['pka_value'])
        plt.ylabel('pKa Value')
        plt.title('pKa Distribution (Box Plot)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / 'filtered_pka_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved distribution plot: {output_path / 'filtered_pka_distribution.png'}")
        
    def save_filtered_data(self, output_file="training_data/filtered_pka_dataset.csv"):
        """Save the filtered dataset."""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        # Add data source tracking
        self.df['filtered'] = True
        self.df['filter_version'] = '1.0'
        
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Filtered dataset saved: {output_path}")
        print(f"   Records: {len(self.df)}")
        print(f"   pKa range: {self.df['pka_value'].min():.2f} to {self.df['pka_value'].max():.2f}")
        
        # Save filtering statistics (convert numpy types to Python types)
        stats_file = output_path.parent / "filtering_stats.json"
        import json
        
        # Convert numpy types to Python types for JSON serialization
        json_stats = {}
        for key, value in self.filter_stats.items():
            if isinstance(value, (np.integer, np.floating)):
                json_stats[key] = value.item()
            elif isinstance(value, tuple) and len(value) == 2:
                json_stats[key] = [float(value[0]), float(value[1])]
            else:
                json_stats[key] = value
        
        with open(stats_file, 'w') as f:
            json.dump(json_stats, f, indent=2)
        
        print(f"   Filtering stats saved: {stats_file}")
        
    def print_summary(self):
        """Print filtering summary."""
        print(f"\n🎉 Filtering Complete!")
        print("=" * 50)
        
        initial = self.filter_stats['initial_count']
        final = len(self.df)
        total_removed = initial - final
        
        print(f"Initial dataset: {initial:,} records")
        print(f"Final dataset:   {final:,} records")
        print(f"Total removed:   {total_removed:,} records ({100*total_removed/initial:.1f}%)")
        
        print(f"\nFiltering breakdown:")
        for key, value in self.filter_stats.items():
            if key.endswith('_removed'):
                step_name = key.replace('_', ' ').title()
                print(f"  {step_name}: {value:,} records")
        
        print(f"\nFinal pKa range: {self.df['pka_value'].min():.2f} to {self.df['pka_value'].max():.2f}")
        print(f"Final mean pKa: {self.df['pka_value'].mean():.2f} ± {self.df['pka_value'].std():.2f}")
        
    def run_full_filtering(self):
        """Run the complete filtering pipeline."""
        print("🎯 Advanced pKa Dataset Filtering")
        print("=" * 50)
        
        self.load_data()
        self.analyze_data_quality()
        self.filter_extreme_pka_values()
        self.filter_invalid_molecules()
        self.filter_molecular_weight_outliers()
        self.handle_duplicates_intelligently()
        self.apply_statistical_outlier_detection()
        self.create_visualization()
        self.save_filtered_data()
        self.print_summary()

def main():
    """Main filtering function."""
    filter_obj = pKaDataFilter()
    filter_obj.run_full_filtering()
    return 0

if __name__ == "__main__":
    exit(main())