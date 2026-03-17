import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


def preprocess_application_data(input_file='datasets/application_data.csv', 
                                output_file='datasets/application_data.csv'):
    """
    Preprocess application_data.csv for XGBoost model.
    
    Operations:
    1. Load the dataset
    2. Convert categorical variables to numerical using LabelEncoder
    3. Move TARGET column to first position
    4. Save the transformed dataset
    
    Args:
        input_file: Path to input CSV file
        output_file: Path to save transformed CSV file
    """
    
    print(f"\n{'='*80}")
    print("PREPROCESSING: application_data.csv")
    print(f"{'='*80}\n")
    
    # Load data
    print("Step 1: Loading data...")
    df = pd.read_csv(input_file)
    print(f"  ✓ Loaded shape: {df.shape}")
    print(f"  ✓ Columns: {df.shape[1]}")
    print(f"  ✓ Rows: {df.shape[0]}")
    
    # Display initial info
    print(f"\nInitial data info:")
    print(f"  Data types:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"    - {dtype}: {count} columns")
    
    # Identify categorical and numerical columns
    print(f"\nStep 2: Identifying categorical columns...")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"  ✓ Categorical columns: {len(categorical_cols)}")
    print(f"    {categorical_cols[:10]}..." if len(categorical_cols) > 10 else f"    {categorical_cols}")
    print(f"  ✓ Numerical columns: {len(numerical_cols)}")
    
    # Convert categorical variables to numerical
    print(f"\nStep 3: Converting categorical to numerical...")
    
    label_encoders = {}
    for col in categorical_cols:
        # Skip TARGET column (will handle separately)
        if col == 'is_fraud':
            continue
        
        # Count unique values before encoding
        n_unique = df[col].nunique()
        
        # Use LabelEncoder
        le = LabelEncoder()
        
        # Handle missing values: NaN gets encoded as -1 before encoding
        # LabelEncoder will then assign it a numeric value
        df[col] = df[col].astype(str)  # Convert to string to handle NaN uniformly
        
        # Fit and transform
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        
        print(f"    ✓ {col:40s} ({n_unique:4d} unique values)")
    
    # Rearrange columns: TARGET first, then all others
    print(f"\nStep 4: Rearranging columns...")
    
    if 'is_fraud' in df.columns:
        # Move TARGET to first position
        columns = ['is_fraud'] + [col for col in df.columns if col != 'is_fraud']
        df = df[columns]
        print(f"  ✓ TARGET moved to first position")
    else:
        print(f"  ⚠️  WARNING: TARGET column not found!")
    
    # Display final info
    print(f"\nStep 5: Verifying transformation...")
    print(f"  ✓ Final shape: {df.shape}")
    print(f"  ✓ First column: {df.columns[0]}")
    print(f"  ✓ Data types after transformation:")
    dtype_counts = df.dtypes.value_counts()
    for dtype, count in dtype_counts.items():
        print(f"    - {dtype}: {count} columns")
    
    print(f"\n  ✓ First few rows (columns 0-5):")
    print(df.iloc[:3, :6].to_string())
    
    # Check for missing values
    missing_count = df.isnull().sum().sum()
    print(f"\n  ✓ Total missing values: {missing_count}")
    
    # Save transformed data
    print(f"\nStep 6: Saving transformed data...")
    df.to_csv(output_file, index=False)
    print(f"  ✓ Saved to: {output_file}")
    print(f"  ✓ File size: {os.path.getsize(output_file) / 1024 / 1024:.2f} MB")
    
    print(f"\n{'='*80}")
    print("✓ PREPROCESSING COMPLETE")
    print(f"{'='*80}\n")
    
    return df, label_encoders


def verify_processed_data(csv_file='datasets/application_data.csv'):
    """
    Verify that processed data is in correct format for XGBoost.
    
    Args:
        csv_file: Path to processed CSV file
    """
    print(f"\n{'='*80}")
    print("VERIFYING PROCESSED DATA")
    print(f"{'='*80}\n")
    
    df = pd.read_csv(csv_file)
    
    print(f"Dataset Info:")
    print(f"  Shape: {df.shape}")
    print(f"  First column (Target): {df.columns[0]}")
    
    # Check data types
    all_numeric = df.dtypes.isin([np.int64, np.float64]).all()
    print(f"  All columns numeric: {all_numeric}")
    
    # Check for non-numeric columns
    non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"  ⚠️  Non-numeric columns found: {non_numeric}")
    else:
        print(f"  ✓ All columns are numeric")
    
    # Check TARGET column
    target_col = df.iloc[:, 0]
    print(f"\nTarget Column ({df.columns[0]}) Stats:")
    print(f"  Unique values: {target_col.nunique()}")
    print(f"  Value distribution:\n{target_col.value_counts()}")
    print(f"  Min: {target_col.min()}, Max: {target_col.max()}")
    
    # Check for any remaining object columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    if object_cols:
        print(f"\n⚠️  WARNING: Object columns still exist: {object_cols}")
    else:
        print(f"\n✓ No object columns remaining")
    
    # Sample data
    print(f"\nSample Data (first 3 rows, first 8 columns):")
    print(df.iloc[:3, :8].to_string())
    
    print(f"\n{'='*80}")
    print("✓ VERIFICATION COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Run preprocessing
    input_path = 'datasets/synthetic_fraud_dataset.csv'
    output_path = 'datasets/synthetic_fraud_dataset.csv'
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        exit(1)
    
    # Preprocess the data
    df, encoders = preprocess_application_data(input_path, output_path)
    
    # Verify the processed data
    verify_processed_data(output_path)
    
    print("\n✓ Data preprocessing and verification complete!")
    print(f"Processed file ready for XGBoost model at: {output_path}")
