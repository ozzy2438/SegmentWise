# src/data_processor.py

import pandas as pd
import numpy as np
import re
from datetime import datetime

# --- Column Name Mapping --- 
# Define potential variations for key columns
COLUMN_MAPPING = {
    'customer_id': ['customerid', 'customer_id', 'cust_id', 'userid', 'user_id', 'memberid', 'member_id', 'id'],
    'transaction_date': ['transactiondate', 'transaction_date', 'orderdate', 'order_date', 'date', 'invoicedate', 'invoice_date', 'timestamp', 'time'],
    'amount': ['amount', 'transactionamount', 'transaction_amount', 'totalamount', 'total_amount', 'revenue', 'sales', 'price', 'value', 'monetary'],
    'recency': ['recency', 'days_since_last_purchase'],
    'frequency': ['frequency', 'transaction_count', 'order_count'],
    'monetary_value': ['monetary', 'monetaryvalue', 'total_spent', 'total_revenue'],
    # Add more potential demographic/behavioral columns as needed
    'age': ['age', 'customer_age'],
    'gender': ['gender', 'sex'],
    'city': ['city', 'location', 'address_city'],
    'country': ['country', 'address_country'],
    'product': ['product', 'product_name', 'item', 'sku'],
    'quantity': ['quantity', 'qty']
}

def normalize_column_name(col_name):
    """Converts column name to lowercase and removes non-alphanumeric characters."""
    if not isinstance(col_name, str):
        col_name = str(col_name)
    return re.sub(r'[^a-z0-9]', '', col_name.lower())

def detect_columns(df):
    """Automatically detects key columns based on predefined mapping."""
    detected_cols = {}
    normalized_df_cols = {normalize_column_name(col): col for col in df.columns}
    
    for key, potential_names in COLUMN_MAPPING.items():
        for name in potential_names:
            normalized_name = normalize_column_name(name)
            if normalized_name in normalized_df_cols:
                detected_cols[key] = normalized_df_cols[normalized_name]
                break # Found the column for this key, move to the next key
    
    # Basic validation (e.g., check if essential columns for RFM are found if needed)
    # This can be expanded later
    print(f"Detected columns: {detected_cols}") # For debugging
    return detected_cols

def load_csv(filepath):
    """Loads a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(filepath, encoding='latin1')
        except Exception as e:
            raise ValueError(f"Could not read CSV file: {e}")
    except Exception as e:
        raise ValueError(f"Error loading CSV: {e}")
    return df

def clean_data(df, detected_cols):
    """Performs basic data cleaning."""
    # Handle missing values (simple strategy: fill numeric with median, categorical with mode)
    for col_key, col_name in detected_cols.items():
        if col_name in df.columns:
            if pd.api.types.is_numeric_dtype(df[col_name]):
                median_val = df[col_name].median()
                df[col_name].fillna(median_val, inplace=True)
            elif pd.api.types.is_object_dtype(df[col_name]): # Includes strings
                mode_val = df[col_name].mode()
                if not mode_val.empty:
                    df[col_name].fillna(mode_val[0], inplace=True)
            # Handle date columns specifically if needed
            if col_key == 'transaction_date':
                # Attempt to convert to datetime, coerce errors to NaT
                df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                # Drop rows where date conversion failed
                df.dropna(subset=[col_name], inplace=True)

    # Convert amount column to numeric if detected and not already numeric
    if 'amount' in detected_cols and detected_cols['amount'] in df.columns:
        col_name = detected_cols['amount']
        if not pd.api.types.is_numeric_dtype(df[col_name]):
             # Remove currency symbols, commas, etc. and convert
            df[col_name] = df[col_name].astype(str).str.replace(r'[$,€£,]', '', regex=True)
            df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
            # Fill any NaNs created during conversion (e.g., with median)
            median_val = df[col_name].median()
            df[col_name].fillna(median_val, inplace=True)
            df.dropna(subset=[col_name], inplace=True) # Drop if still NaN after fill

    # Drop rows with missing customer ID if detected
    if 'customer_id' in detected_cols and detected_cols['customer_id'] in df.columns:
        df.dropna(subset=[detected_cols['customer_id']], inplace=True)
        # Optional: Convert CustomerID to string to handle mixed types
        df[detected_cols['customer_id']] = df[detected_cols['customer_id']].astype(str)

    print(f"Data shape after cleaning: {df.shape}") # For debugging
    return df

def calculate_rfm(df, detected_cols):
    """Calculates Recency, Frequency, Monetary value for each customer."""
    # Check if necessary columns are detected
    required_rfm_cols = ['customer_id', 'transaction_date', 'amount']
    if not all(col in detected_cols for col in required_rfm_cols):
        print("Skipping RFM calculation: Required columns (customer_id, transaction_date, amount) not detected.")
        return df # Return original df if RFM cannot be calculated

    cust_id_col = detected_cols['customer_id']
    date_col = detected_cols['transaction_date']
    amount_col = detected_cols['amount']

    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
         df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
         df.dropna(subset=[date_col], inplace=True) # Drop rows where conversion failed
         if df.empty:
             print("Skipping RFM calculation: No valid dates after conversion.")
             return df

    # Ensure amount is numeric
    if not pd.api.types.is_numeric_dtype(df[amount_col]):
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        df.dropna(subset=[amount_col], inplace=True) # Drop rows where conversion failed
        if df.empty:
             print("Skipping RFM calculation: No valid amounts after conversion.")
             return df

    print("Calculating RFM...")
    # Use the day after the most recent transaction date in the dataset as the reference point
    snapshot_date = df[date_col].max() + pd.Timedelta(days=1)

    # Aggregate data per customer
    rfm_data = df.groupby(cust_id_col).agg({
        date_col: lambda x: (snapshot_date - x.max()).days, # Recency
        cust_id_col: 'count', # Frequency (using count of transactions)
        amount_col: 'sum' # Monetary Value
    })

    # Rename columns
    rfm_data.rename(columns={
        date_col: 'Recency',
        cust_id_col: 'Frequency',
        amount_col: 'MonetaryValue'
    }, inplace=True)

    # Ensure Recency is non-negative (can happen if snapshot_date logic has edge cases)
    rfm_data['Recency'] = rfm_data['Recency'].apply(lambda x: max(0, x))

    print(f"RFM calculation complete. Shape: {rfm_data.shape}")
    # Reset index to make customer_id a column again
    rfm_data.reset_index(inplace=True)
    rfm_data.rename(columns={cust_id_col: detected_cols['customer_id']}, inplace=True) # Ensure original name is kept

    # Merge RFM back with original data (or just return RFM - depends on goal)
    # For segmentation, we often use the RFM table directly or merge selected original features back.
    # Let's return the RFM table for now, as it's the primary input for RFM-based segmentation.
    # If other features (demographics) are needed, they should be merged.

    # --- Option: Merge with demographic data --- 
    # If demographic columns were detected, merge them back.
    # Get unique customer demographics (assuming one record per customer or taking the first)
    demographic_cols_keys = ['age', 'gender', 'city', 'country'] # Add others if needed
    demographic_cols_to_merge = [detected_cols[k] for k in demographic_cols_keys if k in detected_cols and detected_cols[k] in df.columns]
    
    if demographic_cols_to_merge:
        # Ensure we have the customer ID column in the list for merging
        merge_cols = [detected_cols['customer_id']] + demographic_cols_to_merge
        customer_data = df[merge_cols].drop_duplicates(subset=[detected_cols['customer_id']]).set_index(detected_cols['customer_id'])
        rfm_data = rfm_data.set_index(detected_cols['customer_id']).join(customer_data)
        rfm_data.reset_index(inplace=True)
        print(f"Merged demographics. Shape after merge: {rfm_data.shape}")

    return rfm_data

def process_data(filepath):
    """Main function to load, detect columns, clean, and calculate RFM."""
    print(f"Processing file: {filepath}")
    df = load_csv(filepath)
    print(f"Initial data shape: {df.shape}")
    
    detected_cols = detect_columns(df)
    if not detected_cols:
        raise ValueError("Could not automatically detect essential columns. Please check CSV format.")
        
    df_cleaned = clean_data(df.copy(), detected_cols) # Use copy to avoid modifying original df during cleaning
    if df_cleaned.empty:
        raise ValueError("Data is empty after cleaning. Check input file and cleaning steps.")

    # Decide whether to calculate RFM or use other features
    # For now, prioritize RFM if possible
    if all(col in detected_cols for col in ['customer_id', 'transaction_date', 'amount']):
        processed_df = calculate_rfm(df_cleaned, detected_cols)
    else:
        print("RFM columns not fully detected. Using cleaned data directly for segmentation (potential features: demographics etc.)")
        # Select relevant columns for segmentation (e.g., detected demographics)
        segmentation_features = [detected_cols[k] for k in detected_cols if k != 'transaction_date'] # Exclude raw date
        if not segmentation_features:
             raise ValueError("No suitable features detected for segmentation after cleaning.")
        # Ensure customer ID is present if it exists
        if 'customer_id' in detected_cols and detected_cols.get('customer_id') not in segmentation_features:
             segmentation_features.insert(0, detected_cols['customer_id'])
        # If customer_id exists, use it for deduplication, otherwise use all columns
        if 'customer_id' in detected_cols:
            processed_df = df_cleaned[segmentation_features].drop_duplicates(subset=[detected_cols['customer_id']])
        else:
            # If no customer ID, we can't deduplicate on it, so just use the features
            processed_df = df_cleaned[segmentation_features].copy()

    if processed_df.empty:
        raise ValueError("No data available for segmentation after processing.")

    print(f"Final processed data shape for segmentation: {processed_df.shape}")
    # In a real scenario, this function would return the processed_df
    # For now, just return it for potential use in the next steps
    return processed_df, detected_cols

# Example usage (for testing standalone)
if __name__ == '__main__':
    # Create a dummy CSV for testing
    dummy_data = {
        'CustomerID': [1, 2, 1, 3, 2, 1, 4, 5, 4, 5],
        'TransactionDate': pd.to_datetime(['2023-01-15', '2023-01-20', '2023-02-10', '2023-02-15', '2023-03-01', '2023-03-05', '2023-01-10', '2023-02-20', '2023-03-15', '2023-03-25']),
        'Amount': [100, 150, 50, 200, 120, 80, 300, 90, 75, 110],
        'Age': [35, 45, 35, 28, 45, 35, 55, 40, 55, 40],
        'Gender': ['M', 'F', 'M', 'F', 'F', 'M', 'M', 'F', 'M', 'F']
    }
    dummy_filepath = '/home/ubuntu/dummy_sales.csv'
    pd.DataFrame(dummy_data).to_csv(dummy_filepath, index=False)
    
    try:
        processed_data, detected = process_data(dummy_filepath)
        print("\n--- Processed Data Sample ---")
        print(processed_data.head())
        print("\n--- Detected Columns ---")
        print(detected)
    except Exception as e:
        print(f"Error during standalone test: {e}")
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_filepath):
            os.remove(dummy_filepath)

