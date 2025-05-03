# src/segmentation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
# Import OpenAI client if needed later for K determination
# from openai import OpenAI

# Placeholder for OpenAI API Key - Should be set via environment variable or config
# OPENAI_API_KEY = "YOUR_API_KEY"

def select_features_for_segmentation(df, detected_cols):
    """Selects appropriate features for segmentation based on detected columns and data types."""
    potential_features = []
    numeric_features = []
    categorical_features = []

    # Prioritize RFM if available
    rfm_cols = [detected_cols.get("recency"), detected_cols.get("frequency"), detected_cols.get("monetary_value")]
    rfm_cols = [col for col in rfm_cols if col and col in df.columns] # Filter out None or missing columns

    if len(rfm_cols) == 3:
        print("Using RFM features for segmentation.")
        numeric_features.extend([detected_cols["recency"], detected_cols["frequency"], detected_cols["monetary_value"]])
        potential_features.extend(numeric_features)
    else:
        print("RFM features not fully available. Using other detected numeric and categorical features.")
        # Use other numeric/categorical columns if RFM is not complete
        for key, col_name in detected_cols.items():
            if col_name in df.columns and key not in ["customer_id", "transaction_date"]: # Exclude ID and date
                if pd.api.types.is_numeric_dtype(df[col_name]):
                    numeric_features.append(col_name)
                    potential_features.append(col_name)
                elif pd.api.types.is_object_dtype(df[col_name]) or pd.api.types.is_categorical_dtype(df[col_name]):
                     # Limit cardinality for one-hot encoding
                    if df[col_name].nunique() < 50: # Threshold for OHE
                         categorical_features.append(col_name)
                         potential_features.append(col_name)
                    else:
                        print(f"Skipping categorical feature 	'{col_name}	' due to high cardinality ({df[col_name].nunique()}).")

    if not potential_features:
        raise ValueError("No suitable features found for segmentation.")

    print(f"Selected features for segmentation: {potential_features}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Return the DataFrame with only selected features + customer_id
    customer_id_col = detected_cols.get("customer_id")
    if customer_id_col and customer_id_col in df.columns:
        features_df = df[[customer_id_col] + potential_features].copy()
    else:
        # If no customer ID, segmentation can still proceed but results can't be mapped back easily
        print("Warning: Customer ID column not found or not selected. Segmentation will proceed without it.")
        features_df = df[potential_features].copy()
        # Add a temporary index if needed later, but clustering doesn't require it

    return features_df, numeric_features, categorical_features

def preprocess_features(df, numeric_features, categorical_features):
    """Applies scaling to numeric features and one-hot encoding to categorical features."""
    
    preprocessor_steps = []
    
    if numeric_features:
        numeric_transformer = StandardScaler()
        preprocessor_steps.append(("num", numeric_transformer, numeric_features))
        
    if categorical_features:
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        preprocessor_steps.append(("cat", categorical_transformer, categorical_features))
        
    if not preprocessor_steps:
        print("No features to preprocess.")
        # If only one type of feature exists and it's numeric, maybe just scale?
        if numeric_features and not categorical_features:
             scaler = StandardScaler()
             scaled_data = scaler.fit_transform(df[numeric_features])
             # Get feature names if needed later
             feature_names = numeric_features 
             return scaled_data, feature_names, scaler # Return scaler if needed later
        else:
             # Should not happen if feature selection worked
             return df.values, df.columns.tolist(), None 

    preprocessor = ColumnTransformer(transformers=preprocessor_steps)
    
    # Exclude customer_id if present before fitting/transforming
    features_to_process = df.drop(columns=[col for col in df.columns if col not in numeric_features + categorical_features], errors='ignore')
    
    processed_data = preprocessor.fit_transform(features_to_process)
    
    # Get feature names after transformation (important for interpreting results later)
    feature_names = preprocessor.get_feature_names_out()
    
    print(f"Data shape after preprocessing: {processed_data.shape}")
    return processed_data, feature_names, preprocessor

def find_optimal_k(data, max_k=10):
    """Finds the optimal number of clusters using Elbow method and Silhouette score."""
    if data.shape[0] < 2: # Need at least 2 samples for clustering
        print("Not enough data points to determine optimal K. Defaulting to K=1.")
        return 1
        
    # Ensure max_k is not greater than n_samples - 1 for silhouette score
    max_k = min(max_k, data.shape[0] - 1)
    if max_k < 2:
        print(f"Not enough data points ({data.shape[0]}) to perform meaningful clustering beyond K=1. Defaulting to K=1.")
        return 1
        
    inertias = []
    silhouette_scores = []
    k_range = range(2, max_k + 1)

    print(f"Finding optimal K in range 2 to {max_k}...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
        # Silhouette score requires at least 2 labels
        if k > 1:
            try:
                score = silhouette_score(data, kmeans.labels_)
                silhouette_scores.append(score)
                print(f"  K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={score:.4f}")
            except ValueError as e:
                 print(f"  K={k}, Inertia={kmeans.inertia_:.2f}, Could not calculate Silhouette Score: {e}")
                 silhouette_scores.append(-1) # Indicate failure
        else:
             print(f"  K={k}, Inertia={kmeans.inertia_:.2f}")

    # --- Determining Optimal K --- 
    # Simple approach: Look for max silhouette score
    optimal_k_silhouette = -1
    if silhouette_scores:
        best_silhouette_score = max(silhouette_scores)
        if best_silhouette_score > -1: # Check if calculation was successful
             optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
             print(f"Optimal K based on Silhouette Score: {optimal_k_silhouette} (Score: {best_silhouette_score:.4f})")
        else:
            print("Silhouette scores could not be reliably calculated.")

    # TODO: Implement Elbow method analysis (more complex, involves finding the 'elbow' point)
    # TODO: Integrate OpenAI suggestion for K based on data profile (requires API call)

    # Decision Logic (simple version): Prefer silhouette, fallback if needed
    if optimal_k_silhouette != -1:
        optimal_k = optimal_k_silhouette
    else:
        # Fallback logic (e.g., choose a default like 3 or 4, or use Elbow method result)
        print("Could not determine optimal K from Silhouette. Using default K=4.")
        optimal_k = 4 # Default fallback
        # Ensure default K is valid
        optimal_k = min(optimal_k, max_k) 
        optimal_k = max(optimal_k, 2) # Ensure at least 2 clusters if possible

    # User preference: Fewer segments are better
    # We could potentially adjust optimal_k downwards if it's high, e.g., max(2, optimal_k - 1)
    # For now, stick to the calculated/default optimal_k
    print(f"Selected Optimal K: {optimal_k}")
    return optimal_k

def perform_segmentation(data, k):
    """Performs K-Means clustering."""
    if data.shape[0] < k:
        print(f"Warning: Number of data points ({data.shape[0]}) is less than K ({k}). Reducing K to {data.shape[0]}.")
        k = data.shape[0]
    if k < 1:
        print("Warning: K is less than 1. Cannot perform clustering.")
        return None, None # Indicate failure
        
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(data)
    print(f"Segmentation complete. {k} clusters found.")
    return kmeans, labels

# --- Main Segmentation Function --- 
def run_segmentation_pipeline(processed_df, detected_cols):
    """Runs the full segmentation pipeline: feature selection, preprocessing, finding K, clustering."""
    print("\n--- Starting Segmentation Pipeline ---")
    
    # 1. Select Features
    features_df, numeric_features, categorical_features = select_features_for_segmentation(processed_df, detected_cols)
    
    # Store customer IDs before preprocessing if they exist
    customer_ids = None
    customer_id_col = detected_cols.get("customer_id")
    if customer_id_col and customer_id_col in features_df.columns:
        customer_ids = features_df[customer_id_col].copy()
        features_for_processing = features_df.drop(columns=[customer_id_col])
    else:
        features_for_processing = features_df
        
    if features_for_processing.empty:
        raise ValueError("No features available for preprocessing after selection.")

    # 2. Preprocess Features
    preprocessed_data, feature_names, preprocessor = preprocess_features(features_for_processing, numeric_features, categorical_features)
    
    if preprocessed_data is None or preprocessed_data.shape[0] == 0:
        raise ValueError("Preprocessing failed or resulted in empty data.")

    # 3. Find Optimal K
    # Limit max_k based on user preference for fewer segments, e.g., max 6-8? Let's use 8 for now.
    optimal_k = find_optimal_k(preprocessed_data, max_k=8)

    # 4. Perform Segmentation
    kmeans_model, labels = perform_segmentation(preprocessed_data, optimal_k)
    
    if kmeans_model is None:
        raise ValueError("Clustering failed.")

    # 5. Combine results
    results_df = processed_df.copy() # Start with the data before feature selection/preprocessing
    results_df['Segment'] = labels
    
    # Ensure customer ID is the first column if it exists
    if customer_id_col and customer_id_col in results_df.columns:
        cols = [customer_id_col] + [col for col in results_df.columns if col != customer_id_col]
        results_df = results_df[cols]
        
    print("--- Segmentation Pipeline Complete ---")
    # Return the original data with segment labels, the model, preprocessor info, etc.
    return results_df, kmeans_model, preprocessor, feature_names, optimal_k

# Example usage (integrated with data_processor output)
if __name__ == '__main__':
    # Assuming data_processor.py is in the same directory or path
    from data_processor import process_data
    import os

    # Create a dummy CSV for testing
    dummy_data = {
        'CustID': [f'C{i:03}' for i in range(1, 101)],
        'Last Purchase Date': pd.to_datetime(np.random.choice(pd.date_range('2024-01-01', '2024-05-01'), 100)),
        'Total Spend': np.random.randint(50, 1000, 100),
        'Order Count': np.random.randint(1, 20, 100),
            'Demographic_Age': np.random.randint(18, 75, 100),
            'Location': np.random.choice(['North', 'South', 'East', 'West'], 100, p=[0.25, 0.25, 0.25, 0.25])
    }
    # Simulate some missing data
    # dummy_data['Total Spend'][np.random.choice(100, 5, replace=False)] = np.nan
    # dummy_data['Location'][np.random.choice(100, 3, replace=False)] = np.nan
    
    dummy_filepath = 'dummy_segmentation_test.csv'
    pd.DataFrame(dummy_data).to_csv(dummy_filepath, index=False)

    try:
        print("--- Running Data Processor --- ")
        # Use process_data to get the dataframe ready for segmentation
        # Note: process_data calculates RFM if possible, otherwise returns cleaned demographics
        processed_df, detected = process_data(dummy_filepath)
        print("\n--- Data Processor Output (Head) ---")
        print(processed_df.head())
        
        # Run segmentation pipeline
        segmentation_results, model, preprocessor_info, f_names, k = run_segmentation_pipeline(processed_df, detected)
        
        print("\n--- Segmentation Results (Head) ---")
        print(segmentation_results.head())
        print(f"\nNumber of segments (K): {k}")
        print(f"\nSegment counts:\n{segmentation_results['Segment'].value_counts()}")

    except Exception as e:
        print(f"\nError during segmentation pipeline test: {e}")
    finally:
        # Clean up dummy file
        if os.path.exists(dummy_filepath):
            os.remove(dummy_filepath)

