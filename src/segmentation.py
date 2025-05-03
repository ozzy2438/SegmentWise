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
    
    # Performans optimizasyonu: Büyük veri setleri için örnekleme yapalım
    sample_size = 10000  # Maksimum örneklem boyutu
    if data.shape[0] > sample_size:
        print(f"Data size ({data.shape[0]} rows) is large. Using {sample_size} random samples for optimal K calculation.")
        # Rastgele örnekleme
        sample_indices = np.random.choice(data.shape[0], sample_size, replace=False)
        data_sample = data[sample_indices, :]
    else:
        data_sample = data
    
    from sklearn.metrics import silhouette_score
    
    k_range = range(2, max_k + 1)
    inertia_values = []
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(data_sample)
        inertia_values.append(kmeans.inertia_)
        
        # Silhouette score hesaplaması (daha yavaş ama daha doğru bir metrik)
        cluster_labels = kmeans.labels_
        silhouette_avg = silhouette_score(data_sample, cluster_labels)
        silhouette_scores.append(silhouette_avg)
        
        print(f"  K={k}, Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_avg:.4f}")
    
    # Optimal K'yı silhouette score'a göre belirleyelim (daha yüksek daha iyi)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal K based on Silhouette Score: {optimal_k_silhouette} (Score: {max(silhouette_scores):.4f})")
    
    return optimal_k_silhouette

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
def run_segmentation_pipeline(df, detected_cols, user_k=None):
    """Runs the full segmentation pipeline with preprocessing and clustering."""
    print("--- Starting Segmentation Pipeline ---")
    
    # --- Feature Selection & Preparation ---
    # 1. Check if RFM features are available, otherwise use other numeric features
    use_rfm = all(col in df.columns for col in ["Recency", "Frequency", "MonetaryValue"])
    
    if use_rfm:
        print("Using RFM features for segmentation.")
        selected_features = ["Recency", "Frequency", "MonetaryValue"]
        numeric_features = selected_features.copy()
        categorical_features = []
    else:
        print("RFM features not fully available. Using other detected numeric and categorical features.")
        
        # Filter for numeric features
        numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
        
        # Filter for categorical features (with reasonable cardinality)
        categorical_cols = []
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                # Check cardinality (too many unique values can cause dimensionality issues)
                n_unique = df[col].nunique()
                if n_unique < 20:  # Arbitrary threshold, adjust based on domain knowledge
                    categorical_cols.append(col)
                else:
                    print(f"Skipping categorical feature {col} due to high cardinality ({n_unique}).")
        
        # Skip ID-like columns
        id_keywords = ["id", "customer", "cust", "user", "member", "transaction", "order"]
        
        # Segment için uygun sayısal ve kategorik özellikleri seç
        numeric_features = []
        for col in numeric_cols:
            col_lower = col.lower()
            # ID-like kolonları atla
            if any(keyword in col_lower for keyword in id_keywords):
                continue
            numeric_features.append(col)
        
        categorical_features = categorical_cols
        
        # Combine features for preprocessing
        selected_features = numeric_features + categorical_features
        
        if not selected_features:
            raise ValueError("No suitable features found for segmentation. Please check your data.")
    
    print(f"Selected features for segmentation: {selected_features}")
    print(f"Numeric features: {numeric_features}")
    print(f"Categorical features: {categorical_features}")
    
    # Check if customer ID column is present (needed for later)
    customer_id_col = None
    if "CustomerID" in df.columns:
        customer_id_col = "CustomerID"
    elif detected_cols.get("customer_id") and detected_cols["customer_id"] in df.columns:
        customer_id_col = detected_cols["customer_id"]
    else:
        # Try to find a column with "id" in the name
        id_cols = [col for col in df.columns if "id" in col.lower()]
        if id_cols:
            customer_id_col = id_cols[0]
            print(f"Using {customer_id_col} as customer identifier.")
        else:
            print("Warning: Customer ID column not found or not selected. Segmentation will proceed without it.")
    
    # --- Preprocessing ---
    # Create a preprocessing pipeline
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    
    # Numeric transformation: scaling
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Categorical transformation: one-hot encoding 
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine transformers in a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop other columns
    )
    
    # Select features from the dataframe and apply preprocessing
    X = df[selected_features].copy()
    
    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    print(f"Data shape after preprocessing: {X_processed.shape}")
    
    # --- Clustering ---
    # Determine optimal k if not provided by user
    if user_k:
        k = user_k
        print(f"Using user-specified k={k} for clustering")
    else:
        # Optimal k finding (you can adjust the max k range)
        max_k = min(8, X_processed.shape[0] // 2)  # Reasonable upper limit
        print(f"Finding optimal K in range 2 to {max_k}...")
        k = find_optimal_k(X_processed, max_k=max_k)
    
    print(f"Selected Optimal K: {k}")
    
    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_processed)
    
    # Add cluster label to the original dataframe
    df_result = df.copy()
    df_result['Segment'] = clusters
    
    print(f"Segmentation complete. {k} clusters found.")
    print("--- Segmentation Pipeline Complete ---")
    
    # Return the results along with the model and preprocessing info for later use
    return df_result, kmeans, preprocessor, selected_features, k

# --- Visualization Functions ---
def create_segment_visualizations(df, segment_col='Segment', feature_cols=None):
    """Creates enhanced visualizations for segment analysis"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import BytesIO
    import base64
    
    visualizations = {}
    
    # If no feature columns are provided, use numeric columns
    if not feature_cols:
        feature_cols = df.select_dtypes(include=['number']).columns.tolist()
        # Remove segment column if it's in feature_cols
        if segment_col in feature_cols:
            feature_cols.remove(segment_col)
    
    if len(feature_cols) < 2:
        print("Not enough numeric features for visualization")
        return visualizations
    
    # 1. Scatter Plot Matrix - Up to 4 dimensions
    if len(feature_cols) >= 2:
        plot_features = feature_cols[:min(4, len(feature_cols))]
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter_data = df[plot_features + [segment_col]].copy()
        pairs_grid = sns.pairplot(scatter_data, hue=segment_col, palette='viridis', 
                                height=2.5, diag_kind='kde')
        
        buf = BytesIO()
        pairs_grid.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['scatter_matrix'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(pairs_grid.fig)
    
    # 2. 3D Scatter Plot - If at least 3 dimensions
    if len(feature_cols) >= 3:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        scatter = ax.scatter(
            df[feature_cols[0]], 
            df[feature_cols[1]], 
            df[feature_cols[2]],
            c=df[segment_col], 
            cmap='viridis', 
            s=30, 
            alpha=0.7
        )
        
        ax.set_xlabel(feature_cols[0])
        ax.set_ylabel(feature_cols[1])
        ax.set_zlabel(feature_cols[2])
        ax.set_title('3D Segment Visualization')
        
        # Add a colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label('Segment')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['scatter_3d'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    # 3. Cluster Centers Visualization (requires trained kmeans model)
    # This would be implemented when kmeans model is passed to the function
    
    # 4. Segment Distribution
    fig, ax = plt.subplots(figsize=(8, 6))
    segment_counts = df[segment_col].value_counts().sort_index()
    colors = plt.cm.viridis(np.linspace(0, 1, len(segment_counts)))
    
    bars = ax.bar(segment_counts.index.astype(str), segment_counts.values, color=colors)
    ax.set_xlabel('Segment')
    ax.set_ylabel('Müşteri Sayısı')
    ax.set_title('Segment Dağılımı')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height}', ha='center', va='bottom')
    
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    visualizations['segment_distribution'] = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    
    # 5. Feature Importance/Correlation Heatmap
    if len(feature_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[feature_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        ax.set_title('Feature Correlation Heatmap')
        
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        visualizations['correlation_heatmap'] = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
    
    return visualizations

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

