# test_pipeline.py
import sys
import os
import pandas as pd
import json
import plotly

# Add src directory to Python path to import modules
sys.path.insert(0, 	"/home/ubuntu/customer_segmentation_project"	)

from src.data_processor import process_data
from src.segmentation import run_segmentation_pipeline
from src.recommendation import create_segment_profiles, get_openai_recommendations
from src.app import generate_segment_plot # Import plot function from app

# --- Configuration ---
sample_csv_path = 	"/home/ubuntu/upload/sales_data_sample.csv"	
output_dir = 	"/home/ubuntu/customer_segmentation_project/test_results"	
os.makedirs(output_dir, exist_ok=True)

# --- Main Test Execution ---
def run_test():
    print(f"--- Starting Test Pipeline with: {sample_csv_path} ---")
    all_results = {}
    try:
        # 1. Process Data
        print("\nStep 1: Processing Data...")
        processed_df, detected_cols = process_data(sample_csv_path)
        print(f"Data processing complete. Shape: {processed_df.shape}")
        print("Detected Columns:", detected_cols)
        processed_df.to_csv(os.path.join(output_dir, 	"processed_data.csv"	), index=False)
        all_results[	"processed_data_head"	] = processed_df.head().to_dict()
        all_results[	"detected_columns"	] = detected_cols

        # 2. Run Segmentation
        print("\nStep 2: Running Segmentation...")
        segmentation_results_df, kmeans_model, preprocessor, feature_names, k = run_segmentation_pipeline(processed_df, detected_cols)
        print(f"Segmentation complete. Found {k} segments.")
        print("Segment Counts:", segmentation_results_df[	"Segment"	].value_counts().to_dict())
        segmentation_results_df.to_csv(os.path.join(output_dir, 	"segmentation_results.csv"	), index=False)
        all_results[	"segmentation_results_head"	] = segmentation_results_df.head().to_dict()
        all_results[	"k"	] = k
        all_results[	"segment_counts"	] = segmentation_results_df[	"Segment"	].value_counts().to_dict()

        # 3. Create Segment Profiles
        print("\nStep 3: Creating Segment Profiles...")
        profiles = create_segment_profiles(segmentation_results_df, detected_cols)
        print(f"Created profiles for {len(profiles)} segments.")
        with open(os.path.join(output_dir, 	"segment_profiles.json"	), 	"w"	) as f:
            json.dump(profiles, f, indent=2)
        all_results[	"profiles"	] = profiles

        # 4. Get Recommendations (will be dummy)
        print("\nStep 4: Getting Recommendations (Dummy)...")
        recommendations = get_openai_recommendations(profiles)
        print(f"Generated recommendations for {len(recommendations)} segments.")
        with open(os.path.join(output_dir, 	"recommendations.json"	), 	"w"	) as f:
            json.dump(recommendations, f, indent=2)
        all_results[	"recommendations"	] = recommendations

        # 5. Generate Plot
        print("\nStep 5: Generating Plot...")
        plot_json = None
        # Check if RFM columns exist for the default plot
        rfm_cols_present = all(col in segmentation_results_df.columns for col in [	"Recency"	, 	"Frequency"	, 	"MonetaryValue"	])
        if rfm_cols_present:
            plot_json = generate_segment_plot(segmentation_results_df, segment_col=	"Segment"	, 
                                             x_col=	"Recency"	, y_col=	"Frequency"	, color_col=	"MonetaryValue"	)
            if plot_json:
                print("Plot generated successfully.")
                with open(os.path.join(output_dir, 	"plot.json"	), 	"w"	) as f:
                    f.write(plot_json)
                all_results[	"plot_generated"	] = True
            else:
                print("Plot generation failed.")
                all_results[	"plot_generated"	] = False
        else:
            print("Skipping plot generation (RFM columns not found).")
            all_results[	"plot_generated"	] = False
            
        print("\n--- Test Pipeline Completed Successfully ---")
        return True, all_results

    except Exception as e:
        import traceback
        error_message = f"Error during test pipeline: {str(e)}\n{traceback.format_exc()}"
        print(f"\n--- Test Pipeline Failed --- \n{error_message}")
        all_results[	"error"	] = error_message
        return False, all_results

if __name__ == 	"__main__"	:
    success, final_results = run_test()
    # Save overall results summary
    with open(os.path.join(output_dir, 	"test_summary.json"	), 	"w"	) as f:
        # Convert non-serializable items (like DataFrames) if necessary, though we saved heads as dicts
        json.dump(final_results, f, indent=2, default=str) 
        
    if success:
        print(f"\nTest results saved in {output_dir}")
    else:
        print(f"\nTest failed. Check logs and {output_dir}/test_summary.json for details.")

