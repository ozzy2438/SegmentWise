 # src/recommendation.py

import pandas as pd
import numpy as np
import os
import json
from openai import OpenAI

# --- OpenAI API Key Configuration ---
# Best practice: Load API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    print("WARNING: OpenAI API key is not configured. Recommendation engine will not function.")
    client = None
else:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception as e:
        print(f"Error initializing OpenAI client: {e}")
        client = None

def create_segment_profiles(results_df, detected_cols):
    """Generates descriptive profiles for each customer segment."""
    print("\n--- Creating Segment Profiles ---")
    profiles = {}
    
    # Identify potential profiling features (numeric and categorical, excluding ID)
    profiling_features = []
    numeric_profiling = []
    categorical_profiling = []

    for key, col_name in detected_cols.items():
        if col_name in results_df.columns and key not in ["customer_id", "transaction_date"]:
            if pd.api.types.is_numeric_dtype(results_df[col_name]):
                numeric_profiling.append(col_name)
                profiling_features.append(col_name)
            elif results_df[col_name].nunique() < 50: # Use same cardinality limit as segmentation
                categorical_profiling.append(col_name)
                profiling_features.append(col_name)

    if "Segment" not in results_df.columns:
        raise ValueError("Segment column not found in results DataFrame.")

    grouped = results_df.groupby("Segment")

    for segment_id, group in grouped:
        profile = {"segment_id": segment_id, "size": len(group)}
        description = [f"Segment {segment_id} ({len(group)} customers):"]

        # Numeric features summary (mean)
        if numeric_profiling:
            numeric_summary = group[numeric_profiling].mean().round(2).to_dict()
            profile["numeric_summary"] = numeric_summary
            desc_num = ", ".join([f"Avg {k}: {v}" for k, v in numeric_summary.items()])
            description.append(f"- Averages: {desc_num}")

        # Categorical features summary (top values)
        if categorical_profiling:
            categorical_summary = {}
            cat_desc_parts = []
            for col in categorical_profiling:
                # Get top 3 most frequent values and their percentages
                top_values = group[col].value_counts(normalize=True).head(3).round(3).to_dict()
                categorical_summary[col] = top_values
                cat_desc_parts.append(f"{col}: {', '.join([f'{val} ({pct*100:.1f}%)' for val, pct in top_values.items()])}")
            profile["categorical_summary"] = categorical_summary
            description.append("- Key Characteristics: " + "; ".join(cat_desc_parts))
            
        profile["description"] = "\n".join(description)
        profiles[segment_id] = profile
        print(f"Profile created for Segment {segment_id}")

    return profiles

def get_openai_recommendations(segment_profiles):
    """Gets marketing recommendations from OpenAI API based on segment profiles."""
    if not client:
        print("OpenAI client not initialized. Skipping recommendations.")
        # Return dummy recommendations
        recommendations = {}
        for seg_id in segment_profiles:
            recommendations[seg_id] = {
                "segment_id": seg_id,
                "campaign": "N/A (OpenAI not configured)",
                "channel": "N/A",
                "offer": "N/A",
                "rationale": "OpenAI API key needs to be configured to generate recommendations.",
                "openai_comment": "Please configure the OpenAI API key."
            }
        return recommendations

    print("\n--- Getting Recommendations from OpenAI ---")
    recommendations = {}
    system_prompt = (
        "You are a marketing strategist AI. Based on the provided customer segment profile, "
        "recommend the most suitable marketing campaign, primary communication channel, a relevant offer/discount, "
        "and a brief rationale for your choices. Also provide a short overall comment about the segment's potential. "
        "Focus on actionable insights for marketing teams. Format the output as a JSON object with keys: "
        "'campaign', 'channel', 'offer', 'rationale', 'openai_comment'."
    )

    for segment_id, profile in segment_profiles.items():
        print(f"Requesting recommendations for Segment {segment_id}...")
        user_prompt = f"Analyze the following customer segment profile and provide recommendations:\n\n{profile['description']}" 
        
        try:
            response = client.chat.completions.create(
                model="gpt-4o", # Or another suitable model like gpt-3.5-turbo
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"}, # Request JSON output directly
                temperature=0.7, # Adjust creativity vs. consistency
                max_tokens=300
            )
            
            # Extract JSON content
            content = response.choices[0].message.content
            rec_data = json.loads(content)
            
            # Validate expected keys (optional but good practice)
            expected_keys = ["campaign", "channel", "offer", "rationale", "openai_comment"]
            if all(key in rec_data for key in expected_keys):
                rec_data["segment_id"] = segment_id # Add segment ID back
                recommendations[segment_id] = rec_data
                print(f"Recommendations received for Segment {segment_id}")
            else:
                print(f"Warning: OpenAI response for Segment {segment_id} missing expected keys. Response: {content}")
                recommendations[segment_id] = {"segment_id": segment_id, "error": "Invalid format from OpenAI", "raw_response": content}

        except Exception as e:
            print(f"Error getting recommendations for Segment {segment_id} from OpenAI: {e}")
            recommendations[segment_id] = {"segment_id": segment_id, "error": str(e)}

    return recommendations

# Example usage (integrated with segmentation output)
if __name__ == '__main__':
    # Need dummy data similar to segmentation output
    # Let's create a dummy segmentation_results DataFrame
    dummy_seg_data = {
             "CustomerID": [f"C{i:03}" for i in range(1, 11)],
        "Recency": np.random.randint(10, 365, 10),
        "Frequency": np.random.randint(1, 15, 10),
        "MonetaryValue": np.random.randint(50, 1000, 10),
        "Age": np.random.randint(20, 60, 10),
        "City": np.random.choice(["London", "Paris", "Tokyo"], 10),
        "Segment": np.random.randint(0, 3, 10) # Assume 3 segments (0, 1, 2)
    }
    segmentation_results_df = pd.DataFrame(dummy_seg_data)
    
    # Dummy detected_cols matching the dataframe
    dummy_detected_cols = {
        "customer_id": "CustomerID",
        "recency": "Recency",
        "frequency": "Frequency",
        "monetary_value": "MonetaryValue",
        "age": "Age",
        "city": "City"
    }

    try:
        profiles = create_segment_profiles(segmentation_results_df, dummy_detected_cols)
        print("\n--- Generated Profiles ---")
        print(json.dumps(profiles, indent=2))

        # Only run OpenAI call if key is likely configured
        if client:
            recommendations = get_openai_recommendations(profiles)
            print("\n--- Generated Recommendations ---")
            print(json.dumps(recommendations, indent=2))
        else:
            print("\nSkipping OpenAI recommendations as client is not configured.")
            # Show dummy recommendations
            dummy_recs = get_openai_recommendations(profiles)
            print("\n--- Dummy Recommendations ---")
            print(json.dumps(dummy_recs, indent=2))

    except Exception as e:
        print(f"\nError during recommendation engine test: {e}")