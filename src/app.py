# src/app.py
import os
import json
from flask import Flask, request, render_template, redirect, url_for, flash, session
import pandas as pd
from werkzeug.utils import secure_filename
import plotly
import plotly.express as px

# Import project modules
from data_processor import process_data
from segmentation import run_segmentation_pipeline
from recommendation import create_segment_profiles, get_openai_recommendations

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}

app = Flask(__name__, template_folder="../templates") # Point to the correct template folder
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.secret_key = os.urandom(24) # Needed for flash messages and session

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and \
           filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_segment_plot(df, segment_col="Segment", x_col="Recency", y_col="Frequency", color_col="MonetaryValue"):
    """Generates an interactive Plotly scatter plot for segments."""
    if not all(col in df.columns for col in [segment_col, x_col, y_col, color_col]):
        print(f"Skipping plot: Missing one or more columns: {segment_col}, {x_col}, {y_col}, {color_col}")
        return None
    
    try:
        # Ensure segment column is categorical for coloring
        df[segment_col] = df[segment_col].astype("category")
        
        fig = px.scatter(
            df,
            x=x_col,
            y=y_col,
            color=segment_col, # Color by segment
            size=color_col,  # Size by MonetaryValue (or another numeric feature)
            hover_data=[df.columns[0]], # Show CustomerID (assuming it	"s the first col)
            title=f"Customer Segments ({x_col} vs {y_col}, Size by {color_col})",
            labels={segment_col: "Segment"}
        )
        fig.update_layout(xaxis_title=x_col, yaxis_title=y_col)
        
        # Convert plot to JSON
        graph_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        return graph_json
    except Exception as e:
        print(f"Error generating plot: {e}")
        return None

@app.route("/", methods=["GET"])
def index():
    # Clear previous results from session if any
    session.pop("results", None)
    return render_template("index.html", results=None)

@app.route("/upload", methods=["POST"])
def upload_and_process():
    if "file" not in request.files:
        flash("No file part", "danger")
        return redirect(url_for("index"))
    
    file = request.files["file"]
    if file.filename == "":
        flash("No selected file", "warning")
        return redirect(url_for("index"))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config[	"UPLOAD_FOLDER"	], filename)
        try:
            file.save(filepath)
            flash(f"File '{filename}' uploaded successfully. Processing...", "info")

            # --- Full Processing Pipeline ---
            # 1. Process Data (Load, Clean, RFM/Feature Prep)
            processed_df, detected_cols = process_data(filepath)
            
            # 2. Run Segmentation
            segmentation_results_df, kmeans_model, preprocessor, feature_names, k = run_segmentation_pipeline(processed_df, detected_cols)
            
            # 3. Create Segment Profiles
            profiles = create_segment_profiles(segmentation_results_df, detected_cols)
            
            # 4. Get Recommendations (Handles OpenAI key check internally)
            recommendations = get_openai_recommendations(profiles)
            
            # 5. Generate Plot (Example: RFM plot if available)
            plot_json = None
            # Check if RFM columns were used/exist for plotting
            rfm_cols_present = all(col in segmentation_results_df.columns for col in ["Recency", "Frequency", "MonetaryValue"])
            if rfm_cols_present:
                 # Generate one plot for all segments
                 plot_json = generate_segment_plot(segmentation_results_df, segment_col="Segment", 
                                                 x_col="Recency", y_col="Frequency", color_col="MonetaryValue")
            else:
                # TODO: Add alternative plot for non-RFM features if needed
                print("RFM columns not found in results, skipping default plot.")

            # Prepare results dictionary for the template
            results_data = {
                "total_customers": len(segmentation_results_df),
                "k": k,
                "profiles": profiles,
                "recommendations": recommendations,
                "plots": {0: plot_json} if plot_json else {} # Assign plot to a dummy key 0 for now, template expects dict
                # Optionally pass segmentation_results_df if needed for download later
            }
            
            # Store results in session to avoid reprocessing on refresh (optional)
            # session[	"results"	] = results_data 
            # flash(	"Processing complete!"	, 	"success"	)
            # return redirect(url_for(	"show_results_page"	)) # Redirect to a dedicated results page

            # Or render index directly with results
            flash("Processing complete!", "success")
            return render_template("index.html", results=results_data)

        except ValueError as ve:
            flash(f"Processing Error: {str(ve)}", "danger")
            print(f"ValueError during processing: {ve}")
            return redirect(url_for("index"))
        except FileNotFoundError as fnf:
             flash(f"Error: Could not find uploaded file for processing. {str(fnf)}", "danger")
             print(f"FileNotFoundError: {fnf}")
             return redirect(url_for("index"))
        except ImportError as ie:
             flash(f"Internal Error: A required library might be missing. {str(ie)}", "danger")
             print(f"ImportError: {ie}")
             return redirect(url_for("index"))
        except Exception as e:
            flash(f"An unexpected error occurred during processing: {str(e)}", "danger")
            # Log the full traceback for debugging
            import traceback
            print(f"Unexpected Error: {traceback.format_exc()}")
            return redirect(url_for(	"index"	))
        finally:
            # Clean up uploaded file after processing (optional)
            # if os.path.exists(filepath):
            #     os.remove(filepath)
            pass

    else:
        flash("Invalid file type. Please upload a CSV file.", "warning")
        return redirect(url_for("index"))

# Optional: Dedicated route to show results if using session
# @app.route(	"/results"	)
# def show_results_page():
#     results = session.get(	"results"	, None)
#     if not results:
#         flash(	"No results found. Please upload a file first.	", 	"warning"	)
#         return redirect(url_for(	"index"	))
#     return render_template(	"index.html"	, results=results) # Render index with results

if __name__ == "__main__":
    # Listen on 0.0.0.0 to be accessible externally if needed
    app.run(host="0.0.0.0", port=8080, debug=True) # debug=True for development

