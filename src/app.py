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
        flash("Dosya seçilmedi", "danger")
        return redirect(url_for("index"))
    
    file = request.files["file"]
    if file.filename == "":
        flash("Dosya seçilmedi", "warning")
        return redirect(url_for("index"))
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        try:
            file.save(filepath)
            flash(f"'{filename}' başarıyla yüklendi. İşleniyor...", "info")
            
            # Kullanıcının belirlediği segment sayısını al
            user_k = request.form.get("segment_count", None)
            if user_k:
                try:
                    user_k = int(user_k)
                    if user_k < 2 or user_k > 10:
                        print(f"Uygun olmayan segment sayısı: {user_k}, otomatik belirlemeye dönülüyor.")
                        user_k = None
                except (ValueError, TypeError):
                    user_k = None
                    print("Segment sayısı geçerli bir sayı değil, otomatik belirlemeye dönülüyor.")
            
            print(f"Kullanıcı tarafından belirtilen segment sayısı: {user_k if user_k else 'Belirtilmemiş'}")

            # --- Full Processing Pipeline ---
            # 1. Process Data (Load, Clean, RFM/Feature Prep)
            processed_df, detected_cols = process_data(filepath)
            
            # 2. Run Segmentation
            segmentation_results_df, kmeans_model, preprocessor, feature_names, k = run_segmentation_pipeline(processed_df, detected_cols, user_k)
            
            # 3. Create Segment Profiles
            profiles = create_segment_profiles(segmentation_results_df, detected_cols)
            
            # 4. Get Recommendations (Handles OpenAI key check internally)
            recommendations = get_openai_recommendations(profiles)
            
            # 5. Generate Plot (Example: RFM plot if available)
            plot_json = None
            individual_plots = {}
            segment_plots = {}
            
            # Check if RFM columns were used/exist for plotting
            rfm_cols_present = all(col in segmentation_results_df.columns for col in ["Recency", "Frequency", "MonetaryValue"])
            if rfm_cols_present:
                # Generate one plot for all segments
                plot_json = generate_segment_plot(segmentation_results_df, segment_col="Segment", 
                                                x_col="Recency", y_col="Frequency", color_col="MonetaryValue")
            else:
                # Farklı sütunlarla grafik oluşturmayı deneyelim
                numeric_cols = [col for col in segmentation_results_df.columns 
                              if pd.api.types.is_numeric_dtype(segmentation_results_df[col]) 
                              and col != 'Segment']
                
                if len(numeric_cols) >= 2:  # En az 2 sayısal sütun varsa grafik oluştur
                    print(f"RFM columns not found. Creating plot with columns: {numeric_cols[0]}, {numeric_cols[1]}")
                    color_col = numeric_cols[2] if len(numeric_cols) > 2 else numeric_cols[0]
                    plot_json = generate_segment_plot(segmentation_results_df, segment_col="Segment", 
                                                 x_col=numeric_cols[0], y_col=numeric_cols[1], 
                                                 color_col=color_col)
                    
                    # Gelişmiş segment görselleştirmeleri oluştur
                    try:
                        from segmentation import create_segment_visualizations
                        segment_plots = create_segment_visualizations(segmentation_results_df, segment_col="Segment", feature_cols=numeric_cols[:4])
                    except Exception as e:
                        print(f"Error creating segment visualizations: {e}")
                        segment_plots = {}
                    
                    # Her segment için ayrı görselleştirmeler
                    for segment_id in range(k):
                        segment_df = segmentation_results_df[segmentation_results_df['Segment'] == segment_id]
                        if len(segment_df) > 0:
                            try:
                                individual_plot = generate_segment_plot(segment_df, segment_col="Segment", 
                                                                       x_col=numeric_cols[0], y_col=numeric_cols[1], 
                                                                       color_col=color_col)
                                individual_plots[str(segment_id)] = individual_plot
                            except Exception as e:
                                print(f"Error creating individual plot for segment {segment_id}: {e}")
                else:
                    print("Not enough numeric columns to create a plot")

            # Prepare results dictionary for the template
            results_data = {
                "total_customers": len(segmentation_results_df),
                "k": k,
                "profiles": profiles,
                "recommendations": recommendations,
                "plots": {0: plot_json} if plot_json else {}, # Assign plot to a dummy key 0 for now
                "segment_plots": segment_plots,
                "individual_plots": individual_plots
                # Optionally pass segmentation_results_df if needed for download later
            }

            flash("İşlem tamamlandı!", "success")
            return render_template("index.html", results=results_data)

        except ValueError as ve:
            error_msg = str(ve)
            
            # Daha kullanıcı dostu hata mesajları
            if "Data is empty after cleaning" in error_msg:
                flash(f"İşleme hatası: Dosyanız temizleme sonrası boş kaldı. Lütfen veri formatını kontrol edin veya başka bir CSV dosyası deneyin.", "danger")
            elif "No suitable features found for segmentation" in error_msg:
                flash(f"İşleme hatası: Segmentasyon için uygun özellikler bulunamadı. Dosyanızda Amount, Quantity gibi sayısal kolonların olduğundan emin olun.", "danger")
            elif "Could not automatically detect essential columns" in error_msg:
                flash(f"İşleme hatası: Gerekli kolonlar tespit edilemedi. Dosyanızda müşteri ID, tarih, miktar gibi kolonlar olup olmadığını kontrol edin.", "danger")
            else:
                flash(f"İşleme hatası: {error_msg}", "danger")
                
            print(f"ValueError during processing: {ve}")
            return redirect(url_for("index"))
        except FileNotFoundError as fnf:
             flash(f"Hata: Yüklenen dosya işlem için bulunamadı. {str(fnf)}", "danger")
             print(f"FileNotFoundError: {fnf}")
             return redirect(url_for("index"))
        except ImportError as ie:
             flash(f"İç Hata: Gerekli bir kütüphane eksik olabilir. {str(ie)}", "danger")
             print(f"ImportError: {ie}")
             return redirect(url_for("index"))
        except Exception as e:
            flash(f"Beklenmeyen bir hata oluştu: {str(e)}", "danger")
            # Log the full traceback for debugging
            import traceback
            print(f"Unexpected Error: {traceback.format_exc()}")
            return redirect(url_for("index"))
        finally:
            # Clean up uploaded file after processing (optional)
            # if os.path.exists(filepath):
            #     os.remove(filepath)
            pass

    else:
        flash("Geçersiz dosya türü. Lütfen bir CSV dosyası yükleyin.", "warning")
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

