# SegmentWise: Intelligent Customer Segmentation Platform

<p align="center">
  <img src="img/segmentwise_logo.png" alt="SegmentWise Logo" width="200"/>
</p>

<p align="center">
  <b>Create automatic segments from your customer data and develop AI-powered marketing strategies</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python 3.8+"/>
  <img src="https://img.shields.io/badge/Flask-2.0.1-green" alt="Flask 2.0.1"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.0.2-orange" alt="scikit-learn 1.0.2"/>
  <img src="https://img.shields.io/badge/OpenAI-API-blueviolet" alt="OpenAI API"/>
  <img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT License"/>
</p>

## 📊 Project Overview

SegmentWise is a comprehensive platform that automatically analyzes customer data to create meaningful customer segments, provides AI-powered marketing strategies for each segment, and delivers visual data analyses.

### ✨ Demo Screenshots

<p align="center">
  <img src="img/dashboard.png" alt="SegmentWise Dashboard" width="800"/>
  <br/>
  <em>Main Dashboard: Segments and Marketing Strategies</em>
</p>

<p align="center">
  <img src="img/segments.png" alt="Segment Analysis" width="800"/>
  <br/>
  <em>Segment Details and Distribution Analysis</em>
</p>

## 🚀 Key Features

- **🔍 Automatic Data Detection**: Automatically identifies relevant customer data from CSV files
- **🧹 Smart Data Cleaning**: Auto-fills missing data and filters erroneous data
- **📊 Dynamic Segmentation**: Determines the optimal number of segments using K-means algorithm
- **📈 Interactive Charts**: Visualizations for segment distribution and characteristic features
- **🤖 AI Marketing Recommendations**: Custom marketing strategies for each segment using OpenAI API
- **📱 Responsive Design**: Modern interface that works seamlessly on different devices

## 🛠️ Technologies

- **Backend**: Python, Flask, Pandas, NumPy, scikit-learn
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Data Visualization**: Plotly, Seaborn
- **Artificial Intelligence**: OpenAI GPT API integration

## ⚙️ Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- OpenAI API key (optional)

### Step-by-Step Setup

1. Clone the project:
   ```
   git clone https://github.com/ozzy2438/SegmentWise.git
   cd SegmentWise
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate     # For Windows
   ```

3. Install required packages:
   ```
   pip install -r requirements.txt
   ```

4. (Optional) Create a `.env` file for OpenAI API integration:
   ```
   touch .env
   echo "OPENAI_API_KEY=your_api_key" >> .env
   ```

5. Start the application:
   ```
   python src/app.py
   ```

6. Open the following address in your browser: `http://localhost:8080`

## 📊 User Guide

### 1. Data Upload and Segmentation

1. Click the "Upload CSV" button on the main page
2. Select the CSV file containing your customer data
3. Choose the number of segments you want or leave it blank for automatic determination
4. Click the "Start Analysis" button to begin the segmentation process

### 2. Exploring Segments

- Detailed profiles are displayed for each created segment
- Demographic and behavioral characteristics are presented with charts and tables
- Segment distributions and sizes are visually displayed

### 3. Marketing Strategies

- AI-recommended for each segment:
  - Campaign ideas
  - Communication channels
  - Special offers and discounts
  - Strategy rationales

## 📋 Project Structure

```
SegmentWise/
├── src/                  # Source code
│   ├── app.py            # Flask application and routes
│   ├── data_processor.py # Data processing and cleaning
│   ├── segmentation.py   # Segment creation algorithms
│   └── recommendation.py # OpenAI API integration
├── templates/            # HTML templates  
├── static/               # CSS, JS and visuals
├── uploads/              # Uploaded CSV files
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## 🤝 Contributing

We welcome your contributions! Please reach out via GitHub for feature requests, bug reports, or pull requests.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

This project was developed with the help of the following open source libraries:
- [Flask](https://flask.palletsprojects.com/)
- [scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [Plotly](https://plotly.com/)
- [OpenAI](https://openai.com/)

---

<p align="center">
  <a href="https://linkedin.com/in/your-linkedin-profile">Connect on LinkedIn</a> • 
  <a href="mailto:your-email@example.com">Contact</a>
</p> 