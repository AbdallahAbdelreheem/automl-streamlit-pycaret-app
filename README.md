# ⚡ AutoML Streamlit App using PyCaret

A no-code machine learning web app built using **PyCaret** and **Streamlit**.  
It allows you to upload a dataset, preprocess it, train classification models automatically, and visualize the results — all in a few clicks!

---

## 🌟 Features

- 📁 Upload CSV or Excel datasets
- 🧹 Handle missing values automatically
- 🔤 Encode categorical variables
- 🎯 Select target column for classification
- ⚙️ Train and compare multiple models using PyCaret
- 📈 View performance metrics:
  - Confusion Matrix
  - AUC-ROC Curve
  - Feature Importance
- 💾 Download model comparison results
- ✅ Save the best-performing model as `.pkl`

---

## 🖼️ App Walkthrough

### 1. Upload & Preview Data  
![Upload & Preview](Upload%20&%20Preview.png)  
*This step allows users to upload their dataset and preview the first few rows for a quick glance at the data.*

### 2. Preprocessing  
![Preprocessing](Preprocessing.png)  
*In this step, the target variable is selected, and necessary preprocessing steps like handling missing values, encoding categorical variables, and scaling features are performed.*

### 3. Model Training  
![Train](Train.png)  
*This step trains the model using the preprocessed data and selected features.*

### 4. Evaluation Metrics  
![Evaluation Metrics](Evaluation%20metrics.png)  
*This section displays various evaluation metrics like accuracy, precision, recall, and F1-score to assess the model's performance.*

### 5. Feature Visualization  
![Feature Visualization](Feature%20Visualization.png)  
*Here, the features of the dataset are visualized to identify patterns, correlations, and outliers in the data.*

---

## 🌐 Live Demo

Try the app online (note: performance may vary):  
🔗 [Streamlit Cloud App](https://automl-app-pycaret-app-zoeaujxozfb4pj746sugrk.streamlit.app/)

---

## ⚠️ Performance Note

Due to limited resources on **Streamlit Cloud**, the training process might be **slow or timeout** for large datasets.

💡 **Recommended:**  
Clone the repository and run the app locally in your virtual environment for best performance.

---

## 🛠 Tech Stack

- Python 3.10
- PyCaret 3.0.4
- Streamlit 1.28
- Pandas, Scikit-learn, Matplotlib, Seaborn

---

## 🚀 Run Locally

```bash
# Clone the repository
git clone https://github.com/yourusername/automl-streamlit-pycaret-app.git
cd automl-streamlit-pycaret-app

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run the app
streamlit run streamlit_app.py
