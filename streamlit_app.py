import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import ClassificationExperiment
from pycaret.regression import RegressionExperiment

# Set Streamlit page configuration
st.set_page_config(page_title="AutoML App", layout="wide")
st.title("📊 AutoML Web App using PyCaret")

# Sidebar Info
with st.sidebar:
    st.header("App Navigation")
    st.markdown("This app allows you to upload a dataset, handle missing values, encode categorical variables, drop columns, and train models using PyCaret.")

# Tabs for better navigation
tabs = st.tabs(["Upload & Preview", "Preprocessing", "Train & Evaluate", "Feature Visualization"])

# Step 1: Upload and Preview Dataset
with tabs[0]:
    st.header("1. Upload Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV or Excel)", type=["csv", "xlsx"])

    if uploaded_file is not None:
        file_name = uploaded_file.name.lower()
        if file_name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        df.columns = df.columns.str.replace(' ', '_').str.strip().str.lower()

        st.success("Dataset loaded successfully!")
        st.write("### Preview of the Data")
        st.dataframe(df.head(50), use_container_width=True)
        st.session_state.df = df

# Step 2: Handle Missing Values, Drop Columns, and Encoding
with tabs[1]:
    st.header("2. Data Preprocessing")

    if "df" in st.session_state:
        df = st.session_state.df.copy()

        # Drop columns
        st.subheader("Drop Unwanted Columns")
        columns_to_drop = st.multiselect("Select columns to drop:", df.columns)
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)

        # Missing value imputation
        st.subheader("Missing Value Imputation")
        imputation_options = {}
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(df[col]):
                    imputation_options[col] = st.selectbox(
                        f"Imputation method for numeric column '{col}'", ["Mean", "Mode"])
                else:
                    imputation_options[col] = st.selectbox(
                        f"Imputation method for categorical column '{col}'", ["Most Frequent", "New Category"])

        for col, method in imputation_options.items():
            if method == "Mean":
                df[col].fillna(df[col].mean(), inplace=True)
            elif method in ["Mode", "Most Frequent"]:
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "New Category":
                df[col].fillna("Missing", inplace=True)

        st.success("Missing values handled successfully!")

        # Categorical Encoding
        st.subheader("Categorical Encoding")
        encoding_method = st.radio("Choose encoding method:", ["Label Encoding", "One-Hot Encoding"])
        df_encoded = df.copy()

        if encoding_method == "Label Encoding":
            for col in df_encoded.select_dtypes(include='object').columns:
                le = LabelEncoder()
                df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        elif encoding_method == "One-Hot Encoding":
            df_encoded = pd.get_dummies(df_encoded)

        st.success("Encoding completed successfully!")

        # Select Target
        st.subheader("Select Target Variable")
        target = st.selectbox("Choose the target column (Y):", df_encoded.columns)

        if target:
            task_type = "regression" if pd.api.types.is_numeric_dtype(df_encoded[target]) and df_encoded[target].nunique() > 10 else "classification"
            st.session_state.df_encoded = df_encoded
            st.session_state.target = target
            st.session_state.task_type = task_type
            st.success(f"Data ready for {task_type} task ✅")

# Step 3: Train and Compare Models
with tabs[2]:
    st.header("3. Model Training")

    if all(k in st.session_state for k in ["df_encoded", "target", "task_type"]):
        df_encoded = st.session_state.df_encoded
        target = st.session_state.target
        task_type = st.session_state.task_type

        if "best_model" not in st.session_state:
            if st.button("Start Training"):
                with st.spinner("Training in progress..."):
                    try:
                        if task_type == "classification":
                            exp = ClassificationExperiment()
                        else:
                            exp = RegressionExperiment()

                        exp.setup(
                            data=df_encoded,
                            target=target,
                            use_gpu=False,
                            verbose=False,
                            profile=False,
                            session_id=42
                        )

                        best_model = exp.compare_models()
                        results = exp.pull()

                        st.session_state.best_model = best_model
                        st.session_state.results = results
                        st.session_state.exp = exp

                        st.success("Training completed ✅")
                        st.toast("Best model trained successfully!", icon="🎉")
                    except Exception as e:
                        st.error(f"Training failed: {str(e)}")

        if "results" in st.session_state:
            st.subheader("Model Comparison Results")
            st.dataframe(st.session_state.results, use_container_width=True)

            csv = st.session_state.results.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Results as CSV",
                data=csv,
                file_name="model_comparison_results.csv",
                mime="text/csv"
            )

            if st.button("Save Best Model"):
                with open("best_model.pkl", "wb") as f:
                    pickle.dump(st.session_state.best_model, f)
                st.success("Model saved as 'best_model.pkl' ✅")

            st.subheader("Model Evaluation Metrics")
            if st.session_state.exp and st.session_state.best_model:
                try:
                    eval_tabs = st.tabs(["Confusion Matrix" if task_type == "classification" else "Residuals Plot", "AUC Curve" if task_type == "classification" else "Prediction Error", "Feature Importance"])

                    with eval_tabs[0]:
                        st.write("### Confusion Matrix" if task_type == "classification" else "### Residuals Plot")
                        plot = st.session_state.exp.plot_model(
                            st.session_state.best_model,
                            plot="confusion_matrix" if task_type == "classification" else "residuals",
                            display_format="streamlit"
                        )
                        if isinstance(plot, plt.Figure):
                            st.pyplot(plot)
                            plt.close(plot)

                    with eval_tabs[1]:
                        st.write("### AUC-ROC Curve" if task_type == "classification" else "### Prediction Error")
                        plot = st.session_state.exp.plot_model(
                            st.session_state.best_model,
                            plot="auc" if task_type == "classification" else "error",
                            display_format="streamlit"
                        )
                        if isinstance(plot, plt.Figure):
                            st.pyplot(plot)
                            plt.close(plot)

                    with eval_tabs[2]:
                        st.write("### Feature Importance")
                        try:
                            plot = st.session_state.exp.plot_model(
                                st.session_state.best_model,
                                plot="feature",
                                display_format="streamlit"
                            )
                            if isinstance(plot, plt.Figure):
                                st.pyplot(plot)
                                plt.close(plot)
                            else:
                                st.warning("Feature importance plot not available for this model type")
                        except Exception as e:
                            st.error(f"Could not generate feature importance: {str(e)}")
                except Exception as e:
                    st.error(f"Evaluation error: {str(e)}")
                    st.warning("Some plots may not be available for this model type")
    else:
        st.warning("Please complete the previous steps to train the model.")

# Step 4: Feature Visualization
with tabs[3]:
    st.header("4. Feature Visualization")
    if "df" in st.session_state:
        df = st.session_state.df
        feature = st.selectbox("Select a feature to visualize:", df.columns)

        if feature:
            st.subheader(f"Distribution of '{feature}'")
            if pd.api.types.is_numeric_dtype(df[feature]):
                fig, ax = plt.subplots()
                sns.histplot(df[feature], kde=True, ax=ax)
                st.pyplot(fig)
            else:
                fig, ax = plt.subplots()
                df[feature].value_counts().plot(kind='bar', ax=ax)
                st.pyplot(fig)
    else:
        st.info("Please upload and preprocess a dataset first.")
