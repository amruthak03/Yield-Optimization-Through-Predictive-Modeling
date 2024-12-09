import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.metrics import  roc_auc_score
from yield_prediction import remove_low_variance_features, remove_collinear_features, evaluate_model, models  # import functions and models from my script

# Streamlit App
st.title("Semiconductor Manufacturing: Yield Prediction")
st.write("""
This application demonstrates the classification of pass/fail instances in semiconductor manufacturing.
""")

# Sidebar for dataset info
st.sidebar.header("Dataset Information")
st.sidebar.write("Shape of Dataset: 1567 samples, 592 features")
st.sidebar.write("Class Distribution: 93.36% Pass, 6.64% Fail")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Choose a Model", list(models.keys()))

# Sidebar for applying SMOTE and scaling options
smote_choice = st.sidebar.radio("Apply SMOTE?", ["Yes", "No"])
scale_choice = st.sidebar.radio("Apply Scaling?", ["Yes", "No"])

# Loading the dataset
st.write("Loading dataset...")
df = pd.read_csv('uci-secom.csv')

# Pie Chart for Pass/Fail Distribution
st.subheader("Target Distribution (Pass/Fail) - Pie Chart")
labels = ['Pass', 'Fail']
size = df['Pass/Fail'].value_counts()
colors = ['blue', 'green']
explode = [0, 0.1]

plt.rcParams['figure.figsize'] = (6, 6)
plt.pie(size, labels=labels, colors=colors, explode=explode, autopct="%.2f%%", shadow=True)
plt.axis('off')
plt.title('Target: Pass or Fail', fontsize=10)
plt.legend()
st.pyplot()  # Display the pie chart in the app

# Bar Chart for Pass/Fail Count
st.subheader("Pass/Fail Distribution - Bar Chart")
df['Pass/Fail'].value_counts().plot(kind="bar", color=['blue', 'green'])
plt.title("Pass/Fail Count", fontsize=10)
plt.xlabel("Pass/Fail")
plt.ylabel("Count")
st.pyplot()  # Display the bar chart in the app

X = df.iloc[:, 1:-1]  # Exclude 'Time' and 'Pass/Fail'
y = df['Pass/Fail']

# Handle missing values and feature engineering
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)
X = remove_low_variance_features(X)
X = remove_collinear_features(X, threshold=0.8)

# SMOTE
if smote_choice == "Yes":
    smote = SMOTE(random_state=1)
    X, y = smote.fit_resample(X, y)

    st.write("SMOTE applied.")

# Scaling
if scale_choice == "Yes":
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    st.write("Scaling applied.")

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

# Choose and evaluate the model
st.write(f"Evaluating {model_choice}...")
selected_model = models[model_choice]
conf_matrix = evaluate_model(selected_model, X_train, X_test, y_train, y_test)

# Display Results
st.write("**Confusion Matrix:**")
st.write(conf_matrix)

# Showing ROC-AUC Score for the selected model
y_pred_proba = selected_model.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
st.write(f"ROC-AUC Score: {roc_auc:.2f}")

