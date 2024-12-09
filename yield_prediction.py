import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

# Preprocessing and model evaluation functions
# Remove low variance Features
def remove_low_variance_features(X, threshold=0.0001):
    """
    Removes features with variance below a given threshold.
    """
    var_thresh = VarianceThreshold(threshold=threshold)
    X_reduced = var_thresh.fit_transform(X)
    return pd.DataFrame(X_reduced)

#Remove the highly collinear features from the data
def remove_collinear_features(x, threshold):
    '''
        Remove the collinear features from the dataset where a correlation coefficient is
        greater than the threshold. Removing collinear features can help the model to generalize
        and improve the interpretability of the model.
    '''
    # Calculate the correlation matrix
    corr_matrix = x.corr()
    iters = range(len(corr_matrix.columns) - 1)
    drop_cols = []

    # Iterate through the correlation matrix and compare correlations
    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            # If correlation exceeds the threshold
            if val >= threshold:
                # Print the correlated features and the correlation value
                print(col.values[0], "|", row.values[0], "|", round(val[0][0], 2))
                drop_cols.append(col.values[0])

    # Drop one of each pair of correlated columns
    drops = set(drop_cols)
    x = x.drop(columns=drops)

    return x

# Define a function for model training and evaluation
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"\nModel: {model.__class__.__name__}")
    print(classification_report(y_test, y_pred))
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"ROC-AUC Score: {roc_auc:.2f}")
    return conf_matrix

# Initialize models dictionary
models = {
    "Random Forest": RandomForestClassifier(random_state=1),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=1),
    "SVM (Linear Kernel)": SVC(probability=True, kernel='linear', random_state=1),
}

