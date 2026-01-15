import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extractor import URLFeatureExtractor
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """
    Load dataset and preprocess it
    """
    print("Loading dataset...")
    df = pd.read_csv(filepath)

    # Assuming the dataset has 'url' and 'label' columns
    # label: 1 for phishing, 0 for legitimate
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")

    # Handle missing values
    df = df.dropna()

    return df

def extract_features(df):
    """
    Extract features from URLs
    """
    print("Extracting features...")
    extractor = URLFeatureExtractor()
    df_features = extractor.extract_features_from_df(df, url_column='url')

    # Select numerical features for modeling
    numerical_features = [
        'url_length', 'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
        'num_question_marks', 'num_equals', 'num_ampersands', 'num_digits', 'num_alphabets',
        'domain_length', 'subdomain_length', 'num_subdomains', 'has_https', 'has_ip',
        'has_at_symbol', 'has_double_slash', 'suspicious_keywords_count', 'special_chars_ratio',
        'path_length', 'num_path_segments', 'query_length', 'num_query_params'
    ]

    X = df_features[numerical_features]
    y = df_features['label']

    return X, y, numerical_features

def train_baseline_model(X_train, y_train):
    """
    Train baseline Logistic Regression model
    """
    print("Training baseline Logistic Regression model...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler

def train_main_model(X_train, y_train):
    """
    Train main Random Forest model
    """
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    return model

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate model performance
    """
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png')
    plt.show()

    return accuracy, precision, recall, f1

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for Random Forest
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(len(feature_names)), importances[indices],
                color="r", align="center")
        plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
        plt.xlim([-1, len(feature_names)])
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        plt.show()

def save_model(model, filename):
    """
    Save trained model to disk
    """
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

def main():
    # For demonstration, we'll create a sample dataset
    # In real scenario, load from actual dataset files
    print("Creating sample dataset for demonstration...")

    # Sample URLs (replace with actual dataset loading)
    sample_urls = [
        "https://www.google.com",
        "https://www.facebook.com/login",
        "https://secure-bank-login.com",
        "https://paypal-verify-account.net",
        "https://amazon-update-info.org",
        "https://github.com",
        "https://stackoverflow.com",
        "https://login-secure-update.com/verify",
        "https://bank-account-secure.com",
        "https://microsoft-login-update.net"
    ]

    sample_labels = [0, 0, 1, 1, 1, 0, 0, 1, 1, 1]  # 0: legitimate, 1: phishing

    df = pd.DataFrame({'url': sample_urls, 'label': sample_labels})

    # Extract features
    X, y, feature_names = extract_features(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")

    # Train baseline model
    baseline_model, scaler = train_baseline_model(X_train, y_train)
    baseline_metrics = evaluate_model(baseline_model, scaler.transform(X_test), y_test, "Logistic Regression")

    # Train main model
    rf_model = train_main_model(X_train, y_train)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    # Plot feature importance
    plot_feature_importance(rf_model, feature_names)

    # Save the best model
    save_model(rf_model, 'phishing_url_detector.pkl')

    print("\nModel training completed!")
    print("Files saved:")
    print("- phishing_url_detector.pkl (trained model)")
    print("- confusion_matrix_random_forest.png")
    print("- feature_importance.png")

if __name__ == "__main__":
    main()
