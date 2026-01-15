import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from feature_extractor import URLFeatureExtractor

# ==============================
# CONFIG
# ==============================
DATASET_PATH = "dataset.csv"   # change if your dataset filename is different
MODEL_OUTPUT = "phishing_url_detector.pkl"
RANDOM_STATE = 42

# ==============================
# LOAD DATASET
# ==============================
print("[INFO] Loading dataset...")
df = pd.read_csv(DATASET_PATH)

# Expected columns:
# url,label
# label: 0 = legitimate, 1 = phishing

if "url" not in df.columns or "label" not in df.columns:
    raise ValueError("Dataset must contain 'url' and 'label' columns")

print(f"[INFO] Dataset size: {len(df)} rows")

# ==============================
# FEATURE EXTRACTION
# ==============================
print("[INFO] Extracting features...")
extractor = URLFeatureExtractor()

X = df["url"].apply(extractor.extract_features).tolist()
X = pd.DataFrame(X, columns=extractor.feature_names())
y = df["label"]

print(f"[INFO] Feature shape: {X.shape}")

# ==============================
# TRAIN / TEST SPLIT
# ==============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# ==============================
# MODEL TRAINING
# ==============================
print("[INFO] Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=RANDOM_STATE,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ==============================
# EVALUATION
# ==============================
print("[INFO] Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Training Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ==============================
# SAVE MODEL
# ==============================
joblib.dump(model, MODEL_OUTPUT)
print(f"\n💾 Model saved as: {MODEL_OUTPUT}")
