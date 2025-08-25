import os
import warnings
from datetime import datetime

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")


def categorize_bmi(bmi):
    if bmi < 18.5:
        return 'Underweight'
    elif bmi < 25:
        return 'Normal'
    elif bmi < 30:
        return 'Overweight'
    else:
        return 'Obese'


def categorize_age(age):
    if age < 30:
        return 'Young'
    elif age < 50:
        return 'Middle-aged'
    elif age < 65:
        return 'Senior'
    else:
        return 'Elderly'


# this is where the results will be stored
base_dir = "results"
os.makedirs(base_dir, exist_ok=True)

existing_attempts = [d for d in os.listdir(base_dir) if d.startswith("attempt-")]
attempt_num = len(existing_attempts) + 1
attempt_dir = os.path.join(base_dir, f"attempt-{attempt_num}")
os.makedirs(attempt_dir)

# load and remove data leaking columns and feature engineering
df = pd.read_csv("./hypertension_dataset.csv")

df = df.drop(columns=["Medication", "BP_History"])
df['BMI_Category'] = df['BMI'].apply(categorize_bmi)
df['Age_Group'] = df['Age'].apply(categorize_age)
df['Salt_Level'] = pd.cut(df['Salt_Intake'], bins=[0, 8, 10, 15], labels=['Low', 'Moderate', 'High'])

df['Risk_Score'] = (
    (df['Age'] / 100) * 0.3 +
    (df['BMI'] / 40) * 0.25 +
    (df['Salt_Intake'] / 15) * 0.2 +
    (df['Stress_Score'] / 10) * 0.15 +
    (df['Family_History'] == 'Yes').astype(int) * 0.1
)

# one hot encoding
catc = df.select_dtypes(include=["object", "category"])
string_cols = catc.columns.tolist()
df_encoded = pd.get_dummies(df, columns=string_cols, drop_first=True)

target_col = "Has_Hypertension_Yes"
X = df_encoded.drop(columns=[target_col])
y = df_encoded[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, os.path.join(attempt_dir, "scaler.joblib"))

# all the models we will train
# all the models we will train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "Support Vector Machine": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss"
    ),
    "CatBoost": CatBoostClassifier(
        iterations=200,
        learning_rate=0.1,
        depth=6,
        loss_function="Logloss",
        verbose=0,
        random_state=7
    ),
    "LightGBM": LGBMClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=-1,
        random_state=42
    ),
}

# PCA for visualization (but this is very shit lmao, 2 components aint cutting it)
pca = PCA(n_components=2)
X_vis = pca.fit_transform(X_train_scaled)

plt.figure(figsize=(6, 5))
plt.scatter(X_vis[:, 0], X_vis[:, 1], c=y_train, cmap="coolwarm", edgecolor="k")
plt.title("PCA Projection of Training Data")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label="Target")
plt.tight_layout()
plt.savefig(os.path.join(attempt_dir, "pca_projection.png"))
plt.close()

# training each model
for name, model in models.items():
    print(f"{name} Trained")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    model_folder = os.path.join(attempt_dir, name.replace(" ", "_"))
    os.makedirs(model_folder, exist_ok=True)

    # save models
    joblib.dump(model, os.path.join(model_folder, "model.joblib"))

    # save the resuls
    with open(os.path.join(model_folder, "metrics.txt"), "w") as f:
        f.write(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)) + "\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred) + "\n")

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "confusion_matrix.png"))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{name} - ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "roc_curve.png"))
    plt.close()

    # precision recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    plt.figure(figsize=(5, 4))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{name} - Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(os.path.join(model_folder, "precision_recall_curve.png"))
    plt.close()
