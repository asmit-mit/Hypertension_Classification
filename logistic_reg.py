import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    cross_val_score,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

df = pd.read_csv("./hypertension_dataset.csv")


df["Has_Hypertension"] = df["Has_Hypertension"].map({"Yes": 1, "No": 0})


# remove nan values
df["Medication"] = df["Medication"].fillna("No Medication")

print("Non numerical data: ")
print("--------------------")
catc = df.select_dtypes(include=["object"])
string_cols = []
for col in catc.columns:
    string_cols.append(col)
    print(f"{col}: {catc[col].unique()}")


# print("\nOne-hot encoding the data:")
# print("---------------------------")
# df_encoded = pd.get_dummies(
#     df,
#     columns=string_cols,
#     drop_first=True,
# )
#
# print(df_encoded.columns)

X = df.drop(columns=["Has_Hypertension", "BP_History", "Medication"])
y = df["Has_Hypertension"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7, stratify=y
)

cat_col = X.select_dtypes("object").columns.to_list()
num_col = X.select_dtypes("number").columns.to_list()

num_pipe = Pipeline([("numeric_col", StandardScaler())])
cat_pipe = Pipeline([("cat_col", OneHotEncoder())])

preprocessor = ColumnTransformer([("num", num_pipe, num_col), ("cat", cat_pipe, cat_col)])

from sklearn.model_selection import StratifiedKFold

models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=7),
    "XGBoost": XGBClassifier(
        use_label_encoder=False, eval_metric="logloss", n_jobs=-1, random_state=7
    ),
    "LightGBM": LGBMClassifier(n_jobs=-1, random_state=7),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=7),
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
results = []

for name, model in models.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
    results.append({"Model": name, "CV Mean Accuracy": scores.mean(), "CV Std": scores.std()})

print("\n")
print(pd.DataFrame(results).sort_values(by="CV Mean Accuracy", ascending=False))


# y_pred = best_model.predict(X_test)
# print("\nClassification Report on Test Set:")
# print(classification_report(y_test, y_pred))
