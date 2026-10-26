"""Train, compare, evaluate and export the Spaceship Titanic model."""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, OrdinalEncoder, StandardScaler

from app.features import CATEGORICAL, NUMERIC, RAW_COLUMNS, engineer

SEED = 42
df = pd.read_csv("data/train.csv")
X, y = df[RAW_COLUMNS], df["Transported"].astype(int)

# Hold out 20% that is NEVER used for model selection
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

def make_pipeline(estimator, linear: bool):
    if linear:  # linear models need scaling + one-hot
        cat = OneHotEncoder(handle_unknown="ignore")
        num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    else:       # trees: ordinal codes, unseen -> -1, NaNs handled natively
        cat = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        num = SimpleImputer(strategy="median")
    pre = ColumnTransformer([("cat", cat, CATEGORICAL), ("num", num, NUMERIC)])
    return Pipeline([("features", FunctionTransformer(engineer)), ("pre", pre), ("model", estimator)])

candidates = {
    "LogisticRegression": make_pipeline(LogisticRegression(max_iter=2000), linear=True),
    "RandomForest": make_pipeline(RandomForestClassifier(n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=SEED), linear=False),
    "HistGradientBoosting": make_pipeline(HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0, random_state=SEED), linear=False),
}

cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
scores = {}
for name, pipe in candidates.items():
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="accuracy")
    scores[name] = float(s.mean())
    print(f"{name:22s} CV accuracy = {s.mean():.4f} (+/- {s.std():.4f})")

best_name = max(scores, key=scores.get)
print("\nBest model:", best_name)

# Honest estimate on the untouched hold-out set
best = candidates[best_name].fit(X_tr, y_tr)
pred = best.predict(X_val)
proba = best.predict_proba(X_val)[:, 1]
acc, auc = accuracy_score(y_val, pred), roc_auc_score(y_val, proba)
print(f"\nHold-out accuracy = {acc:.4f} | ROC-AUC = {auc:.4f}")
print(confusion_matrix(y_val, pred))
print(classification_report(y_val, pred, target_names=["Not transported", "Transported"]))

# Refit on ALL labelled data for the final exported model
final = make_pipeline(
    candidates[best_name].named_steps["model"].__class__(**candidates[best_name].named_steps["model"].get_params()),
    linear=(best_name == "LogisticRegression"),
).fit(X, y)

joblib.dump(final, "artifacts/model.joblib")
meta = {"model": best_name, "cv_accuracy": scores, "holdout_accuracy": acc, "holdout_roc_auc": auc,
        "n_train": int(len(X)), "raw_columns": RAW_COLUMNS}
json.dump(meta, open("artifacts/metadata.json", "w"), indent=2)

# Kaggle-style predictions for test.csv
test = pd.read_csv("data/test.csv")
sub = pd.DataFrame({"PassengerId": test["PassengerId"], "Transported": final.predict(test[RAW_COLUMNS]).astype(bool)})
sub.to_csv("artifacts/submission.csv", index=False)
print("\nSaved model.joblib, metadata.json, submission.csv")
