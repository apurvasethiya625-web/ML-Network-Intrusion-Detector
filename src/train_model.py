"""Train and evaluate a simple ML intrusion detector on a labelled CSV."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a network intrusion classifier.")
    parser.add_argument("--data", required=True, help="Path to labelled CSV file")
    parser.add_argument("--target", default="Label", help="Target/label column name")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--sample", type=int, default=100_000,
                        help="Maximum rows to load for laptop-friendly training; use 0 for all rows")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"Dataset not found: {data_path}. Download a labelled CSV and retry.")

    # Load a manageable sample for beginner/laptop-friendly training.
    if args.sample > 0:
        df = pd.read_csv(data_path, nrows=args.sample, low_memory=False)
    else:
        df = pd.read_csv(data_path, low_memory=False)

    df.columns = [str(c).strip() for c in df.columns]
    if args.target not in df.columns:
        raise SystemExit(
            f"Target column '{args.target}' not found. Available columns: {df.columns.tolist()}"
        )

    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna(subset=[args.target])
    y_raw = df.pop(args.target).astype(str).str.strip()
    X = df.drop(columns=[c for c in df.columns if c.lower() in {"flow id", "source ip", "destination ip", "src ip", "dst ip"}],
                errors="ignore")

    # Drop columns that are entirely empty and rows with no labels.
    X = X.dropna(axis=1, how="all")
    if X.shape[1] == 0:
        raise SystemExit("No usable feature columns found in the CSV.")

    # Keep only features with some variation; avoids trivial constant columns.
    X = X.loc[:, X.nunique(dropna=False) > 1]
    if X.shape[1] == 0:
        raise SystemExit("All feature columns are constant in the selected data.")

    # Stratify only when every class has at least two samples.
    counts = y_raw.value_counts()
    stratify = y_raw if len(counts) > 1 and counts.min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_raw, test_size=args.test_size, random_state=42, stratify=stratify
    )

    numeric_cols = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X_train.columns if c not in numeric_cols]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ],
        remainder="drop",
    )

    candidates = {
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=120, max_depth=18, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
    }

    results = {}
    best_name, best_f1, best_pipeline = None, -1.0, None
    for name, model in candidates.items():
        pipe = Pipeline([("preprocess", preprocess), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        score = f1_score(y_test, pred, average="macro", zero_division=0)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision_macro": float(precision_score(y_test, pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, pred, average="macro", zero_division=0)),
            "f1_macro": float(score),
            "classification_report": classification_report(
                y_test, pred, output_dict=True, zero_division=0
            ),
        }
        print(f"\n{name}")
        print(classification_report(y_test, pred, zero_division=0))
        if score > best_f1:
            best_name, best_f1, best_pipeline = name, score, pipe
            best_pred = pred

    model_dir, report_dir = ROOT / "models", ROOT / "reports"
    model_dir.mkdir(exist_ok=True)
    report_dir.mkdir(exist_ok=True)
    joblib.dump(
        {"pipeline": best_pipeline, "feature_columns": X.columns.tolist(),
         "target_column": args.target, "model_name": best_name},
        model_dir / "intrusion_model.joblib",
    )
    with open(report_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"selected_model": best_name, "models": results}, f, indent=2)

    labels = sorted(y_test.unique().tolist())
    cm = confusion_matrix(y_test, best_pred, labels=labels)
    plt.figure(figsize=(max(7, len(labels) * 0.7), max(5, len(labels) * 0.6)))
    sns.heatmap(cm, annot=len(labels) <= 12, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {best_name}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(report_dir / "confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance works for tree-based estimators; use transformed names.
    model = best_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        try:
            names = best_pipeline.named_steps["preprocess"].get_feature_names_out()
            importance = pd.Series(model.feature_importances_, index=names).sort_values().tail(20)
            plt.figure(figsize=(9, 7))
            importance.plot(kind="barh")
            plt.title(f"Top Feature Importances — {best_name}")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(report_dir / "feature_importance.png", dpi=150)
            plt.close()
        except Exception as exc:
            print(f"Feature importance chart skipped: {exc}")

    print(f"\nSelected model: {best_name} (macro F1={best_f1:.4f})")
    print(f"Saved model: {model_dir / 'intrusion_model.joblib'}")
    print(f"Saved reports to: {report_dir}")


if __name__ == "__main__":
    main()
