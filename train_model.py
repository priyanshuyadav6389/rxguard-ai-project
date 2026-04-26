"""
RxGuard AI — ML Model Training
Uses: DrugBank + TWOSIDES + FAERS merged dataset
Trains: Random Forest, Logistic Regression, SVM (Module 3)
        + ANN via Keras (Module 4)
Run: python train_model.py
"""

import os, pickle, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.pipeline import Pipeline

DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
PLOTS_DIR  = os.path.join(os.path.dirname(__file__), "static", "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

SEP = "=" * 60

def load_data():
    master = os.path.join(DATA_DIR, "master_interactions.csv")
    if os.path.exists(master):
        df = pd.read_csv(master)
        print(f"   ✓  Loaded master dataset: {len(df)} rows")
    else:
        df = pd.read_csv(os.path.join(DATA_DIR, "interactions.csv"))
        print(f"   ✓  Loaded curated interactions: {len(df)} rows")

    # Load drug categories
    drugs_df = pd.read_csv(os.path.join(DATA_DIR, "drugs.csv"))
    cat_map = drugs_df.set_index("name")["category"].to_dict()
    df["drug1_category"] = df["drug1"].map(cat_map).fillna("Unknown")
    df["drug2_category"] = df["drug2"].map(cat_map).fillna("Unknown")

    return df

def engineer_features(df):
    """Build feature matrix from interaction data."""
    le1 = LabelEncoder()
    le2 = LabelEncoder()

    all_cats = list(set(df["drug1_category"].tolist() + df["drug2_category"].tolist()))
    le1.fit(all_cats); le2.fit(all_cats)

    features = pd.DataFrame({
        "cat1_enc":    le1.transform(df["drug1_category"]),
        "cat2_enc":    le2.transform(df["drug2_category"]),
        "max_prr":     df.get("max_prr", pd.Series([0.0]*len(df))).fillna(0),
        "has_death":   df.get("has_death_report", pd.Series([0]*len(df))).fillna(0),
        "report_cnt":  df.get("report_count", pd.Series([0]*len(df))).fillna(0),
        "same_class":  (df["drug1_category"] == df["drug2_category"]).astype(int),
    })
    return features, le1, le2

def plot_confusion_matrix(cm, labels, title, filename):
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(labels)), yticks=range(len(labels)),
           xticklabels=labels, yticklabels=labels,
           title=title, ylabel="True label", xlabel="Predicted label")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   ✓  Plot saved: static/plots/{filename}")

def plot_feature_importance(rf, feature_names, filename):
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(8,4))
    ax.bar(range(len(importances)), importances[indices], color="#C0392B", alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=30, ha="right")
    ax.set_title("Random Forest — Feature Importance")
    ax.set_ylabel("Importance Score")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   ✓  Plot saved: static/plots/{filename}")

def plot_model_comparison(results, filename):
    models = list(results.keys())
    accs   = [results[m]["accuracy"] for m in models]
    colors = ["#C0392B","#E67E22","#2980B9","#27AE60"]
    fig, ax = plt.subplots(figsize=(8,4))
    bars = ax.bar(models, accs, color=colors[:len(models)], alpha=0.85, edgecolor="white")
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{acc:.1%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_title("Model Comparison — Accuracy on Test Set", fontsize=13)
    ax.set_ylabel("Accuracy")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xticklabels(models, rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, filename), dpi=100, bbox_inches="tight")
    plt.close()
    print(f"   ✓  Plot saved: static/plots/{filename}")

def train_ann(X_train, X_test, y_train, y_test, n_classes):
    """Train ANN using Keras (Module 4 — Deep Learning)."""
    try:
        import tensorflow as tf
        from tensorflow import keras

        model = keras.Sequential([
            keras.layers.Input(shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(n_classes, activation="softmax")
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])

        history = model.fit(X_train, y_train,
                            epochs=50, batch_size=8,
                            validation_split=0.2, verbose=0)

        _, acc = model.evaluate(X_test, y_test, verbose=0)

        # Plot training curves
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
        ax1.plot(history.history["accuracy"], label="Train", color="#C0392B")
        ax1.plot(history.history["val_accuracy"], label="Validation", color="#2980B9")
        ax1.set_title("ANN — Accuracy Curve"); ax1.set_xlabel("Epoch")
        ax1.legend(); ax1.spines[["top","right"]].set_visible(False)

        ax2.plot(history.history["loss"], label="Train", color="#C0392B")
        ax2.plot(history.history["val_loss"], label="Validation", color="#2980B9")
        ax2.set_title("ANN — Loss Curve"); ax2.set_xlabel("Epoch")
        ax2.legend(); ax2.spines[["top","right"]].set_visible(False)

        plt.tight_layout()
        plt.savefig(os.path.join(PLOTS_DIR, "ann_training_curves.png"),
                    dpi=100, bbox_inches="tight")
        plt.close()
        print(f"   ✓  ANN trained — Test Accuracy: {acc:.1%}")
        print(f"   ✓  Plot saved: static/plots/ann_training_curves.png")

        model.save(os.path.join(MODELS_DIR, "ann_model.keras"))
        print(f"   ✓  ANN model saved: models/ann_model.keras")

        return model, acc
    except ImportError:
        print("   ⚠  TensorFlow not installed. Skipping ANN.")
        print("      Install with: pip install tensorflow")
        return None, 0.0


# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print(SEP)
    print("RxGuard AI — Model Training Pipeline")
    print(SEP)

    # ── Load & prepare data
    print("\n[1/5] Loading data...")
    df = load_data()
    X, le1, le2 = engineer_features(df)
    sev_map = {"severe": 3, "moderate": 2, "mild": 1, "none": 0}
    y_raw = df["severity"].map(sev_map).fillna(0).astype(int)
    le_sev = LabelEncoder()
    y = le_sev.fit_transform(y_raw)
    n_classes = len(le_sev.classes_)
    feature_names = X.columns.tolist()
    print(f"   ✓  Features: {feature_names}")
    print(f"   ✓  Classes: {le_sev.classes_} → {list(range(n_classes))}")
    print(f"   ✓  Samples: {len(y)}")

    X_arr = X.values.astype(float)
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y, test_size=0.2, random_state=42, stratify=y if len(set(y))>1 else None)

    # ── Train classical ML models (Module 3)
    print("\n[2/5] Training Classical ML Models (Module 3 — Supervised Learning)...")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    models_config = {
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Logistic Regression": Pipeline([("scaler", StandardScaler()),
                                          ("clf", LogisticRegression(max_iter=1000, random_state=42))]),
        "SVM": Pipeline([("scaler", StandardScaler()),
                         ("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    }

    results = {}
    label_names = ["none","mild","moderate","severe"]

    for name, model in models_config.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cv  = cross_val_score(model, X_arr, y, cv=min(5,len(y)), scoring="accuracy").mean()
        results[name] = {"accuracy": acc, "cv_accuracy": cv, "model": model}
        print(f"   ✓  {name}: Acc={acc:.1%}, CV={cv:.1%}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        actual_labels = [label_names[c] for c in sorted(set(y_test)|set(y_pred))
                         if c < len(label_names)]
        plot_confusion_matrix(cm, actual_labels,
                              f"{name} — Confusion Matrix",
                              f"cm_{name.lower().replace(' ','_')}.png")

    # Feature importance plot (Random Forest)
    rf_model = results["Random Forest"]["model"]
    plot_feature_importance(rf_model, feature_names, "feature_importance.png")

    # Model comparison plot
    plot_model_comparison(results, "model_comparison.png")

    # ── Save best classical model
    print("\n[3/5] Saving best model...")
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    print(f"   ✓  Best model: {best_name} ({results[best_name]['accuracy']:.1%})")

    with open(os.path.join(MODELS_DIR, "best_model.pkl"), "wb") as f:
        pickle.dump({
            "model":    best_model,
            "le1":      le1,
            "le2":      le2,
            "le_sev":   le_sev,
            "scaler":   scaler,
            "features": feature_names,
            "results":  {k: {"accuracy": v["accuracy"],
                             "cv_accuracy": v["cv_accuracy"]} for k,v in results.items()}
        }, f)
    print(f"   ✓  Saved: models/best_model.pkl")

    # ── Train ANN (Module 4)
    print("\n[4/5] Training ANN (Module 4 — Deep Learning)...")
    ann_model, ann_acc = train_ann(X_train_sc, X_test_sc, y_train, y_test, n_classes)

    # ── Summary report
    print("\n[5/5] Training Summary")
    print("─" * 40)
    for name, res in results.items():
        print(f"   {name:<25} Acc: {res['accuracy']:.1%}  CV: {res['cv_accuracy']:.1%}")
    if ann_acc > 0:
        print(f"   {'ANN (Keras)':<25} Acc: {ann_acc:.1%}")

    print("\n" + SEP)
    print("✅ Training complete!")
    print(f"   Models saved in: models/")
    print(f"   Plots saved in:  static/plots/")
    print(f"   Run: python app.py")
    print(SEP)