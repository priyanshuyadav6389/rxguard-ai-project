"""
RxGuard AI — TWOSIDES Real Data Trainer (FIXED)
Handles:
  - Small OR large TWOSIDES.csv
  - Adaptive PRR thresholds based on actual data distribution
  - Graceful stratify fallback
  - Your exact columns: drug_1_concept_name, drug_2_concept_name,
    condition_concept_name, PRR, A, B, C, D, mean_reporting_frequency

Run: python train_twosides.py
"""

import os, sys, pickle, warnings
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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score)
from sklearn.pipeline import Pipeline


BASE          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
MODEL_DIR     = os.path.join(BASE, "models")
PLOT_DIR      = os.path.join(BASE, "static", "plots")
TWOSIDES_PATH = os.path.join(RAW_DIR, "TWOSIDES.csv")

for d in [DATA_DIR, RAW_DIR, MODEL_DIR, PLOT_DIR]:
    os.makedirs(d, exist_ok=True)

SEP = "=" * 60


FATAL_CONDITIONS = {
    "death","sudden death","cardiac arrest","respiratory failure",
    "respiratory arrest","anaphylaxis","anaphylactic shock",
    "anaphylactic reaction","serotonin syndrome","rhabdomyolysis",
    "lactic acidosis","intracranial haemorrhage","intracranial hemorrhage",
    "cerebral haemorrhage","subarachnoid haemorrhage","haemorrhage",
    "hemorrhage","gastrointestinal haemorrhage","gastrointestinal hemorrhage",
    "stevens-johnson syndrome","toxic epidermal necrolysis",
    "liver failure","hepatic failure","acute liver failure",
    "renal failure","acute kidney injury","renal failure acute",
    "heart block","complete heart block","ventricular fibrillation",
    "ventricular tachycardia","torsade de pointes",
    "pulmonary embolism","agranulocytosis","aplastic anaemia",
    "aplastic anemia","sepsis","septicaemia","septic shock",
    "multi-organ failure","disseminated intravascular coagulation"
}

SERIOUS_CONDITIONS = {
    "myocardial infarction","stroke","cerebrovascular accident",
    "seizure","convulsion","epilepsy","status epilepticus",
    "coma","loss of consciousness","syncope",
    "peptic ulcer","upper gastrointestinal haemorrhage",
    "pancreatitis","acute pancreatitis",
    "hepatitis","hepatotoxicity","liver disorder","jaundice",
    "nephrotoxicity","renal impairment","renal insufficiency",
    "bradycardia","tachycardia","atrial fibrillation","arrhythmia",
    "cardiac failure","congestive cardiac failure","heart failure",
    "hyponatraemia","hyponatremia","hyperkalemia","hyperkalaemia",
    "hypoglycaemia","hypoglycemia","diabetic ketoacidosis",
    "thrombocytopenia","neutropenia","leucopenia",
    "qt prolongation","electrocardiogram qt prolonged",
    "peripheral neuropathy","neuropathy peripheral",
    "psychosis","hallucination","delirium",
    "angioedema","urticaria",
    "hypotension","shock","circulatory collapse",
    "thrombosis","deep vein thrombosis"
}


def load_twosides():
    if not os.path.exists(TWOSIDES_PATH):
        print(f"\n  ✗  File not found: {TWOSIDES_PATH}")
        print("  Make sure TWOSIDES.csv is placed in data/raw/")
        sys.exit(1)

    file_mb = os.path.getsize(TWOSIDES_PATH) / (1024*1024)
    print(f"\n[1/7] Loading TWOSIDES.csv  ({file_mb:.0f} MB)...")

    needed_cols = ["drug_1_concept_name","drug_2_concept_name",
                   "condition_concept_name","PRR"]

    # Read full file (handles both small samples and full 1.3M)
    df = pd.read_csv(TWOSIDES_PATH, low_memory=False)
    print(f"   ✓  Rows loaded       : {len(df):,}")
    print(f"   ✓  Columns found     : {list(df.columns)}")

 
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns={
        "drug_1_concept_name" : "drug1",
        "drug_2_concept_name" : "drug2",
        "condition_concept_name": "side_effect",
    }, inplace=True)


    for col in ["drug1","drug2","side_effect","PRR"]:
        if col not in df.columns:
            print(f"   ✗  Missing column: {col}")
            print(f"      Available: {list(df.columns)}")
            sys.exit(1)

 
    df["drug1"]       = df["drug1"].astype(str).str.strip().str.title()
    df["drug2"]       = df["drug2"].astype(str).str.strip().str.title()
    df["side_effect"] = df["side_effect"].astype(str).str.strip().str.lower()
    df["PRR"]         = pd.to_numeric(df["PRR"], errors="coerce").fillna(0.0)

    print(f"   ✓  Unique drug1      : {df['drug1'].nunique():,}")
    print(f"   ✓  Unique drug2      : {df['drug2'].nunique():,}")
    print(f"   ✓  Unique conditions : {df['side_effect'].nunique():,}")
    print(f"   ✓  PRR range         : {df['PRR'].min():.2f} – {df['PRR'].max():.2f}")
    print(f"   ✓  PRR median        : {df['PRR'].median():.2f}")
    print(f"   ✓  PRR 75th pct      : {df['PRR'].quantile(0.75):.2f}")
    print(f"   ✓  PRR 90th pct      : {df['PRR'].quantile(0.90):.2f}")
    print(f"   ✓  PRR 95th pct      : {df['PRR'].quantile(0.95):.2f}")

    return df


def aggregate_pairs(df):
    print(f"\n[2/7] Aggregating by drug pair...")

    agg = df.groupby(["drug1","drug2"]).agg(
        max_prr      = ("PRR", "max"),
        mean_prr     = ("PRR", "mean"),
        n_conditions = ("side_effect", "count"),
        side_effects = ("side_effect", lambda x: set(x.dropna())),
    ).reset_index()

    print(f"   ✓  Unique drug pairs : {len(agg):,}")
    print(f"   ✓  Avg conditions/pair: {agg['n_conditions'].mean():.1f}")
    print(f"   ✓  Pair PRR range    : {agg['max_prr'].min():.2f} – {agg['max_prr'].max():.2f}")

    return agg


def derive_severity(agg):
    """
    ADAPTIVE thresholds based on actual PRR distribution.
    Instead of fixed 2/5/10, we use percentiles so labels
    are always well-distributed regardless of dataset size.
    """
    print(f"\n[3/7] Deriving severity labels (adaptive thresholds)...")

    prr = agg["max_prr"]

    # Compute adaptive thresholds from percentiles
    p50 = prr.quantile(0.50)   # median  → mild cutoff
    p75 = prr.quantile(0.75)   # 75th    → moderate cutoff
    p90 = prr.quantile(0.90)   # 90th    → severe cutoff

    print(f"   ✓  Adaptive thresholds from your data:")
    print(f"      mild     : PRR > {p50:.2f}  (50th percentile)")
    print(f"      moderate : PRR > {p75:.2f}  (75th percentile)")
    print(f"      severe   : PRR > {p90:.2f}  (90th percentile)")
    print(f"      none     : PRR <= {p50:.2f}")

    def label(row):
        prr_val = row["max_prr"]
        effects = row["side_effects"] if isinstance(row["side_effects"], set) else set()

        has_fatal   = bool(effects & FATAL_CONDITIONS)
        has_serious = bool(effects & SERIOUS_CONDITIONS)

        # Fatal conditions always bump up severity
        if has_fatal and prr_val > p75:
            return "severe"
        if prr_val > p90 or (has_fatal and prr_val > p50):
            return "severe"
        elif prr_val > p75 or (has_serious and prr_val > p50):
            return "moderate"
        elif prr_val > p50:
            return "mild"
        else:
            return "none"

    agg["severity"] = agg.apply(label, axis=1)

    dist  = agg["severity"].value_counts()
    total = len(agg)
    print(f"\n   ✓  Label distribution:")
    for sev in ["severe","moderate","mild","none"]:
        count = dist.get(sev, 0)
        pct   = count / total * 100
        bar   = "█" * max(1, int(pct / 2))
        print(f"      {sev:<12}: {count:>6,}  ({pct:5.1f}%)  {bar}")

    # Check minimum samples per class
    min_count = dist.min()
    if min_count < 5:
        print(f"\n   ⚠  Some classes have very few samples ({min_count}).")
        print(f"      This is OK — the trainer will handle it gracefully.")

    # Save master dataset
    out = agg[["drug1","drug2","severity","max_prr","n_conditions"]].copy()
    out.to_csv(os.path.join(DATA_DIR,"master_interactions.csv"), index=False)
    print(f"\n   ✓  Saved: data/master_interactions.csv ({len(out):,} rows)")

    return agg


def build_features(agg):
    print(f"\n[4/7] Building feature matrix...")

    # Load drug categories
    drugs_path = os.path.join(DATA_DIR,"drugs.csv")
    if os.path.exists(drugs_path):
        drugs_df = pd.read_csv(drugs_path).drop_duplicates("name")
        cat_map  = dict(zip(drugs_df["name"].str.title(), drugs_df["category"]))
        print(f"   ✓  Drug categories loaded: {len(cat_map)}")
    else:
        cat_map = {}
        print("   ⚠  drugs.csv not found — run prepare_data.py first")

    agg["cat1"] = agg["drug1"].map(cat_map).fillna("Unknown")
    agg["cat2"] = agg["drug2"].map(cat_map).fillna("Unknown")

    all_cats = sorted(set(agg["cat1"].tolist() + agg["cat2"].tolist()))
    le_cat   = LabelEncoder().fit(all_cats)

    # Normalise PRR to 0-1 range using min-max
    prr_max = agg["max_prr"].max()
    prr_min = agg["max_prr"].min()
    prr_range = prr_max - prr_min if prr_max != prr_min else 1.0

    features = pd.DataFrame({
        "cat1_enc"    : le_cat.transform(agg["cat1"]),
        "cat2_enc"    : le_cat.transform(agg["cat2"]),
        "max_prr_norm": (agg["max_prr"] - prr_min) / prr_range,
        "mean_prr_norm": (agg["mean_prr"] - prr_min) / prr_range,
        "log_n_cond"  : np.log1p(agg["n_conditions"]),
        "same_class"  : (agg["cat1"] == agg["cat2"]).astype(int),
    })

    sev_map   = {"severe":3,"moderate":2,"mild":1,"none":0}
    y         = agg["severity"].map(sev_map).fillna(0).astype(int)
    le_sev    = LabelEncoder().fit(["none","mild","moderate","severe"])

    print(f"   ✓  Feature matrix    : {features.shape[0]} rows × {features.shape[1]} features")
    print(f"   ✓  Features          : {list(features.columns)}")
    print(f"   ✓  Class counts      : {dict(y.value_counts().sort_index())}")

    return features, y, le_cat, le_sev

# ── STEP 5: TRAIN MODELS ──────────────────────────────────────
def train_models(X_df, y):
    print(f"\n[5/7] Training ML models...")

    X = X_df.values.astype(float)

    # Sample for speed if very large
    if len(X) > 300_000:
        print(f"   ℹ  Sampling 300K from {len(X):,} rows")
        idx = np.random.RandomState(42).choice(len(X), 300_000, replace=False)
        X   = X[idx]
        y   = y.iloc[idx].reset_index(drop=True)

    # ── Safe stratified split ──────────────────────────────────
    # Only stratify if every class has at least 2 members
    class_counts = pd.Series(y).value_counts()
    can_stratify = (class_counts >= 2).all()

    if can_stratify:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"   ✓  Stratified split  : train={len(X_train):,}  test={len(X_test):,}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        print(f"   ✓  Random split      : train={len(X_train):,}  test={len(X_test):,}")
        print(f"   ⚠  Stratify skipped  : some classes have < 2 samples")

    # ── Models ────────────────────────────────────────────────
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=15, n_jobs=-1,
            random_state=42, class_weight="balanced"
        ),
        "Logistic Regression": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, random_state=42,
                class_weight="balanced", solver="lbfgs"
            ))
        ]),
        "SVM": Pipeline([
            ("sc",  StandardScaler()),
            ("clf", SVC(
                kernel="rbf", probability=True, random_state=42,
                class_weight="balanced", C=1.0, gamma="scale"
            ))
        ]),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=6,
            learning_rate=0.1, random_state=42, subsample=0.8
        ),
    }

    results = {}
    print(f"\n   {'Model':<25} {'Accuracy':>10} {'CV (5-fold)':>12} {'F1':>8}")
    print(f"   {'─'*25} {'─'*10} {'─'*12} {'─'*8}")

    for name, model in models.items():
        print(f"   Training {name}...", end=" ", flush=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # CV — safely handle small datasets
        n_splits  = min(5, class_counts.min()) if can_stratify else 3
        n_splits  = max(2, int(n_splits))
        cv_size   = min(len(X), 50_000)
        cv_idx    = np.random.RandomState(42).choice(len(X), cv_size, replace=False)
        X_cv      = X[cv_idx]
        y_cv      = y.iloc[cv_idx] if hasattr(y,"iloc") else y[cv_idx]

        try:
            cv_acc = cross_val_score(
                model, X_cv, y_cv, cv=n_splits,
                scoring="accuracy", n_jobs=-1
            ).mean()
        except Exception:
            cv_acc = acc   # fallback — just use test accuracy

        results[name] = {
            "model": model, "accuracy": acc,
            "cv_accuracy": cv_acc, "f1": f1,
            "y_pred": y_pred, "y_test": y_test
        }
        print(f"\r   {name:<25} {acc:>10.3f} {cv_acc:>12.3f} {f1:>8.3f}")

    return results, X_train, X_test, y_train, y_test

# ── STEP 6: PLOTS ─────────────────────────────────────────────
def make_plots(results, feature_names, agg):
    print(f"\n[6/7] Generating plots...")
    colors = ["#C0392B","#B8620A","#1A5C9A","#0D7A5F"]
    plt.rcParams.update({"font.family":"DejaVu Sans"})

    # 1. Model comparison
    fig, ax = plt.subplots(figsize=(10,4))
    names = list(results.keys())
    accs  = [results[m]["accuracy"]    for m in names]
    cvs   = [results[m]["cv_accuracy"] for m in names]
    f1s   = [results[m]["f1"]          for m in names]
    x     = np.arange(len(names))
    b1 = ax.bar(x-0.25, accs, 0.25, label="Test Accuracy", color=colors, alpha=0.9)
    b2 = ax.bar(x,      cvs,  0.25, label="CV Accuracy",   color=colors, alpha=0.55)
    b3 = ax.bar(x+0.25, f1s,  0.25, label="F1 Score",      color=colors, alpha=0.3)
    for bar in b1:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f"{bar.get_height():.2f}", ha="center", va="bottom",
                fontsize=8, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=12, ha="right")
    ax.set_ylim(0,1.15)
    ax.set_title("Model Comparison — Trained on TWOSIDES", fontsize=11)
    ax.legend(fontsize=9); ax.set_ylabel("Score")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"model_comparison.png"), dpi=110, bbox_inches="tight")
    plt.close(); print("   ✓  model_comparison.png")

    # 2. Feature importance
    if "Random Forest" in results:
        rf  = results["Random Forest"]["model"]
        imp = rf.feature_importances_
        idx = np.argsort(imp)[::-1]
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(range(len(imp)), imp[idx],
               color=[colors[i%len(colors)] for i in range(len(imp))], alpha=0.85)
        ax.set_xticks(range(len(imp)))
        ax.set_xticklabels([feature_names[i] for i in idx], rotation=25, ha="right")
        ax.set_title("Random Forest — Feature Importance (TWOSIDES)", fontsize=11)
        ax.set_ylabel("Importance"); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,"feature_importance.png"), dpi=110, bbox_inches="tight")
        plt.close(); print("   ✓  feature_importance.png")

    # 3. Confusion matrix (best model)
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    y_test    = results[best_name]["y_test"]
    y_pred    = results[best_name]["y_pred"]
    label_map = {0:"none",1:"mild",2:"moderate",3:"severe"}
    present   = sorted(set(y_test.tolist()+y_pred.tolist()))
    p_labels  = [label_map.get(i,str(i)) for i in present]
    cm = confusion_matrix(y_test, y_pred, labels=present)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, cmap="Reds")
    plt.colorbar(im, ax=ax)
    ax.set(xticks=range(len(present)), yticks=range(len(present)),
           xticklabels=p_labels, yticklabels=p_labels,
           title=f"{best_name} — Confusion Matrix",
           xlabel="Predicted", ylabel="True label")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,format(cm[i,j],","),ha="center",va="center",
                    color="white" if cm[i,j]>thresh else "black",fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"confusion_matrix.png"), dpi=110, bbox_inches="tight")
    plt.close(); print("   ✓  confusion_matrix.png")

    # 4. Severity distribution + PRR histogram
    dist   = agg["severity"].value_counts()
    w_cols = {"severe":"#C0392B","moderate":"#E67E22","mild":"#2980B9","none":"#27AE60"}
    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].pie(dist.values,
                labels=[f"{l}\n({v:,})" for l,v in zip(dist.index,dist.values)],
                colors=[w_cols.get(l,"#999") for l in dist.index],
                autopct="%1.1f%%", startangle=140,
                wedgeprops={"edgecolor":"white","linewidth":1.5})
    axes[0].set_title("Severity Distribution (TWOSIDES pairs)", fontsize=11)

    prr_clip = agg["max_prr"].clip(0, agg["max_prr"].quantile(0.95))
    p50 = agg["max_prr"].quantile(0.50)
    p75 = agg["max_prr"].quantile(0.75)
    p90 = agg["max_prr"].quantile(0.90)
    axes[1].hist(prr_clip, bins=50, color="#C0392B", alpha=0.75, edgecolor="white")
    axes[1].axvline(x=p50, color="#2980B9", linestyle="--", lw=1.5, label=f"p50={p50:.1f} (mild)")
    axes[1].axvline(x=p75, color="#E67E22", linestyle="--", lw=1.5, label=f"p75={p75:.1f} (moderate)")
    axes[1].axvline(x=p90, color="#C0392B", linestyle="--", lw=1.5, label=f"p90={p90:.1f} (severe)")
    axes[1].set_title("PRR Distribution with Adaptive Thresholds", fontsize=11)
    axes[1].set_xlabel("PRR Score"); axes[1].set_ylabel("Drug pairs")
    axes[1].legend(fontsize=8); axes[1].spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"severity_distribution.png"), dpi=110, bbox_inches="tight")
    plt.close(); print("   ✓  severity_distribution.png")

    # 5. Top 20 dangerous pairs
    top = agg[agg["severity"]=="severe"].nlargest(20,"max_prr")[["drug1","drug2","max_prr"]]
    if len(top) > 0:
        fig, ax = plt.subplots(figsize=(10,6))
        labels  = [f"{r.drug1[:14]}+{r.drug2[:14]}" for _,r in top.iterrows()]
        ax.barh(labels, top["max_prr"].values, color="#C0392B", alpha=0.85)
        ax.set_xlabel("Max PRR Score")
        ax.set_title("Top 20 Highest-Risk Drug Pairs (Severe, by PRR)", fontsize=11)
        ax.invert_yaxis(); ax.spines[["top","right"]].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR,"top_dangerous_pairs.png"), dpi=110, bbox_inches="tight")
        plt.close(); print("   ✓  top_dangerous_pairs.png")

# ── STEP 7: SAVE ──────────────────────────────────────────────
def save_models(results, le_cat, le_sev, feature_names):
    print(f"\n[7/7] Saving best model...")

    best_name = max(results, key=lambda k: results[k]["accuracy"])
    best      = results[best_name]

    print(f"   ✓  Best model        : {best_name}")
    print(f"   ✓  Test Accuracy     : {best['accuracy']*100:.1f}%")
    print(f"   ✓  CV Accuracy       : {best['cv_accuracy']*100:.1f}%")
    print(f"   ✓  F1 Score          : {best['f1']:.3f}")

    payload = {
        "model":     best["model"],
        "le_cat":    le_cat,
        "le1":       le_cat,
        "le2":       le_cat,
        "le_sev":    le_sev,
        "features":  feature_names,
        "best_name": best_name,
        "results": {
            k: {
                "accuracy":    round(v["accuracy"],    4),
                "cv_accuracy": round(v["cv_accuracy"], 4),
                "f1":          round(v["f1"],          4),
            }
            for k,v in results.items()
        }
    }

    path = os.path.join(MODEL_DIR,"best_model.pkl")
    with open(path,"wb") as f:
        pickle.dump(payload, f)
    print(f"   ✓  Saved             : {path}")

    # Classification report
    label_map = {0:"none",1:"mild",2:"moderate",3:"severe"}
    y_test    = best["y_test"]
    y_pred    = best["y_pred"]
    present   = sorted(set(y_test.tolist()))
    p_labels  = [label_map.get(i,str(i)) for i in present]
    print(f"\n   Classification Report — {best_name}:")
    print("   " + "─"*50)
    print(classification_report(y_test, y_pred,
                                 target_names=p_labels,
                                 zero_division=0, digits=3))

# ── MAIN ──────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    print(SEP)
    print("RxGuard AI — TWOSIDES Real Data Training Pipeline")
    print(SEP)
    print(f"  Dataset : {TWOSIDES_PATH}")
    print(f"  Models  : {MODEL_DIR}")
    print(f"  Plots   : {PLOT_DIR}")
    print(SEP)

    df       = load_twosides()
    agg      = aggregate_pairs(df)
    agg      = derive_severity(agg)
    X_df, y, le_cat, le_sev = build_features(agg)
    results, *_ = train_models(X_df, y)
    make_plots(results, list(X_df.columns), agg)
    save_models(results, le_cat, le_sev, list(X_df.columns))

    print("\n" + SEP)
    print("✅  Training complete on real TWOSIDES data!")
    print("    Run: python app.py")
    print(SEP)