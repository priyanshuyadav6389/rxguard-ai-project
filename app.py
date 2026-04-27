from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os, pickle, base64
from itertools import combinations

app = Flask(__name__)
CORS(app)

BASE      = os.path.dirname(__file__)
DATA      = os.path.join(BASE, "data")
MODELS    = os.path.join(BASE, "models")
PLOTS_DIR = os.path.join(BASE, "static", "plots")

# ── Load datasets ──────────────────────────────────────────────
drug_df     = pd.read_csv(os.path.join(DATA, "drugs.csv"))
interact_df = pd.read_csv(os.path.join(DATA, "interactions.csv"))
brand_df    = pd.read_csv(os.path.join(DATA, "brand_names.csv"))
preg_df     = pd.read_csv(os.path.join(DATA, "pregnancy_unsafe.csv"))
allergy_df  = pd.read_csv(os.path.join(DATA, "allergy_classes.csv"))

interaction_map = {}
for _, row in interact_df.iterrows():
    key = frozenset([str(row["drug1"]).strip().lower(), str(row["drug2"]).strip().lower()])
    interaction_map[key] = {
        "severity":    str(row.get("severity","moderate")).strip().lower(),
        "description": str(row.get("description","Potential interaction detected.")),
        "alternative": str(row.get("alternative","Consult your physician.")),
        "mechanism":   str(row.get("mechanism","Pharmacokinetic/pharmacodynamic interaction."))
    }

brand_map = {str(r["brand"]).strip().lower(): str(r["generic"]).strip()
             for _, r in brand_df.iterrows()}

all_drug_names = sorted(set(
    drug_df["name"].dropna().str.strip().tolist() +
    brand_df["brand"].dropna().str.strip().tolist()
))

preg_unsafe  = set(preg_df["drug"].str.strip().str.lower().tolist())
allergy_map  = {str(r["drug"]).strip().lower(): str(r["allergy_class"]).strip()
                for _, r in allergy_df.iterrows()}
drug_cat_map = drug_df.drop_duplicates("name").set_index("name")["category"].to_dict()

# ── Load ML model ──────────────────────────────────────────────
model_data = None
model_path = os.path.join(MODELS, "best_model.pkl")
if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)

def ml_predict(drug1, drug2):
    if not model_data:
        return "unknown", 50, {}
    try:
        le1    = model_data["le1"]
        le2    = model_data["le2"]
        le_sev = model_data["le_sev"]
        model  = model_data["model"]
        cat1   = drug_cat_map.get(drug1, "Unknown")
        cat2   = drug_cat_map.get(drug2, "Unknown")
        c1 = cat1 if cat1 in le1.classes_ else "Unknown"
        c2 = cat2 if cat2 in le2.classes_ else "Unknown"
        x = np.array([[
            le1.transform([c1])[0] if c1 in le1.classes_ else 0,
            le2.transform([c2])[0] if c2 in le2.classes_ else 0,
            0, 0, 0, int(cat1 == cat2)
        ]])
        pred   = model.predict(x)[0]
        proba  = model.predict_proba(x)[0]
        label  = le_sev.inverse_transform([pred])[0]
        conf   = round(float(proba.max()) * 100)
        sev_inv = {3:"severe", 2:"moderate", 1:"mild", 0:"none"}
        label_str = sev_inv.get(int(label), "unknown")
        all_probs = {sev_inv.get(int(le_sev.inverse_transform([i])[0]), "none"):
                     round(float(p)*100) for i, p in enumerate(proba)}
        return label_str, conf, all_probs
    except:
        return "unknown", 50, {}

# ── Helpers ────────────────────────────────────────────────────
SEVERITY_ORDER = {"severe":3,"moderate":2,"mild":1,"none":0,"unknown":0}

def resolve_drug(name):
    return brand_map.get(name.strip().lower(), name.strip())

def normalise(name): return name.strip().lower()

def check_interactions(drugs):
    results = []
    for d1, d2 in combinations(drugs, 2):
        key = frozenset([normalise(d1), normalise(d2)])
        ml_sev, ml_conf, ml_probs = ml_predict(d1, d2)
        if key in interaction_map:
            info = interaction_map[key]
            results.append({
                "drug1":d1,"drug2":d2,
                "severity":    info["severity"],
                "description": info["description"],
                "alternative": info["alternative"],
                "mechanism":   info["mechanism"],
                "ml_severity": ml_sev,
                "ml_confidence": ml_conf,
                "ml_probs":    ml_probs,
                "source":      "database"
            })
        else:
            results.append({
                "drug1":d1,"drug2":d2,
                "severity":    "none",
                "description": "No known interaction found.",
                "alternative": "—",
                "mechanism":   "—",
                "ml_severity": ml_sev,
                "ml_confidence": ml_conf,
                "ml_probs":    ml_probs,
                "source":      "ml_only"
            })
    results.sort(key=lambda x: SEVERITY_ORDER.get(x["severity"],0), reverse=True)
    return results

def overall_risk(interactions):
    sevs = [SEVERITY_ORDER.get(i["severity"],0) for i in interactions]
    return ["none","mild","moderate","severe"][max(sevs) if sevs else 0]

def risk_score(interactions):
    weights = {"severe":100,"moderate":50,"mild":20,"none":0}
    if not interactions: return 0
    scores = [weights.get(i["severity"],0) for i in interactions]
    return min(100, int(sum(scores)/len(scores) + max(scores)*0.3))

# ── Routes ─────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/drugs")
def get_drugs():
    q = request.args.get("q","").lower()
    matches = [d for d in all_drug_names if q in d.lower()][:20]
    return jsonify(matches)

@app.route("/api/check", methods=["POST"])
def check():
    data       = request.get_json()
    raw_drugs  = [d.strip() for d in data.get("drugs",[]) if d.strip()]
    age        = data.get("age","")
    gender     = data.get("gender","")
    pregnant   = data.get("pregnant", False)
    conditions = data.get("conditions",[])
    allergies  = data.get("allergies","").strip().lower()

    if len(raw_drugs) < 2:
        return jsonify({"error":"Please enter at least 2 drugs."}), 400

    drugs    = [resolve_drug(d) for d in raw_drugs]
    resolved = {orig:res for orig,res in zip(raw_drugs,drugs)
                if orig.strip().lower() != res.strip().lower()}

    interactions = check_interactions(drugs)
    risk  = overall_risk(interactions)
    score = risk_score(interactions)

    warnings = []
    cond_lower = [c.lower() for c in conditions]

    for drug in drugs:
        dn = normalise(drug)
        if "kidney" in cond_lower and dn in ["metformin","ibuprofen","naproxen","diclofenac","celecoxib"]:
            warnings.append({"type":"contraindication","icon":"🔴","text":f"{drug} may worsen kidney function."})
        if "liver" in cond_lower and dn in ["paracetamol","methotrexate","amiodarone","atorvastatin"]:
            warnings.append({"type":"contraindication","icon":"🔴","text":f"{drug} requires caution in liver disease."})
        if "heart" in cond_lower and dn in ["ibuprofen","naproxen","diclofenac","celecoxib"]:
            warnings.append({"type":"contraindication","icon":"🔴","text":f"{drug} (NSAID) increases cardiovascular risk."})
        if "diabetes" in cond_lower and dn in ["prednisone","prednisolone","dexamethasone"]:
            warnings.append({"type":"contraindication","icon":"🔴","text":f"{drug} raises blood glucose — monitor carefully."})
        if pregnant and dn in preg_unsafe:
            warnings.append({"type":"pregnancy","icon":"🤰","text":f"{drug} is UNSAFE during pregnancy."})
        drug_class = allergy_map.get(dn)
        if drug_class and drug_class.lower() in allergies:
            warnings.append({"type":"allergy","icon":"⚠️","text":f"{drug} belongs to {drug_class} — allergy risk."})
        if gender == "female" and dn in ["finasteride","dutasteride"]:
            warnings.append({"type":"gender","icon":"⚠️","text":f"{drug} is contraindicated in women."})

    if age:
        try:
            age_int = int(age)
            for drug in drugs:
                dn = normalise(drug)
                if age_int > 65 and dn in ["diazepam","zolpidem","alprazolam","amitriptyline","diphenhydramine"]:
                    warnings.append({"type":"age","icon":"👴","text":f"{drug} is high-risk in patients over 65 (Beers Criteria)."})
                if age_int < 18 and dn == "aspirin":
                    warnings.append({"type":"age","icon":"👶","text":"Aspirin is contraindicated under 18 (Reye's syndrome)."})
        except: pass

    nodes = [{"id":d,"category":drug_cat_map.get(d,"Unknown")} for d in drugs]
    links = [{"source":i["drug1"],"target":i["drug2"],"severity":i["severity"]}
             for i in interactions if i["severity"] != "none"]

    return jsonify({
        "drugs":         drugs,
        "raw_drugs":     raw_drugs,
        "resolved":      resolved,
        "risk":          risk,
        "risk_score":    score,
        "interactions":  interactions,
        "warnings":      warnings,
        "graph":         {"nodes":nodes,"links":links},
        "model_results": model_data.get("results",{}) if model_data else {},
    })

@app.route("/api/plots")
def plots_api():
    result = {}
    for name in ["model_comparison","feature_importance","ann_training_curves"]:
        path = os.path.join(PLOTS_DIR, f"{name}.png")
        if os.path.exists(path):
            with open(path,"rb") as f:
                result[name] = base64.b64encode(f.read()).decode()
    return jsonify(result)

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port)