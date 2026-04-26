"""
RxGuard AI v2 — Complete Data Preparation
Run once before starting the app.
"""
import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

print("=" * 60)
print("RxGuard AI v2 — Data Setup")
print("=" * 60)

# ── 1. Drugs ──────────────────────────────────────────────────
print("\n[1/6] Creating drug database...")

drugs = [
    ("Aspirin","Analgesic"),("Ibuprofen","NSAID"),("Paracetamol","Analgesic"),
    ("Metformin","Antidiabetic"),("Atorvastatin","Statin"),
    ("Amlodipine","CCB"),("Lisinopril","ACE Inhibitor"),("Omeprazole","PPI"),
    ("Metoprolol","Beta Blocker"),("Losartan","ARB"),
    ("Warfarin","Anticoagulant"),("Clopidogrel","Antiplatelet"),
    ("Amoxicillin","Antibiotic"),("Azithromycin","Antibiotic"),
    ("Ciprofloxacin","Antibiotic"),("Doxycycline","Antibiotic"),
    ("Metronidazole","Antibiotic"),("Fluconazole","Antifungal"),
    ("Prednisone","Corticosteroid"),("Prednisolone","Corticosteroid"),
    ("Dexamethasone","Corticosteroid"),("Salbutamol","Bronchodilator"),
    ("Montelukast","Antiasthmatic"),("Cetirizine","Antihistamine"),
    ("Loratadine","Antihistamine"),("Fexofenadine","Antihistamine"),
    ("Diazepam","Benzodiazepine"),("Alprazolam","Benzodiazepine"),
    ("Zolpidem","Sedative"),("Sertraline","SSRI"),
    ("Fluoxetine","SSRI"),("Escitalopram","SSRI"),
    ("Amitriptyline","TCA"),("Gabapentin","Anticonvulsant"),
    ("Pregabalin","Anticonvulsant"),("Carbamazepine","Anticonvulsant"),
    ("Phenytoin","Anticonvulsant"),("Valproate","Anticonvulsant"),
    ("Levodopa","Antiparkinsonian"),("Donepezil","Cholinesterase Inhibitor"),
    ("Insulin","Antidiabetic"),("Glibenclamide","Antidiabetic"),
    ("Sitagliptin","Antidiabetic"),("Empagliflozin","Antidiabetic"),
    ("Levothyroxine","Thyroid"),("Hydrochlorothiazide","Diuretic"),
    ("Furosemide","Diuretic"),("Spironolactone","Diuretic"),
    ("Digoxin","Cardiac Glycoside"),("Amiodarone","Antiarrhythmic"),
    ("Atenolol","Beta Blocker"),("Bisoprolol","Beta Blocker"),
    ("Ramipril","ACE Inhibitor"),("Valsartan","ARB"),
    ("Diltiazem","CCB"),("Verapamil","CCB"),
    ("Simvastatin","Statin"),("Rosuvastatin","Statin"),
    ("Allopurinol","Antigout"),("Colchicine","Antigout"),
    ("Naproxen","NSAID"),("Diclofenac","NSAID"),
    ("Celecoxib","NSAID"),("Morphine","Opioid"),
    ("Tramadol","Opioid"),("Codeine","Opioid"),
    ("Ondansetron","Antiemetic"),("Metoclopramide","Antiemetic"),
    ("Pantoprazole","PPI"),("Ranitidine","H2 Blocker"),
    ("Methotrexate","DMARD"),("Hydroxychloroquine","DMARD"),
    ("Azathioprine","Immunosuppressant"),("Warfarin","Anticoagulant"),
    ("Heparin","Anticoagulant"),("Rivaroxaban","Anticoagulant"),
    ("Apixaban","Anticoagulant"),("Dabigatran","Anticoagulant"),
    ("Erythromycin","Antibiotic"),("Clarithromycin","Antibiotic"),
    ("Clindamycin","Antibiotic"),("Vancomycin","Antibiotic"),
    ("Acyclovir","Antiviral"),("Oseltamivir","Antiviral"),
    ("Finasteride","Hormone"),("Dutasteride","Hormone"),
    ("Tamoxifen","Anticancer"),("Imatinib","Anticancer"),
    ("Diphenhydramine","Antihistamine"),("Promethazine","Antihistamine"),
    ("Adrenaline","Vasopressor"),("Dopamine","Vasopressor"),
    ("Nitroglycerin","Nitrate"),("Isosorbide","Nitrate"),
]

drug_df = pd.DataFrame([{"id": i+1, "name": d[0], "category": d[1]} for i, d in enumerate(drugs)])
drug_df.drop_duplicates(subset=["name"], inplace=True)
drug_df.to_csv(os.path.join(DATA_DIR, "drugs.csv"), index=False)
print(f"   ✓ {len(drug_df)} drugs saved")

# ── 2. Interactions ───────────────────────────────────────────
print("\n[2/6] Building interaction database...")

interactions = [
    # Severe
    ("Warfarin","Aspirin","severe",
     "Concurrent use significantly increases bleeding risk through complementary anticoagulant mechanisms.",
     "Use Paracetamol for pain relief instead of Aspirin.",
     "Pharmacodynamic synergy — both inhibit clotting cascade"),
    ("Warfarin","Ibuprofen","severe",
     "NSAIDs potentiate anticoagulant effect of Warfarin and damage gastric mucosa causing serious GI bleeding.",
     "Use Paracetamol. Monitor INR closely if NSAID unavoidable.",
     "CYP2C9 inhibition increases Warfarin plasma levels"),
    ("Metformin","Alcohol","severe",
     "Combination increases risk of lactic acidosis — rare but potentially fatal metabolic complication.",
     "Avoid alcohol completely while on Metformin.",
     "Both inhibit hepatic gluconeogenesis, increasing lactate"),
    ("Simvastatin","Amiodarone","severe",
     "Amiodarone inhibits CYP3A4 metabolism of Simvastatin causing dangerous accumulation and rhabdomyolysis.",
     "Switch to Rosuvastatin or Pravastatin (less CYP3A4 dependent).",
     "CYP3A4 inhibition by Amiodarone increases statin AUC by 75%"),
    ("Digoxin","Amiodarone","severe",
     "Amiodarone increases Digoxin levels by 70–100% risking toxicity: nausea, arrhythmias, visual disturbances.",
     "Reduce Digoxin dose by 50% and monitor serum levels closely.",
     "P-glycoprotein inhibition reduces Digoxin renal clearance"),
    ("Warfarin","Ciprofloxacin","severe",
     "Ciprofloxacin inhibits Warfarin metabolism dramatically increasing bleeding risk.",
     "Use alternative antibiotic if possible. Monitor INR daily.",
     "CYP1A2 inhibition raises Warfarin (R-enantiomer) levels"),
    ("Sertraline","Tramadol","severe",
     "Risk of serotonin syndrome — potentially life-threatening. Both increase serotonin activity.",
     "Avoid combination. Use non-serotonergic analgesics like Paracetamol.",
     "Combined serotonergic effect exceeds safe threshold"),
    ("Fluoxetine","Tramadol","severe",
     "High risk of serotonin syndrome. Symptoms: agitation, confusion, rapid heart rate, high blood pressure.",
     "Switch to non-SSRI or use different analgesic entirely.",
     "Fluoxetine also inhibits CYP2D6, impairing Tramadol metabolism"),
    ("Methotrexate","Ibuprofen","severe",
     "NSAIDs reduce renal clearance of Methotrexate leading to toxic accumulation.",
     "Avoid all NSAIDs with Methotrexate. Use Paracetamol cautiously.",
     "NSAID-mediated reduction in renal prostaglandins decreases MTX clearance"),
    ("Carbamazepine","Erythromycin","severe",
     "Erythromycin inhibits Carbamazepine metabolism causing toxicity: diplopia, ataxia, drowsiness.",
     "Use Azithromycin instead — significantly less CYP3A4 inhibition.",
     "CYP3A4 inhibition raises Carbamazepine levels 2–3 fold"),
    ("Warfarin","Clarithromycin","severe",
     "Clarithromycin strongly inhibits CYP3A4 and CYP2C9 causing marked rise in Warfarin effect.",
     "Use Azithromycin. If Clarithromycin necessary, reduce Warfarin dose and monitor INR daily.",
     "Dual CYP inhibition — most dangerous macrolide-Warfarin combination"),
    ("Metoprolol","Verapamil","severe",
     "Both slow heart rate — combination causes dangerous bradycardia and potential heart block.",
     "Avoid combination. Use Amlodipine (dihydropyridine CCB) instead of Verapamil.",
     "Additive AV node suppression through complementary mechanisms"),

    # Moderate
    ("Aspirin","Ibuprofen","moderate",
     "Ibuprofen blocks Aspirin binding to COX-1, reducing its antiplatelet effect and cardiovascular protection.",
     "Take Aspirin 2 hours before Ibuprofen or use Paracetamol for pain.",
     "Competitive COX-1 binding — Ibuprofen occupies active site first"),
    ("Atorvastatin","Amlodipine","moderate",
     "Amlodipine increases Atorvastatin exposure by ~18%. Monitor for muscle pain (myalgia).",
     "Cap Atorvastatin dose at 40mg. Switch to Rosuvastatin if symptoms occur.",
     "Mild CYP3A4 inhibition by Amlodipine"),
    ("Metformin","Furosemide","moderate",
     "Furosemide may increase Metformin plasma levels. Risk of lactic acidosis in dehydration states.",
     "Monitor renal function and hydration status closely.",
     "Furosemide reduces renal tubular secretion of Metformin"),
    ("Lisinopril","Spironolactone","moderate",
     "Combination causes dangerous hyperkalemia (high potassium) — can cause fatal arrhythmias.",
     "Monitor serum potassium regularly. Avoid potassium supplements.",
     "Both retain potassium through different mechanisms — additive effect"),
    ("Diazepam","Omeprazole","moderate",
     "Omeprazole inhibits CYP2C19 slowing Diazepam metabolism and increasing sedation risk.",
     "Reduce Diazepam dose. Consider Pantoprazole as alternative PPI.",
     "CYP2C19 inhibition increases Diazepam half-life significantly"),
    ("Amlodipine","Simvastatin","moderate",
     "Amlodipine inhibits Simvastatin metabolism increasing myopathy and rhabdomyolysis risk.",
     "Do not exceed 20mg Simvastatin with Amlodipine. Switch to Rosuvastatin.",
     "CYP3A4 inhibition by Amlodipine raises Simvastatin AUC"),
    ("Prednisolone","Ibuprofen","moderate",
     "Combination greatly increases risk of peptic ulcers and gastrointestinal bleeding.",
     "Add Omeprazole for gastric protection. Use Paracetamol for pain where possible.",
     "Corticosteroids impair mucosal defence; NSAIDs inhibit prostaglandin synthesis"),
    ("Levothyroxine","Omeprazole","moderate",
     "PPIs raise gastric pH reducing absorption of Levothyroxine and may cause hypothyroidism.",
     "Take Levothyroxine 30–60 min before Omeprazole on empty stomach.",
     "pH-dependent dissolution — Levothyroxine requires acidic environment for absorption"),
    ("Gabapentin","Morphine","moderate",
     "Additive CNS depression with increased risk of respiratory depression — especially in elderly.",
     "Use lowest effective doses. Monitor respiratory rate and sedation level.",
     "Synergistic CNS depression through different receptor mechanisms"),
    ("Allopurinol","Amoxicillin","moderate",
     "Combination significantly increases risk of maculopapular skin rash (up to 20% incidence).",
     "Use Cephalexin or Azithromycin instead of Amoxicillin if possible.",
     "Mechanism unclear — possibly immune-mediated hypersensitivity reaction"),
    ("Furosemide","Digoxin","moderate",
     "Furosemide causes potassium and magnesium loss which sensitises heart to Digoxin toxicity.",
     "Monitor electrolytes regularly. Supplement potassium or add Spironolactone.",
     "Hypokalemia increases binding of Digoxin to myocardial Na/K-ATPase"),
    ("Omeprazole","Clopidogrel","moderate",
     "Omeprazole inhibits CYP2C19 activation of Clopidogrel reducing its antiplatelet effect by ~40%.",
     "Switch to Pantoprazole or Rabeprazole — less CYP2C19 inhibition.",
     "Clopidogrel is a prodrug requiring CYP2C19 activation to active thiol metabolite"),
    ("Ciprofloxacin","Metformin","moderate",
     "Ciprofloxacin may cause both hypo and hyperglycaemia unpredictably in diabetic patients.",
     "Monitor blood glucose frequently. Consider alternative antibiotic.",
     "Ciprofloxacin stimulates insulin secretion and reduces hepatic glucose output"),

    # Mild
    ("Aspirin","Atorvastatin","mild",
     "Minor pharmacokinetic interaction — generally well tolerated at standard doses.",
     "No significant action needed. Routine monitoring.",
     "Minor competition for plasma protein binding"),
    ("Cetirizine","Diazepam","mild",
     "Additive sedative effect. May cause drowsiness — avoid driving or operating machinery.",
     "Take Cetirizine at night. Warn patient about sedation.",
     "Additive CNS depression through different receptor pathways"),
    ("Metformin","Amlodipine","mild",
     "Calcium channel blockers may slightly impair insulin secretion and glucose tolerance.",
     "Monitor blood glucose. Usually not clinically significant at standard doses.",
     "CCBs block calcium-dependent insulin secretion from pancreatic beta cells"),
    ("Paracetamol","Warfarin","mild",
     "High or prolonged doses of Paracetamol (>2g/day) can increase INR slightly.",
     "Limit Paracetamol to 2g/day maximum in patients on Warfarin. Monitor INR weekly.",
     "Paracetamol metabolite inhibits Vitamin K epoxide reductase at high doses"),
    ("Lisinopril","Ibuprofen","mild",
     "NSAIDs may reduce antihypertensive effect of ACE inhibitors by blocking prostaglandin synthesis.",
     "Monitor blood pressure. Use Paracetamol for pain where possible.",
     "NSAIDs reduce renal prostaglandins that mediate ACE inhibitor vasodilation"),
    ("Atenolol","Metformin","mild",
     "Beta-blockers may mask tachycardia — an early warning sign of hypoglycaemia.",
     "Monitor for other hypoglycaemia signs (sweating, hunger). Patient education important.",
     "Beta-blockade masks adrenergic hypoglycaemia symptoms"),
]

int_df = pd.DataFrame(interactions,
    columns=["drug1","drug2","severity","description","alternative","mechanism"])
int_df.to_csv(os.path.join(DATA_DIR, "interactions.csv"), index=False)
print(f"   ✓ {len(int_df)} interactions saved")

# ── 3. Brand names ────────────────────────────────────────────
print("\n[3/6] Creating brand name database...")

brands = [
    ("Crocin","Paracetamol"),("Calpol","Paracetamol"),("Tylenol","Paracetamol"),
    ("Brufen","Ibuprofen"),("Advil","Ibuprofen"),("Nurofen","Ibuprofen"),
    ("Disprin","Aspirin"),("Ecosprin","Aspirin"),("Bayer","Aspirin"),
    ("Glucophage","Metformin"),("Glycomet","Metformin"),("Fortamet","Metformin"),
    ("Lipitor","Atorvastatin"),("Sortis","Atorvastatin"),
    ("Norvasc","Amlodipine"),("Amlip","Amlodipine"),("Amlong","Amlodipine"),
    ("Zestril","Lisinopril"),("Prinivil","Lisinopril"),
    ("Prilosec","Omeprazole"),("Losec","Omeprazole"),("Ocid","Omeprazole"),
    ("Lopressor","Metoprolol"),("Toprol","Metoprolol"),("Betaloc","Metoprolol"),
    ("Cozaar","Losartan"),("Repace","Losartan"),
    ("Coumadin","Warfarin"),("Warf","Warfarin"),
    ("Plavix","Clopidogrel"),("Clopilet","Clopidogrel"),
    ("Amoxil","Amoxicillin"),("Novamox","Amoxicillin"),
    ("Zithromax","Azithromycin"),("Azee","Azithromycin"),("Zithrocin","Azithromycin"),
    ("Cipro","Ciprofloxacin"),("Ciplox","Ciprofloxacin"),
    ("Vibramycin","Doxycycline"),("Doxy","Doxycycline"),
    ("Flagyl","Metronidazole"),("Metrogyl","Metronidazole"),
    ("Diflucan","Fluconazole"),("Zocon","Fluconazole"),
    ("Deltasone","Prednisone"),("Omnacortil","Prednisolone"),
    ("Ventolin","Salbutamol"),("Asthalin","Salbutamol"),
    ("Singulair","Montelukast"),("Montair","Montelukast"),
    ("Zyrtec","Cetirizine"),("Cetzine","Cetirizine"),
    ("Claritin","Loratadine"),("Lorfast","Loratadine"),
    ("Valium","Diazepam"),("Calmpose","Diazepam"),
    ("Xanax","Alprazolam"),("Restyl","Alprazolam"),
    ("Ambien","Zolpidem"),("Stilnoct","Zolpidem"),
    ("Zoloft","Sertraline"),("Daxid","Sertraline"),("Serlift","Sertraline"),
    ("Prozac","Fluoxetine"),("Fludac","Fluoxetine"),
    ("Lexapro","Escitalopram"),("Nexito","Escitalopram"),
    ("Elavil","Amitriptyline"),("Tryptomer","Amitriptyline"),
    ("Neurontin","Gabapentin"),("Gabapin","Gabapentin"),
    ("Lyrica","Pregabalin"),("Pregeb","Pregabalin"),
    ("Tegretol","Carbamazepine"),("Mazetol","Carbamazepine"),
    ("Dilantin","Phenytoin"),("Eptoin","Phenytoin"),
    ("Depakote","Valproate"),("Valparin","Valproate"),
    ("Syndopa","Levodopa"),("Sinemet","Levodopa"),
    ("Aricept","Donepezil"),("Donamem","Donepezil"),
    ("Glucovance","Glibenclamide"),("Daonil","Glibenclamide"),
    ("Januvia","Sitagliptin"),
    ("Jardiance","Empagliflozin"),
    ("Synthroid","Levothyroxine"),("Thyronorm","Levothyroxine"),("Eltroxin","Levothyroxine"),
    ("Lasix","Furosemide"),("Frusemide","Furosemide"),
    ("Aldactone","Spironolactone"),("Aldactide","Spironolactone"),
    ("Lanoxin","Digoxin"),("Digoxin","Digoxin"),
    ("Cordarone","Amiodarone"),("Amiodar","Amiodarone"),
    ("Tenormin","Atenolol"),("Tenolol","Atenolol"),
    ("Altace","Ramipril"),("Cardace","Ramipril"),
    ("Diovan","Valsartan"),("Valzaar","Valsartan"),
    ("Cardizem","Diltiazem"),("Dilzem","Diltiazem"),
    ("Isoptin","Verapamil"),("Calaptin","Verapamil"),
    ("Zocor","Simvastatin"),("Simvas","Simvastatin"),
    ("Crestor","Rosuvastatin"),("Rozat","Rosuvastatin"),
    ("Zyloprim","Allopurinol"),("Zyloric","Allopurinol"),
    ("Naprosyn","Naproxen"),("Naprogesic","Naproxen"),
    ("Voltaren","Diclofenac"),("Voveran","Diclofenac"),
    ("Celebrex","Celecoxib"),("Cobix","Celecoxib"),
    ("MSContin","Morphine"),("Ultracet","Tramadol"),
    ("Zofran","Ondansetron"),("Emeset","Ondansetron"),
    ("Reglan","Metoclopramide"),("Perinorm","Metoclopramide"),
    ("Protonix","Pantoprazole"),("Pan","Pantoprazole"),
    ("Zantac","Ranitidine"),("Rantac","Ranitidine"),
    ("Rheumatrex","Methotrexate"),("Folitrax","Methotrexate"),
    ("Plaquenil","Hydroxychloroquine"),("HCQS","Hydroxychloroquine"),
    ("Imuran","Azathioprine"),
    ("Xarelto","Rivaroxaban"),
    ("Eliquis","Apixaban"),
    ("Pradaxa","Dabigatran"),
    ("Erythrocin","Erythromycin"),("Biaxin","Clarithromycin"),("Klaricid","Clarithromycin"),
    ("Cleocin","Clindamycin"),("Vancocin","Vancomycin"),
    ("Zovirax","Acyclovir"),("Acivir","Acyclovir"),
    ("Tamiflu","Oseltamivir"),
    ("Proscar","Finasteride"),("Finpecia","Finasteride"),
    ("Nolvadex","Tamoxifen"),("Tamodex","Tamoxifen"),
    ("Gleevec","Imatinib"),
    ("Benadryl","Diphenhydramine"),
    ("GTN","Nitroglycerin"),("Sorbitrate","Isosorbide"),
]

brand_df = pd.DataFrame(brands, columns=["brand","generic"])
brand_df.to_csv(os.path.join(DATA_DIR, "brand_names.csv"), index=False)
print(f"   ✓ {len(brand_df)} brand names saved")

# ── 4. Pregnancy unsafe drugs ─────────────────────────────────
print("\n[4/6] Creating pregnancy safety database...")

preg_unsafe = [
    "warfarin","methotrexate","thalidomide","isotretinoin","valproate",
    "carbamazepine","phenytoin","tetracycline","doxycycline","ciprofloxacin",
    "ibuprofen","naproxen","diclofenac","celecoxib","aspirin",
    "atorvastatin","simvastatin","rosuvastatin","lisinopril","ramipril",
    "losartan","valsartan","amiodarone","lithium","fluconazole",
    "misoprostol","tamoxifen","finasteride","dutasteride","ribavirin",
]
pd.DataFrame({"drug": preg_unsafe}).to_csv(
    os.path.join(DATA_DIR, "pregnancy_unsafe.csv"), index=False)
print(f"   ✓ {len(preg_unsafe)} pregnancy-unsafe drugs saved")

# ── 5. Allergy classes ────────────────────────────────────────
print("\n[5/6] Creating allergy class database...")

allergy_data = [
    ("Amoxicillin","Penicillin"),("Ampicillin","Penicillin"),("Piperacillin","Penicillin"),
    ("Ciprofloxacin","Fluoroquinolone"),("Levofloxacin","Fluoroquinolone"),("Moxifloxacin","Fluoroquinolone"),
    ("Ibuprofen","NSAID"),("Naproxen","NSAID"),("Diclofenac","NSAID"),("Celecoxib","NSAID"),
    ("Codeine","Opioid"),("Morphine","Opioid"),("Tramadol","Opioid"),("Fentanyl","Opioid"),
    ("Cetirizine","Antihistamine"),("Loratadine","Antihistamine"),("Diphenhydramine","Antihistamine"),
    ("Sulfasalazine","Sulfonamide"),("Trimethoprim","Sulfonamide"),
    ("Carbamazepine","Anticonvulsant"),("Phenytoin","Anticonvulsant"),("Valproate","Anticonvulsant"),
    ("Azithromycin","Macrolide"),("Erythromycin","Macrolide"),("Clarithromycin","Macrolide"),
]
pd.DataFrame(allergy_data, columns=["drug","allergy_class"]).to_csv(
    os.path.join(DATA_DIR, "allergy_classes.csv"), index=False)
print(f"   ✓ {len(allergy_data)} allergy mappings saved")

# ── 6. Verify ─────────────────────────────────────────────────
print("\n[6/6] Verifying all data files...")
files = ["drugs.csv","interactions.csv","brand_names.csv","pregnancy_unsafe.csv","allergy_classes.csv"]
for f in files:
    path = os.path.join(DATA_DIR, f)
    df = pd.read_csv(path)
    print(f"   ✓ {f}: {len(df)} rows")

print("\n" + "=" * 60)
print("✅ Data setup complete! Run: python app.py")
print("=" * 60)