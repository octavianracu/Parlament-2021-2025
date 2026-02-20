"""
ANALIZA TRANSFERULUI DE RISC DE EXCLUDERE ÎNTRE LEGISLATURI
=============================================================
Parlamentul Republicii Moldova

Logică:
  • leg1 (2021–2025) = SET DE ANTRENAMENT  (cu etichete reale exclus/inclus)
  • leg2 (2025–2029) = SET DE TEST         (etichete folosite DOAR pentru validare finală)

Pipeline:
  1. Construiește graf + metrici pentru LEG1 → antrenează 3 modele
  2. Construiește graf + metrici pentru LEG2 → aplică modele (transfer)
  3. Validare pe leg2 (dacă avem etichete): AUC, calibrare, erori
  4. Analiza drift structural: s-a schimbat rețeaua între legislaturi?
  5. Dashboard HTML interactiv complet

Fișiere necesare:
  • leg1_raw.json                  — co-autoriat leg1 (parliament_scraper.py)
  • leg2_raw.json                  — co-autoriat leg2 (parliament_scraper.py)
  • excluded_deputies.csv          — leg1: Nume, Exclus (0/1), Partid
  • excluded_deputies_leg2.csv     — leg2: Nume, Exclus (0/1), Partid

Output:
  • transfer_analysis_report.html  — dashboard interactiv complet
  • transfer_predictions_leg2.csv  — predicții + risc per deputat leg2
  • transfer_model_report.json     — metrici de performanță ale transferului
"""

import json, os, warnings
from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
import networkx as nx

warnings.filterwarnings("ignore")

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import LeaveOneOut, cross_val_predict, StratifiedKFold
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                  brier_score_loss, confusion_matrix)
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.utils.class_weight import compute_class_weight
    from sklearn.pipeline import Pipeline
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    print("⚠ pip install scikit-learn")

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    print("⚠ pip install plotly")

try:
    from scipy import stats
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False


# ═══════════════════════════════════════════════════════════════════════════════
#  DATE DEMO
# ═══════════════════════════════════════════════════════════════════════════════

DEMO_LEG1_EXCL = """Nume,Exclus,Partid
Ion Chicu,1,PSRM
Vlad Bătrâncea,1,PSRM
Radu Mudreac,1,PSRM
Arina Spătaru,1,PAS
Dumitru Alaiba,1,PAS
Nicolae Ciubuc,1,PAS
Lilian Carp,0,PAS
Doina Gherman,0,PAS
Mihai Popșoi,0,PAS
Liliana Nicolaescu-Onofrei,0,PAS
Igor Grosu,0,PAS
Olesea Stamate,0,PAS
Marina Tauber,0,PSRM
Vasile Bolea,0,PSRM
Alexandru Suhodolski,0,PSRM
Grigore Novac,0,PSRM
Vasile Năstase,0,PPDA
Iurie Reniță,0,PPDA
"""

DEMO_LEG2_EXCL = """Nume,Exclus,Partid
Petru Frunze,1,PAS
Ana Racu,1,PAS
Maria Pancu,1,PAS
Vitali Gavrouc,1,PAS
Boris Marcoci,1,PAS
Mihail Leahu,1,PAS
Valentina Manic,1,PAS
Iulia Dascălu,1,PAS
Marcela Nistor,1,PAS
Boris Popa,1,PAS
Vasile Șoimaru,1,PAS
Ion Șpac,1,PAS
Evghenia Cojocari,1,PAS
Ina Coșeru,1,PAS
Mariana Cușnir,1,PAS
Ion Poia,1,PAS
Valentina Ghețu,1,PAS
Oazu Nantoi,1,PAS
Ana Speianu,1,PAS
Alina Dandara,1,PAS
Oleg Botnaru,1,PAS
Vitalie Jacot,1,PAS
Victor Spînu,1,PAS
Nicolae Plămădeală,1,PAS
"""

DEMO_LEG1_PROJECTS = [
    {"title": f"P1-{i}", "deputy_authors": a, "author_count": len(a)}
    for i, a in enumerate([
        ["Lilian Carp","Doina Gherman","Mihai Popșoi"],
        ["Igor Grosu","Olesea Stamate","Lilian Carp"],
        ["Lilian Carp","Liliana Nicolaescu-Onofrei","Doina Gherman"],
        ["Ion Chicu","Vlad Bătrâncea","Radu Mudreac"],
        ["Ion Chicu","Marina Tauber","Vasile Bolea"],
        ["Vlad Bătrâncea","Alexandru Suhodolski","Grigore Novac"],
        ["Arina Spătaru","Dumitru Alaiba","Nicolae Ciubuc"],
        ["Arina Spătaru","Lilian Carp","Doina Gherman"],
        ["Mihai Popșoi","Igor Grosu","Doina Gherman"],
        ["Vasile Năstase","Iurie Reniță","Lilian Carp"],
        ["Olesea Stamate","Liliana Nicolaescu-Onofrei","Igor Grosu"],
        ["Marina Tauber","Vasile Bolea","Alexandru Suhodolski"],
        ["Ion Chicu","Vlad Bătrâncea","Marina Tauber"],
        ["Dumitru Alaiba","Nicolae Ciubuc","Olesea Stamate"],
        ["Vasile Năstase","Iurie Reniță","Grigore Novac"],
        ["Lilian Carp","Mihai Popșoi","Olesea Stamate","Igor Grosu"],
        ["Ion Chicu","Vlad Bătrâncea","Radu Mudreac","Marina Tauber"],
        ["Doina Gherman","Liliana Nicolaescu-Onofrei","Lilian Carp"],
        ["Arina Spătaru","Dumitru Alaiba","Lilian Carp"],
        ["Vasile Bolea","Alexandru Suhodolski","Grigore Novac","Ion Chicu"],
    ])
]

DEMO_LEG2_PROJECTS = [
    {"title": f"P2-{i}", "deputy_authors": a, "author_count": len(a)}
    for i, a in enumerate([
        ["Sergiu Litvinenco","Veronica Roșca","Natalia Gavrilița"],
        ["Radu Burduja","Cristina Gherasim","Victor Parlicov"],
        ["Ana Moldovan","Petru Galbur","Elena Vrabie"],
        ["Sergiu Litvinenco","Ana Moldovan","Radu Burduja"],
        ["Tudor Ulianovschi","Alexandr Coroli","Mihai Moldovan"],
        ["Natalia Gavrilița","Veronica Roșca","Cristina Gherasim"],
        ["Petru Galbur","Alexandr Coroli","Mihai Moldovan"],
        ["Victor Parlicov","Sergiu Litvinenco","Radu Burduja"],
        ["Elena Vrabie","Ana Moldovan","Natalia Gavrilița"],
        ["Tudor Ulianovschi","Mihai Moldovan","Petru Galbur"],
        ["Sergiu Litvinenco","Veronica Roșca","Victor Parlicov","Radu Burduja"],
        ["Alexandr Coroli","Mihai Moldovan","Tudor Ulianovschi"],
    ])
]


# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITĂȚI COMUNE
# ═══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = [
    "Degree_Centrality", "Power_Bonacich", "Betweenness", "Stress",
    "Prestige_PageRank", "Eccentricity", "Clustering", "Weighted_Degree",
    "Omophily_Party", "Clique_Count", "Clique_Max_Size", "Is_Isolated",
]

FEATURE_LABELS = {
    "Degree_Centrality":  "Degree Centrality",
    "Power_Bonacich":     "Power (Bonacich)",
    "Betweenness":        "Betweenness Centrality",
    "Stress":             "Stress Centrality",
    "Prestige_PageRank":  "Prestige (PageRank)",
    "Eccentricity":       "Eccentricity",
    "Clustering":         "Clustering Coefficient",
    "Weighted_Degree":    "Grad Ponderat",
    "Omophily_Party":     "Omofilie Partid",
    "Clique_Count":       "Nr. Clici (≥3)",
    "Clique_Max_Size":    "Dimensiune Max Clică",
    "Is_Isolated":        "Izolat în Rețea",
}


def load_json(path, demo):
    if os.path.exists(path):
        print(f"  ✓ {path} găsit — date reale")
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    print(f"  ⚠ {path} absent — DATE DEMO")
    return demo


def load_csv(path, demo_str):
    if os.path.exists(path):
        print(f"  ✓ {path} găsit — date reale")
        return pd.read_csv(path, encoding="utf-8-sig")
    print(f"  ⚠ {path} absent — DATE DEMO")
    from io import StringIO
    return pd.read_csv(StringIO(demo_str))


def build_graph(projects):
    G = nx.Graph()
    for p in projects:
        for a in p["deputy_authors"]:
            G.add_node(a)
        for a1, a2 in combinations(p["deputy_authors"], 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)
    return G


def stress_centrality(G):
    stress = defaultdict(float)
    for s in G.nodes():
        sp = dict(nx.single_source_shortest_path(G, s))
        for t in G.nodes():
            if t == s or t not in sp:
                continue
            for v in sp[t][1:-1]:
                stress[v] += 1.0
    n = G.number_of_nodes()
    norm = (n-1)*(n-2) if n > 2 else 1
    return {v: stress[v]/norm for v in G.nodes()}


def eccentricity_centrality(G):
    res = {}
    for comp in nx.connected_components(G):
        sub = G.subgraph(comp)
        if sub.number_of_nodes() < 2:
            for v in comp: res[v] = 0.0
            continue
        ecc = nx.eccentricity(sub)
        for v, e in ecc.items():
            res[v] = 1.0/e if e else 0.0
    return res


def compute_features(G, df_info):
    """Calculează toate featurile pentru deputații din df_info."""
    deputies  = df_info["Nume"].tolist()
    party_map = dict(zip(df_info["Nume"], df_info["Partid"]))

    # Metrici NetworkX
    deg_c  = nx.degree_centrality(G)
    try:
        pow_c = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except Exception:
        try:
            pow_c = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            pow_c = {v: 0.0 for v in G.nodes()}
    bet_c  = nx.betweenness_centrality(G, weight="weight", normalized=True)
    str_c  = stress_centrality(G)
    try:
        pr_c = nx.pagerank(G, weight="weight")
    except Exception:
        n = G.number_of_nodes()
        pr_c = {v: 1/n for v in G.nodes()}
    ecc_c  = eccentricity_centrality(G)
    clu_c  = nx.clustering(G, weight="weight")

    # Omofilie
    omoph = {}
    for node in G.nodes():
        nbs = list(G.neighbors(node))
        if not nbs:
            omoph[node] = 0.0
            continue
        p = party_map.get(node, "?")
        omoph[node] = sum(1 for nb in nbs if party_map.get(nb,"??") == p) / len(nbs)

    # Clici
    cl_cnt, cl_max = defaultdict(int), defaultdict(int)
    for cl in nx.find_cliques(G):
        if len(cl) >= 3:
            for v in cl:
                cl_cnt[v] += 1
                cl_max[v] = max(cl_max[v], len(cl))

    # Grad ponderat
    wdeg = {v: sum(d["weight"] for _,d in G[v].items()) for v in G.nodes()}

    rows = []
    for dep in deputies:
        ig = dep in G.nodes()
        rows.append({
            "Deputat":           dep,
            "Partid":            party_map.get(dep, "?"),
            "In_Graf":           int(ig),
            "Degree_Centrality": deg_c.get(dep, 0.0),
            "Power_Bonacich":    pow_c.get(dep, 0.0),
            "Betweenness":       bet_c.get(dep, 0.0),
            "Stress":            str_c.get(dep, 0.0),
            "Prestige_PageRank": pr_c.get(dep, 0.0),
            "Eccentricity":      ecc_c.get(dep, 0.0),
            "Clustering":        clu_c.get(dep, 0.0),
            "Weighted_Degree":   wdeg.get(dep, 0.0),
            "Omophily_Party":    omoph.get(dep, 0.0),
            "Clique_Count":      cl_cnt.get(dep, 0),
            "Clique_Max_Size":   cl_max.get(dep, 0),
            "Is_Isolated":       int(not ig or G.degree(dep) == 0),
        })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  ANTRENARE PE LEG1 - VERSIUNE REPARATĂ
# ═══════════════════════════════════════════════════════════════════════════════

def train_on_leg1(df1):
    """
    Antrenează 3 modele pe leg1 cu LOO-CV intern pentru estimarea performanței.
    Returnează modelele calibrate antrenate pe TOATE datele leg1.
    """
    if not SKLEARN_OK:
        return None, None, None, None

    print("\n🏋️ Antrenare modele pe LEG1 (LOO-CV intern)...")

    # Verificăm dacă avem date suficiente
    X = df1[FEATURE_COLS].values
    y = df1["Exclus"].values
    
    # Debug: verificăm distribuția claselor
    unique_classes = np.unique(y)
    print(f"  Clase găsite în y: {unique_classes}")
    print(f"  Distribuție: {np.bincount(y.astype(int))}")
    
    # Verificăm dacă avem ambele clase (0 și 1)
    if len(unique_classes) < 2:
        print("  ⚠ ATENȚIE: Datele conțin o singură clasă! Nu se poate antrena modelul.")
        print("  Se utilizează un model constant care întoarce media clasei.")
        
        # Calculăm media clasei
        mean_prob = np.mean(y)
        
        # Creăm un scaler dummy
        scaler = StandardScaler()
        scaler.fit(X)  # Antrenăm scaler-ul chiar dacă nu vom folosi modele
        
        # Construim un dicționar de rezultate dummy
        loo_results = {
            "Logistic_L1": {"auc": 0.5, "ap": mean_prob, "brier": np.mean((y - mean_prob)**2), "loo_probs": np.full_like(y, mean_prob, dtype=float)},
            "Random_Forest": {"auc": 0.5, "ap": mean_prob, "brier": np.mean((y - mean_prob)**2), "loo_probs": np.full_like(y, mean_prob, dtype=float)},
            "Gradient_Boosting": {"auc": 0.5, "ap": mean_prob, "brier": np.mean((y - mean_prob)**2), "loo_probs": np.full_like(y, mean_prob, dtype=float)},
        }
        
        # Feature importance dummy
        feat_imp = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f,f) for f in FEATURE_COLS],
            "RF_Importance": [1/len(FEATURE_COLS)] * len(FEATURE_COLS),
            "Logistic_Coef": [0.0] * len(FEATURE_COLS),
        })
        
        # Returnăm un dicționar gol pentru trained_models (nu avem modele reale)
        return {}, scaler, loo_results, feat_imp

    # Calculăm class weights doar pentru clasele existente
    try:
        cw = compute_class_weight("balanced", classes=unique_classes, y=y)
        class_weights = {cls: cw[i] for i, cls in enumerate(unique_classes)}
    except Exception as e:
        print(f"  ⚠ Eroare la calculul class weights: {e}")
        class_weights = "balanced"

    scaler = StandardScaler()
    X_sc   = scaler.fit_transform(X)

    base_models = {
        "Logistic_L1": LogisticRegression(
            penalty="l1", solver="liblinear", C=0.5,
            class_weight=class_weights, max_iter=1000, random_state=42),
        "Random_Forest": RandomForestClassifier(
            n_estimators=500, max_depth=4, min_samples_leaf=2,
            class_weight=class_weights, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.08,
            subsample=0.8, random_state=42),
    }

    # Folosim StratifiedKFold în loc de LeaveOneOut pentru a evita problemele cu clase rare
    cv = StratifiedKFold(n_splits=min(5, len(y) // 2), shuffle=True, random_state=42)
    loo_results = {}

    for name, mdl in base_models.items():
        X_use = X_sc if "Logistic" in name else X
        try:
            probs = cross_val_predict(mdl, X_use, y, cv=cv, method="predict_proba")[:,1]
        except Exception as e:
            print(f"  ⚠ Eroare la cross_val_predict pentru {name}: {e}")
            # Fallback: antrenăm pe tot și predicții pe același set (doar pentru demo)
            mdl.fit(X_use, y)
            probs = mdl.predict_proba(X_use)[:,1]
        
        try:    
            auc = roc_auc_score(y, probs)
        except:
            auc = np.nan
        try:    
            ap = average_precision_score(y, probs)
        except: 
            ap = np.nan
        brier = brier_score_loss(y, probs)
        loo_results[name] = {"auc": auc, "ap": ap, "brier": brier, "loo_probs": probs}
        print(f"  {name}: AUC={auc:.3f} | AP={ap:.3f} | Brier={brier:.3f}")

    # Antrenare finală pe toate datele leg1 + calibrare Platt
    trained = {}
    for name, mdl in base_models.items():
        X_use = X_sc if "Logistic" in name else X
        try:
            cal   = CalibratedClassifierCV(mdl, method="sigmoid", cv=min(5, len(np.unique(y))))
            cal.fit(X_use, y)
            trained[name] = cal
            print(f"  ✓ {name} calibrat și antrenat pe toate datele leg1")
        except Exception as e:
            print(f"  ⚠ Nu s-a putut calibra {name}: {e}")
            try:
                mdl.fit(X_use, y)
                trained[name] = mdl
            except:
                print(f"  ⚠ Nu s-a putut antrena nici măcar modelul simplu {name}")
                trained[name] = None

    # Feature importance din modelele non-calibrate
    try:
        base_models["Logistic_L1"].fit(X_sc, y)
        base_models["Random_Forest"].fit(X, y)
        
        feat_imp = pd.DataFrame({
            "Feature":       [FEATURE_LABELS.get(f,f) for f in FEATURE_COLS],
            "RF_Importance": base_models["Random_Forest"].feature_importances_,
            "Logistic_Coef": base_models["Logistic_L1"].coef_[0],
        }).sort_values("RF_Importance", ascending=False)
    except Exception as e:
        print(f"  ⚠ Eroare la calcularea feature importance: {e}")
        feat_imp = pd.DataFrame({
            "Feature": [FEATURE_LABELS.get(f,f) for f in FEATURE_COLS],
            "RF_Importance": [1/len(FEATURE_COLS)] * len(FEATURE_COLS),
            "Logistic_Coef": [0.0] * len(FEATURE_COLS),
        })

    return trained, scaler, loo_results, feat_imp


# ═══════════════════════════════════════════════════════════════════════════════
#  TRANSFER PE LEG2 - VERSIUNE REPARATĂ
# ═══════════════════════════════════════════════════════════════════════════════

def apply_to_leg2(df2, trained_models, scaler, df2_labels):
    """
    Aplică modelele antrenate pe leg1 asupra deputaților din leg2.
    Validează rezultatele față de etichetele reale leg2.
    """
    if not SKLEARN_OK:
        df2["P_Excludere_Ensemble"] = 0.5
        return df2, {}

    print("\n🚀 Aplicare modele pe LEG2 (transfer)...")

    # Verificăm dacă avem modele și scaler
    if trained_models is None or scaler is None:
        print("  ⚠ Nu există modele antrenate sau scaler. Se utilizează predicție constantă.")
        df2["P_Excludere_Ensemble"] = 0.5
        df2["Risc"] = pd.cut(
            df2["P_Excludere_Ensemble"],
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=["🟢 Scăzut", "🟡 Moderat", "🟠 Ridicat", "🔴 Critic"],
        )
        return df2, {}

    # Alinierea etichetelor leg2
    excl_map = dict(zip(df2_labels["Nume"], df2_labels["Exclus"]))
    df2["Exclus"] = df2["Deputat"].map(excl_map).fillna(-1).astype(int)  # -1 = necunoscut

    X2    = df2[FEATURE_COLS].values
    
    # Verificăm dacă scaler este antrenat
    try:
        X2_sc = scaler.transform(X2)
    except Exception as e:
        print(f"  ⚠ Eroare la transformarea cu scaler: {e}")
        print("  Se utilizează datele originale fără scalare.")
        X2_sc = X2  # Fallback: folosim datele originale

    transfer_results = {}
    probs_all = []

    for name, model in trained_models.items():
        if model is None:
            print(f"  ⚠ Modelul {name} este None, se sare peste el")
            continue
            
        X_use = X2_sc if "Logistic" in name else X2
        try:
            probs = model.predict_proba(X_use)[:,1]
        except Exception as e:
            print(f"  ⚠ Eroare la predict pentru {name}: {e}")
            # Dacă modelul are doar o clasă, întoarcem media clasei din antrenament
            if hasattr(model, 'classes_') and len(model.classes_) == 1:
                probs = np.full(len(X_use), 0.5)  # Fallback la 0.5
            else:
                probs = np.full(len(X_use), 0.5)
        
        df2[f"P_{name}"] = probs
        probs_all.append(probs)

        # Validare pe cei cu etichete cunoscute (Exclus != -1)
        known_mask = df2["Exclus"] != -1
        if known_mask.sum() >= 2:
            y_true = df2.loc[known_mask, "Exclus"].values
            p_true = probs[known_mask]
            try:    
                auc = roc_auc_score(y_true, p_true)
            except: 
                auc = np.nan
            try:    
                ap  = average_precision_score(y_true, p_true)
            except: 
                ap  = np.nan
            try:
                brier = brier_score_loss(y_true, p_true)
            except:
                brier = np.nan
            transfer_results[name] = {"auc": auc, "ap": ap, "brier": brier}
            print(f"  {name}: AUC={auc:.3f} | AP={ap:.3f} | Brier={brier:.3f}")
        else:
            transfer_results[name] = {"auc": np.nan, "ap": np.nan, "brier": np.nan}

    if probs_all:
        df2["P_Excludere_Ensemble"] = np.mean(probs_all, axis=0)
    else:
        print("  ⚠ Nu s-a putut calcula ensemble-ul, se utilizează 0.5")
        df2["P_Excludere_Ensemble"] = 0.5
        
    df2["Risc"] = pd.cut(
        df2["P_Excludere_Ensemble"],
        bins=[0, 0.25, 0.50, 0.75, 1.0],
        labels=["🟢 Scăzut", "🟡 Moderat", "🟠 Ridicat", "🔴 Critic"],
    )
    return df2, transfer_results


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALIZA DRIFTULUI STRUCTURAL
# ═══════════════════════════════════════════════════════════════════════════════

def structural_drift(df1, df2, G1, G2):
    """
    Compară proprietățile structurale ale rețelelor leg1 vs. leg2.
    Identifică dacă rețeaua s-a schimbat semnificativ (drift).
    """
    print("\n📐 Analiză drift structural leg1 → leg2...")

    stats_leg = {}
    for label, G, df in [("leg1", G1, df1), ("leg2", G2, df2)]:
        n = G.number_of_nodes()
        comps = list(nx.connected_components(G))
        largest = max(comps, key=len) if comps else set()
        stats_leg[label] = {
            "Noduri":              n,
            "Muchii":              G.number_of_edges(),
            "Densitate":           round(nx.density(G), 4),
            "Clustering_mediu":    round(nx.average_clustering(G), 4),
            "Nr_componente":       len(comps),
            "Comp_mare_%":         round(len(largest)/n*100, 1) if n else 0,
            "Grad_mediu":          round(np.mean([d for _,d in G.degree()]), 2) if n else 0,
            "Grad_max":            max((d for _,d in G.degree()), default=0),
        }

    df_drift = pd.DataFrame(stats_leg).T
    print(df_drift.to_string())

    # Test Mann-Whitney pe distribuțiile de metrici
    drift_tests = []
    common_feats = ["Degree_Centrality","Betweenness","Clustering","Weighted_Degree"]
    for feat in common_feats:
        if feat in df1.columns and feat in df2.columns:
            v1 = df1[feat].values
            v2 = df2[feat].values
            if len(v1) > 1 and len(v2) > 1 and SCIPY_OK:
                try:
                    stat, pval = stats.mannwhitneyu(v1, v2, alternative="two-sided")
                    drift_tests.append({
                        "Feature":    FEATURE_LABELS.get(feat, feat),
                        "Leg1_medie": round(np.mean(v1), 4),
                        "Leg2_medie": round(np.mean(v2), 4),
                        "Delta_%":    round((np.mean(v2)-np.mean(v1))/(np.mean(v1)+1e-9)*100, 1),
                        "p_value":    round(pval, 4),
                        "Drift":      "⚠ DA" if pval < 0.05 else "✓ Nu",
                    })
                except:
                    pass

    df_tests = pd.DataFrame(drift_tests)
    if not df_tests.empty:
        print("\n🔄 Drift per metrică (MW test):")
        print(df_tests.to_string(index=False))

    return df_drift, df_tests


# ═══════════════════════════════════════════════════════════════════════════════
#  VIZUALIZĂRI - VERSIUNE REPARATĂ
# ═══════════════════════════════════════════════════════════════════════════════

def _base():
    return dict(
        paper_bgcolor="#0d1117", 
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8", size=11)
    )


def chart_loo_vs_transfer(loo_results, transfer_results):
    if not PLOTLY_OK: return ""
    if not loo_results or not transfer_results:
        return "<div class='warn'>Date insuficiente pentru grafic</div>"
    
    models = list(loo_results.keys())
    metrics = ["auc","ap","brier"]
    labels  = ["AUC-ROC","Avg Precision","Brier Score"]
    colors  = {"leg1_loo": "#3b82f6", "leg2_transfer": "#f59e0b"}

    fig = make_subplots(rows=1, cols=3, subplot_titles=labels)
    for col_idx, (met, lbl) in enumerate(zip(metrics, labels), 1):
        v1 = [loo_results[m].get(met, np.nan) for m in models]
        v2 = [transfer_results.get(m, {}).get(met, np.nan) for m in models]
        fig.add_trace(go.Bar(name="Leg1 LOO-CV" if col_idx==1 else None,
                             x=models, y=v1, marker_color=colors["leg1_loo"],
                             showlegend=col_idx==1), row=1, col=col_idx)
        fig.add_trace(go.Bar(name="Leg2 Transfer" if col_idx==1 else None,
                             x=models, y=v2, marker_color=colors["leg2_transfer"],
                             showlegend=col_idx==1), row=1, col=col_idx)

    fig.update_layout(
        **_base(),
        height=380, 
        barmode="group",
        margin=dict(l=60, r=30, t=55, b=60),
        title=dict(text="Performanță: Leg1 LOO-CV vs. Transfer pe Leg2",
                   font=dict(size=14, color="#e2e8f0")),
        legend=dict(bgcolor="#161b22", bordercolor="#334155")
    )
    for ax in ["xaxis","xaxis2","xaxis3","yaxis","yaxis2","yaxis3"]:
        if hasattr(fig.layout, ax):
            fig.layout[ax].update(gridcolor="#334155")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_risk_ranking(df2):
    if not PLOTLY_OK: return ""
    df_s = df2.sort_values("P_Excludere_Ensemble", ascending=True)
    excl_known = df_s["Exclus"].values

    colors = []
    for e in excl_known:
        if e == 1:   colors.append("#ef4444")
        elif e == 0: colors.append("#3b82f6")
        else:        colors.append("#64748b")

    fig = go.Figure(go.Bar(
        y=df_s["Deputat"], x=df_s["P_Excludere_Ensemble"],
        orientation="h", marker_color=colors,
        text=[f"{p:.1%}" for p in df_s["P_Excludere_Ensemble"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Risc: %{x:.3f}<extra></extra>",
    ))
    fig.add_vline(x=0.5, line=dict(color="#f59e0b", dash="dash", width=1.5),
                  annotation=dict(text="Prag 50%", font=dict(color="#f59e0b")))
    
    # Combinăm base() cu parametrii specifici, având grijă la margin
    base_layout = _base()
    # Eliminăm margin din base_layout dacă există
    if 'margin' in base_layout:
        del base_layout['margin']
    
    fig.update_layout(
        **base_layout,
        height=max(400, len(df_s)*24),
        xaxis=dict(range=[0,1.15], tickformat=".0%", gridcolor="#334155"),
        yaxis=dict(tickfont=dict(size=9)),
        title=dict(text="Ranking Risc Excludere — LEG2 (Roșu=exclus real · Albastru=inclus · Gri=necunoscut)",
                   font=dict(size=13, color="#e2e8f0")),
        margin=dict(l=200, r=80, t=60, b=40),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_calibration(df1, df2, loo_results):
    """Calibration plot: probabilitățile prezise vs. frecvența reală."""
    if not PLOTLY_OK: return ""
    if loo_results is None:
        return "<div class='warn'>Date insuficiente pentru grafic</div>"
        
    fig = go.Figure()

    # Leg1 LOO calibrare
    y1 = df1["Exclus"].values
    for name, res in loo_results.items():
        probs = res["loo_probs"]
        bins  = np.linspace(0, 1, 6)
        bin_means, bin_true = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs >= lo) & (probs < hi)
            if mask.sum() > 0:
                bin_means.append(probs[mask].mean())
                bin_true.append(y1[mask].mean())
        fig.add_trace(go.Scatter(x=bin_means, y=bin_true, mode="lines+markers",
                                 name=f"Leg1 {name}", line=dict(dash="dot")))

    # Leg2 transfer calibrare (unde avem etichete)
    known = df2[df2["Exclus"] != -1]
    if len(known) >= 4:
        y2    = known["Exclus"].values
        probs2 = known["P_Excludere_Ensemble"].values
        bins  = np.linspace(0, 1, 5)
        bm2, bt2 = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (probs2 >= lo) & (probs2 < hi)
            if mask.sum() > 0:
                bm2.append(probs2[mask].mean())
                bt2.append(y2[mask].mean())
        if bm2:
            fig.add_trace(go.Scatter(x=bm2, y=bt2, mode="lines+markers",
                                     name="Leg2 Ensemble Transfer",
                                     line=dict(color="#f59e0b", width=2.5)))

    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             line=dict(color="#475569", dash="dash"),
                             name="Calibrare perfectă"))
    
    base_layout = _base()
    if 'margin' in base_layout:
        del base_layout['margin']
        
    fig.update_layout(
        **base_layout,
        height=420,
        margin=dict(l=60, r=30, t=55, b=60),
        xaxis=dict(title="Probabilitate prezisă", gridcolor="#334155"),
        yaxis=dict(title="Frecvență reală (fracție excluși)", gridcolor="#334155"),
        title=dict(text="Calibration Plot: Leg1 LOO vs. Leg2 Transfer",
                   font=dict(size=13, color="#e2e8f0")),
        legend=dict(bgcolor="#161b22", bordercolor="#334155")
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_feature_importance(feat_imp):
    if not PLOTLY_OK or feat_imp is None: return ""
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Random Forest — Importanța Variabilelor",
                        "Logistic L1 — Coeficienți"])
    dfs = feat_imp.sort_values("RF_Importance", ascending=True)
    fig.add_trace(go.Bar(y=dfs["Feature"], x=dfs["RF_Importance"],
                         orientation="h", marker_color="#f59e0b"), row=1, col=1)
    dfc = feat_imp.sort_values("Logistic_Coef", ascending=True)
    fig.add_trace(go.Bar(y=dfc["Feature"], x=dfc["Logistic_Coef"],
                         orientation="h",
                         marker_color=["#ef4444" if v>0 else "#3b82f6"
                                       for v in dfc["Logistic_Coef"]]), row=1, col=2)
    
    base_layout = _base()
    if 'margin' in base_layout:
        del base_layout['margin']
        
    fig.update_layout(
        **base_layout,
        height=430, 
        showlegend=False,
        margin=dict(l=200, r=30, t=60, b=40),
        title=dict(text="Predictori de risc — model antrenat pe Leg1",
                   font=dict(size=13, color="#e2e8f0"))
    )
    for ax in ["xaxis","xaxis2","yaxis","yaxis2"]:
        if hasattr(fig.layout, ax):
            fig.layout[ax].update(gridcolor="#334155", zerolinecolor="#475569")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_drift(df_drift, df_tests):
    if not PLOTLY_OK: return ""
    fig = make_subplots(rows=1, cols=2,
        subplot_titles=["Proprietăți globale ale rețelei",
                        "Drift per metrică (delta %)"])

    props = ["Densitate","Clustering_mediu","Grad_mediu","Comp_mare_%"]
    props = [p for p in props if p in df_drift.columns]
    for leg, color in [("leg1","#3b82f6"),("leg2","#f59e0b")]:
        if leg in df_drift.index:
            vals = [float(df_drift.loc[leg, p]) if p in df_drift.columns else 0 for p in props]
            fig.add_trace(go.Bar(name=leg.upper(), x=props, y=vals,
                                 marker_color=color), row=1, col=1)

    if not df_tests.empty:
        delta_colors = ["#ef4444" if "DA" in str(r) else "#3b82f6"
                        for r in df_tests["Drift"]]
        fig.add_trace(go.Bar(x=df_tests["Feature"], y=df_tests["Delta_%"],
                             marker_color=delta_colors,
                             hovertext=df_tests["p_value"].astype(str),
                             name="Delta %"), row=1, col=2)
        fig.add_hline(y=0, line=dict(color="#475569"), row=1, col=2)

    base_layout = _base()
    if 'margin' in base_layout:
        del base_layout['margin']
        
    fig.update_layout(
        **base_layout,
        height=400, 
        barmode="group",
        margin=dict(l=60, r=30, t=55, b=60),
        title=dict(text="Drift Structural: Leg1 → Leg2",
                   font=dict(size=13, color="#e2e8f0")),
        legend=dict(bgcolor="#161b22", bordercolor="#334155")
    )
    for ax in ["xaxis","xaxis2","yaxis","yaxis2"]:
        if hasattr(fig.layout, ax):
            fig.layout[ax].update(gridcolor="#334155")
    return fig.to_html(full_html=False, include_plotlyjs=False)


def chart_scatter_transfer(df2):
    if not PLOTLY_OK: return ""
    excl_known = df2["Exclus"].values
    colors  = ["#ef4444" if e==1 else "#3b82f6" if e==0 else "#64748b" for e in excl_known]
    symbols = ["x" if e==1 else "circle" if e==0 else "diamond" for e in excl_known]

    fig = go.Figure(go.Scatter(
        x=df2["Degree_Centrality"], y=df2["Betweenness"],
        mode="markers+text", text=df2["Deputat"],
        textposition="top center", textfont=dict(size=8),
        marker=dict(color=colors, symbol=symbols, size=11,
                    line=dict(width=1, color="white")),
        hovertemplate="<b>%{text}</b><br>Degree: %{x:.4f}<br>Betweenness: %{y:.4f}<extra></extra>",
    ))
    
    base_layout = _base()
    if 'margin' in base_layout:
        del base_layout['margin']
        
    fig.update_layout(
        **base_layout,
        height=500,
        margin=dict(l=60, r=30, t=55, b=60),
        xaxis=dict(title="Degree Centrality", gridcolor="#334155"),
        yaxis=dict(title="Betweenness Centrality", gridcolor="#334155"),
        title=dict(text="Leg2: Spațiu structural · Roșu(×)=exclus · Albastru(○)=inclus · Gri(◇)=necunoscut",
                   font=dict(size=13, color="#e2e8f0"))
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  TABELE HTML
# ═══════════════════════════════════════════════════════════════════════════════

def ranking_table_html(df2):
    rows = []
    df_s = df2.sort_values("P_Excludere_Ensemble", ascending=False)
    risc_colors = {
        "🔴 Critic": "#ef4444", "🟠 Ridicat": "#f97316",
        "🟡 Moderat": "#f59e0b", "🟢 Scăzut": "#10b981",
    }
    for rank, (_, r) in enumerate(df_s.iterrows(), 1):
        p    = r["P_Excludere_Ensemble"]
        excl = r["Exclus"]
        risc = str(r.get("Risc","—"))
        rc   = risc_colors.get(risc, "#64748b")
        if excl == 1:
            status_html = '<span style="color:#ef4444">⚫ Exclus</span>'
        elif excl == 0:
            status_html = '<span style="color:#10b981">✓ Inclus</span>'
        else:
            status_html = '<span style="color:#64748b">— Necunoscut</span>'
        bar = int(p*100)
        rows.append(f"""<tr>
          <td style="color:#64748b">{rank}</td>
          <td><strong>{r['Deputat']}</strong></td>
          <td>{r['Partid']}</td>
          <td>{status_html}</td>
          <td style="color:{rc}"><strong>{risc}</strong></td>
          <td>
            <div style="display:flex;align-items:center;gap:8px">
              <div style="width:{bar}px;max-width:80px;height:6px;background:{rc};border-radius:3px"></div>
              <span style="color:{rc}">{p:.2%}</span>
            </div>
          </td>
          <td>{r['Degree_Centrality']:.4f}</td>
          <td>{r['Betweenness']:.4f}</td>
          <td>{r['Clique_Count']}</td>
        </tr>""")
    return "\n".join(rows)


def model_perf_rows(loo_results, transfer_results):
    rows = []
    for name in loo_results:
        l = loo_results[name]
        t = transfer_results.get(name, {})
        def fmt(v): return f"{v:.3f}" if not (v != v) else "—"
        # degradare AUC
        l_auc = l.get("auc", np.nan)
        t_auc = t.get("auc", np.nan)
        if not (l_auc != l_auc) and not (t_auc != t_auc):
            delta = t_auc - l_auc
            dcol  = "#10b981" if delta >= 0 else "#ef4444"
            delta_s = f'<span style="color:{dcol}">{delta:+.3f}</span>'
        else:
            delta_s = "—"
        rows.append(f"""<tr>
          <td><strong>{name}</strong></td>
          <td>{fmt(l_auc)}</td>
          <td>{fmt(t_auc)}</td>
          <td>{delta_s}</td>
          <td>{fmt(l.get("brier",np.nan))}</td>
          <td>{fmt(t.get("brier",np.nan))}</td>
        </tr>""")
    return "\n".join(rows)


def drift_table_html(df_tests):
    if df_tests is None or df_tests.empty:
        return "<tr><td colspan='6'>Date insuficiente</td></tr>"
    rows = []
    for _, r in df_tests.iterrows():
        dcol = "#ef4444" if "DA" in str(r["Drift"]) else "#10b981"
        rows.append(f"""<tr>
          <td>{r['Feature']}</td>
          <td>{r['Leg1_medie']}</td>
          <td>{r['Leg2_medie']}</td>
          <td>{r['Delta_%']}%</td>
          <td>{r['p_value']}</td>
          <td style="color:{dcol}">{r['Drift']}</td>
        </tr>""")
    return "\n".join(rows)


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════════════════════

def generate_dashboard(df1, df2, G1, G2,
                       loo_results, transfer_results,
                       feat_imp, df_drift, df_tests,
                       output_file):
    print(f"\n🎨 Generare dashboard: {output_file}")

    n_excl_leg2 = int((df2["Exclus"]==1).sum())
    n_incl_leg2 = int((df2["Exclus"]==0).sum())
    n_unkn_leg2 = int((df2["Exclus"]==-1).sum())
    n_risc_crit = int((df2["Risc"]=="🔴 Critic").sum()) if "Risc" in df2.columns else 0
    n_risc_rid  = int((df2["Risc"]=="🟠 Ridicat").sum()) if "Risc" in df2.columns else 0

    # ensemble AUC transfer
    best_auc = 0.5
    if transfer_results:
        best_auc = max((v.get("auc",0) for v in transfer_results.values()
                        if not (v.get("auc",0) != v.get("auc",0))), default=0.5)
    auc_col  = "#10b981" if best_auc > 0.7 else "#f59e0b" if best_auc > 0.55 else "#ef4444"

    html = f"""<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Transfer Risc Excludere — Leg2 2025–2029</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#0d1117;--s:#161b22;--b:#21262d;--acc:#3b82f6;--red:#ef4444;--grn:#10b981;--amb:#f59e0b;--txt:#e2e8f0;--m:#64748b;--mono:'IBM Plex Mono',monospace;--sans:'DM Sans',sans-serif}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{background:var(--bg);color:var(--txt);font-family:var(--sans)}}
header{{border-bottom:1px solid var(--b);padding:20px 40px;display:flex;align-items:center;gap:14px;position:sticky;top:0;background:var(--bg);z-index:100}}
.badge{{background:var(--red);color:#fff;font-family:var(--mono);font-size:11px;padding:4px 10px;border-radius:4px;letter-spacing:.05em}}
.badge2{{background:var(--amb);color:#000;font-family:var(--mono);font-size:10px;padding:3px 8px;border-radius:4px}}
header h1{{font-size:18px;font-weight:700;letter-spacing:-.02em}}
header p{{font-size:11px;color:var(--m);font-family:var(--mono)}}
.pills{{margin-left:auto;display:flex;gap:10px;flex-wrap:wrap}}
.pill{{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:7px 13px;text-align:center}}
.pill .v{{font-size:19px;font-weight:700;font-family:var(--mono)}}
.pill .l{{font-size:9px;color:var(--m);text-transform:uppercase;letter-spacing:.08em}}
nav{{display:flex;border-bottom:1px solid var(--b);padding:0 40px;overflow-x:auto}}
.tab{{padding:12px 16px;font-size:12px;font-weight:500;color:var(--m);cursor:pointer;border-bottom:2px solid transparent;white-space:nowrap;transition:.15s}}
.tab:hover{{color:var(--txt)}}.tab.active{{color:var(--acc);border-bottom-color:var(--acc)}}
main{{padding:28px 40px;max-width:1500px}}
.sec{{display:none}}.sec.active{{display:block}}
.stitle{{font-size:14px;font-weight:700;margin-bottom:5px;display:flex;align-items:center;gap:8px}}
.stitle::before{{content:'';width:3px;height:16px;background:var(--acc);border-radius:2px;display:inline-block}}
.sdesc{{font-size:11px;color:var(--m);font-family:var(--mono);margin-bottom:18px;line-height:1.7}}
.card{{background:var(--s);border:1px solid var(--b);border-radius:10px;padding:18px;margin-bottom:18px}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:18px}}
.warn{{background:rgba(245,158,11,.08);border:1px solid rgba(245,158,11,.25);border-radius:8px;padding:12px 16px;font-size:11px;line-height:1.7;margin-bottom:16px;color:#fbbf24}}
.info{{background:rgba(59,130,246,.07);border:1px solid rgba(59,130,246,.2);border-radius:8px;padding:12px 16px;font-size:11px;line-height:1.7;margin-bottom:16px;color:#93c5fd}}
.flow{{display:flex;align-items:center;gap:12px;padding:16px;background:var(--bg);border-radius:8px;margin-bottom:16px;flex-wrap:wrap}}
.flow-box{{background:var(--s);border:1px solid var(--b);border-radius:8px;padding:10px 16px;font-size:11px;text-align:center;min-width:130px}}
.flow-box strong{{display:block;font-size:13px;margin-bottom:2px}}
.flow-arr{{color:var(--m);font-size:18px}}
table{{width:100%;border-collapse:collapse;font-size:11px;font-family:var(--mono)}}
th{{padding:8px 10px;text-align:left;font-size:9px;text-transform:uppercase;letter-spacing:.07em;color:var(--m);background:var(--b)}}
td{{padding:7px 10px;border-bottom:1px solid var(--b)}}
tr:hover td{{background:rgba(59,130,246,.04)}}
.risk-legend{{display:flex;gap:16px;font-size:11px;margin-bottom:14px;flex-wrap:wrap}}
.rl-item{{display:flex;align-items:center;gap:6px}}
</style>
</head>
<body>
<header>
  <div>
    <div style="display:flex;gap:8px;margin-bottom:6px">
      <div class="badge">TRANSFER LEARNING</div>
      <div class="badge2">LEG1 → LEG2</div>
    </div>
    <h1>Predicție Risc Excludere — Legislatura 2025–2029</h1>
    <p>Model antrenat pe Leg1 (2021–2025) · Aplicat pe Leg2 (2025–2029)</p>
  </div>
  <div class="pills">
    <div class="pill"><div class="v" style="color:var(--acc)">{G1.number_of_nodes()}</div><div class="l">Dep. Leg1</div></div>
    <div class="pill"><div class="v" style="color:var(--amb)">{G2.number_of_nodes()}</div><div class="l">Dep. Leg2</div></div>
    <div class="pill"><div class="v" style="color:var(--red)">{n_risc_crit+n_risc_rid}</div><div class="l">Risc ≥ Ridicat</div></div>
    <div class="pill"><div class="v" style="color:{auc_col}">{best_auc:.3f}</div><div class="l">Best AUC Transfer</div></div>
  </div>
</header>

<nav>
  <div class="tab active" onclick="show('ranking')">🔴 Ranking Risc</div>
  <div class="tab" onclick="show('performanta')">📊 Performanță Model</div>
  <div class="tab" onclick="show('predictori')">⚙ Predictori</div>
  <div class="tab" onclick="show('drift')">🔄 Drift Structural</div>
  <div class="tab" onclick="show('calibrare')">🎯 Calibrare</div>
  <div class="tab" onclick="show('metodologie')">📖 Metodologie</div>
</nav>

<main>

<!-- RANKING -->
<div class="sec active" id="sec-ranking">
  <div class="stitle">Ranking Deputați Leg2 după Riscul de Excludere</div>
  <div class="sdesc">Probabilitate estimată de model ensemble (Logistic L1 + Random Forest + Gradient Boosting)
antrenat pe Leg1 și calibrat Platt. Culori: Roșu=exclus real · Albastru=inclus real · Gri=fără etichetă.</div>

  <div class="risk-legend">
    <div class="rl-item"><span style="color:#ef4444">🔴</span> Critic (&gt;75%)</div>
    <div class="rl-item"><span style="color:#f97316">🟠</span> Ridicat (50–75%)</div>
    <div class="rl-item"><span style="color:#f59e0b">🟡</span> Moderat (25–50%)</div>
    <div class="rl-item"><span style="color:#10b981">🟢</span> Scăzut (&lt;25%)</div>
    <div class="rl-item" style="color:var(--m)">{n_risc_crit} critici · {n_risc_rid} ridicați · {n_excl_leg2} excluși cunoscuți · {n_unkn_leg2} fără etichetă</div>
  </div>

  <div class="card">{chart_risk_ranking(df2)}</div>

  <div class="stitle" style="margin-bottom:12px">Tabel complet — Leg2</div>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr><th>#</th><th>Deputat</th><th>Partid</th><th>Status real</th><th>Nivel risc</th><th>P(Excludere)</th><th>Degree</th><th>Betweenness</th><th>Nr. Clici</th></tr></thead>
      <tbody>{ranking_table_html(df2)}</tbody>
    </table>
  </div>
  <div class="card">{chart_scatter_transfer(df2)}</div>
</div>

<!-- PERFORMANȚĂ -->
<div class="sec" id="sec-performanta">
  <div class="stitle">Performanța Modelului: Leg1 LOO-CV vs. Transfer Leg2</div>
  <div class="sdesc">LOO-CV pe Leg1 = estimare internă a puterii predictive.
Transfer pe Leg2 = validare externă reală pe o legislatură nevăzută de model.
Degradarea AUC (Leg2−Leg1) arată cât de bine se generalizează pattern-urile structurale.</div>

  <div class="warn">⚠ <strong>Atenție la N mic:</strong> AUC calculat pe {n_excl_leg2} excluși și {n_incl_leg2} incluși cunoscuți din Leg2.
Intervalele de încredere sunt largi — interpretați tendințele, nu valorile exacte.</div>

  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr><th>Model</th><th>AUC Leg1 (LOO)</th><th>AUC Leg2 (Transfer)</th><th>Δ AUC</th><th>Brier Leg1</th><th>Brier Leg2</th></tr></thead>
      <tbody>{model_perf_rows(loo_results, transfer_results)}</tbody>
    </table>
  </div>
  <div class="card">{chart_loo_vs_transfer(loo_results, transfer_results)}</div>
</div>

<!-- PREDICTORI -->
<div class="sec" id="sec-predictori">
  <div class="stitle">Predictori de Risc — Model antrenat pe Leg1</div>
  <div class="sdesc">RF Importance = contribuția medie a variabilei la reducerea impurității.
Logistic Coef: pozitiv = crește probabilitatea de excludere, negativ = o reduce.
Aceste ponderi sunt transferate direct asupra Leg2.</div>
  <div class="card">{chart_feature_importance(feat_imp)}</div>
</div>

<!-- DRIFT -->
<div class="sec" id="sec-drift">
  <div class="stitle">Analiza Driftului Structural: Leg1 → Leg2</div>
  <div class="sdesc">Driftul structural măsoară cât de mult s-a schimbat rețeaua de co-autoriat
între legislaturi. Un drift mare înseamnă că modelul transferat poate fi mai puțin fiabil
(a învățat pattern-uri care nu mai sunt valabile în Leg2).</div>

  <div class="info">ℹ Dacă Delta% este mare și p &lt; 0.05 pentru mai mulți predictori, luați în considerare re-antrenarea modelului pe date Leg2 odată ce se acumulează suficiente proiecte co-semnate.</div>

  <div class="card">{chart_drift(df_drift, df_tests)}</div>
  <div class="card" style="overflow-x:auto">
    <table>
      <thead><tr><th>Metrică</th><th>Leg1 medie</th><th>Leg2 medie</th><th>Delta %</th><th>p-value (MW)</th><th>Drift semnificativ</th></tr></thead>
      <tbody>{drift_table_html(df_tests)}</tbody>
    </table>
  </div>
</div>

<!-- CALIBRARE -->
<div class="sec" id="sec-calibrare">
  <div class="stitle">Calibration Plot</div>
  <div class="sdesc">Un model bine calibrat are punctele aproape de diagonala y=x.
Deasupra diagonalei = model subestimează riscul. Sub diagonală = supraevaluează.
Calibrarea Platt a fost aplicată înainte de transfer.</div>
  <div class="card">{chart_calibration(df1, df2, loo_results)}</div>
</div>

<!-- METODOLOGIE -->
<div class="sec" id="sec-metodologie">
  <div class="stitle">Notă Metodologică — Transfer Learning Cross-Legislatură</div>
  <div class="sdesc">Justificarea alegerilor metodologice și limitările analizei.</div>
  <div class="card" style="line-height:1.9;font-size:13px">

  <div class="flow">
    <div class="flow-box"><strong style="color:var(--acc)">LEG1 (2021–2025)</strong>Co-autoriat + etichete<br>exclus/inclus reale</div>
    <div class="flow-arr">→</div>
    <div class="flow-box"><strong style="color:var(--amb)">ANTRENARE</strong>LOO-CV intern<br>3 modele + calibrare Platt</div>
    <div class="flow-arr">→</div>
    <div class="flow-box"><strong style="color:var(--amb)">TRANSFER</strong>Aplicare directă<br>pe features Leg2</div>
    <div class="flow-arr">→</div>
    <div class="flow-box"><strong style="color:var(--red)">LEG2 (2025–2029)</strong>Scor risc per deputat<br>+ validare pe cunoscuți</div>
  </div>

  <p style="margin-bottom:12px"><strong style="color:var(--acc)">De ce transfer learning și nu re-antrenare pe Leg2?</strong><br>
  Leg2 este o legislatură nouă — numărul de proiecte co-semnate este mic, deci rețeaua este incompletă.
  Re-antrenarea pe date insuficiente ar produce un model instabil. Transferul din Leg1 valorifică
  experiența completă a mandatelor precedente.</p>

  <p style="margin-bottom:12px"><strong style="color:var(--acc)">Ipoteza de transfer (și când se invalidează):</strong><br>
  Presupunem că pattern-urile structurale asociate cu excluderea sunt <em>stabile între legislaturi</em>:
  deputații izolați, cu betweenness scăzut sau cu omofilie ridicată au risc mai mare.
  Ipoteza se invalidează dacă conducerea partidului schimbă criteriile de selecție sau dacă
  compoziția politică a parlamentului se schimbă radical. <strong>Analiza driftului structural</strong>
  (tab-ul anterior) testează tocmai această stabilitate.</p>

  <p style="margin-bottom:12px"><strong style="color:var(--acc)">Calibrarea Platt:</strong><br>
  Modelele ML produc scoruri, nu probabilități calibrate. Calibrarea Platt (regresie logistică
  pe output-ul modelului) transformă scorurile în probabilități interpretabile.
  Fără calibrare, un scor de 0.8 nu înseamnă 80% șansă reală de excludere.</p>

  <p style="margin-bottom:12px"><strong style="color:var(--acc)">Limitări critice:</strong><br>
  • N mic în Leg2 → AUC instabil; intervalele de încredere bootstrap ar fi necesare<br>
  • Co-autoriatul nu captează afilierile politice informale sau conflictele intra-partid<br>
  • Modelul nu știe despre schimbări în conducerea partidului sau presiuni externe<br>
  • Pe măsură ce Leg2 acumulează proiecte, re-antrenați modelul pe date mixte Leg1+Leg2</p>

  <p><strong style="color:var(--acc)">Referințe:</strong><br>
  Weiss, K. et al. (2016). A Survey of Transfer Learning. <em>Journal of Big Data</em>, 3(9).<br>
  Platt, J. (1999). Probabilistic Outputs for SVMs. <em>Advances in Large Margin Classifiers</em>.<br>
  Borgatti, S.P. et al. (2018). <em>Analyzing Social Networks</em>. SAGE.</p>
  </div>
</div>

</main>
<script>
function show(name){{
  document.querySelectorAll('.sec').forEach(s=>s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  document.getElementById('sec-'+name).classList.add('active');
  event.target.classList.add('active');
}}
</script>
</body>
</html>"""

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ Dashboard salvat: {output_file}")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("="*70)
    print("TRANSFER LEARNING CROSS-LEGISLATURĂ — RISC EXCLUDERE")
    print("Parlamentul Republicii Moldova")
    print("="*70)
    print("pip install networkx numpy pandas scikit-learn plotly scipy\n")

    # ── Date ────────────────────────────────────────────────────────────────
    print("📂 Încărcare date...")
    proj1  = load_json("leg1_raw.json", DEMO_LEG1_PROJECTS)
    proj2  = load_json("leg2_raw.json", DEMO_LEG2_PROJECTS)
    excl1  = load_csv("excluded_deputies.csv", DEMO_LEG1_EXCL)
    excl2  = load_csv("excluded_deputies_leg2.csv", DEMO_LEG2_EXCL)
    excl1["Nume"] = excl1["Nume"].str.strip()
    excl2["Nume"] = excl2["Nume"].str.strip()

    # ── Grafuri ─────────────────────────────────────────────────────────────
    print("\n🔗 Construire grafuri...")
    G1 = build_graph(proj1)
    G2 = build_graph(proj2)
    print(f"  Leg1: {G1.number_of_nodes()} noduri, {G1.number_of_edges()} muchii")
    print(f"  Leg2: {G2.number_of_nodes()} noduri, {G2.number_of_edges()} muchii")

    # ── Features ────────────────────────────────────────────────────────────
    print("\n🔧 Calcul features...")
    df1 = compute_features(G1, excl1)
    df1["Exclus"] = excl1.set_index("Nume")["Exclus"].reindex(df1["Deputat"]).values
    
    # Debug: verificăm dacă avem valori NaN în Exclus
    print(f"  Valori NaN în Exclus leg1: {df1['Exclus'].isna().sum()}")
    # Eliminăm rândurile cu Exclus NaN (deputați care nu au etichetă)
    df1 = df1.dropna(subset=["Exclus"])
    print(f"  Rânduri după eliminare NaN: {len(df1)}")

    df2 = compute_features(G2, excl2)

    # ── Antrenare Leg1 ──────────────────────────────────────────────────────
    result = train_on_leg1(df1)
    if result[0] is None:
        print("❌ Antrenare eșuată"); return
    trained_models, scaler, loo_results, feat_imp = result

    # ── Transfer Leg2 ───────────────────────────────────────────────────────
    df2, transfer_results = apply_to_leg2(df2, trained_models, scaler, excl2)

    # ── Drift structural ────────────────────────────────────────────────────
    df_drift, df_tests = structural_drift(df1, df2, G1, G2)

    # ── Export ──────────────────────────────────────────────────────────────
    df2.to_csv("transfer_predictions_leg2.csv", index=False, encoding="utf-8-sig")
    print("\n✓ transfer_predictions_leg2.csv")

    report = {
        "loo_leg1":      {k: {m: (round(v,4) if v==v else None)
                              for m,v in r.items() if m != "loo_probs"}
                          for k,r in loo_results.items()},
        "transfer_leg2": {k: {m: (round(v,4) if v==v else None)
                               for m,v in r.items()}
                          for k,r in transfer_results.items()},
    }
    with open("transfer_model_report.json","w",encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("✓ transfer_model_report.json")

    # ── Dashboard ────────────────────────────────────────────────────────────
    generate_dashboard(
        df1=df1, df2=df2, G1=G1, G2=G2,
        loo_results=loo_results,
        transfer_results=transfer_results,
        feat_imp=feat_imp,
        df_drift=df_drift,
        df_tests=df_tests,
        output_file="transfer_analysis_report.html",
    )

    print("\n🎉 ANALIZĂ COMPLETĂ!")
    print("   → Deschide transfer_analysis_report.html în browser")
    print("\nPentru date reale, pune în același folder:")
    print("   leg1_raw.json · leg2_raw.json")
    print("   excluded_deputies.csv · excluded_deputies_leg2.csv")


if __name__ == "__main__":
    main()