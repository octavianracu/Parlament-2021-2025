"""
ANALIZA REȚELEI DE COOPERARE — DEPUTAȚI PARLAMENTUL R. MOLDOVA
===============================================================
Metrici implementate:
  1. Centralitate de Grad (Degree Centrality)
  2. Centralitate de Putere Bonacich (Power / Eigenvector Centrality)
  3. Centralitate de Intermediere (Betweenness Centrality)
  4. Centralitate de Stres (Stress Centrality)
  5. Centralitate de Prestigiu (PageRank ca proxy Prestige)
  6. Centralitate de Accesibilitate (Eccentricity Centrality)
  7. Identificarea clicilor (Clique detection)
  8. Matricea coeficienților de corelație Pearson
  9. Matricea coeficienților de similaritate (Matching / Jaccard)

Output:
  • network_analysis_<prefix>.html  — dashboard interactiv (Pyvis + Plotly)
  • network_metrics_<prefix>.csv    — toate metricile per deputat
  • cliques_<prefix>.json           — clici identificate
  • pearson_matrix_<prefix>.csv     — matricea Pearson
  • similarity_matrix_<prefix>.csv  — matricea similaritate
"""

import json
import math
import os
import warnings
from collections import defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── opțional, pentru grafuri interactive ─────────────────────────────────────
try:
    from pyvis.network import Network
    PYVIS_OK = True
except ImportError:
    PYVIS_OK = False
    print("⚠ pyvis nu este instalat. Grafurile interactive nu vor fi generate.")
    print("  Instalați cu:  pip install pyvis")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_OK = True
except ImportError:
    PLOTLY_OK = False
    print("⚠ plotly nu este instalat. Heatmap-urile nu vor fi generate.")
    print("  Instalați cu:  pip install plotly")


# ═══════════════════════════════════════════════════════════════════════════════
#  1. ÎNCĂRCARE DATE
# ═══════════════════════════════════════════════════════════════════════════════

def load_projects(json_file: str) -> list:
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(projects: list) -> nx.Graph:
    """Construiește graf neorientat ponderat din lista de proiecte."""
    G = nx.Graph()
    for project in projects:
        authors = project["deputy_authors"]
        for a in authors:
            G.add_node(a)
        for a1, a2 in combinations(authors, 2):
            if G.has_edge(a1, a2):
                G[a1][a2]["weight"] += 1
            else:
                G.add_edge(a1, a2, weight=1)
    return G


# ═══════════════════════════════════════════════════════════════════════════════
#  2. METRICI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_stress_centrality(G: nx.Graph) -> dict:
    """
    Stress Centrality = numărul de drumuri scurte (geodezice) care
    TREC PRIN nodul v (fără a-l include ca endpoint).
    Similară cu betweenness, dar nenormalizată și neîmpărțită la σ_st.
    """
    stress = dict.fromkeys(G.nodes(), 0.0)
    nodes = list(G.nodes())

    for s in nodes:
        # BFS / Dijkstra pentru toate drumurile scurte de la s
        try:
            paths = dict(nx.all_pairs_shortest_path(G))
        except Exception:
            break

    # Implementare manuală eficientă
    stress = defaultdict(float)
    for s in nodes:
        # shortest path lengths + number of shortest paths
        spl = dict(nx.single_source_shortest_path_length(G, s))
        # toate drumurile scurte (neponderate) de la s
        sp_gen = nx.single_source_shortest_path(G, s)
        sp = dict(sp_gen)

        for t in nodes:
            if t == s or t not in sp:
                continue
            path = sp[t]
            # nodurile intermediare
            for v in path[1:-1]:
                stress[v] += 1.0

    # normalizare opțională: împărțim la (n-1)(n-2)
    n = G.number_of_nodes()
    norm = (n - 1) * (n - 2) if n > 2 else 1
    return {v: stress[v] / norm for v in G.nodes()}


def compute_eccentricity_centrality(G: nx.Graph) -> dict:
    """
    Eccentricity Centrality = 1 / eccentricity(v).
    Lucrăm pe componenta cea mai mare dacă graful nu este conex.
    """
    result = {}
    if nx.is_connected(G):
        ecc = nx.eccentricity(G)
        for v, e in ecc.items():
            result[v] = 1.0 / e if e > 0 else 0.0
    else:
        # calculăm pe fiecare componentă
        for component in nx.connected_components(G):
            subG = G.subgraph(component)
            if subG.number_of_nodes() < 2:
                for v in component:
                    result[v] = 0.0
                continue
            ecc = nx.eccentricity(subG)
            for v, e in ecc.items():
                result[v] = 1.0 / e if e > 0 else 0.0
    return result


def compute_all_metrics(G: nx.Graph) -> pd.DataFrame:
    """Calculează toate metricile și returnează un DataFrame."""
    print("\n📐 Calcul metrici de centralitate...")

    nodes = list(G.nodes())
    n = len(nodes)

    print(f"  Graf: {n} noduri, {G.number_of_edges()} muchii")

    # 1. Degree Centrality
    print("  [1/6] Degree Centrality...")
    degree_c = nx.degree_centrality(G)

    # 2. Power / Eigenvector Centrality (Bonacich)
    print("  [2/6] Power Centrality (Bonacich / Eigenvector)...")
    try:
        power_c = nx.eigenvector_centrality(G, max_iter=1000, weight="weight")
    except nx.PowerIterationFailedConvergence:
        print("    ⚠ Convergență eșuată, se folosește eigenvector fără ponderi...")
        try:
            power_c = nx.eigenvector_centrality(G, max_iter=1000)
        except Exception:
            power_c = {v: 0.0 for v in nodes}

    # 3. Betweenness Centrality
    print("  [3/6] Betweenness Centrality...")
    betweenness_c = nx.betweenness_centrality(G, weight="weight", normalized=True)

    # 4. Stress Centrality
    print("  [4/6] Stress Centrality...")
    stress_c = compute_stress_centrality(G)

    # 5. Prestige Centrality (PageRank)
    print("  [5/6] Prestige Centrality (PageRank)...")
    try:
        prestige_c = nx.pagerank(G, weight="weight", max_iter=500)
    except Exception:
        prestige_c = {v: 1.0 / n for v in nodes}

    # 6. Eccentricity Centrality
    print("  [6/6] Eccentricity Centrality...")
    eccentricity_c = compute_eccentricity_centrality(G)

    # Asamblare DataFrame
    df = pd.DataFrame(
        {
            "Deputat": nodes,
            "Grad": [G.degree(v) for v in nodes],
            "Grad_Ponderat": [sum(data.get("weight", 1) for _, data in G[v].items()) for v in nodes],
            "Degree_Centrality": [degree_c.get(v, 0) for v in nodes],
            "Power_Centrality_Bonacich": [power_c.get(v, 0) for v in nodes],
            "Betweenness_Centrality": [betweenness_c.get(v, 0) for v in nodes],
            "Stress_Centrality": [stress_c.get(v, 0) for v in nodes],
            "Prestige_Centrality_PageRank": [prestige_c.get(v, 0) for v in nodes],
            "Eccentricity_Centrality": [eccentricity_c.get(v, 0) for v in nodes],
        }
    )
    df = df.sort_values("Degree_Centrality", ascending=False).reset_index(drop=True)
    df.index = df.index + 1  # 1-based rank

    print(f"  ✅ Metrici calculate pentru {n} deputați")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  3. CLICI
# ═══════════════════════════════════════════════════════════════════════════════

def find_cliques(G: nx.Graph) -> dict:
    print("\n🔍 Identificare clici...")
    all_cliques = list(nx.find_cliques(G))
    all_cliques.sort(key=len, reverse=True)

    max_size = max(len(c) for c in all_cliques) if all_cliques else 0
    max_cliques = [c for c in all_cliques if len(c) == max_size]

    result = {
        "total_cliques": len(all_cliques),
        "max_clique_size": max_size,
        "max_cliques": max_cliques,
        "cliques_by_size": {},
    }

    for clique in all_cliques:
        size = len(clique)
        if size >= 3:  # salvăm doar clicile de cel puțin 3 noduri
            result["cliques_by_size"].setdefault(str(size), []).append(clique)

    print(f"  ✅ Clici găsite: {len(all_cliques)} | Dimensiune maximă: {max_size}")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  4. MATRICE PEARSON
# ═══════════════════════════════════════════════════════════════════════════════

def compute_pearson_matrix(G: nx.Graph) -> pd.DataFrame:
    """
    Matricea coeficienților de corelație Pearson între profilurile de conectivitate
    ale deputaților (vectorii de adiacență ponderată).
    """
    print("\n📊 Calcul matrice Pearson...")
    nodes = sorted(G.nodes())
    n = len(nodes)
    node_idx = {v: i for i, v in enumerate(nodes)}

    # Matrice de adiacență ponderată
    adj = np.zeros((n, n))
    for u, v, data in G.edges(data=True):
        i, j = node_idx[u], node_idx[v]
        adj[i][j] = data.get("weight", 1)
        adj[j][i] = data.get("weight", 1)

    # Corelație Pearson între rânduri
    pearson = np.corrcoef(adj)
    pearson = np.nan_to_num(pearson, nan=0.0)

    df = pd.DataFrame(pearson, index=nodes, columns=nodes)
    print(f"  ✅ Matrice Pearson {n}x{n} calculată")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  5. MATRICE SIMILARITATE (Matching / Jaccard)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_similarity_matrix(G: nx.Graph, method: str = "jaccard") -> pd.DataFrame:
    """
    Matrice de similaritate structurală între deputați.
    method='jaccard'   → |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
    method='matching'  → matching coefficient (coeficientul de potrivire simplu)
    """
    print(f"\n📊 Calcul matrice similaritate ({method})...")
    nodes = sorted(G.nodes())
    n = len(nodes)
    sim = np.zeros((n, n))
    node_idx = {v: i for i, v in enumerate(nodes)}

    for i, u in enumerate(nodes):
        Nu = set(G.neighbors(u))
        for j, v in enumerate(nodes):
            if i == j:
                sim[i][j] = 1.0
                continue
            Nv = set(G.neighbors(v))
            if method == "jaccard":
                union = Nu | Nv
                inter = Nu & Nv
                sim[i][j] = len(inter) / len(union) if union else 0.0
            else:  # matching
                inter = Nu & Nv
                sim[i][j] = len(inter) / n

    df = pd.DataFrame(sim, index=nodes, columns=nodes)
    print(f"  ✅ Matrice similaritate {n}x{n} calculată ({method})")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  6. VIZUALIZARE INTERACTIVĂ — HTML
# ═══════════════════════════════════════════════════════════════════════════════

def _get_node_color_by_metric(value: float, metric: str) -> str:
    """Mapare valoare → culoare hex (gradient albastru→roșu)."""
    v = max(0.0, min(1.0, value))
    r = int(30 + v * 200)
    g = int(100 - v * 60)
    b = int(220 - v * 180)
    return f"#{r:02x}{g:02x}{b:02x}"


def create_pyvis_graph(G: nx.Graph, df_metrics: pd.DataFrame, metric_col: str, title: str) -> str:
    """Returnează HTML-ul unui graf Pyvis colorat după metrica dată."""
    if not PYVIS_OK:
        return "<p>pyvis nu este disponibil</p>"

    net = Network(height="600px", width="100%", bgcolor="#0d1117", font_color="white")
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -8000,
          "springLength": 120,
          "springConstant": 0.04
        },
        "stabilization": {"iterations": 200}
      },
      "edges": {
        "color": {"color": "#334155"},
        "smooth": {"type": "continuous"}
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100
      }
    }
    """)

    metric_values = dict(zip(df_metrics["Deputat"], df_metrics[metric_col]))
    vals = list(metric_values.values())
    vmin, vmax = min(vals), max(vals)
    vrange = vmax - vmin if vmax != vmin else 1

    for _, row in df_metrics.iterrows():
        node = row["Deputat"]
        val = row[metric_col]
        norm = (val - vmin) / vrange
        color = _get_node_color_by_metric(norm, metric_col)
        size = 10 + norm * 30

        tooltip = (
            f"<b>{node}</b><br>"
            f"Grad: {row['Grad']}<br>"
            f"{metric_col.replace('_', ' ')}: {val:.4f}<br>"
            f"Betweenness: {row['Betweenness_Centrality']:.4f}"
        )

        net.add_node(
            node,
            label=node,
            title=tooltip,
            color=color,
            size=size,
            font={"size": 9, "color": "white"},
        )

    for u, v, data in G.edges(data=True):
        w = data.get("weight", 1)
        net.add_edge(u, v, value=w, title=f"Colaborări: {w}", color="#334155")

    net.heading = title
    html = net.generate_html()
    return html


def create_heatmap_html(matrix_df: pd.DataFrame, title: str, colorscale: str = "RdBu") -> str:
    """Returnează HTML Plotly pentru un heatmap al matricei."""
    if not PLOTLY_OK:
        return "<p>plotly nu este disponibil</p>"

    nodes = matrix_df.columns.tolist()
    # Dacă matricea e mare, afișăm doar top 50 după sumă
    if len(nodes) > 50:
        sums = matrix_df.sum(axis=1)
        top = sums.nlargest(50).index.tolist()
        matrix_df = matrix_df.loc[top, top]
        nodes = top

    fig = go.Figure(
        data=go.Heatmap(
            z=matrix_df.values,
            x=nodes,
            y=nodes,
            colorscale=colorscale,
            zmid=0,
            hoverongaps=False,
            hovertemplate="%{y} — %{x}<br>Valoare: %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#e2e8f0")),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#0d1117",
        font=dict(color="#94a3b8", size=9),
        xaxis=dict(tickangle=-45, tickfont=dict(size=7)),
        yaxis=dict(tickfont=dict(size=7)),
        height=700,
        margin=dict(l=120, r=20, t=60, b=120),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_bar_chart_html(df: pd.DataFrame, x_col: str, y_col: str, title: str, color: str = "#3b82f6") -> str:
    """Returnează HTML Plotly pentru un bar chart al top deputați."""
    if not PLOTLY_OK:
        return "<p>plotly nu este disponibil</p>"

    top = df.nlargest(20, y_col)
    fig = go.Figure(
        go.Bar(
            x=top[x_col],
            y=top[y_col],
            marker_color=color,
            hovertemplate="%{x}<br>" + y_col.replace("_", " ") + ": %{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8"),
        xaxis=dict(tickangle=-40, tickfont=dict(size=8)),
        yaxis=dict(gridcolor="#334155"),
        height=400,
        margin=dict(l=40, r=20, t=50, b=120),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_clique_chart_html(cliques_data: dict) -> str:
    """Bar chart cu distribuția clicilor după dimensiune."""
    if not PLOTLY_OK:
        return ""

    sizes = sorted(cliques_data["cliques_by_size"].keys(), key=int)
    counts = [len(cliques_data["cliques_by_size"][s]) for s in sizes]

    fig = go.Figure(
        go.Bar(
            x=[f"Dimensiune {s}" for s in sizes],
            y=counts,
            marker_color="#f59e0b",
            hovertemplate="%{x}<br>Număr clici: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Distribuția clicilor după dimensiune", font=dict(size=15, color="#e2e8f0")),
        paper_bgcolor="#0d1117",
        plot_bgcolor="#1e293b",
        font=dict(color="#94a3b8"),
        xaxis=dict(tickangle=-30),
        yaxis=dict(gridcolor="#334155"),
        height=350,
        margin=dict(l=40, r=20, t=50, b=80),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


# ═══════════════════════════════════════════════════════════════════════════════
#  7. ASAMBLARE HTML DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="ro">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analiza Rețelei — {leg_display}</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg: #0d1117;
    --surface: #161b22;
    --border: #21262d;
    --accent: #3b82f6;
    --accent2: #f59e0b;
    --accent3: #10b981;
    --text: #e2e8f0;
    --muted: #64748b;
    --mono: 'IBM Plex Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
  }}

  /* HEADER */
  header {{
    border-bottom: 1px solid var(--border);
    padding: 24px 40px;
    display: flex;
    align-items: center;
    gap: 20px;
    position: sticky;
    top: 0;
    background: var(--bg);
    z-index: 100;
  }}
  .header-badge {{
    background: var(--accent);
    color: white;
    font-family: var(--mono);
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 4px;
    letter-spacing: 0.05em;
  }}
  header h1 {{
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
  }}
  header p {{
    font-size: 13px;
    color: var(--muted);
    font-family: var(--mono);
  }}
  .stat-pills {{
    margin-left: auto;
    display: flex;
    gap: 12px;
  }}
  .stat-pill {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 16px;
    text-align: center;
  }}
  .stat-pill .val {{
    font-size: 22px;
    font-weight: 700;
    font-family: var(--mono);
    color: var(--accent);
  }}
  .stat-pill .lbl {{
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  /* NAV TABS */
  nav {{
    display: flex;
    gap: 0;
    border-bottom: 1px solid var(--border);
    padding: 0 40px;
    overflow-x: auto;
  }}
  .tab {{
    padding: 14px 20px;
    font-size: 13px;
    font-weight: 500;
    color: var(--muted);
    cursor: pointer;
    border-bottom: 2px solid transparent;
    white-space: nowrap;
    transition: all 0.15s;
  }}
  .tab:hover {{ color: var(--text); }}
  .tab.active {{
    color: var(--accent);
    border-bottom-color: var(--accent);
  }}

  /* CONTENT */
  main {{ padding: 32px 40px; max-width: 1600px; }}

  .section {{ display: none; }}
  .section.active {{ display: block; }}

  .section-title {{
    font-size: 15px;
    font-weight: 700;
    letter-spacing: -0.01em;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}
  .section-title::before {{
    content: '';
    width: 3px;
    height: 18px;
    background: var(--accent);
    border-radius: 2px;
    display: inline-block;
  }}
  .section-desc {{
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 20px;
    font-family: var(--mono);
    line-height: 1.6;
  }}

  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 24px;
  }}

  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    margin-bottom: 24px;
  }}

  /* TABLE */
  .table-wrap {{ overflow-x: auto; }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-family: var(--mono);
  }}
  thead tr {{
    background: var(--border);
  }}
  th {{
    padding: 10px 12px;
    text-align: left;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--muted);
    white-space: nowrap;
  }}
  td {{
    padding: 8px 12px;
    border-bottom: 1px solid var(--border);
    color: var(--text);
  }}
  tr:hover td {{ background: rgba(59,130,246,0.04); }}

  .rank {{
    font-weight: 700;
    color: var(--accent);
    font-size: 11px;
  }}
  .bar-cell {{
    min-width: 120px;
  }}
  .bar-fill {{
    height: 6px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    border-radius: 3px;
    min-width: 2px;
  }}

  /* METRIC CARDS */
  .metric-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 14px;
    margin-bottom: 24px;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    cursor: pointer;
    transition: border-color 0.15s;
  }}
  .metric-card:hover {{ border-color: var(--accent); }}
  .metric-card.active {{ border-color: var(--accent); background: rgba(59,130,246,0.06); }}
  .metric-card h3 {{
    font-size: 12px;
    font-weight: 600;
    margin-bottom: 4px;
  }}
  .metric-card p {{
    font-size: 10px;
    color: var(--muted);
    line-height: 1.5;
  }}

  /* GRAPH CONTAINER */
  .graph-container {{
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
  }}
  .graph-container iframe {{
    border: none;
    width: 100%;
    height: 600px;
    background: #0d1117;
  }}

  /* CLIQUE LIST */
  .clique-list {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
    gap: 12px;
    max-height: 500px;
    overflow-y: auto;
    padding-right: 4px;
  }}
  .clique-item {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }}
  .clique-header {{
    font-size: 11px;
    font-weight: 600;
    color: var(--accent2);
    margin-bottom: 6px;
    font-family: var(--mono);
  }}
  .clique-members {{
    font-size: 10px;
    color: var(--muted);
    line-height: 1.6;
  }}
  .badge {{
    display: inline-block;
    background: rgba(245,158,11,0.15);
    color: var(--accent2);
    border-radius: 4px;
    padding: 1px 6px;
    font-size: 10px;
    font-family: var(--mono);
    margin-right: 4px;
  }}

  /* MATRIX NOTE */
  .matrix-note {{
    font-size: 11px;
    color: var(--muted);
    font-family: var(--mono);
    padding: 10px;
    background: var(--bg);
    border-radius: 6px;
    margin-bottom: 16px;
  }}

  @media (max-width: 900px) {{
    header {{ flex-wrap: wrap; padding: 16px 20px; }}
    nav {{ padding: 0 20px; }}
    main {{ padding: 20px; }}
    .metric-grid {{ grid-template-columns: 1fr 1fr; }}
    .grid-2 {{ grid-template-columns: 1fr; }}
    .stat-pills {{ flex-wrap: wrap; }}
  }}
</style>
</head>
<body>

<header>
  <div>
    <div class="header-badge">ANALIZA REȚELEI</div>
    <h1>Parlamentul R. Moldova — {leg_display}</h1>
    <p>Co-autoriat legislative · Metrici de centralitate · Structuri de rețea</p>
  </div>
  <div class="stat-pills">
    <div class="stat-pill">
      <div class="val">{n_nodes}</div>
      <div class="lbl">Deputați</div>
    </div>
    <div class="stat-pill">
      <div class="val">{n_edges}</div>
      <div class="lbl">Colaborări</div>
    </div>
    <div class="stat-pill">
      <div class="val">{n_projects}</div>
      <div class="lbl">Proiecte</div>
    </div>
    <div class="stat-pill">
      <div class="val">{n_cliques}</div>
      <div class="lbl">Clici (≥3)</div>
    </div>
  </div>
</header>

<nav>
  <div class="tab active" onclick="showSection('centralitate')">📐 Centralitate</div>
  <div class="tab" onclick="showSection('grafuri')">🕸 Grafuri</div>
  <div class="tab" onclick="showSection('clici')">🔵 Clici</div>
  <div class="tab" onclick="showSection('pearson')">📊 Pearson</div>
  <div class="tab" onclick="showSection('similaritate')">🔗 Similaritate</div>
  <div class="tab" onclick="showSection('tabel')">📋 Tabel complet</div>
</nav>

<main>

<!-- ── CENTRALITATE ───────────────────────────────────────────────── -->
<div class="section active" id="sec-centralitate">
  <div class="section-title">Metrici de Centralitate</div>
  <div class="section-desc">Top 20 deputați per metrică · Click pe o metrică pentru a vedea graficul corespunzător</div>

  <div class="metric-grid">
    <div class="metric-card active" id="mc-degree" onclick="showMetric('degree')">
      <h3>🎯 Degree Centrality</h3>
      <p>Proporția de noduri la care un deputat este conectat direct. Măsoară popularitatea în rețea.</p>
    </div>
    <div class="metric-card" id="mc-power" onclick="showMetric('power')">
      <h3>⚡ Power Centrality (Bonacich)</h3>
      <p>Centralitate bazată pe puterea vecinilor. Un nod este important dacă vecinii săi sunt importanți.</p>
    </div>
    <div class="metric-card" id="mc-betweenness" onclick="showMetric('betweenness')">
      <h3>🌉 Betweenness Centrality</h3>
      <p>Proporția drumurilor scurte care trec prin nod. Măsoară rolul de broker/intermediar.</p>
    </div>
    <div class="metric-card" id="mc-stress" onclick="showMetric('stress')">
      <h3>💥 Stress Centrality</h3>
      <p>Numărul absolut de drumuri geodezice care trec prin nod. Similar cu betweenness, neîmpărțit la σ_st.</p>
    </div>
    <div class="metric-card" id="mc-prestige" onclick="showMetric('prestige')">
      <h3>👑 Prestige Centrality (PageRank)</h3>
      <p>Importanța unui nod bazată pe importanța celor care îl recomandă (analog Google PageRank).</p>
    </div>
    <div class="metric-card" id="mc-eccentricity" onclick="showMetric('eccentricity')">
      <h3>🎪 Eccentricity Centrality</h3>
      <p>Inversul excentricității: nodurile cu distanță maximă mică față de celelalte sunt centrale.</p>
    </div>
  </div>

  <div class="card" id="chart-degree">{chart_degree}</div>
  <div class="card" id="chart-power" style="display:none">{chart_power}</div>
  <div class="card" id="chart-betweenness" style="display:none">{chart_betweenness}</div>
  <div class="card" id="chart-stress" style="display:none">{chart_stress}</div>
  <div class="card" id="chart-prestige" style="display:none">{chart_prestige}</div>
  <div class="card" id="chart-eccentricity" style="display:none">{chart_eccentricity}</div>
</div>

<!-- ── GRAFURI ────────────────────────────────────────────────────── -->
<div class="section" id="sec-grafuri">
  <div class="section-title">Grafuri Interactive ale Rețelei</div>
  <div class="section-desc">Nodurile sunt colorate și dimensionate după metrica selectată · Hover pentru detalii · Drag pentru repoziționare</div>

  <div class="metric-grid">
    <div class="metric-card active" id="gmc-degree" onclick="showGraph('gdegree')">
      <h3>🎯 Degree Centrality</h3>
      <p>Graf colorat după gradul de conectare al fiecărui deputat.</p>
    </div>
    <div class="metric-card" id="gmc-betweenness" onclick="showGraph('gbetweenness')">
      <h3>🌉 Betweenness Centrality</h3>
      <p>Graf colorat după rolul de intermediar al fiecărui deputat.</p>
    </div>
    <div class="metric-card" id="gmc-power" onclick="showGraph('gpower')">
      <h3>⚡ Power Centrality</h3>
      <p>Graf colorat după puterea Bonacich / Eigenvector.</p>
    </div>
    <div class="metric-card" id="gmc-prestige" onclick="showGraph('gprestige')">
      <h3>👑 Prestige (PageRank)</h3>
      <p>Graf colorat după scorul de prestigiu PageRank.</p>
    </div>
    <div class="metric-card" id="gmc-stress" onclick="showGraph('gstress')">
      <h3>💥 Stress Centrality</h3>
      <p>Graf colorat după stress centrality (drumuri geodezice).</p>
    </div>
    <div class="metric-card" id="gmc-eccentricity" onclick="showGraph('geccentricity')">
      <h3>🎪 Eccentricity Centrality</h3>
      <p>Graf colorat după accesibilitate în rețea.</p>
    </div>
  </div>

  <div class="graph-container" id="graph-gdegree">
    <iframe srcdoc="{graph_degree}" id="iframe-gdegree"></iframe>
  </div>
  <div class="graph-container" id="graph-gbetweenness" style="display:none">
    <iframe srcdoc="{graph_betweenness}" id="iframe-gbetweenness"></iframe>
  </div>
  <div class="graph-container" id="graph-gpower" style="display:none">
    <iframe srcdoc="{graph_power}" id="iframe-gpower"></iframe>
  </div>
  <div class="graph-container" id="graph-gprestige" style="display:none">
    <iframe srcdoc="{graph_prestige}" id="iframe-gprestige"></iframe>
  </div>
  <div class="graph-container" id="graph-gstress" style="display:none">
    <iframe srcdoc="{graph_stress}" id="iframe-gstress"></iframe>
  </div>
  <div class="graph-container" id="graph-geccentricity" style="display:none">
    <iframe srcdoc="{graph_eccentricity}" id="iframe-geccentricity"></iframe>
  </div>
</div>

<!-- ── CLICI ─────────────────────────────────────────────────────── -->
<div class="section" id="sec-clici">
  <div class="section-title">Identificarea Clicilor</div>
  <div class="section-desc">
    O clică este un subgraf complet — fiecare pereche de noduri din clică este direct conectată.
    Se afișează clicile cu dimensiunea ≥ 3.
    Total clici găsite: {n_cliques_total} · Dimensiune maximă: {max_clique_size}
  </div>

  <div class="card">{chart_cliques}</div>

  <div class="section-title" style="margin-bottom:14px">Clicile maxime ({n_max_cliques} clici de dimensiune {max_clique_size})</div>
  <div class="clique-list">
    {max_cliques_html}
  </div>

  <div class="section-title" style="margin-top:28px;margin-bottom:14px">Toate clicile (dimensiune ≥ 3)</div>
  <div class="clique-list">
    {all_cliques_html}
  </div>
</div>

<!-- ── PEARSON ────────────────────────────────────────────────────── -->
<div class="section" id="sec-pearson">
  <div class="section-title">Matricea Coeficienților de Corelație Pearson</div>
  <div class="section-desc">
    Corelează profilurile de conectivitate ale deputaților (vectorii lor de adiacență ponderată).
    r ≈ +1 → structuri de colaborare similare · r ≈ -1 → structuri opuse · r ≈ 0 → independente.
    Afișați top 50 de noduri dacă rețeaua este mare.
  </div>
  <div class="matrix-note">ℹ Heatmap interactiv — hover pentru valori exacte · Roșu = corelație pozitivă · Albastru = corelație negativă</div>
  <div class="card">{heatmap_pearson}</div>
</div>

<!-- ── SIMILARITATE ───────────────────────────────────────────────── -->
<div class="section" id="sec-similaritate">
  <div class="section-title">Matricea Coeficienților de Similaritate (Jaccard)</div>
  <div class="section-desc">
    Similaritatea Jaccard: |N(u) ∩ N(v)| / |N(u) ∪ N(v)|.
    Măsoară în ce măsură doi deputați colaborează cu aceiași parteneri.
    1 = identic · 0 = fără parteneri comuni.
  </div>
  <div class="matrix-note">ℹ Heatmap interactiv — hover pentru valori exacte</div>
  <div class="card">{heatmap_similarity}</div>
</div>

<!-- ── TABEL COMPLET ──────────────────────────────────────────────── -->
<div class="section" id="sec-tabel">
  <div class="section-title">Tabel Complet — Toate Metricile</div>
  <div class="section-desc">Toți {n_nodes} deputații, ordonați după Degree Centrality descrescător</div>
  <div class="card table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Deputat</th>
          <th>Grad</th>
          <th>Degree</th>
          <th>Power Bonacich</th>
          <th>Betweenness</th>
          <th>Stress</th>
          <th>Prestige (PR)</th>
          <th>Eccentricity</th>
        </tr>
      </thead>
      <tbody>
        {table_rows}
      </tbody>
    </table>
  </div>
</div>

</main>

<script>
function showSection(name) {{
  document.querySelectorAll('.section').forEach(s => s.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById('sec-' + name).classList.add('active');
  event.target.classList.add('active');
}}

// metric bar charts
const metricKeys = ['degree','power','betweenness','stress','prestige','eccentricity'];
function showMetric(key) {{
  metricKeys.forEach(k => {{
    document.getElementById('chart-' + k).style.display = k === key ? '' : 'none';
    document.getElementById('mc-' + k).classList.toggle('active', k === key);
  }});
}}

// network graphs
const graphKeys = ['gdegree','gbetweenness','gpower','gprestige','gstress','geccentricity'];
function showGraph(key) {{
  graphKeys.forEach(k => {{
    document.getElementById('graph-' + k).style.display = k === key ? '' : 'none';
    const mc = document.getElementById('gmc-' + k.replace('g',''));
    if(mc) mc.classList.toggle('active', k === key);
  }});
}}
</script>
</body>
</html>
"""


def build_table_rows(df: pd.DataFrame) -> str:
    cols = [
        "Degree_Centrality", "Power_Centrality_Bonacich",
        "Betweenness_Centrality", "Stress_Centrality",
        "Prestige_Centrality_PageRank", "Eccentricity_Centrality",
    ]
    maxvals = {c: df[c].max() or 1 for c in cols}
    rows = []
    for rank, row in df.iterrows():
        cells = [f'<td class="rank">{rank}</td>', f'<td>{row["Deputat"]}</td>', f'<td>{int(row["Grad"])}</td>']
        for c in cols:
            v = row[c]
            pct = int(v / maxvals[c] * 100) if maxvals[c] else 0
            cells.append(
                f'<td><div style="font-size:10px;margin-bottom:2px">{v:.4f}</div>'
                f'<div class="bar-fill" style="width:{pct}%"></div></td>'
            )
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return "\n".join(rows)


def build_clique_items_html(cliques: list) -> str:
    items = []
    for i, clique in enumerate(cliques):
        members_html = ", ".join(f"<span class='badge'>{m}</span>" for m in sorted(clique))
        items.append(
            f'<div class="clique-item">'
            f'<div class="clique-header">Clică #{i+1} · {len(clique)} membri</div>'
            f'<div class="clique-members">{members_html}</div>'
            f'</div>'
        )
    return "\n".join(items)


def generate_dashboard(
    G: nx.Graph,
    df_metrics: pd.DataFrame,
    cliques_data: dict,
    pearson_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    projects: list,
    leg_display: str,
    output_file: str,
):
    print(f"\n🎨 Generare dashboard HTML: {output_file}")

    # Bar charts per metrică
    colors = {
        "degree": "#3b82f6",
        "power": "#8b5cf6",
        "betweenness": "#ef4444",
        "stress": "#f97316",
        "prestige": "#f59e0b",
        "eccentricity": "#10b981",
    }
    metric_map = {
        "degree": "Degree_Centrality",
        "power": "Power_Centrality_Bonacich",
        "betweenness": "Betweenness_Centrality",
        "stress": "Stress_Centrality",
        "prestige": "Prestige_Centrality_PageRank",
        "eccentricity": "Eccentricity_Centrality",
    }
    titles_map = {
        "degree": "Top 20 — Degree Centrality",
        "power": "Top 20 — Power Centrality (Bonacich)",
        "betweenness": "Top 20 — Betweenness Centrality",
        "stress": "Top 20 — Stress Centrality",
        "prestige": "Top 20 — Prestige Centrality (PageRank)",
        "eccentricity": "Top 20 — Eccentricity Centrality",
    }
    charts = {}
    for key, col in metric_map.items():
        charts[f"chart_{key}"] = create_bar_chart_html(
            df_metrics, "Deputat", col, titles_map[key], colors[key]
        )

    # Graf pyvis per metrică
    graphs = {}
    for key, col in metric_map.items():
        print(f"  🕸 Graf: {key}...")
        html = create_pyvis_graph(G, df_metrics, col, titles_map[key])
        # Escape pentru srcdoc
        graphs[f"graph_{key}"] = html.replace('"', "&quot;").replace("'", "&#39;")

    # Heatmaps
    heatmap_pearson = create_heatmap_html(pearson_df, "Matricea Coeficienților Pearson", "RdBu")
    heatmap_similarity = create_heatmap_html(similarity_df, "Matricea Similarității Jaccard", "YlOrRd")

    # Clici
    all_cliques_flat = []
    for size_key in sorted(cliques_data["cliques_by_size"].keys(), key=int, reverse=True):
        all_cliques_flat.extend(cliques_data["cliques_by_size"][size_key])

    n_cliques_total = sum(
        len(v) for v in cliques_data["cliques_by_size"].values()
    )

    html = HTML_TEMPLATE.format(
        leg_display=leg_display,
        n_nodes=G.number_of_nodes(),
        n_edges=G.number_of_edges(),
        n_projects=len(projects),
        n_cliques=n_cliques_total,
        n_cliques_total=n_cliques_total,
        max_clique_size=cliques_data["max_clique_size"],
        n_max_cliques=len(cliques_data["max_cliques"]),
        chart_degree=charts["chart_degree"],
        chart_power=charts["chart_power"],
        chart_betweenness=charts["chart_betweenness"],
        chart_stress=charts["chart_stress"],
        chart_prestige=charts["chart_prestige"],
        chart_eccentricity=charts["chart_eccentricity"],
        graph_degree=graphs["graph_degree"],
        graph_betweenness=graphs["graph_betweenness"],
        graph_power=graphs["graph_power"],
        graph_prestige=graphs["graph_prestige"],
        graph_stress=graphs["graph_stress"],
        graph_eccentricity=graphs["graph_eccentricity"],
        heatmap_pearson=heatmap_pearson,
        heatmap_similarity=heatmap_similarity,
        chart_cliques=create_clique_chart_html(cliques_data),
        max_cliques_html=build_clique_items_html(cliques_data["max_cliques"]),
        all_cliques_html=build_clique_items_html(all_cliques_flat),
        table_rows=build_table_rows(df_metrics),
    )

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  ✅ Dashboard salvat: {output_file}")


# ═══════════════════════════════════════════════════════════════════════════════
#  8. MAIN
# ═══════════════════════════════════════════════════════════════════════════════

LEGISLATURES = [
    {
        "label": "leg1",
        "display": "26.07.2021 — 21.10.2025",
        "json_file": "leg1_raw.json",
    },
    {
        "label": "leg2",
        "display": "22.10.2025 — 22.10.2029",
        "json_file": "leg2_raw.json",
    },
]


def analyze_legislature(leg: dict):
    label = leg["label"]
    display = leg["display"]
    json_file = leg["json_file"]

    print(f"\n{'═'*70}")
    print(f"  LEGISLATURA: {display}")
    print(f"{'═'*70}")

    if not os.path.exists(json_file):
        print(f"  ⚠ Fișierul {json_file} nu există. Săriți legislatura.")
        return

    projects = load_projects(json_file)
    print(f"  📂 Proiecte încărcate: {len(projects)}")

    G = build_graph(projects)
    print(f"  📊 Graf: {G.number_of_nodes()} noduri, {G.number_of_edges()} muchii")

    if G.number_of_nodes() == 0:
        print("  ⚠ Graful este gol. Fișier JSON poate fi vid.")
        return

    # Metrici
    df_metrics = compute_all_metrics(G)
    df_metrics.to_csv(f"network_metrics_{label}.csv", index=True, encoding="utf-8-sig")
    print(f"  ✓ network_metrics_{label}.csv")

    # Clici
    cliques_data = find_cliques(G)
    with open(f"cliques_{label}.json", "w", encoding="utf-8") as f:
        json.dump(cliques_data, f, ensure_ascii=False, indent=2)
    print(f"  ✓ cliques_{label}.json")

    # Matrice Pearson
    pearson_df = compute_pearson_matrix(G)
    pearson_df.to_csv(f"pearson_matrix_{label}.csv", encoding="utf-8-sig")
    print(f"  ✓ pearson_matrix_{label}.csv")

    # Matrice Similaritate (Jaccard)
    sim_df = compute_similarity_matrix(G, method="jaccard")
    sim_df.to_csv(f"similarity_matrix_{label}.csv", encoding="utf-8-sig")
    print(f"  ✓ similarity_matrix_{label}.csv")

    # Matching coefficient
    matching_df = compute_similarity_matrix(G, method="matching")
    matching_df.to_csv(f"matching_matrix_{label}.csv", encoding="utf-8-sig")
    print(f"  ✓ matching_matrix_{label}.csv")

    # Dashboard HTML interactiv
    generate_dashboard(
        G=G,
        df_metrics=df_metrics,
        cliques_data=cliques_data,
        pearson_df=pearson_df,
        similarity_df=sim_df,
        projects=projects,
        leg_display=display,
        output_file=f"network_analysis_{label}.html",
    )

    print(f"\n✅ Analiza pentru {display} completă!")
    print(f"   → Deschide network_analysis_{label}.html în browser pentru dashboard interactiv")


def main():
    print("=" * 70)
    print("ANALIZA REȚELEI DE COOPERARE — PARLAMENTUL R. MOLDOVA")
    print("=" * 70)
    print("\nDependențe necesare:")
    print("  pip install networkx numpy pandas pyvis plotly")
    print()

    for leg in LEGISLATURES:
        analyze_legislature(leg)

    print("\n🎉 PROCESARE COMPLETĂ!")
    print("\nFișiere generate (per legislatură):")
    print("  network_metrics_<leg>.csv       — toate metricile per deputat")
    print("  cliques_<leg>.json              — clici identificate")
    print("  pearson_matrix_<leg>.csv        — matricea coeficienților Pearson")
    print("  similarity_matrix_<leg>.csv     — matricea similaritate Jaccard")
    print("  matching_matrix_<leg>.csv       — matricea coeficienților de potrivire")
    print("  network_analysis_<leg>.html     — dashboard interactiv complet")


if __name__ == "__main__":
    main()