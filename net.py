import json
import pandas as pd
import networkx as nx
import os
import community as community_louvain 

class NetworkVisualizer:
    def __init__(self, collab_file='colaborari_deputati.csv'):
        self.collab_file = collab_file
        self.G = self._load_network_from_csv()

        if self.G.number_of_nodes() > 0:
            self.calculate_metrics()
            self.detect_communities()
            self.categorize_centrality('degree_centrality')
            self.categorize_centrality('betweenness_centrality')

    def _load_network_from_csv(self):
        if not os.path.exists(self.collab_file):
            print(f"❌ Eroare: Fișierul de colaborări '{self.collab_file}' nu a fost găsit.")
            print("Rulați mai întâi 'scrap.py' pentru a genera datele.")
            return nx.Graph()

        print(f"🔗 Încărcare muchii din '{self.collab_file}'...")
        df = pd.read_csv(self.collab_file, encoding='utf-8-sig')
        G = nx.Graph()

        for index, row in df.iterrows():
            deputat1 = row['Deputat_1']
            deputat2 = row['Deputat_2'] 
            greutate = row['Numar_Colaborari']
            G.add_edge(deputat1, deputat2, weight=greutate)
            
        print(f"  👥 Noduri (deputați): {G.number_of_nodes()}")
        print(f"  🔗 Muchii (colaborări): {G.number_of_edges()}")
        
        return G

    def calculate_metrics(self):
        if self.G.number_of_nodes() == 0:
            return

        print("📊 Calcul centralități...")
        degree_centrality = nx.degree_centrality(self.G)
        betweenness_centrality = nx.betweenness_centrality(self.G)

        for node in self.G.nodes():
            self.G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
            self.G.nodes[node]['betweenness_centrality'] = betweenness_centrality.get(node, 0)
    
    def detect_communities(self):
        print("🤝 Detectare Comunități (algoritmul Louvain)...")
        if self.G.number_of_nodes() < 3 or self.G.number_of_edges() == 0:
            print("  ℹ Nu există suficientă structură pentru a detecta comunități.")
            return

        try:
            partition = community_louvain.best_partition(self.G, weight='weight')
            
            for node, comm_id in partition.items():
                self.G.nodes[node]['community'] = comm_id
            
            modularity = community_louvain.modularity(partition, self.G, weight='weight')
            print(f"  ✅ Modularity score: {modularity:.4f}")
            print(f"  📊 Comunități identificate: {len(set(partition.values()))}")
            
        except Exception as e:
            print(f"  ❌ Eroare la detectarea comunităților: {e.__class__.__name__}.")
            
    def categorize_centrality(self, centrality_type):
        if centrality_type not in ['degree_centrality', 'betweenness_centrality']:
            return
        
        centrality_values = [data.get(centrality_type, 0) for node, data in self.G.nodes(data=True) if data.get(centrality_type, 0) > 0]
        if not centrality_values:
            return

        sorted_values = sorted(centrality_values)
        n = len(sorted_values)
        
        q1_index = int(n * 0.33)
        q2_index = int(n * 0.66)
        
        low_cutoff = sorted_values[q1_index] if n > q1_index else 0
        high_cutoff = sorted_values[q2_index] if n > q2_index else max(centrality_values)

        for node, data in self.G.nodes(data=True):
            value = data.get(centrality_type, 0)
            
            tier = 'Low'
            if value >= high_cutoff and value > 0:
                tier = 'High'
            elif value >= low_cutoff and value > 0:
                tier = 'Medium'
            
            self.G.nodes[node][f'{centrality_type}_tier'] = tier
        
        print(f"  ✅ Categorii definite pentru {centrality_type}")


    def export_interactive_network(self, output_file='retea_colaborare_interactiva.html'):
        if self.G.number_of_nodes() == 0:
            print("  ❌ Rețeaua este goală. Nu se poate genera vizualizarea.")
            return

        print(f"🌐 Export rețea interactivă către '{output_file}'...")
        
        nodes_data = []
        edges_data = []
        
        color_tiers = {
            'High': '#e74c3c', 
            'Medium': '#f1c40f',
            'Low': '#3498db'
        }

        community_colors = [
            '#1abc9c', '#f39c12', '#9b59b6', '#34495e', '#e74c3c', 
            '#2ecc71', '#e67e22', '#3498db', '#95a5a6', '#f1c40f',
            '#d35400', '#2980b9', '#16a085', '#27ae60', '#8e44ad',
            '#c0392b'
        ]
        
        community_ids = set(nx.get_node_attributes(self.G, 'community').values())
        community_id_map = {comm_id: community_colors[i % len(community_colors)] for i, comm_id in enumerate(sorted(list(community_ids)))}
        
        all_degree_vals = [data.get('degree_centrality', 0) for _, data in self.G.nodes(data=True)]
        max_degree_val = max(all_degree_vals) if all_degree_vals else 1
        
        all_betweenness_vals = [data.get('betweenness_centrality', 0) for _, data in self.G.nodes(data=True)]
        max_betweenness_val = max(all_betweenness_vals) if all_betweenness_vals else 1
        
        min_size = 15 
        max_size = 55 
        
        for i, node in enumerate(self.G.nodes()):
            data = self.G.nodes[node]
            degree = self.G.degree(node)

            # Folosim <b> și <br> pentru a formata HTML-ul în tooltip
            title = (f"<b>{node}</b><br>"
                     f"Comunitate: {data.get('community', -1) + 1}<br>"
                     f"Grad Rețea: {degree}<br>"
                     f"Centralitate Grad: {data.get('degree_centrality', 0):.3f} ({data.get('degree_centrality_tier', 'N/A')})<br>"
                     f"Centralitate Betweenness: {data.get('betweenness_centrality', 0):.3f} ({data.get('betweenness_centrality_tier', 'N/A')})")
            
            nodes_data.append({
                'id': node,
                'label': node,
                'community': data.get('community', -1),
                'community_color': community_id_map.get(data.get('community', -1), '#AAAAAA'),
                'degree_val': data.get('degree_centrality', 0),
                'betweenness_val': data.get('betweenness_centrality', 0),
                'degree_tier': data.get('degree_centrality_tier', 'Low'),
                'betweenness_tier': data.get('betweenness_centrality_tier', 'Low'),
                'title': title.replace('\n', '\\n'),
            })

        max_weight = max([d['weight'] for u, v, d in self.G.edges(data=True)]) if self.G.number_of_edges() > 0 else 1

        for i, (u, v, data) in enumerate(self.G.edges(data=True)):
            weight = data['weight']
            edges_data.append({
                'id': i,
                'from': u,
                'to': v,
                'value': weight,
                'title': f"Colaborări: {weight}",
                'width': 1.0 + 4.0 * (weight / max_weight), 
                'smooth': {'enabled': True, 'type': 'dynamic'}
            })
            
        nodes_json = json.dumps(nodes_data, indent=2)
        edges_json = json.dumps(edges_data, indent=2)
        color_tiers_json = json.dumps(color_tiers)
        community_map_json = json.dumps(community_id_map)

        # 2. Generarea conținutului HTML (Vis.js template)
        html_content = f"""
<!DOCTYPE html>
<html lang="ro">
<head>
    <meta charset="utf-8">
    <title>Rețeaua de Colaborare a Deputaților (Vizualizare Interactivă)</title>
    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style type="text/css">
        body {{ 
            font: 12pt 'Arial', sans-serif; 
            color: #333;
            margin: 0;
            padding: 0;
            height: 100vh;
            overflow: hidden;
            background-color: #f4f7f6;
        }}
        #container {{
            position: relative;
            width: 100%;
            height: 100vh;
        }}
        #mynetwork {{
            width: 100%;
            height: 100%; 
            background-color: #333;
        }}
        .header {{ 
            position: absolute; 
            top: 0;
            left: 0;
            right: 0;
            text-align: center; 
            padding: 10px 0;
            background-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            z-index: 10;
        }}
        .controls {{ 
            position: absolute; 
            top: 70px;
            left: 20px; 
            padding: 10px; 
            border-radius: 8px; 
            background-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            z-index: 10;
        }}
        .legend {{ 
            position: absolute; 
            bottom: 20px; 
            right: 20px; 
            padding: 15px; 
            border-radius: 8px; 
            background-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
            z-index: 10;
        }}
        .color-box {{
            width: 15px;
            height: 15px;
            display: inline-block;
            margin-right: 5px;
            border: 1px solid #333;
            border-radius: 3px;
        }}
        .community-legend-item {{ {{ display: flex; align-items: center; margin-bottom: 5px; }} }}
        select {{ padding: 8px; border-radius: 4px; border: 1px solid #ccc; }}
        strong {{ color: #2c3e50; }}
        
        .vis-tooltip {{
            padding: 10px !important;
            border-radius: 6px !important;
            border: 1px solid #777 !important;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
            white-space: normal !important;
        }}
    </style>
</head>
<body>

<div id="container">
    <div class="header">
        <h1>Rețeaua de Co-autoriat a Deputaților</h1>
    </div>

    <div class="controls">
        <label for="centralitySelect">Alegeți metrica pentru vizualizare:</label>
        <select id="centralitySelect" onchange="updateNetwork()">
            <option value="degree" selected>Centralitate Grad (Dimensiune/Culoare)</option>
            <option value="betweenness">Centralitate Betweenness (Dimensiune/Culoare)</option>
            <option value="community">Comunități (Culoare)</option>
        </select>
    </div>

    <div id="mynetwork"></div>

    <div class="legend">
        <div id="centrality-legend">
            <strong>Evidențiere Centralitate (Culoare / Dimensiune):</strong>
            <p><span class="color-box" style="background-color: #e74c3c;"></span> Înaltă (Top 33%)</p>
            <p><span class="color-box" style="background-color: #f1c40f;"></span> Medie</p>
            <p><span class="color-box" style="background-color: #3498db;"></span> Scăzută</p>
            <p>Grosimea muchiei: Număr de proiecte comune</p>
        </div>
        <div id="community-legend" style="display:none;">
            <strong>Evidențiere Comunități (Culoare):</strong>
            <div id="community-legend-list"></div>
        </div>
    </div>
</div>

<script type="text/javascript">
    const RAW_NODES = {nodes_json};
    const EDGES = new vis.DataSet({edges_json});
    const COLOR_TIERS = {color_tiers_json};
    const COMMUNITY_MAP = {community_map_json};
    
    const MAX_DEGREE = {max_degree_val};
    const MAX_BETWEENNESS = {max_betweenness_val};

    const MIN_SIZE = 15;
    const MAX_SIZE = 55;
    
    let network = null;
    let NODES = new vis.DataSet(RAW_NODES);

    function calculateSize(value, max_value) {{
        if (max_value === 0 || value === 0) return MIN_SIZE;
        return MIN_SIZE + (MAX_SIZE - MIN_SIZE) * (value / max_value);
    }}

    function updateNetwork() {{
        const mode = document.getElementById('centralitySelect').value;
        const tempNodes = [];

        RAW_NODES.forEach(node => {{
            let color, size;
            let title = node.title;

            if (mode === 'community') {{
                color = node.community_color;
                size = calculateSize(node.degree_val, MAX_DEGREE); 
                title += "<br>Vizualizare Culoare: Comunitate";

            }} else if (mode === 'degree') {{
                color = COLOR_TIERS[node.degree_tier];
                size = calculateSize(node.degree_val, MAX_DEGREE);
                title += "<br>Vizualizare: Grad";

            }} else if (mode === 'betweenness') {{
                color = COLOR_TIERS[node.betweenness_tier];
                size = calculateSize(node.betweenness_val, MAX_BETWEENNESS);
                title += "<br>Vizualizare: Betweenness";
            }}

            tempNodes.push({{
                ...node,
                color: {{
                    border: '#ffffff',
                    background: color,
                    highlight: {{ border: '#000000', background: color }}
                }},
                size: size,
                title: title,
                shadow: true
            }});
        }});
        
        NODES.update(tempNodes);
        updateLegend(mode);
    }}
    
    function updateLegend(mode) {{
        const centralityLegend = document.getElementById('centrality-legend');
        const communityLegend = document.getElementById('community-legend');
        const communityListDiv = document.getElementById('community-legend-list');

        if (mode === 'community') {{
            centralityLegend.style.display = 'none';
            communityLegend.style.display = 'block';
            
            let html = '';
            Object.keys(COMMUNITY_MAP).sort((a, b) => parseInt(a) - parseInt(b)).forEach(id_key => {{
                const color = COMMUNITY_MAP[id_key];
                const displayId = parseInt(id_key) + 1; 
                html += '<div class="community-legend-item"><span class="color-box" style="background-color: ' + color + ';"></span> Comunitatea ' + displayId + '</div>';
            }});
            communityListDiv.innerHTML = html;

        }} else {{
            centralityLegend.style.display = 'block';
            communityLegend.style.display = 'none';
        }}
    }}

    function drawNetwork() {{
        var container = document.getElementById('mynetwork');
        var data = {{ nodes: NODES, edges: EDGES }};
        
        var options = {{
            nodes: {{
                shape: 'dot',
                scaling: {{ min: MIN_SIZE, max: MAX_SIZE }},
                font: {{ 
                    face: 'Arial', 
                    size: 14, 
                    color: '#ffffff',
                    strokeWidth: 2, 
                    strokeColor: '#333'
                }}
            }},
            edges: {{
                color: {{ color: '#ccc', highlight: '#fff' }},
                width: 1,
                smooth: {{ type: 'dynamic' }}
            }},
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based', 
                forceAtlas2Based: {{
                    gravitationalConstant: -20, 
                    centralGravity: 0.005, 
                    springLength: 100,
                    springConstant: 0.1, 
                    damping: 0.4
                }},
                minVelocity: 0.75,
                stabilization: {{ iterations: 2000 }}
            }},
            // CORECȚIE FINALĂ: Setăm proprietatea allowHtml direct în interaction, 
            // deși am setat-o și în configure în pasul anterior, 
            // menținem structura simplă și adăugăm asigurări CSS.
            interaction: {{ 
                tooltipDelay: 200, 
                hover: true,
                hoverConnectedEdges: true,
                tooltip: {{ 
                    allowHtml: true // Cea mai directă și (sperăm) funcțională setare
                }}
            }}
        }};

        network = new vis.Network(container, data, options);
        updateNetwork(); 
    }}

    document.addEventListener('DOMContentLoaded', drawNetwork);
</script>

</body>
</html>
"""
        # 3. Salvarea fișierului
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"✅ Vizualizare interactivă generată! Deschideți '{output_file}' în browser.")


def main():
    print("=" * 70)
    print("VIZUALIZARE INTERACTIVĂ REȚEA COOPERARE")
    print("Datele sunt preluate din colaborari_deputati.csv")
    print("=" * 70)
    
    visualizer = NetworkVisualizer()
    visualizer.export_interactive_network()
    
    print("\n🎉 VIZUALIZARE COMPLETĂ!")

if __name__ == "__main__":
    main()