import time
import json
import csv
import pandas as pd
from collections import defaultdict
from itertools import combinations
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import re


# ============================================================
#  CONFIGURARE LEGISLATURI
# ============================================================

LEGISLATURES = [
    {
        "label": "leg1",
        "display": "26.07.2021-21.10.2025",
        "search_text": "26.07.2021-21.10.2025",
    },
    {
        "label": "leg2",
        "display": "22.10.2025-22.10.2029",
        "search_text": "22.10.2025-22.10.2029",
    },
]


# ============================================================
#  SCRAPER
# ============================================================

class ParliamentSeleniumScraper:
    def __init__(self, headless=False):
        print("Inițializare browser Selenium...")

        chrome_options = Options()
        if headless:
            chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument(
            "--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        prefs = {"profile.default_content_setting_values.notifications": 2}
        chrome_options.add_experimental_option("prefs", prefs)

        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 30)

        self.base_url = "https://parlament.md"
        self.projects_url = f"{self.base_url}/proiecte-de-acte-legislative.nspx"

        print("✓ Browser inițializat cu succes")

    # ------------------------------------------------------------------ helpers

    def wait_for_angular(self):
        time.sleep(3)

    def _handle_cookie_consent(self):
        print("  🍪 Verificare fereastră Cookies...")
        cookie_selectors = [
            (By.ID, "btnOK"),
            (By.XPATH, "//button[contains(text(), 'Accept')]"),
            (By.XPATH, "//a[contains(text(), 'Înțeleg')]"),
        ]
        for selector_type, selector_value in cookie_selectors:
            try:
                btn = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((selector_type, selector_value))
                )
                if btn.is_displayed():
                    self.driver.execute_script("arguments[0].click();", btn)
                    time.sleep(2)
                    print("    ✅ Fereastră Cookies închisă.")
                    return True
            except TimeoutException:
                continue
            except Exception:
                continue
        print("    ℹ Fereastră Cookies nu a fost găsită sau este deja închisă.")
        return False

    # ------------------------------------------------------------------ filters

    def set_filters_and_apply(self, legislature_search_text):
        print(f"\n🎯 SETARE FILTRE — Legislatură: {legislature_search_text}")
        try:
            print("  🌐 Navigare către pagina proiectelor...")
            self.driver.get(self.projects_url)
            self._handle_cookie_consent()

            print("  ⏳ Așteptare element 'select#structure'...")
            self.wait.until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "select#structure"))
            )

            if not self._set_legislature_structure(legislature_search_text):
                print("  ❌ Eroare la setarea legislaturii")
                return False

            if not self._set_initiators_deputy():
                print("  ⚠ Eroare la setarea inițiatorilor")
                return False

            if not self._click_apply_button():
                print("  ❌ Eroare la apăsarea butonului Aplică")
                return False

            print("  ✅ Filtre aplicate cu succes!")
            return True

        except TimeoutException:
            print("  ❌ Timeout la așteptarea elementului 'select#structure'.")
            return False
        except Exception as e:
            print(f"  ❌ Eroare la setarea filtrelor: {e}")
            return False

    def _set_legislature_structure(self, target_legislature):
        print("    🔍 Căutare dropdown legislatură...")
        try:
            dropdown = self.driver.find_element(By.CSS_SELECTOR, "select#structure")
            select = Select(dropdown)
            for option in select.options:
                if target_legislature in option.text:
                    select.select_by_visible_text(option.text)
                    print(f"    ✅ Legislatură selectată: {option.text}")
                    self.driver.execute_script(
                        """
                        arguments[0].dispatchEvent(new Event('change', {bubbles: true}));
                        arguments[0].dispatchEvent(new Event('input', {bubbles: true}));
                        """,
                        dropdown,
                    )
                    time.sleep(2)
                    return True
            print("    ❌ Legislatura nu a fost găsită în meniu")
            return False
        except Exception as e:
            print(f"    ❌ Eroare la setarea legislaturii: {e}")
            return False

    def _set_initiators_deputy(self):
        print("    🔍 Setare inițiatori = Deputat...")
        try:
            proposer_dropdown = self.wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "select#responsible-committee")
                )
            )
            self.driver.execute_script(
                """
                var sel = arguments[0];
                var scope = angular.element(sel).scope();
                scope.postData.InitiatorTypeId = 1;
                scope.$digest();
                """,
                proposer_dropdown,
            )
            time.sleep(5)
            print("    ✅ Inițiatori setați (Deputat).")
            return True
        except Exception as e:
            print(f"    ❌ Eroare Angular Scope: {e.__class__.__name__}")
            return False

    def _click_apply_button(self):
        print("    🔍 Apăsare buton Aplică...")
        try:
            btn = self.wait.until(
                EC.element_to_be_clickable(
                    (By.CSS_SELECTOR, "a[ng-click='loadData(true)']")
                )
            )
            self.driver.execute_script("arguments[0].click();", btn)
            print("    ✅ Buton Aplică apăsat!")
            time.sleep(8)
            self.wait_for_angular()
            return True
        except Exception as e:
            print(f"    ❌ Eroare la Aplică: {e}")
            return False

    # ------------------------------------------------------------------ extract

    def _filter_institutional_names(self, author_list):
        institutional_keywords = [
            "Biroul permanent", "Guvernul", "Comisia", "Parlamentul",
            "Președintele", "Ministerul", "Deputat", "Grupul",
            "Fracțiunea", "Senat", "Guvern", "Comisie", "Procurorul General",
        ]
        filtered = set()
        for author in author_list:
            clean = author.strip()
            if not clean:
                continue
            is_institutional = any(
                re.match(
                    r"\b" + re.escape(kw) + r"\b", clean, re.IGNORECASE | re.UNICODE
                )
                for kw in institutional_keywords
            )
            if not is_institutional and len(clean.split()) >= 2:
                filtered.add(clean)
        return sorted(filtered)

    def extract_deputy_authors_from_table(self):
        print("  📊 Extragere autori din tabel...")
        projects = []
        try:
            time.sleep(5)
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            print(f"  📋 Găsite {len(rows)} rânduri")

            for idx, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.CSS_SELECTOR, "td")
                    if len(cells) < 4:
                        continue
                    title = cells[0].text.strip()
                    author_cell = cells[3]
                    spans = author_cell.find_elements(
                        By.CSS_SELECTOR, "div.authors span.ng-scope"
                    )
                    raw_authors = [s.text.strip() for s in spans if s.text.strip()]
                    deputy_authors = self._filter_institutional_names(raw_authors)

                    if len(deputy_authors) >= 2:
                        projects.append(
                            {
                                "title": title,
                                "deputy_authors": deputy_authors,
                                "author_count": len(deputy_authors),
                            }
                        )
                        print(
                            f"    ✅ Rând {idx+1}: {len(deputy_authors)} autori — "
                            + ", ".join(deputy_authors)
                        )
                except Exception as e:
                    print(f"    ⚠ Rând {idx+1}: {e.__class__.__name__} — continuare")
                    continue

            print(f"  ✅ Extrase {len(projects)} proiecte cu 2+ autori")
            return projects
        except Exception as e:
            print(f"  ❌ Eroare extragere tabel: {e}")
            return []

    def navigate_to_next_page(self):
        try:
            buttons = self.driver.find_elements(
                By.XPATH, "//a[contains(text(), 'Următoarea')]"
            )
            for btn in buttons:
                if btn.is_displayed() and btn.is_enabled():
                    if "disabled" not in btn.get_attribute("class"):
                        self.driver.execute_script("arguments[0].click();", btn)
                        print("  ✅ Pagina următoare...")
                        time.sleep(6)
                        self.wait_for_angular()
                        return True
            print("  ℹ Ultima pagină atinsă.")
            return False
        except Exception as e:
            print(f"  ❌ Eroare navigare: {e}")
            return False

    def get_all_deputy_projects(self, legislature_search_text, max_pages=None):
        print(f"\n{'='*70}")
        print(f"📚 EXTRAGERE — {legislature_search_text}")
        print(f"{'='*70}")

        all_projects = []

        if not self.set_filters_and_apply(legislature_search_text):
            print("❌ Nu s-au putut seta filtrele")
            return all_projects

        page = 1
        has_more = True

        while has_more:
            print(f"\n📄 Pagina {page}")
            page_projects = self.extract_deputy_authors_from_table()
            if page_projects:
                all_projects.extend(page_projects)
                print(f"  📊 Total acumulat: {len(all_projects)} proiecte")

            if max_pages and page >= max_pages:
                print(f"  ℹ Limită {max_pages} pagini atinsă")
                break

            has_more = self.navigate_to_next_page()
            page += 1

            if page > 500:
                print("  ⚠ Limită de siguranță (500 pagini) atinsă")
                break

        print(f"\n✅ TOTAL: {len(all_projects)} proiecte din {page-1} pagini")
        return all_projects

    def close(self):
        if self.driver:
            self.driver.quit()
            print("\n✓ Browser închis")


# ============================================================
#  EXPORT PENTRU INSTRUMENTE DE REȚEA SOCIALĂ
# ============================================================

class NetworkExporter:
    """
    Exportă datele co-autoriatului în formate compatibile cu:
      • Gephi  → GEXF + GraphML + CSV edge list
      • Social Network Visualizer (SNV/SocNetV) → Pajek (.net)
      • UCINET → DL format
    """

    @staticmethod
    def build_cooperation_data(projects):
        """Construiește structurile de bază: noduri și muchii ponderate."""
        nodes = set()
        edge_weights = defaultdict(int)

        for project in projects:
            authors = project["deputy_authors"]
            for author in authors:
                nodes.add(author)
            for a1, a2 in combinations(sorted(authors), 2):
                edge_weights[(a1, a2)] += 1

        node_list = sorted(nodes)
        node_index = {name: i + 1 for i, name in enumerate(node_list)}  # 1-based

        edges = [
            {"source": src, "target": tgt, "weight": w}
            for (src, tgt), w in sorted(
                edge_weights.items(), key=lambda x: x[1], reverse=True
            )
        ]

        print(f"  👥 Noduri: {len(node_list)}")
        print(f"  🔗 Muchii: {len(edges)}")

        return node_list, node_index, edges

    # ------------------------------------------------------------------ CSV

    @staticmethod
    def export_csv(node_list, edges, prefix):
        """Edge list CSV (compatibil Gephi CSV import)."""
        nodes_file = f"{prefix}_nodes.csv"
        edges_file = f"{prefix}_edges.csv"

        with open(nodes_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Id", "Label"])
            for node in node_list:
                writer.writerow([node, node])
        print(f"  ✓ {nodes_file}")

        with open(edges_file, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Source", "Target", "Weight", "Type"])
            for e in edges:
                writer.writerow([e["source"], e["target"], e["weight"], "Undirected"])
        print(f"  ✓ {edges_file}")

    # ------------------------------------------------------------------ GEXF (Gephi)

    @staticmethod
    def export_gexf(node_list, edges, prefix):
        """GEXF — format nativ Gephi, suportă atribute și dinamică."""
        filename = f"{prefix}.gexf"
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<gexf xmlns="http://gexf.net/1.3" xmlns:viz="http://gexf.net/1.3/viz" version="1.3">',
            '  <meta>',
            f'    <description>Rețea cooperare deputați — {prefix}</description>',
            '  </meta>',
            '  <graph defaultedgetype="undirected">',
            '    <attributes class="edge">',
            '      <attribute id="0" title="weight" type="integer"/>',
            '    </attributes>',
            '    <nodes>',
        ]
        for node in node_list:
            safe = node.replace("&", "&amp;").replace('"', "&quot;")
            lines.append(f'      <node id="{safe}" label="{safe}"/>')
        lines.append("    </nodes>")
        lines.append("    <edges>")
        for i, e in enumerate(edges):
            s = e["source"].replace("&", "&amp;").replace('"', "&quot;")
            t = e["target"].replace("&", "&amp;").replace('"', "&quot;")
            lines.append(
                f'      <edge id="{i}" source="{s}" target="{t}">'
                f'<attvalues><attvalue for="0" value="{e["weight"]}"/></attvalues></edge>'
            )
        lines += ["    </edges>", "  </graph>", "</gexf>"]

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  ✓ {filename}  (Gephi — GEXF)")

    # ------------------------------------------------------------------ GraphML (Gephi)

    @staticmethod
    def export_graphml(node_list, edges, prefix):
        """GraphML — alt format Gephi, compatibil și cu yEd."""
        filename = f"{prefix}.graphml"
        lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<graphml xmlns="http://graphml.graphdrawing.org/graphml">',
            '  <key id="label" for="node" attr.name="label" attr.type="string"/>',
            '  <key id="weight" for="edge" attr.name="weight" attr.type="int"/>',
            '  <graph id="G" edgedefault="undirected">',
        ]
        for node in node_list:
            safe = node.replace("&", "&amp;").replace('"', "&quot;")
            lines.append(f'    <node id="{safe}"><data key="label">{safe}</data></node>')
        for i, e in enumerate(edges):
            s = e["source"].replace("&", "&amp;").replace('"', "&quot;")
            t = e["target"].replace("&", "&amp;").replace('"', "&quot;")
            lines.append(
                f'    <edge id="e{i}" source="{s}" target="{t}">'
                f'<data key="weight">{e["weight"]}</data></edge>'
            )
        lines += ["  </graph>", "</graphml>"]

        with open(filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"  ✓ {filename}  (Gephi / yEd — GraphML)")

    # ------------------------------------------------------------------ Pajek .net (SNV)

    @staticmethod
    def export_pajek(node_list, node_index, edges, prefix):
        """Format Pajek .net — citit direct de Social Network Visualizer (SocNetV)."""
        filename = f"{prefix}.net"
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"*Vertices {len(node_list)}\n")
            for node in node_list:
                idx = node_index[node]
                f.write(f'{idx} "{node}"\n')
            f.write("*Edges\n")
            for e in edges:
                s = node_index[e["source"]]
                t = node_index[e["target"]]
                f.write(f"{s} {t} {e['weight']}\n")
        print(f"  ✓ {filename}  (SocNetV / Pajek — .net)")

    # ------------------------------------------------------------------ UCINET DL

    @staticmethod
    def export_ucinet_dl(node_list, node_index, edges, prefix):
        """
        Format DL pentru UCINET — matrice de adiacență cu etichete.
        Folosim formatul NODELIST2 pentru eficiență (graf rar).
        """
        filename = f"{prefix}.dl"
        n = len(node_list)

        # Construim matricea de adiacență (dicționar sparse)
        matrix = defaultdict(lambda: defaultdict(int))
        for e in edges:
            si = node_index[e["source"]]
            ti = node_index[e["target"]]
            matrix[si][ti] = e["weight"]
            matrix[ti][si] = e["weight"]

        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"DL N={n} FORMAT=MATRIX\n")
            f.write("LABELS:\n")
            f.write(",".join(node_list) + "\n")
            f.write("DATA:\n")
            for i in range(1, n + 1):
                row = [str(matrix[i][j]) for j in range(1, n + 1)]
                f.write(" ".join(row) + "\n")
        print(f"  ✓ {filename}  (UCINET — DL matrix)")

    # ------------------------------------------------------------------ master export

    @classmethod
    def export_all(cls, projects, prefix):
        print(f"\n💾 Export date rețea — prefix: '{prefix}'")
        node_list, node_index, edges = cls.build_cooperation_data(projects)

        # CSV (edge list + node list)
        cls.export_csv(node_list, edges, prefix)

        # Gephi
        cls.export_gexf(node_list, edges, prefix)
        cls.export_graphml(node_list, edges, prefix)

        # SocNetV / Pajek
        cls.export_pajek(node_list, node_index, edges, prefix)

        # UCINET
        cls.export_ucinet_dl(node_list, node_index, edges, prefix)

        # JSON brut
        json_file = f"{prefix}_raw.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)
        print(f"  ✓ {json_file}  (date brute JSON)")

        print(f"  ✅ Export complet pentru '{prefix}'!")
        return node_list, edges


# ============================================================
#  MAIN
# ============================================================

def main():
    print("=" * 70)
    print("EXTRAGERE DATE CO-AUTORIAT DEPUTAȚI")
    print("Parlamentul Republicii Moldova")
    print("Legislaturi: 2021–2025 și 2025–2029")
    print("=" * 70)

    scraper = None

    try:
        scraper = ParliamentSeleniumScraper(headless=False)

        for leg in LEGISLATURES:
            label = leg["label"]
            search_text = leg["search_text"]
            display = leg["display"]

            print(f"\n{'#'*70}")
            print(f"# LEGISLATURA: {display}")
            print(f"{'#'*70}")

            projects = scraper.get_all_deputy_projects(
                legislature_search_text=search_text,
                max_pages=None,          # None = toate paginile; pune un număr pentru test
            )

            if not projects:
                print(f"⚠ Nu s-au găsit proiecte pentru {display}. Continuare...")
                continue

            # Export toate formatele
            NetworkExporter.export_all(projects, prefix=label)

            print(f"\n📋 Rezumat {display}:")
            print(f"   Proiecte cu 2+ autori: {len(projects)}")
            all_deputies = set()
            for p in projects:
                all_deputies.update(p["deputy_authors"])
            print(f"   Deputați unici: {len(all_deputies)}")

        print("\n🎉 TOATE LEGISLATURILE AU FOST PROCESATE!")
        print("\nFișiere generate (per legislatură):")
        print("  leg1_nodes.csv / leg2_nodes.csv    — noduri Gephi")
        print("  leg1_edges.csv / leg2_edges.csv    — muchii Gephi (CSV)")
        print("  leg1.gexf      / leg2.gexf         — Gephi (GEXF nativ)")
        print("  leg1.graphml   / leg2.graphml      — Gephi / yEd (GraphML)")
        print("  leg1.net       / leg2.net           — SocNetV / Pajek")
        print("  leg1.dl        / leg2.dl            — UCINET (DL matrix)")
        print("  leg1_raw.json  / leg2_raw.json      — date brute JSON")

    except Exception as e:
        print(f"\n❌ EROARE GENERALĂ: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if scraper:
            scraper.close()


if __name__ == "__main__":
    main()