import time
import json
import pandas as pd
import networkx as nx
from collections import defaultdict
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager
import re 

class ParliamentSeleniumScraper:
    def __init__(self, headless=False):
        """
        Inițializează scraper-ul cu Selenium și dezactivează notificările.
        """
        print("Inițializare browser Selenium...")
        
        chrome_options = Options()
        # Puteți schimba headless=True dacă doriți să ruleze în fundal
        if headless: 
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-gpu')
        chrome_options.add_argument('--window-size=1920,1080')
        chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
        
        prefs = {"profile.default_content_setting_values.notifications": 2}
        chrome_options.add_experimental_option("prefs", prefs)
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.wait = WebDriverWait(self.driver, 30)
        
        self.base_url = "https://parlament.md"
        self.projects_url = f"{self.base_url}/proiecte-de-acte-legislative.nspx"
        
        print("✓ Browser inițializat cu succes")
    
    def wait_for_angular(self, timeout=15):
        """
        Funcție de așteptare simplificată pentru a da timp Angular să proceseze.
        """
        try:
            time.sleep(3) 
            return
        except:
            time.sleep(5)
            return
    
    def _handle_cookie_consent(self):
        """Încearcă să închidă fereastra de consimțământ Cookie-uri."""
        print("  🍪 Verificare fereastră Cookies...")
        
        cookie_selectors = [
            (By.ID, "btnOK"), 
            (By.XPATH, "//button[contains(text(), 'Accept')]"),
            (By.XPATH, "//a[contains(text(), 'Înțeleg')]"),
        ]
        
        for selector_type, selector_value in cookie_selectors:
            try:
                cookie_button = WebDriverWait(self.driver, 5).until(
                    EC.element_to_be_clickable((selector_type, selector_value))
                )
                
                if cookie_button.is_displayed():
                    print(f"    ✅ Buton Cookie găsit ({selector_value}). Apasare...")
                    self.driver.execute_script("arguments[0].click();", cookie_button)
                    time.sleep(2) 
                    print("    ✅ Fereastră Cookies închisă.")
                    return True
                    
            except TimeoutException:
                continue 
            except Exception:
                continue
                
        print("    ℹ Fereastră Cookies nu a fost găsită sau este deja închisă.")
        return False
    
    def set_filters_and_apply(self):
        """
        Setează filtrele și apasă butonul Aplică.
        (Același cod, deoarece funcționează corect)
        """
        print(f"\n🎯 SETARE FILTRE ȘI APLICARE")
        print("📅 Legislatura: 26.07.2021-21.10.2025")
        print("👥 Inițiatori: Deputat")
        
        try:
            print("  🌐 Navigare către pagina proiectelor...")
            self.driver.get(self.projects_url)
            
            self._handle_cookie_consent()
            
            print("  ⏳ Așteptare ca pagina să se încarce complet...")
            self.wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "select#structure")))
            print("  ✅ Elementul 'select#structure' este prezent.")
            
            # 1. SETARE LEGISLATURĂ
            print("  🏛️ Setare legislatură...")
            if not self._set_legislature_structure():
                print("  ❌ Eroare la setarea legislaturii")
                return False
            
            # 2. SETARE INIȚIATORI (SOLUȚIA FINALĂ - Angular Scope)
            print("  👥 Setare inițiatori...")
            if not self._set_initiators_deputy():
                print("  ⚠ Eroare la setarea inițiatorilor")
                return False
            
            # 3. APĂSARE BUTON APLICĂ
            print("  🔘 Apasare buton 'Aplică'...")
            if not self._click_apply_button():
                print("  ❌ Eroare la apăsarea butonului Aplică")
                return False
            
            print("  ✅ Filtre aplicate cu succes!")
            return True
            
        except TimeoutException:
            print("  ❌ Eroare: Timp expirat la așteptarea elementului 'select#structure'.")
            return False
        except Exception as e:
            print(f"  ❌ Eroare la setarea filtrelor: {e}")
            return False
    
    def _set_legislature_structure(self):
        """Setează legislatura corectă"""
        print("    🔍 Căutare dropdown legislatură...")
        
        try:
            structure_dropdown = self.driver.find_element(By.CSS_SELECTOR, "select#structure")
            select = Select(structure_dropdown)
            
            target_legislature = "26.07.2021-21.10.2025"
            
            for option in select.options:
                if target_legislature in option.text:
                    select.select_by_visible_text(option.text)
                    print(f"    ✅ Legislatură selectată: {option.text}")
                    
                    self.driver.execute_script("""
                        arguments[0].dispatchEvent(new Event('change', {bubbles: true}));
                        arguments[0].dispatchEvent(new Event('input', {bubbles: true}));
                    """, structure_dropdown)
                    
                    time.sleep(2)
                    return True
            
            print("    ❌ Legislatura nu a fost găsită")
            return False
            
        except Exception as e:
            print(f"    ❌ Eroare la setarea legislaturii: {e}")
            return False
    
    def _set_initiators_deputy(self):
        """Setează inițiatorii la Deputat prin injectarea valorii direct în Angular scope."""
        print("    🔍 Căutare dropdown inițiatori...")
        
        selector = "select#responsible-committee"
        
        try:
            proposer_dropdown = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, selector))
            )
            print("    ✅ Dropdown inițiatori găsit.")
            
            print("    ℹ Se forțează selecția prin injectare Angular Scope...")
            
            js_code = f"""
                var selectElement = arguments[0];
                var angularScope = angular.element(selectElement).scope();
                
                angularScope.postData.InitiatorTypeId = 1; 
                
                angularScope.$digest();
            """
            
            self.driver.execute_script(js_code, proposer_dropdown)
            
            current_value = self.driver.execute_script("return arguments[0].value;", proposer_dropdown)
            print(f"    ✅ Inițiatori setați (Forțat Angular Scope). Valoare DOM: {current_value}")

            time.sleep(5) 
            return True

        except Exception as e:
            print(f"    ❌ Eroare la setarea Angular Scope: {e.__class__.__name__}. Verificați dacă Angular este încărcat.")
            return False
    
    def _click_apply_button(self):
        """Apasă butonul Aplică care încarcă datele"""
        print("    🔍 Căutare buton Aplică...")
        
        try:
            apply_button = self.wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "a[ng-click='loadData(true)']"))
            )
            
            self.driver.execute_script("arguments[0].click();", apply_button)
            print("    ✅ Buton Aplică apăsat!")
            
            print("    ⏳ Așteptare încărcare rezultate...")
            time.sleep(8)
            self.wait_for_angular()
            
            return True
            
        except Exception as e:
            print(f"    ❌ Eroare la apăsarea butonului Aplică: {e}")
            return False
    
    def _filter_institutional_names(self, author_list):
        """
        Filtrează autorii din listă, eliminând entitățile instituționale.
        Aceste entități sunt extrase ca autori individuali, dar nu sunt deputați.
        """
        # Entitățile instituționale cunoscute
        institutional_names = [
            "Biroul permanent", "Guvernul", "Comisia", "Parlamentul",
            "Președintele", "Ministerul", "Deputat", "Grupul", 
            "Fracțiunea", "Senat", "Guvern", "Comisie", "Procurorul General"
        ]
        
        filtered_authors = set()
        for author in author_list:
            clean_author = author.strip()
            if not clean_author:
                continue

            # Verificare pentru nume instituționale (case-insensitive)
            is_institutional = False
            for institutional_name in institutional_names:
                # Folosim regex pentru a verifica dacă începe cu un cuvânt cheie instituțional
                if re.match(r'\b' + re.escape(institutional_name) + r'\b', clean_author, re.IGNORECASE | re.UNICODE):
                    is_institutional = True
                    break
            
            # Verificarea finală: trebuie să fie cel puțin un nume de două cuvinte și să nu fie instituțional.
            if not is_institutional and len(clean_author.split()) >= 2:
                filtered_authors.add(clean_author)

        return sorted(list(filtered_authors))

    
    def extract_deputy_authors_from_table(self):
        """
        [CORIJAT] Extrage DOAR numele deputaților prin căutarea directă a elementelor <span>
        pentru proiectele cu 2+ autori
        """
        print("  📊 Extragere autori deputați din tabel...")
        
        projects = []
        
        try:
            time.sleep(5)
            
            rows = self.driver.find_elements(By.CSS_SELECTOR, "table tbody tr")
            print(f"  📋 Găsite {len(rows)} rânduri în tabel")
            
            for idx, row in enumerate(rows):
                try:
                    cells = row.find_elements(By.CSS_SELECTOR, "td")
                    
                    if len(cells) < 4: 
                        continue
                    
                    title = cells[0].text.strip()
                    author_cell = cells[3] 
                    
                    # NOU: Căutăm direct elementele <span> care conțin numele autorilor.
                    # Acestea sunt în interiorul div.authors și au clasa ng-scope/ng-binding.
                    # Folosim span.ng-scope care este mai specific pentru elementele generate de ng-repeat.
                    author_spans = author_cell.find_elements(By.CSS_SELECTOR, "div.authors span.ng-scope")
                    
                    # Extragem textul curat al fiecărui span
                    raw_authors = [span.text.strip() for span in author_spans if span.text.strip()]
                    
                    # Aplicăm filtrul pentru a elimina instituțiile (e.g. "Biroul permanent")
                    deputy_authors = self._filter_institutional_names(raw_authors)
                    
                    
                    if len(deputy_authors) >= 2:
                        project_data = {
                            'title': title,
                            'deputy_authors': deputy_authors,
                            'author_count': len(deputy_authors)
                        }
                        projects.append(project_data)
                        print(f"    ✅ Nume extrase {idx+1}: {len(deputy_authors)} autori - {', '.join(deputy_authors)}")
                    
                except Exception as e:
                    print(f"    ⚠ Eroare la rândul {idx+1}: {e.__class__.__name__}. Continuați...")
                    # traceback.print_exc() # Uncomment for deep debugging
                    continue
            
            print(f"  ✅ Extrase {len(projects)} seturi de nume (pentru 2+ autori)")
            return projects
            
        except Exception as e:
            print(f"  ❌ Eroare la extragerea datelor tabelului: {e}")
            return []
    
    def navigate_to_next_page(self):
        """Navighează la pagina următoare"""
        try:
            print("  🔄 Navigare la pagina următoare...")
            
            next_buttons = self.driver.find_elements(By.XPATH, "//a[contains(text(), 'Următoarea')]")
            
            for button in next_buttons:
                if button.is_displayed() and button.is_enabled():
                    if 'disabled' not in button.get_attribute('class'):
                        self.driver.execute_script("arguments[0].click();", button)
                        print("  ✅ Buton 'Următoarea' apăsat!")
                        
                        time.sleep(6)
                        self.wait_for_angular()
                        return True
            
            print("  ℹ Buton 'Următoarea' nu este disponibil - ultima pagină")
            return False
            
        except Exception as e:
            print(f"  ❌ Eroare la navigare: {e}")
            return False
    
    def get_all_deputy_projects(self, max_pages=None):
        """
        Extrage toate proiectele cu 2+ autori deputați din toate paginile.
        """
        print(f"\n📚 EXTRAGERE PROIECTE CU 2+ AUTORI DEPUTAȚI")
        print(f"📅 Legislatura: 26.07.2021-21.10.2025")
        print("👥 Filtru: Inițiatori = Deputat")
        print("🎯 Se extrag DOAR numele deputaților din coloana Autori din TOATE paginile\n")
        
        all_projects = []
        
        if not self.set_filters_and_apply():
            print("❌ Nu s-au putut seta filtrele")
            return all_projects
        
        page = 1
        has_more_pages = True
        
        while has_more_pages:
            print(f"{'='*70}")
            print(f"📄 PAGINA {page}")
            print(f"{'='*70}")
            
            page_projects = self.extract_deputy_authors_from_table()
            
            if page_projects:
                all_projects.extend(page_projects)
                print(f"  ✅ Adăugate {len(page_projects)} seturi de nume")
                print(f"  📊 Total acumulat: {len(all_projects)} seturi de nume")
            
            if max_pages and page >= max_pages:
                print(f"  ℹ Limită de {max_pages} pagini atinsă")
                break
            
            has_more_pages = self.navigate_to_next_page()
            page += 1
            
            if page > 500:
                print("  ⚠ Limită de siguranță atinsă (500 de pagini)")
                break
        
        print(f"\n{'='*70}")
        print(f"✅ EXTRAGERE COMPLETATĂ")
        print(f"{'='*70}")
        print(f"📊 Total seturi de nume cu 2+ autori deputați: {len(all_projects)}")
        print(f"📄 Pagini procesate: {page-1}")
        print(f"{'='*70}\n")
        
        return all_projects

    def close(self):
        """Închide browser-ul"""
        if self.driver:
            self.driver.quit()
            print("\n✓ Browser închis")


class NetworkAnalyzer:
    """
    Analizează rețeaua de cooperare între deputați
    (Cod neschimbat, funcționează cu datele corecte)
    """
    
    @staticmethod
    def build_cooperation_network(projects):
        """
        Construiește rețeaua de cooperare bazată pe co-autoriat
        """
        print("\n🔗 Construire rețea de cooperare...")
        
        G = nx.Graph()
        cooperation_count = defaultdict(int)
        
        for project in projects:
            authors = project['deputy_authors']
            
            for author in authors:
                G.add_node(author)
            
            for i in range(len(authors)):
                for j in range(i + 1, len(authors)):
                    author1, author2 = authors[i], authors[j]
                    pair = tuple(sorted([author1, author2]))
                    cooperation_count[pair] += 1
                    
                    if G.has_edge(author1, author2):
                        G[author1][author2]['weight'] += 1
                    else:
                        G.add_edge(author1, author2, weight=1)
        
        print(f"  👥 Noduri (deputați): {G.number_of_nodes()}")
        print(f"  🔗 Muchii (colaborări): {G.number_of_edges()}")
        
        return G, cooperation_count
    
    @staticmethod
    def analyze_network(G):
        """
        Analizează rețeaua de cooperare
        """
        print("\n📊 Analiză rețea...")
        
        analysis = {}
        
        if G.number_of_nodes() == 0:
            print("  ℹ Rețeaua este goală. Nu se poate efectua analiza.")
            return None

        # Calculăm centralitățile
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        
        top_by_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        analysis['top_degree'] = top_by_degree
        analysis['top_betweenness'] = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        analysis['degree_centrality'] = degree_centrality
        analysis['betweenness_centrality'] = betweenness_centrality
        
        analysis['density'] = nx.density(G)
        analysis['avg_clustering'] = nx.average_clustering(G)
        
        print("\n🏆 TOP 10 DEPUTAȚI (după centralitate):")
        for i, (deputy, centrality) in enumerate(top_by_degree, 1):
            degree = G.degree(deputy)
            print(f"  {i:2d}. {deputy:<30} {centrality:.3f} (grad: {degree})")
        
        print(f"\n📈 Statistici rețea:")
        print(f"  📏 Densitate: {analysis['density']:.4f}")
        print(f"  🔄 Clustering mediu: {analysis['avg_clustering']:.3f}")
        
        return analysis
    
    @staticmethod
    def export_results(projects, G, analysis, cooperation_count):
        """
        Exportă rezultatele
        """
        print("\n💾 Export rezultate...")
        
        df_projects = pd.DataFrame([
            {
                'Titlu': p['title'],
                'Numar_Autori': p['author_count'],
                'Autori_Deputati': ', '.join(p['deputy_authors'])
            }
            for p in projects
        ])
        df_projects.to_csv('proiecte_deputati_multiplicu.csv', index=False, encoding='utf-8-sig')
        print("✓ proiecte_deputati_multiplicu.csv")
        
        df_collab = pd.DataFrame([
            {
                'Deputat_1': pair[0],
                'Deputat_2': pair[1],
                'Numar_Colaborari': count
            }
            for pair, count in sorted(cooperation_count.items(), key=lambda x: x[1], reverse=True)
        ])
        df_collab.to_csv('colaborari_deputati.csv', index=False, encoding='utf-8-sig')
        print("✓ colaborari_deputati.csv")
        
        df_deputies = pd.DataFrame([
            {
                'Deputat': deputy,
                'Centralitate_Grad': analysis['degree_centrality'].get(deputy, 0),
                'Centralitate_Betweenness': analysis['betweenness_centrality'].get(deputy, 0),
                'Grad_Rețea': G.degree(deputy)
            }
            for deputy in analysis['degree_centrality'].keys()
        ])
        df_deputies.to_csv('top_deputati.csv', index=False, encoding='utf-8-sig')
        print("✓ top_deputati.csv")
        
        print("✅ Export complet!")


def main():
    print("=" * 70)
    print("ANALIZA COOPERĂRII ÎNTRE DEPUTAȚI")
    print("Parlamentul Republicii Moldova")
    print("Legislatura: 26.07.2021-21.10.2025")
    print("=" * 70)
    
    scraper = None
    
    try:
        scraper = ParliamentSeleniumScraper(headless=False)
        
        # Scoateti max_pages pentru extragerea completă
        projects = scraper.get_all_deputy_projects() 
        
        if not projects:
            print("❌ Nu s-au extras proiecte")
            return
        
        with open('date_brute.json', 'w', encoding='utf-8') as f:
            json.dump(projects, f, ensure_ascii=False, indent=2)
        print("✓ Date brute salvate: date_brute.json")
        
        G, cooperation_count = NetworkAnalyzer.build_cooperation_network(projects)
        analysis = NetworkAnalyzer.analyze_network(G)
        
        if analysis:
            NetworkAnalyzer.export_results(projects, G, analysis, cooperation_count)
        else:
            print("❌ Nu s-a putut efectua analiza din cauza lipsei de date valide.")
        
        print("\n🎉 ANALIZA COMPLETĂ!")
        
    except Exception as e:
        print(f"\n❌ EROARE: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if scraper:
            scraper.close()


if __name__ == "__main__":
    main()