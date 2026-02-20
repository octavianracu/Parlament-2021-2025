"""
SCRIPT DE DIAGNOSTIC ȘI PREGĂTIRE DATE
========================================
Rulați PRIMUL, înainte de exclusion.py

Ce face:
  1. Afișează toți deputații din leg1_raw.json (graful real)
  2. Compară cu excluded_deputies.csv — găsește nepotriviri de nume
  3. Generează automat: all_deputies_leg1.csv  (toți 60, Exclus=0 implicit)
     → Dumneavoastră marcați manual cu 1 cei excluși, apoi salvați ca excluded_deputies.csv
  4. Oferă și o variantă de fuzzy matching pentru a detecta diferențe de diacritice
"""

import json
import csv
import os
from collections import defaultdict
from itertools import combinations

# ── Configurare ───────────────────────────────────────────────────────────────
LEG1_JSON        = "leg1_raw.json"
EXCLUDED_CSV     = "excluded_deputies.csv"
OUTPUT_TEMPLATE  = "all_deputies_leg1.csv"   # fișier de completat manual
MATCH_REPORT     = "diagnostic_report.txt"

# ── Fuzzy matching simplu (fără dependențe externe) ───────────────────────────
def normalize(name):
    """Elimină diacritice și convertește la lowercase pentru comparație."""
    replacements = {
        'ă': 'a', 'â': 'a', 'î': 'i', 'ș': 's', 'ț': 't', 'ş': 's', 'ţ': 't',
        'Ă': 'A', 'Â': 'A', 'Î': 'I', 'Ș': 'S', 'Ț': 'T', 'Ş': 'S', 'Ţ': 'T',
    }
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    return result.strip().lower()


def levenshtein(s1, s2):
    """Distanța Levenshtein pentru detectarea typo-urilor."""
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (0 if c1 == c2 else 1)))
        prev = curr
    return prev[len(s2)]


# ── 1. Încărcare date ─────────────────────────────────────────────────────────
print("=" * 65)
print("DIAGNOSTIC DATE — PARLAMENTUL R. MOLDOVA")
print("=" * 65)

if not os.path.exists(LEG1_JSON):
    print(f"\n❌ EROARE: {LEG1_JSON} nu există în directorul curent!")
    print(f"   Director curent: {os.getcwd()}")
    exit(1)

with open(LEG1_JSON, "r", encoding="utf-8") as f:
    projects = json.load(f)

print(f"\n📂 {LEG1_JSON}: {len(projects)} proiecte cu 2+ autori\n")

# Extrage toți deputații unici + statistici
deputy_projects = defaultdict(int)    # câte proiecte a semnat fiecare
deputy_coauthors = defaultdict(set)   # cu cine a co-semnat

for p in projects:
    authors = p["deputy_authors"]
    for a in authors:
        deputy_projects[a] += 1
    for a1, a2 in combinations(authors, 2):
        deputy_coauthors[a1].add(a2)
        deputy_coauthors[a2].add(a1)

all_deputies_json = sorted(deputy_projects.keys())
print(f"👥 Deputați unici în graf: {len(all_deputies_json)}")
print(f"   (aceștia sunt cei care apar în cel puțin un proiect co-semnat)\n")

# ── 2. Afișare toți deputații din graf ───────────────────────────────────────
print("─" * 65)
print(f"{'#':<4} {'Nume Deputat':<35} {'Proiecte':>8} {'Co-autori':>9}")
print("─" * 65)
for i, dep in enumerate(all_deputies_json, 1):
    print(f"{i:<4} {dep:<35} {deputy_projects[dep]:>8} {len(deputy_coauthors[dep]):>9}")
print("─" * 65)

# ── 3. Analiză CSV exclus ─────────────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"ANALIZĂ: {EXCLUDED_CSV}")
print(f"{'=' * 65}")

if not os.path.exists(EXCLUDED_CSV):
    print(f"\n⚠  {EXCLUDED_CSV} nu există — se va genera template-ul.")
    csv_deputies = []
    csv_exclus_map = {}
else:
    with open(EXCLUDED_CSV, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    csv_deputies = [r["Nume"].strip() for r in rows]
    csv_exclus_map = {r["Nume"].strip(): int(r.get("Exclus", 0)) for r in rows}
    csv_partid_map = {r["Nume"].strip(): r.get("Partid", "?") for r in rows}

    n_excl = sum(1 for v in csv_exclus_map.values() if v == 1)
    n_incl = sum(1 for v in csv_exclus_map.values() if v == 0)

    print(f"\n   Rânduri în CSV:  {len(rows)}")
    print(f"   Excluși (1):     {n_excl}")
    print(f"   Incluși (0):     {n_incl}")

    if n_incl == 0:
        print("\n   ⚠  PROBLEMĂ CRITICĂ: Nu există deputați cu Exclus=0!")
        print("      Modelul nu poate învăța fără contraexemple (clasa 0).")
        print("      → Adăugați toți deputații incluși în CSV cu Exclus=0")

    # ── 4. Comparație CSV vs. JSON ───────────────────────────────────────────
    print(f"\n{'─' * 65}")
    print("COMPARAȚIE: Deputați CSV ↔ Deputați din Graf (JSON)")
    print(f"{'─' * 65}")

    json_norm  = {normalize(d): d for d in all_deputies_json}
    csv_norm   = {normalize(d): d for d in csv_deputies}

    # Găsiți în ambele
    matched_exact  = []
    matched_fuzzy  = []
    only_in_csv    = []
    only_in_json   = []

    for csv_name in csv_deputies:
        cn = normalize(csv_name)
        if cn in json_norm:
            matched_exact.append((csv_name, json_norm[cn]))
        else:
            # Fuzzy: caută cel mai apropiat
            best_match = None
            best_dist  = 999
            for jn, jname in json_norm.items():
                dist = levenshtein(cn, jn)
                if dist < best_dist:
                    best_dist = dist
                    best_match = jname
            if best_dist <= 3:
                matched_fuzzy.append((csv_name, best_match, best_dist))
            else:
                only_in_csv.append(csv_name)

    only_in_json = [d for d in all_deputies_json
                    if normalize(d) not in csv_norm and
                    not any(levenshtein(normalize(d), normalize(c)) <= 3
                            for c in csv_deputies)]

    print(f"\n✅ Potriviri exacte (normalizat):  {len(matched_exact)}")
    print(f"⚠  Potriviri fuzzy (diacritice?): {len(matched_fuzzy)}")
    print(f"❌ Doar în CSV (nu în graf):       {len(only_in_csv)}")
    print(f"❓ Doar în Graf (nu în CSV):       {len(only_in_json)}")

    if matched_fuzzy:
        print(f"\n{'─' * 65}")
        print("⚠  POTRIVIRI FUZZY — verificați diacriticele:")
        print(f"   {'CSV (exclus.csv)':<35} {'JSON (graf)':<35} {'Dist':>4}")
        print(f"   {'─'*35} {'─'*35} {'─'*4}")
        for csv_n, json_n, dist in matched_fuzzy:
            status = "✅ EXCLUS" if csv_exclus_map.get(csv_n, 0) == 1 else "  inclus"
            print(f"   {csv_n:<35} {json_n:<35} {dist:>4}   {status}")

    if only_in_csv:
        print(f"\n{'─' * 65}")
        print("❌ NUME DIN CSV CARE NU APAR ÎN GRAF:")
        print("   (acești deputați vor avea toate metricile = 0)")
        for n in only_in_csv:
            status = "EXCLUS" if csv_exclus_map.get(n, 0) == 1 else "inclus"
            print(f"   [{status}] {n}")

    if only_in_json:
        print(f"\n{'─' * 65}")
        print("❓ DEPUTAȚI DIN GRAF CARE LIPSESC DIN CSV:")
        print("   (aceștia nu vor fi incluși în analiză)")
        for n in only_in_json:
            print(f"   {n}  ({deputy_projects[n]} proiecte, {len(deputy_coauthors[n])} co-autori)")

# ── 5. Generare template complet ─────────────────────────────────────────────
print(f"\n{'=' * 65}")
print(f"GENERARE TEMPLATE: {OUTPUT_TEMPLATE}")
print(f"{'=' * 65}")

# Partid placeholder (se va completa manual sau dintr-o sursă externă)
# Dacă CSV-ul există, preluăm partidul din acolo unde există potrivire
partid_known = {}
if os.path.exists(EXCLUDED_CSV) and 'csv_partid_map' in dir():
    for csv_n, partid in csv_partid_map.items():
        cn = normalize(csv_n)
        for jn, jname in {normalize(d): d for d in all_deputies_json}.items():
            if levenshtein(cn, jn) <= 3:
                partid_known[jname] = partid

exclus_known = {}
if os.path.exists(EXCLUDED_CSV) and 'csv_exclus_map' in dir():
    for csv_n, excl in csv_exclus_map.items():
        cn = normalize(csv_n)
        for jn, jname in {normalize(d): d for d in all_deputies_json}.items():
            if levenshtein(cn, jn) <= 3:
                exclus_known[jname] = excl

with open(OUTPUT_TEMPLATE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["Nume", "Exclus", "Partid", "Proiecte_cosemnate", "Nr_coautori"])
    for dep in all_deputies_json:
        excl_val   = exclus_known.get(dep, 0)   # 0 = inclus implicit
        partid_val = partid_known.get(dep, "?")  # ? = necunoscut
        writer.writerow([
            dep,
            excl_val,
            partid_val,
            deputy_projects[dep],
            len(deputy_coauthors[dep]),
        ])

print(f"\n✅ Generat: {OUTPUT_TEMPLATE}")
print(f"   {len(all_deputies_json)} deputați (toți din graf), Exclus=0 implicit")
print(f"\n📋 INSTRUCȚIUNI:")
print(f"   1. Deschideți {OUTPUT_TEMPLATE} în Excel")
print(f"   2. Setați Exclus=1 pentru deputații excluși de pe lista electorală")
print(f"   3. Completați coloana Partid acolo unde apare '?'")
print(f"   4. Salvați ca: excluded_deputies.csv")
print(f"   5. Rulați din nou exclusion.py")
print(f"\n⚠  ATENȚIE: Dacă un deputat exclus NU apare în template")
print(f"   (nu a co-semnat niciun proiect), adăugați-l manual cu rândul:")
print(f"   Nume,1,Partid,0,0")
print(f"   Dar rețineți: metricile sale de rețea vor fi toate 0 (izolat structural).")

# ── 6. Raport text ────────────────────────────────────────────────────────────
with open(MATCH_REPORT, "w", encoding="utf-8") as f:
    f.write("RAPORT DIAGNOSTIC DATE\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"Proiecte în leg1_raw.json: {len(projects)}\n")
    f.write(f"Deputați unici în graf:    {len(all_deputies_json)}\n\n")
    f.write("TOȚI DEPUTAȚII DIN GRAF:\n")
    f.write(f"{'Nume':<35} {'Proiecte':>8} {'Co-autori':>9}\n")
    f.write("─" * 55 + "\n")
    for dep in all_deputies_json:
        f.write(f"{dep:<35} {deputy_projects[dep]:>8} {len(deputy_coauthors[dep]):>9}\n")

print(f"\n✅ Raport salvat: {MATCH_REPORT}")
print(f"\n{'=' * 65}")
print("DIAGNOSTIC COMPLET")
print(f"{'=' * 65}")