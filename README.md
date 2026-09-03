# polyN_pipeline — criblage d'allotropes polyazotés Nₙ

Ensemble d'outils pour générer, optimiser, valider et classer des structures
moléculaires polyazotées (neutres, cations, anions), depuis la génération de
topologies jusqu'aux enveloppes convexes de stabilité, avec vérification
vibrationnelle des minima. Conçu aussi pour constituer un jeu de données
propre en vue d'une méthode générative.

Auteur : Gilles Frapper (IC2MP, UMR 7285 CNRS, Université de Poitiers).

---

## Installation

```bash
conda create -n polyN python=3.11 -y
conda activate polyN
conda install -c conda-forge tblite-python rdkit ase pyyaml pandas scipy matplotlib -y
pip install pubchempy          # uniquement pour harvest_hydrocarbons.py
# nauty (geng) : uniquement pour le generateur geng
#   brew install nauty        # macOS
#   sudo apt install nauty    # Debian/Ubuntu
```

Vérification :
```bash
python3 -c "from tblite.ase import TBLite; from rdkit import Chem; import ase, yaml, pandas; print('OK')"
```

> Note macOS : si `which python3` pointe vers Homebrew au lieu de conda, préfixer
> les commandes par `~/miniconda3/bin/python3`, ou corriger l'ordre du PATH.

---

## Le pipeline principal

### `polyN_pipeline.py`
Orchestrateur en 6 étapes :
1. lecture des topologies (fichiers `.smi`/`.sdf`, un par stœchiométrie) ;
2. génération 3D (RDKit ETKDGv3) + optimisation GFN2-xTB en deux temps
   (checkpoint bon marché -> affinement précis sur liste courte) ;
3. sélection des structures les plus basses par stœchiométrie
   (avec mise à l'échelle 3N-6 de `top_n`) ;
4. **vérification de fréquences** : confirme que chaque structure retenue est
   un vrai minimum (toutes fréquences réelles), et suit automatiquement les
   modes imaginaires vers le vrai minimum le cas échéant ;
5. références par famille via réactions équilibrées (neutres : Nₓ → (x/2)N₂ ;
   chargés : Nₓ± → ((x−5)/2)N₂ + N₅±) + **enveloppes convexes séparées par
   famille de charge** (neutre / cation / anion, jamais mélangées).

Déduplication automatique (après optimisation et après fréquences) : les
structures de même formule+charge dont les énergies coïncident à
`dedup_tol_kcalmol` près sont fusionnées, utile quand plusieurs générateurs
produisent la même topologie.

```bash
python3 polyN_pipeline.py --config config_example.yaml [--skip-freq]
```

Sorties dans `output_dir/` : `results.csv` (toutes les structures retenues,
classées par composition, avec énergies, famille, `rank`, énergie de réaction,
statut de minimum, chemin `.xyz`) ; `summary_after_frequencies.csv` (résumé
compact des espèces vérifiées) ; `best_structures/` (toutes les structures
vérifiées dans un répertoire plat, renommées `<formule>_<rang>.xyz`, ex.
`N4_001.xyz`, `N4_002.xyz`, pour consultation rapide) ; `convex_hull_{neutral,
cation,anion}.{csv,png}` ; et `work/` (intermédiaires).

---

## Générateurs de topologies (sources complémentaires)

### `random_structure_generator.py`
Échantillonnage aléatoire de graphes moléculaires valides (modèle de
configuration / stub-matching), pour toute taille — contourne le mur
combinatoire de l'énumération exhaustive à grand N.
```bash
# valence uniforme (neutre)
python3 random_structure_generator.py -n 16 -k 2000 --valence 3 -o N16.smi
# séquence de valence mixte (motif à charge localisée)
python3 random_structure_generator.py -n 5 -k 100 --valence-pattern "4,3,3,3,3" -o N5_cation.smi
# charge nette cible, environnements de coordination aléatoires
python3 random_structure_generator.py -n 9 -k 100 --target-charge -1 -o N9_anion.smi
```

### `cxhx_to_nx.py`
Substitution isolobale C->N sur des hydrocarbures CxHy (chaque C->N, la charge
formelle émerge du motif de liaison). Bibliothèque intégrée de polyédranes +
mode fouille de masse. Contrôle de pureté final : rejette tout résultat
contenant un H résiduel.
```bash
python3 cxhx_to_nx.py --all -o maygen_output/seeds/                    # bibliothèque intégrée
python3 cxhx_to_nx.py --smiles-file hydrocarbons.smi --max-abs-charge 1 -o maygen_output/seeds/
```

### `harvest_hydrocarbons.py`
Fouille PubChem des hydrocarbures CxHy (avec timeout réseau et reprise), pour
alimenter `cxhx_to_nx.py`.
```bash
python3 harvest_hydrocarbons.py --min-c 4 --max-c 16 -o hydrocarbons.smi [--resume]
```

### `geng_enumerate.py`
Énumération **exhaustive** des topologies via `geng` (nauty) + attribution des
ordres de liaison. Les charges formelles (q = b − 3, où b = somme des ordres de
liaison de l'azote) sont posées **au stade graphe, avant la sanitisation
RDKit**, ce qui empêche toute saturation automatique par hydrogène : les
familles neutre/cation/anion émergent du même jeu de graphes, sans
contamination. Nécessite l'exécutable `geng` (paquet nauty).
```bash
python3 geng_enumerate.py -n 10 --max-abs-charge 1 -o seeds/
python3 geng_enumerate.py -n 6 --max-graphs 500 --max-abs-charge 1 -o seeds/
```

---

## Vérification et diagnostic

### `frequency_check.py`
Vérifie qu'une structure optimisée est un vrai minimum (calcul de Hessienne),
et suit automatiquement les modes imaginaires vers le vrai minimum. Utilisé en
interne par le pipeline (étape 5), aussi utilisable en autonome.
```bash
python3 frequency_check.py --xyz structure.xyz --charge 0 --output-dir ./verif
```

### `check_xyz_purity.py`
Signale tout fichier `.xyz` contenant un élément autre que celui attendu
(détection de contamination, ex. hydrogène parasite).
```bash
python3 check_xyz_purity.py --results-csv resultats/.../results.csv
python3 check_xyz_purity.py --dir ./resultats --element N
```

### `benchmark_frequencies.py`
Mesure les temps d'optimisation et de calcul de fréquences selon la taille, pour
dimensionner les runs.
```bash
python3 benchmark_frequencies.py --sizes 4,6,8,10,12,14,16 -o benchmark.csv
```

### `test_charge_propagation.py`
Test de non-régression sur la propagation des charges et la pureté (aucun H
parasite), à lancer après toute modification d'un générateur ou du chemin
d'optimisation tblite. Vérifie trois choses : (1) la charge nette du SMILES
correspond à la famille annoncée par le nom de fichier, (2) **tblite applique
bien la charge** — N₅⁺ et N₅⁻ doivent donner des énergies différentes (le bug
corrigé où `TBLite(charge=...)` était ignoré), (3) aucune structure ne contient
d'hydrogène. Code de sortie non nul si un test échoue (utilisable en CI).
```bash
python3 test_charge_propagation.py                        # jeu de contrôle généré en direct
python3 test_charge_propagation.py --seeds-dir ./seeds    # sur des graines existantes
```

### `visualize_results.py`
Rend les structures trouvées en une grille 2D annotée (formule, symbole de
charge, rang de stabilité, énergie de réaction). Utilise la colonne `smiles`
du `results.csv` si elle existe (dessins nets), sinon perçoit la connectivité
depuis chaque `.xyz` (approximatif pour les polyazotés — ordres de liaison et
charges imparfaits, mais suffisant pour reconnaître une topologie). Filtres :
par famille, par rang (ex. seulement le plus stable de chaque composition),
minima vérifiés uniquement.
```bash
python3 visualize_results.py --results resultats/seeds_pubchem/results.csv
python3 visualize_results.py --results .../results.csv --family anion
python3 visualize_results.py --results .../results.csv --per-composition 1   # top-1 par composition
python3 visualize_results.py --results .../results.csv --pdf                 # PDF multi-pages
```

### `filter_fragmented.py`
Retire les structures **fragmentées** d'un `results.csv` existant, sans tout
relancer. Une structure est fragmentée si ses atomes se séparent en plusieurs
composantes connexes (cas typique : un N₂ détaché en interaction de van der
Waals — un « N₈ » qui est en réalité N₆ + N₂). Deux atomes sont liés si leur
distance est sous `--threshold` (défaut 2.0 Å : une liaison N–N réelle fait
≤ ~1.6 Å, un contact van der Waals ~3 Å). Écrit un CSV nettoyé, ré-classe les
rangs, et peut reconstruire un `best_structures_clean/`. Le pipeline applique
déjà ce filtre automatiquement (`reject_fragmented: true`) ; ce script sert à
nettoyer un run antérieur.
```bash
python3 filter_fragmented.py --results resultats/seeds_pubchem/results.csv
python3 filter_fragmented.py --results .../results.csv --rebuild-best-dir
python3 filter_fragmented.py --results .../results.csv --in-place   # remplace (backup .bak)
```

### `generate_report.py`
Produit un **rapport LaTeX + PDF** des structures retenues, organisé par
famille (neutre → anion → cation) puis par cluster Nₓ, avec tous les isomères
(ground-state + métastables). Pour chaque isomère : le nom du fichier `.xyz`
**propre lisible par VESTA** (écrit dans `xyz_clean/`), l'**énergie relative**
au ground-state de la série (GS = 0), l'**énergie de réaction** (réaction de
référence par famille), et un **dessin 2D**. Génère `report_<Nom>.tex` +
`report_<Nom>.pdf`. Le ground-state de chaque série est surligné.
```bash
python3 generate_report.py --results resultats/seeds_pubchem/results_clean.csv --author "Gilles Frapper"
python3 generate_report.py --results .../results_clean.csv --author Frapper --no-compile   # .tex seul
```

---

## Configurations

- `config_example.yaml` — modèle documenté, toutes les options commentées.
- `config_seeds.yaml` — dédiée au traitement du jeu de graines PubChem
  (`top_n` large, pensé pour constituer un jeu de données plutôt que pour
  isoler les seuls meilleurs).

---

## Méthodologie

Voir `polyN_pipeline_manual_en.pdf` (et sa source `.tex`) pour l'exposé complet :
séparation stricte des familles de charge, stratégie d'optimisation à deux
étages, contrainte topologique du nombre de cycles, substitution isolobale,
et les découvertes empiriques (dont le N4 tétraédrique point-selle).

---

## Flux de travail type

```bash
# 1. Générer/collecter des topologies (une ou plusieurs sources)
python3 harvest_hydrocarbons.py --min-c 4 --max-c 16 -o hydrocarbons.smi
python3 cxhx_to_nx.py --smiles-file hydrocarbons.smi --max-abs-charge 1 -o maygen_output/seeds/

# 2. (optionnel) vérifier la pureté
python3 check_xyz_purity.py --dir maygen_output/seeds --element N

# 3. Lancer le pipeline complet (avec anti-veille sur macOS)
caffeinate -i nohup python3 polyN_pipeline.py --config config_seeds.yaml \
    > run.log 2>&1 &
tail -f run.log

# 4. Vérifier la pureté des géométries produites
python3 check_xyz_purity.py --results-csv resultats/seeds_pubchem/results.csv
