# polyN_pipeline

Pipeline : isomères MAYGEN → optimisation GFN2-xTB → sélection des structures
les plus basses en énergie → réoptimisation optionnelle DFTB+ → enveloppe
convexe de stabilité par rapport à une référence (N2 par défaut).

## Installation

```bash
pip install rdkit ase pyyaml pandas scipy matplotlib tblite
```

Binaire requis dans le `$PATH` :
- `dftb+` — uniquement si `dftb.enabled: true` dans la config

Sur Mac (conda, recommandé) :
```bash
conda install -c conda-forge tblite-python dftbplus
```

**Pourquoi `tblite` plutôt que le binaire `xtb` en ligne de commande ?**
Le CLI `xtb` (utilisé dans une version antérieure de ce script) contient un
bug connu dans sa routine d'affichage de progression (`optimizer.f90`,
chaîne de format Fortran mal écrite) qui plante avec l'erreur `Fortran
runtime error: Missing comma between descriptors` sur les runtimes gfortran
récents — y compris les builds `xtb` distribués via conda-forge et
Homebrew au moment de l'écriture. Le correctif existe en amont
(grimme-lab/xtb PR #1278) mais n'est pas encore présent dans les paquets
distribués couramment. `tblite` est une bibliothèque Python/C séparée,
développée par la même équipe, qui implémente les mêmes hamiltoniens
GFN1-/GFN2-xTB sans passer par ce code défaillant ; le pipeline pilote
l'optimisation de géométrie via l'optimiseur `LBFGS` d'ASE.

## Utilisation

1. Génère tes isomères avec MAYGEN pour chaque stœchiométrie, en sortie SMILES :
   ```bash
   java -jar MAYGEN-1.8.jar -f N4 -smi -o ./maygen_output/
   java -jar MAYGEN-1.8.jar -f N6 -smi -o ./maygen_output/
   # etc.
   ```
   Renomme si besoin les fichiers produits en `<formule>.smi` (ou indique le
   nom exact via la clé `file:` dans la config).

2. Adapte `config_example.yaml` (stœchiométries, charges, sélection, chemin
   des Slater-Koster pour DFTB+, référence pour l'enveloppe convexe).

3. Lance le pipeline :
   ```bash
   python polyN_pipeline.py --config config_example.yaml
   ```
   Ajoute `--skip-dftb` pour forcer l'arrêt après l'étape xtb, même si
   `dftb.enabled: true` dans la config.

## Étapes du pipeline

1. **Lecture MAYGEN** : un fichier SMILES (ou SDF) par stœchiométrie.
2. **Génération 3D** (RDKit, ETKDGv3 + pré-optimisation MMFF si les
   paramètres existent — sinon xtb corrige la géométrie de toute façon).
   Plusieurs embeddings par isomère sont possibles (`n_conformers`), utile
   pour les clusters d'atomes lourds sans hétéroatomes où la géométrie
   initiale RDKit est parfois pauvre.
3. **Optimisation GFN2-xTB**, en parallèle (`xtb.n_jobs`), avec charge et
   nombre d'électrons non appariés (`uhf`) configurables par stœchiométrie.

   Un **checkpoint bon marché** (activé par défaut) passe d'abord tous les
   candidats embarqués par une optimisation **au même niveau GFN2-xTB**,
   mais plafonnée à très peu de pas (`xtb.prefilter.max_steps`, 30 par
   défaut) et avec un seuil de convergence relâché — utilisée uniquement
   comme **sonde de classement**, pas comme point de départ « préchauffé »
   pour la suite. Seule la liste courte qui en résulte
   (`xtb.prefilter.keep_top_n`, avec la même mise à l'échelle 3N-6 que la
   sélection finale) est ensuite optimisée précisément, **en repartant de
   la géométrie RDKit brute d'origine**, pas de la géométrie partiellement
   relaxée du checkpoint.

   Ce dernier point vient de mesures empiriques, pas d'une intuition : on a
   testé utiliser GFN1-xTB comme sonde bon marché (hamiltonien plus simple),
   mais il n'est pas plus rapide que GFN2-xTB par appel dans `tblite`. On a
   aussi testé repartir de la géométrie du checkpoint pour l'affinement
   (« warm start ») en espérant économiser des pas — mais l'optimiseur
   LBFGS d'ASE perd l'historique de courbure utile quand on crée un nouvel
   objet optimiseur à partir d'une géométrie déjà partiellement relaxée, et
   ça coûte in fine *plus* de pas qu'un redémarrage à froid. Le vrai gain
   (environ ×4 sur des lots de plusieurs milliers d'isomères, mesuré) vient
   uniquement du fait que la grande majorité des candidats ne sont jamais
   poussés jusqu'à convergence complète — pas d'un hamiltonien plus rapide
   ni d'un redémarrage plus malin. Désactivable via
   `xtb.prefilter.enabled: false` si tu préfères tout passer directement en
   GFN2-xTB précis.
4. **Sélection** : pour chaque stœchiométrie, conserve soit les *N*
   structures les plus basses (`top_n`), soit toutes celles dans une fenêtre
   d'énergie donnée en kcal/mol au-dessus du minimum (`energy_window`). Le
   nombre de degrés de liberté d'un cluster croît en 3N-6 avec le nombre
   d'atomes N ; un `top_n` fixe sous-échantillonne donc les grandes tailles.
   Activer `selection.top_n_scaling.enabled: true` fait croître le nombre de
   structures conservées proportionnellement à la taille du cluster
   (`max(minimum, round(top_n * n_atoms / ref_n_atoms))`).
5. **Réoptimisation DFTB+ (optionnelle)** sur le sous-ensemble sélectionné,
   via le calculateur ASE `Dftb`. Les structures sont traitées comme des
   molécules isolées (`pbc=False`) : **aucun bloc `KPointsAndWeights` n'est
   jamais généré ici**, ce qui contourne le problème connu de DFTB+ 2025 —
   celui-ci ne concerne que les calculs périodiques / supercellules.
6. **Référence & enveloppe(s) convexe(s)** : la référence (N2 par défaut) est
   optimisée avec la même méthode, puis sert à calculer l'énergie de
   formation de chaque structure.

   **Les familles de charge ne sont jamais mélangées.** Les systèmes sont
   automatiquement répartis en trois familles — `neutral` (charge 0),
   `cation` (charge > 0), `anion` (charge < 0) — et une enveloppe convexe
   distincte est construite pour chacune (`convex_hull_neutral.*`,
   `convex_hull_cation.*`, `convex_hull_anion.*`). C'est indispensable pour
   deux raisons : (1) une voie de décomposition réelle conserve la charge
   totale — un cation ne peut se fragmenter qu'en cation + neutres, un
   anion qu'en anion + neutres, jamais l'un vers l'autre ; (2) les méthodes
   semi-empiriques (GFN2-xTB, DFTB) ne garantissent pas une échelle
   d'énergie absolue cohérente entre différents états de charge — les
   potentiels d'ionisation et affinités électroniques sont un point faible
   connu de ces méthodes.

   Dans chaque famille, l'enveloppe utilise l'**énergie de formation par
   atome** (axe X = nombre d'atomes, axe Y = énergie de formation par
   atome). Une stœchiométrie est stable si aucune combinaison d'autres
   tailles de la *même famille* n'offre une énergie par atome plus basse.

## Sorties

Dans `output_dir` :
- `results.csv` — toutes les structures sélectionnées, énergies xtb/DFTB+,
  famille de charge (`family`), énergie de formation, chemin vers le
  fichier `.xyz` optimisé.
- `convex_hull_neutral.csv`, `convex_hull_cation.csv`, `convex_hull_anion.csv`
  — une ligne par stœchiométrie de la famille correspondante (structure la
  plus basse), avec le statut `on_hull` et l'énergie au-dessus de
  l'enveloppe (`e_above_hull_ev_per_atom`).
- `convex_hull_neutral.png`, `convex_hull_cation.png`, `convex_hull_anion.png`
  — un graphique par famille.
- `work/` — tous les fichiers intermédiaires (géométries initiales, logs
  xtb, logs DFTB+), utile pour déboguer une structure qui ne converge pas.

## Limitations connues

- L'embedding RDKit part d'une simple structure de graphe (SMILES) ; pour
  des clusters d'azote pur sans hétéroatome, la géométrie 3D initiale peut
  être de mauvaise qualité. Augmenter `n_conformers` améliore les chances
  de retomber sur le bon minimum après optimisation xtb.
- L'enveloppe convexe suppose une seule structure "représentative" par
  stœchiométrie (la plus basse en énergie) ; l'ensemble complet des
  structures triées reste disponible dans `results.csv`.
- Les mots-clés DFTB+ passés au calculateur ASE peuvent nécessiter un
  ajustement selon ta version exacte de DFTB+ (`Hamiltonian_...`,
  `Driver_...`) — vérifie le fichier `dftb_in.hsd` généré dans
  `work/dftb/<tag>/` au premier essai.
