# Plots Directory

Ce dossier contient toutes les visualisations générées par l'analyse exploratoire des données (EDA).

## Visualisations générées

Les plots suivants sont automatiquement générés lors de l'exécution du notebook `notebooks/01_exploratory_data_analysis.ipynb`:

### 1. Distribution du Service Time
**Fichier:** `eda_service_time_distribution.png`
- Histogramme de la distribution
- Boxplot
- Distribution log-transformée
- Fonction de distribution cumulative

### 2. Analyse Temporelle et Catégorielle
**Fichier:** `eda_temporal_categorical.png`
- Service time par heure de la journée
- Service time par jour de la semaine
- Comparaison Aufzug vs. Kein Aufzug
- Comparaison Privatkunde vs. Geschäftskunde

### 3. Analyse des Étages
**Fichier:** `eda_floor_analysis.png`
- Service time en fonction de l'étage
- Effet de l'ascenseur par étage

### 4. Analyse du Poids
**Fichier:** `eda_weight_analysis.png`
- Service time vs. poids total
- Service time vs. nombre de boîtes
- Service time vs. nombre d'articles
- Service time par classe de poids

### 5. Matrice de Corrélation
**Fichier:** `eda_correlation_matrix.png`
- Heatmap de corrélation entre toutes les features

### 6. Top Corrélations
**Fichier:** `eda_top_correlations.png`
- Bar chart des 10 features les plus corrélées avec le service time

### 7. Analyse des Outliers
**Fichier:** `eda_outliers.png`
- Distribution avec outliers (méthode IQR)
- Distribution sans outliers

## Régénérer les plots

Pour régénérer tous les plots, exécutez simplement le notebook EDA:

```bash
# Activer l'environnement virtuel
source myenv/Scripts/activate

# Lancer Jupyter
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

Ou exécutez toutes les cellules du notebook depuis VS Code.

## Format

- **Format:** PNG
- **Résolution:** 300 DPI
- **Optimisation:** `bbox_inches='tight'` pour éviter les marges excessives

## Note

Les fichiers PNG générés dans ce dossier sont ignorés par Git (voir `.gitignore`). Seul ce README et le fichier `.gitkeep` sont versionnés pour maintenir la structure du projet.
