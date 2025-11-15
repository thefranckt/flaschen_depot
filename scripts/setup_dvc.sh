# Add data to DVC tracking
dvc add data/raw/bottles.csv
dvc add data/processed/

# Add models to DVC tracking
dvc add models/model.pkl
dvc add models/preprocessor.pkl

# Commit DVC files
git add data/raw/bottles.csv.dvc data/processed.dvc models/model.pkl.dvc models/preprocessor.pkl.dvc .gitignore
git commit -m "Add data and models to DVC"

# Push to DVC remote
dvc push
