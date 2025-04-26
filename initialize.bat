@echo off
set PROJECT_NAME=complaint_prioritizer

echo Creating project folders...

mkdir %PROJECT_NAME%
cd %PROJECT_NAME%

mkdir data
mkdir data\raw
mkdir data\processed
mkdir notebooks
mkdir src
mkdir src\config
mkdir src\data_loader
mkdir src\preprocessing
mkdir src\utils
mkdir tests

echo Creating empty files...

type nul > README.md
type nul > requirements.txt
type nul > .gitignore
type nul > main.py

type nul > src\__init__.py
type nul > src\config\config.py
type nul > src\data_loader\data_loader.py
type nul > src\preprocessing\__init__.py
type nul > src\preprocessing\cleaner.py
type nul > src\preprocessing\tokenizer.py
type nul > src\preprocessing\vectorizer.py
type nul > src\utils\utils.py
type nul > src\pipeline.py
type nul > tests\test_preprocessing.py
type nul > notebooks\exploration.ipynb

echo Project structure created successfully!
pause
