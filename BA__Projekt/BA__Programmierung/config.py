# BA__Projekt/BA__Programmierung/config.py

from pathlib import Path


# FILEPATHS
BASE_DIR = Path(__file__).resolve().parent.parent

BUILD_DIR = BASE_DIR / "build"

DATA_DIR = BASE_DIR / "data"
DATA_DIR__SOURCE = BASE_DIR / "data" / "source"
DATA_DIR__RAW = BASE_DIR / "data" / "raw"
DATA_DIR__PROCESSED = BASE_DIR / "data" / "processed"

DB_PATH = DATA_DIR / "database/analytics.duckdb"

MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR = BASE_DIR / "output"

PROJECT_DIR = BASE_DIR / "BA__Programmierung"

RAW_DATA_PATH = DATA_DIR / "raw"

SCRIPTS_PATH = DATA_DIR / "scripts"

TESTS_PATH = DATA_DIR / "tests"