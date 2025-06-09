# BA__Projekt/BA__Programmierung/config.py

from pathlib import Path


# FILEPATHS
BASE_DIR = Path(__file__).resolve().parent.parent

ASSET_DIR = BASE_DIR / "assets"

BUILD_DIR = BASE_DIR / "build"

DATA_DIR = ASSET_DIR / "data"
DATA_DIR__SOURCE = ASSET_DIR / "data" / "source"
DATA_DIR__RAW = ASSET_DIR / "data" / "raw"
DATA_DIR__PROCESSED = ASSET_DIR / "data" / "processed"

DB_PATH = ASSET_DIR / "dbs"

DB_PATH__EDNN_REGRESSION__IRIS = DB_PATH / "ednn_regression__iris.duckdb"
CSV_PATH__EDNN_REGRESSION__IRIS = DATA_DIR__RAW / "dataset__iris__dataset" / "Iris.csv"
SQLITE_PATH__EDNN_REGRESSION__IRIS = DATA_DIR__RAW / "dataset__iris__dataset" / "database.sqlite"

MODEL_DIR = BASE_DIR / "models"

NOTEBOOK_PATH = ASSET_DIR / "notebooks"

OUTPUT_DIR = BASE_DIR / "output"

PROJECT_DIR = BASE_DIR / "BA__Programmierung"

SCRIPTS_PATH = DATA_DIR / "scripts"

TESTS_PATH = DATA_DIR / "tests"

VIZ_PATH = ASSET_DIR / "viz"
