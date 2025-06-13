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

# DATASET__BOSTON_HOUSING
DB_PATH__BOSTON_HOUSING = DB_PATH / "dataset__boston_housing.duckdb"
CSV_PATH__BOSTON_HOUSING = DATA_DIR__RAW / "dataset__boston-housing" / "dataset__boston_housing.csv"
# SQLITE_PATH__BOSTON_HOUSING = DATA_DIR__RAW / "dataset__boston_housing" / "dataset__boston_housing.sqlite"

# DATASET__CIFAR10
# DB_PATH__CIFAR10 = DB_PATH / "dataset__cifar10.duckdb"
# CSV_PATH__CIFAR10 = DATA_DIR__RAW / "dataset__cifar10" / "dataset__cifar10.csv"
# SQLITE_PATH__CIFAR10 = DATA_DIR__RAW / "dataset__cifar10" / "dataset__cifar10.sqlite"

# DATASET__COMBINED_CYCLE_POWER_PLANT
DB_PATH__COMBINED_CYCLE_POWER_PLANT = DB_PATH / "dataset__combined_cycle_power_plant.duckdb"
CSV_PATH__COMBINED_CYCLE_POWER_PLANT = DATA_DIR__RAW / "dataset__combined_cycle_power_plant" / "dataset__combined_cycle_power_plant.csv"
# SQLITE_PATH__COMBINED_CYCLE_POWER_PLANT = DATA_DIR__RAW / "dataset__combined_cycle_power_plant" / "database.sqlite"

# DATASET__CONCRETE_COMPRESSIVE_STRENGTH
DB_PATH__CONCRETE_COMPRESSIVE_STRENGTH = DB_PATH / "dataset__concrete_compressive_strength.duckdb"
CSV_PATH__CONCRETE_COMPRESSIVE_STRENGTH = DATA_DIR__RAW / "dataset__concrete_compressive_strength" / "dataset__concrete_compressive_strength.csv"
# SQLITE_PATH__CONCRETE_COMPRESSIVE_STRENGTH = DATA_DIR__RAW / "dataset__concrete_compressive_strength" / "database.sqlite"

# DATASET__CONDITION_BASED_MAINTENANCE_OF_NAVAL_PROPULSION_PLANTS
DB_PATH__CONDITION_BASED_MAINTENANCE_OF_NAVAL_PROPULSION_PLANTS = DB_PATH / "dataset__condition_based_maintenance_of_naval_propulsion_plants.duckdb"
CSV_PATH__CONDITION_BASED_MAINTENANCE_OF_NAVAL_PROPULSION_PLANTS = DATA_DIR__RAW / "dataset__condition_based_maintenance_of_naval_propulsion_plants" / "dataset__condition_based_maintenance_of_naval_propulsion_plants.csv"
# SQLITE_PATH__CONDITION_BASED_MAINTENANCE_OF_NAVAL_PROPULSION_PLANTS = DATA_DIR__RAW / "dataset__condition_based_maintenance_of_naval_propulsion_plants" / "database.sqlite"

# DATASET__GENERATED__MAVANI__FUNC_1
DB_PATH__GENERATED__MAVANI__FUNC_1 = DB_PATH / "dataset__generated__nmavani__func_1.duckdb"
CSV_PATH__GENERATED__MAVANI__FUNC_1 = DATA_DIR__RAW / "dataset__generated-nmavani-func_1" / "dataset__generated-nmavani-func_1.csv"
# SQLITE_PATH__GENERATED__MAVANI__FUNC_1 = DATA_DIR__RAW / "dataset__generated__nmavani__func_1" / "database.sqlite"

# DATASET__EDNN_REGRESSION__IRIS
DB_PATH__EDNN_REGRESSION__IRIS = DB_PATH / "ednn_regression__iris.duckdb"
CSV_PATH__EDNN_REGRESSION__IRIS = DATA_DIR__RAW / "dataset__iris__dataset" / "Iris.csv"
SQLITE_PATH__EDNN_REGRESSION__IRIS = DATA_DIR__RAW / "dataset__iris__dataset" / "database.sqlite"

# DATASET__FMNIST
DB_PATH__FMNIST = DB_PATH / "dataset__fmnist.duckdb"
CSV_PATH__FMNIST = DATA_DIR__RAW / "dataset__fmnist" / "dataset__fmnist.csv"
# SQLITE_PATH__FMNIST = DATA_DIR__RAW / "dataset__fmnist" / "dataset__fmnist.sqlite"

# DATASET__KMNIST
DB_PATH__KMNIST = DB_PATH / "dataset__kmnist.duckdb"
# CSV_PATH__KMNIST = DATA_DIR__RAW / "dataset__kmnist" / "dataset__kmnist.csv"
# SQLITE_PATH__KMNIST = DATA_DIR__RAW / "dataset__kmnist" / "dataset__kmnist.sqlite"

# DATASET__MNIST
# DB_PATH__MNIST = DB_PATH / "dataset__mnist.duckdb"
# CSV_PATH__MNIST = DATA_DIR__RAW / "dataset__mnist" / "dataset__mnist.csv"
# SQLITE_PATH__MNIST = DATA_DIR__RAW / "dataset__mnist" / "dataset__mnist.sqlite"

# DATASET__KIN8NM
DB_PATH__KIN8NM = DB_PATH / "dataset__kin8nm.duckdb"
CSV_PATH__KIN8NM = DATA_DIR__RAW / "dataset__kin8nm" / "dataset__kin8nm-dataset_2175.csv"
# SQLITE_PATH__KIN8NM = DATA_DIR__RAW / "dataset__kin8nm" / "dataset__kin8nm.sqlite"

# DATASET__SVHN
# DB_PATH__SVHN = DB_PATH / "dataset__svhn.duckdb"
# CSV_PATH__SVHN = DATA_DIR__RAW / "dataset__svhn" / "dataset__svhn.csv"
# SQLITE_PATH__SVHN = DATA_DIR__RAW / "dataset__svhn" / "dataset__svhn.sqlite"

# DATASET__WINE+QUALITY
DB_PATH__WINE_QUALITY = DB_PATH / "wine-quality-red.duckdb"
CSV_PATH__WINE_QUALITY = DATA_DIR__RAW / "wine_quality__dataset" / "wine-quality-red.csv"
DB_PATH__WINE_QUALITY = DB_PATH / "wine-quality-white.duckdb"
CSV_PATH__WINE_QUALITY = DATA_DIR__RAW / "wine_quality__dataset" / "wine-quality-white.csv"
# SQLITE_PATH__WINE_QUALITY = DATA_DIR__RAW / "wine_quality__dataset" / "wine_quality__dataset.sqlite"

LOGS_DIR = ASSET_DIR / "logs"

MODEL_DIR = BASE_DIR / "models"

NOTEBOOK_PATH = ASSET_DIR / "notebooks"

OUTPUT_DIR = BASE_DIR / "output"

PROJECT_DIR = BASE_DIR / "BA__Programmierung"

ML_DIR = PROJECT_DIR / "ml"

SCRIPTS_PATH = DATA_DIR / "scripts"

TESTS_PATH = DATA_DIR / "tests"

VIZ_PATH = ASSET_DIR / "viz"
