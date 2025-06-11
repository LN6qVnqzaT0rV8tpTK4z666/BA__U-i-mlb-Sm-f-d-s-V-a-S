# BA__Projekt/BA__Programmierung/db/persist.py

import duckdb

from BA__Programmierung.config import (
    DB_PATH,
    DATA_DIR__RAW,
    DB_PATH__EDNN_REGRESSION__IRIS,
    CSV_PATH__EDNN_REGRESSION__IRIS,
    SQLITE_PATH__EDNN_REGRESSION__IRIS,
    LOGS_DIR,
)
from datetime import datetime
from loguru import logger
from rich.console import Console


# Setup Logging
LOGS_DIR.mkdir(parents=True, exist_ok=True)
log_file_path = (
    LOGS_DIR / f"persist__{datetime.now().strftime('%Y-%m-%d__%H-%M-%S')}.log"
)
logger.add(
    log_file_path,
    rotation="1 day",
    retention="7 days",
    compression="zip",
    enqueue=True,
)
logger.info(f"Logging gestartet: {log_file_path}")


def find_csv_files(base_path):
    return list(base_path.glob("dataset__*/**/*.csv"))


def find_sqlite_for_dataset(dataset_dir):
    sqlite_files = list(dataset_dir.glob("*.sqlite")) + list(
        dataset_dir.glob("*.db")
    )
    return sqlite_files[0] if sqlite_files else None


def derive_dataset_name(csv_path):
    return csv_path.parent.name  # e.g., dataset__boston_housing


def db__persist(): # db__persist_all_raw_datasets()
    console = Console()
    console.log("Starte Persistierung aller Rohdaten...")

    csv_paths = find_csv_files(DATA_DIR__RAW)

    for csv_path in csv_paths:
        dataset_name = derive_dataset_name(csv_path)
        duckdb_path = DB_PATH / f"{dataset_name}.duckdb"

        console.log(f"→ Persistiere: [bold cyan]{dataset_name}[/]")

        con = duckdb.connect(duckdb_path)

        table_name = dataset_name.replace("dataset__", "").replace("-", "_")

        con.execute(f"""
            CREATE OR REPLACE TABLE {table_name}_csv AS
            SELECT * FROM read_csv_auto('{csv_path.as_posix()}')
        """)

        sqlite_path = find_sqlite_for_dataset(csv_path.parent)
        if sqlite_path:
            console.log(f"  + SQLite gefunden: {sqlite_path.name}")
            con.execute(f"""
                ATTACH DATABASE '{sqlite_path.as_posix()}' AS sqlite_db (TYPE SQLITE)
            """)

            sqlite_tables = con.execute("""
                SELECT table_name FROM information_schema.tables WHERE table_schema = 'sqlite_db'
            """).fetchall()

            for tbl in sqlite_tables:
                tbl_name = tbl[0]
                console.log(f"    ↳ Importiere Tabelle: sqlite_db.{tbl_name}")
                con.execute(f"""
                    CREATE OR REPLACE TABLE {table_name}__{tbl_name} AS
                    SELECT * FROM sqlite_db.{tbl_name}
                """)

        con.close()
        console.log(f"[green]✔ Persistiert:[/] {duckdb_path.name}")

    console.log("[bold green]Alle Datensätze wurden persistiert.[/]")


def db__perstist_test():
    console = Console()
    console.log("Hello persistance.")

    db_path__ednn_regression__iris = DB_PATH__EDNN_REGRESSION__IRIS
    csv_path__ednn_regression__iris = CSV_PATH__EDNN_REGRESSION__IRIS
    sqlite_path__ednn_regression__iris = SQLITE_PATH__EDNN_REGRESSION__IRIS

    con = duckdb.connect(db_path__ednn_regression__iris)

    con.execute(f"""
        CREATE OR REPLACE TABLE iris_csv AS
        SELECT * FROM read_csv_auto('{csv_path__ednn_regression__iris.as_posix()}')
    """)

    con.execute(f"""
        ATTACH DATABASE '{sqlite_path__ednn_regression__iris.as_posix()}' AS sqlite_db (TYPE SQLITE)
    """)

    print(
        con.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'sqlite_db'
    """).fetchall()
    )

    con.execute("""
        CREATE OR REPLACE TABLE iris_sql AS
        SELECT * FROM sqlite_db.iris
    """)

    con.close()
    console.log(f"DuckDB gespeichert unter: {db_path__ednn_regression__iris}")
