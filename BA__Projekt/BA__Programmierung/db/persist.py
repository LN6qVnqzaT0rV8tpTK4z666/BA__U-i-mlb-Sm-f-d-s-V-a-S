# BA__Projekt/BA__Programmierung/db/persist.py

from datetime import datetime

import duckdb
from loguru import logger
from rich.console import Console

from BA__Programmierung.config import (
    DATA_DIR__RAW,
    DB_PATH,
    LOGS_DIR,
)

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


def db__persist():
    console = Console()
    console.log("Starte Persistierung aller Rohdaten...")

    csv_paths = find_csv_files(DATA_DIR__RAW)

    for csv_path in csv_paths:
        dataset_name = derive_dataset_name(csv_path)
        duckdb_path = DB_PATH / f"{dataset_name}.duckdb"

        if duckdb_path.exists():
            console.log(f"[yellow]⚠ DuckDB existiert bereits, überspringe:[/] {duckdb_path.name}")
            continue

        console.log(f"→ Persistiere: [bold cyan]{dataset_name}[/]")

        con = duckdb.connect(duckdb_path)
        table_name = dataset_name.replace("dataset__", "").replace("-", "_")

        con.execute(f"""
            CREATE TABLE {table_name}_csv AS
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
                    CREATE TABLE {table_name}__{tbl_name} AS
                    SELECT * FROM sqlite_db.{tbl_name}
                """)

        con.close()
        console.log(f"[green]✔ Persistiert:[/] {duckdb_path.name}")

    console.log("[bold green]Alle nicht-persistierten Datensätze wurden gespeichert.[/]")
