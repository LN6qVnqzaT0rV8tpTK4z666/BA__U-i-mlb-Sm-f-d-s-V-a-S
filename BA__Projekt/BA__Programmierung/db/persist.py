# BA__Projekt/BA__Programmierung/db/persist.py

import duckdb

from BA__Programmierung.config import (
    DB_PATH__EDNN_REGRESSION__IRIS,
    CSV_PATH__EDNN_REGRESSION__IRIS,
    SQLITE_PATH__EDNN_REGRESSION__IRIS,
)
from rich.console import Console


def db__persist():
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
