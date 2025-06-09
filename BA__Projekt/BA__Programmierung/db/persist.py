# BA__Projekt/BA__Programmierung/db/persist.py

import duckdb

from BA__Programmierung.config import DATA_DIR__RAW
from rich.console import Console


def db__persist():
    console = Console()
    console.log("Hello persistance.")

    csv_path = DATA_DIR__RAW / 'dataset__iris__dataset' / 'Iris.csv'
    sqlite_path = DATA_DIR__RAW / 'dataset__iris__dataset' / 'database.sqlite'

    con = duckdb.connect('my_database.duckdb')

    con.execute(f"""
        CREATE OR REPLACE TABLE iris_csv AS
        SELECT * FROM read_csv_auto('{csv_path.as_posix()}')
    """)

    con.execute(f"""
        ATTACH DATABASE '{sqlite_path.as_posix()}' AS sqlite_db (TYPE SQLITE)
    """)

    print(con.execute("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'sqlite_db'
    """).fetchall())

    con.execute("""
        CREATE OR REPLACE TABLE iris_sql AS
        SELECT * FROM sqlite_db.iris
    """)

    con.close()
