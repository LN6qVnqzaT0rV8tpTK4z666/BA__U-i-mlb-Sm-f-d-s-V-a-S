# BA__Projekt/BA__Programmierung/db/load_iris.py

import duckdb
from pathlib import Path
import os

def load_data():
    # Hole den Basis-Pfad aus der Umgebungsvariable oder verwende den Standardpfad
    base_path = Path(__file__).resolve().parent.parent
    csv_path = base_path/ 'data' / 'raw' / 'dataset__iris__dataset' / 'iris.csv'
    sqlite_path = base_path / 'data' / 'raw' / 'dataset__iris__dataset' / 'database.sqlite'

    con = duckdb.connect('my_database.duckdb')

    # con.execute(f"""
    #     CREATE TABLE iris_csv AS
    #     SELECT * FROM read_csv_auto('{csv_path.as_posix()}')
    # """)

    # con.execute(f"""
    #     ATTACH DATABASE '{sqlite_path.as_posix()}' AS sqlite_db (TYPE SQLITE)
    # """)

    # print(con.execute("""
    #     SELECT name FROM sqlite_db.sqlite_master WHERE type='table'
    # """).fetchall())

    # con.execute("""
    #     CREATE TABLE iris_sql AS
    #     SELECT * FROM sqlite_db.iris
    # """)

    con.close()
