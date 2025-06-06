# BA__Projekt/BA__Programmierung/db/load_iris.py

import duckdb # type: ignore
from pathlib import Path

def load_data():
    base_path = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'dataset__iris__dataset'
    csv_path = base_path / 'iris.csv'
    sqlite_path = base_path / 'database.sqlite'

    con = duckdb.connect()

    con.close()
