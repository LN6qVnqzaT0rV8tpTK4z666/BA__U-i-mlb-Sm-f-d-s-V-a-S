# BA__Projekt/BA__Programmierung/db/load_iris.py

import duckdb
from duckdb import DuckDBPyConnection
from pathlib import PureWindowsPath
import logging
import os


# # Ensure the log directory exists
# log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'log')
# os.makedirs(log_dir, exist_ok=True)

# # Log file path
# log_file = os.path.join(log_dir, 'project.log')

# # Configure logging to file
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler(log_file),
#         logging.StreamHandler()  # optional: also log to console
#     ]
# )

# logger = logging.getLogger(__name__)


def load_data():
    # Hole den Basis-Pfad aus der Umgebungsvariable oder verwende den Standardpfad
    # base_path = Path(__file__).resolve().parent.parent
    # csv_path = base_path/ 'data' / 'raw' / 'dataset__iris__dataset' / 'iris.csv'
    # sqlite_path = base_path / 'data' / 'raw' / 'dataset__iris__dataset' / 'database.sqlite'

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
