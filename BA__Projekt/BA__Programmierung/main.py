# BA__Projekt/BA__Programmierung/main.py
from rich.console import Console
# from db.load_iris import load_data

from ml.ednn_regression__iris import main as ednn_main

import sys
import os

def main():
    # set python path up to one folder to get subdirs like /ml/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    # Create a Console instance
    console = Console()
    console.log("Hello Project.")
    #load_data()
    ednn_main()

if __name__ == '__main__':
    main()