# BA__Projekt/BA__Programmierung/main.py

from db.persist import db__persist
from ml.ednn_regression__iris import main as ednn_main
from rich.console import Console


def main():
    console = Console()
    console.log("Hello Project.")
    db__persist()
    ednn_main()

if __name__ == '__main__':
    main()
    