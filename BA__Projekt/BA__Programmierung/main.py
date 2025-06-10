# BA__Projekt/BA__Programmierung/main.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Suppress TensorFlow logging messages: INFO, WARNING, ERROR
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN custom op warnings
import time
from db.persist import db__persist
from ml.ednn_regression__iris import main as ednn__main
from rich.console import Console
from viz.viz__ednn_regression__iris import main as ednn__viz


def main():
    console = Console()
    console.log("Hello Project.")
    db__persist()
    ednn__main()
    time.sleep(1)
    ednn__viz()


if __name__ == "__main__":
    main()
