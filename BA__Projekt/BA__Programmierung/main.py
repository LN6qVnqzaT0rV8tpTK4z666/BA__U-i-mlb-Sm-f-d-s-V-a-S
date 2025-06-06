# BA__Projekt/BA__Programmierung/main.py
from rich.console import Console
from db.load_iris import load_data

def main():
    # Create a Console instance
    console = Console()
    console.log("Hello Project.")
    #load_data()

if __name__ == '__main__':
    main()