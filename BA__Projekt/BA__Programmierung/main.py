# BA__Projekt/BA__Programmierung/main.py

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = (
    "3"  # Suppress TensorFlow logging messages: INFO, WARNING, ERROR
)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN custom op warnings

import importlib
import time

from rich.console import Console

from BA__Programmierung.config import ML_DIR
from BA__Programmierung.db.persist import db__persist

# Define tokens for which training and viz should be skipped, add your manual tokens here, e.g.:
SKIP_TOKENS = {
    "boston_housing",
    "combined_cycle_power_plant",
    "concrete_compressive_strength",
    "condition_based_maintenance_of_naval_propulsion_plants",
    # "energy_efficiency",
    "fmnist",
    # "iris",
    # "kin8nm",
    # "nmavani_func1",
    # "wine_quality_red",
    # "wine_quality_white"
}


def get_target_keys():
    """
    Detect all ednn_regression__*.py training scripts and derive their keys.
    """
    console = Console()
    console.log(ML_DIR)
    return [
        f.stem.replace("ednn_regression__", "")
        for f in ML_DIR.glob("ednn_regression__*.py")
        if f.is_file()
    ]


def dynamic_import_and_run(module_path, description=""):
    """
    Import module dynamically and run its `main()` method.
    """
    try:
        module = importlib.import_module(module_path)
        module.main()
    except Exception as e:
        console = Console()
        console.print(
            f"[red]❌ Error running {description} from {module_path}[/red]"
        )
        console.print_exception()
        raise e


def main():
    console = Console()
    console.log("[bold green]Hello Project.[/bold green]")

    # Initial DB setup
    db__persist()

    # Discover and run all target training + viz modules
    target_keys = get_target_keys()

    console.log(target_keys)

    for key in target_keys:
        if key in SKIP_TOKENS:
            console.log(f"[yellow]Skipping training and visualization for: {key}[/yellow]")
            continue

        ml_module = f"ml.ednn_regression__{key}"
        viz_module = f"viz.viz__ednn_regression__{key}"

        console.rule(f"[bold blue]Running Training: {key}[/bold blue]")
        dynamic_import_and_run(ml_module, description=f"ML Training ({key})")

        time.sleep(1)  # Optional delay between training and viz

        console.rule(
            f"[bold magenta]Running Visualization: {key}[/bold magenta]"
        )
        dynamic_import_and_run(
            viz_module, description=f"Visualization ({key})"
        )

    console.log(
        "[bold green]All trainings and visualizations completed.[/bold green]"
    )


if __name__ == "__main__":
    main()
