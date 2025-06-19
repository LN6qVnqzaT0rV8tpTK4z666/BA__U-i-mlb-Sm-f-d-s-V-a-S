# BA__Projekt/BA__Programmierung/main.py

"""
Main project orchestration script for training and visualizing models.

This script performs the following:
- Sets environment variables to suppress TensorFlow logs.
- Loads all available `ednn_regression__*.py` training scripts from the ML directory.
- Skips training and visualization for manually specified datasets.
- Dynamically imports and runs training and visualization modules.
- Initializes the database on startup.
"""

import os
import importlib
import time
from rich.console import Console

from BA__Programmierung.config import ML_DIR
from BA__Programmierung.db.persist import db__persist

# Suppress TensorFlow logging messages and disable oneDNN custom op warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Suppress properscoring/_brier.py:95: SyntaxWarning: invalid escape sequence '\i' equivalents
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

# Dataset tokens to skip during training and visualization
SKIP_TOKENS = {
    # "boston_housing",
    "combined_cycle_power_plant",
    "concrete_compressive_strength",
    "condition_based_maintenance_of_naval_propulsion_plants",
    "energy_efficiency",
    "iris",
    "kin8nm",
    "nmavani_func1",
    "wine_quality_red",
    "wine_quality_white"
}


def get_target_keys():
    """
    Detect training scripts and derive dataset keys.

    Scans the `ML_DIR` directory for files matching `ednn_regression__*.py`,
    and extracts the dataset key from the filename.

    Returns:
        List[str]: A list of dataset keys such as `iris`, `wine_quality_red`, etc.
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
    Dynamically import a module and execute its `main()` function.

    Args:
        module_path (str): Python module path (e.g., `ml.ednn_regression__iris`).
        description (str): Optional human-readable description for logging.

    Raises:
        Exception: If an error occurs during module import or execution.
    """
    try:
        module = importlib.import_module(module_path)
        module.main()
    except Exception as e:
        console = Console()
        console.print(f"[red]❌ Error running {description} from {module_path}[/red]")
        console.print_exception()
        raise e


def main():
    """
    Entry point for model training and visualization.

    Performs the following:
    - Initializes the database.
    - Detects available training targets.
    - Executes training and visualization scripts unless skipped.
    """
    console = Console()
    console.log("[bold green]Hello Project.[/bold green]")

    # Step 1: DB initialization
    db__persist()

    # Step 2: Discover available training targets
    target_keys = get_target_keys()
    console.log(target_keys)

    # Step 3: Execute training and visualization
    for key in target_keys:
        if key in SKIP_TOKENS:
            console.log(f"[yellow]Skipping training and visualization for: {key}[/yellow]")
            continue

        ml_module = f"ml.ednn_regression__{key}"
        viz_module = f"viz.viz__ednn_regression__{key}"

        console.rule(f"[bold blue]Running Training: {key}[/bold blue]")
        dynamic_import_and_run(ml_module, description=f"ML Training ({key})")

        time.sleep(1)  # Optional delay between training and viz

        console.rule(f"[bold magenta]Running Visualization: {key}[/bold magenta]")
        dynamic_import_and_run(viz_module, description=f"Visualization ({key})")

    console.log("[bold green]All trainings and visualizations completed.[/bold green]")


if __name__ == "__main__":
    main()

