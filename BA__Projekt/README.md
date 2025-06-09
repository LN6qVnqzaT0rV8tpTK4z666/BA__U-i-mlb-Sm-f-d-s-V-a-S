# BA: U-i-mlb-Sm-f-d-s-V-a-S 

Bachelor-Arbeit: Unsicherheiten in machine-learning-basierten Surrogatmodellen für die szenariobasierte Validierung autonomer Systeme

## TOC

- [Initialization](#initialization)
- [Structure](#structure)
- [Changelog](#changelog)

## Initialization

Setup with Windows 11 WSL-2-Ubuntu-24.04.01-LTS.

Set environment variable.
```bash
sh scripts/setup__project_base_path.sh
```

Reload `bashrc`.
```bash
source ~/.bashrc
```

Inspect `bashrc` manually.
```bash
code ~/.bashrc; echo $PROJECT_BASE_PATH
```

Ensure installation of python3, tar, unzip, make.
```bash
sudo apt install python3 tar unzip make
```

Create virtual environment.
```bash
python3 -m venv .venv
```

Install required Python packages via pip.
```bash
pip install duckdb matplotlib notebook numpy pandas torch scikit-learn scipy tensorflow torchvision
```

Activate the virtual environment.
```bash
source .venv/bin/activate
```

Deactivate the virtual environment.
```bash
deactivate
```

Clean up .venv if the project directory was renamed.
```bash
rm -rf BA__Programmierung/.venv
```

LOCAL: Install the Python package.
```bash
pip install .
```

LOCAL: Run the built package from the main entry point `main.py`.
```bash
ba-programmierung
```

DOCKER: Start Docker service. 
```bash
sudo systemctl start docker
```

DOCKER: Build Docker image.
```bash
docker build -t ba__projekt .
```

DOCKER: Run Docker image.
```bash
docker run --rm ba__projekt:latest
```

Optional: Visualize loss-training with TensorBoard or inspect `viz`.
```bash
tensorboard --logdir /root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/data/processed/; http://localhost:6006
```

Optional: Clean build artifacts.
```bash
make clean
```

## Changelog
- [x] Setup basic clean-scripts.
- [x] Start english documentation.
- [x] Setup basic loss-visualization to `/viz/`.
- [x] Setup basic iris EDNN-training to `/data/processed/`.
- [x] Setup basic Docker.
- [x] Setup Readme.