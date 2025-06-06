# BA: U-i-mlb-Sm-f-d-s-V-a-S 

Bachelor-Arbeit: Unsicherheiten in machine-learning-basierten Surrogatmodellen für die szenariobasierte Validierung autonomer Systeme

## Initialisieren

Setup unter Windows 11 WSL-2-Ubuntu-24.04.01-LTS.

Umgebungsvariable setzen.
```bash
sh scripts/setup__project_base_path.sh
```

bashrc neu einlesen.
```bash
source ~/.bashrc
```

bashrc manuell inspizieren.
```bash
code ~/.bashrc; echo $PROJECT_BASE_PATH
```

python3, tar, unzip Installation sicherstellen.
```bash
sudo apt install python3 tar unzip 
```

Virtuelle Umgebung erstellen.
```bash
python3 -m venv .venv
```

PIP Package-Installation ausführen.
```bash
pip install duckdb matplotlib notebook numpy pandas torch scikit-learn scipy tensorflow torchvision
```

Virtuelle Umgebung aktivieren.
```bash
source .venv/bin/activate
```

.venv verlassen.
```bash
deactivate
```

.venv abräumen bei Umbennenung.
```bash
rm -rf BA__Programmierung/.venv
```

LOKAL: Python-Package-Installation ausführen.
```bash
pip install .
```

LOKAL: Gebautes Paket am Einstiegspunkt main.py ausführen.
```bash
ba-programmierung
```

DOCKER: Service starten. 
```bash
sudo systemctl start docker
```

DOCKER: Image bauen.
```bash
docker build -t ba__projekt .
```

DOCKER: Image ausführen.
```bash
docker run --rm ba__projekt:latest
```

## Changelog
- [x] Grundlegend Docker aufsetzen.
- [x] Grundlegende Datenbankverbindung aufsetzen.
- [x] Readme grundlegend aufsetzen.