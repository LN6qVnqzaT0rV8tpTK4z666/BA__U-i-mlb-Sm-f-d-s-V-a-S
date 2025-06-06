# BA: U-i-mlb-Sm-f-d-s-V-a-S 

Bachelor-Arbeit: Unsicherheiten in machine-learning-basierten Surrogatmodellen für die szenariobasierte Validierung autonomer Systeme

## Initialisieren

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

Python-Package-Installation ausführen.
```bash
pip install .
```

Gebautes Paket am Einstiegspunkt main.py ausführen.
```bash
ba-programmierung
```

<!-- 
1. 🛠️ Baue das Docker-Image: Im Projektverzeichnis (BA__Programmierung/):

```bash
docker build -t ba-projekt .
```

2. ▶️ Starte den Container:

```bash
docker run -it -p 8888:8888 ba-projekt
```

3. 🧪 Ergebnis: Sobald der Container läuft, öffne im Browser:

```bash
http://localhost:8888
``` 
## Struktur
-->


## Changelog

- [x] Grundlegende Datenbankverbindung aufsetzen.
- [x] Readme grundlegend aufsetzen.