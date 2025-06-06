# Basis-Image
FROM python:3.11-slim

# Arbeitsverzeichnis setzen
WORKDIR /app

# Systempakete (z. B. build tools, falls nötig)
RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

# Projektdateien in den Container kopieren
COPY . /app

# Abhängigkeiten und Paket installieren
RUN pip install --upgrade pip setuptools wheel \
 && pip install .

# Startbefehl (z. B. main.py oder Einstiegspunkt aus setup/pyproject)
CMD ["ba-programmierung"]
