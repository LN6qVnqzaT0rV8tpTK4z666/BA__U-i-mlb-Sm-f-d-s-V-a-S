# Basis-Image mit Python
FROM python:3.10-slim

# Systemabhängigkeiten installieren (z. B. für opencv oder andere)
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Arbeitsverzeichnis im Container
WORKDIR /app

# Projektdateien kopieren
COPY . .

# Optional: virtualenv nutzen
# RUN python -m venv venv
# ENV PATH="/app/venv/bin:$PATH"

# Installiere Abhängigkeiten über pyproject.toml
RUN pip install --upgrade pip
RUN pip install .

# Öffne Jupyter Notebook Port
EXPOSE 8888

# Standard-Kommando beim Container-Start (z. B. Jupyter starten)
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
