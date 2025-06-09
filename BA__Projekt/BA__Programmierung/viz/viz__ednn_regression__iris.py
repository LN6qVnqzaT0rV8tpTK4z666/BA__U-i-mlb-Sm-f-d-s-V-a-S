# BA__Projekt/BA__Programmierung/viz/vis__ednn_regression__iris.py

import glob
import matplotlib.pyplot as plt
import os
from tensorboard.backend.event_processing import event_accumulator

# ============
# Visualization
# ============

# === Dynamisches Logverzeichnis finden ===
processed_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../data/processed")
)
log_folders = sorted(
    glob.glob(os.path.join(processed_root, "ednn_iris_*")),
    key=os.path.getmtime,
    reverse=True
)

if not log_folders:
    raise FileNotFoundError("No log directory found. Please run training first.")

log_dir = log_folders[0]
print(f"Verwende Log-Verzeichnis: {log_dir}")

# === Zielverzeichnis für Visualisierung ===
viz_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../viz")
)
os.makedirs(viz_dir, exist_ok=True)
output_path = os.path.join(viz_dir, "loss_curve.png")

# === TensorBoard-Logs einlesen ===
ea = event_accumulator.EventAccumulator(log_dir)
ea.Reload()

# Verfügbare Tags anzeigen (optional)
print("Verfügbare Scalar-Tags:", ea.Tags()["scalars"])

# Daten extrahieren
train_scalars = ea.Scalars("Loss/train")
val_scalars = ea.Scalars("Loss/val") if "Loss/val" in ea.Tags()["scalars"] else []

train_steps = [e.step for e in train_scalars]
train_vals = [e.value for e in train_scalars]

plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_vals, label="Train Loss", color="blue")

if val_scalars:
    val_steps = [e.step for e in val_scalars]
    val_vals = [e.value for e in val_scalars]
    plt.plot(val_steps, val_vals, label="Val Loss", color="orange")

plt.title("Train & Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()

# === Bild speichern ===
plt.savefig(output_path)
print(f"Plot gespeichert unter: {output_path}")
