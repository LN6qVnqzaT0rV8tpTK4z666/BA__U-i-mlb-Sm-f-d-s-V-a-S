#!/bin/bash

# Definiere den Pfad, der hinzugefügt werden soll
export_line='export PROJECT_BASE_PATH="/root/BA__U-i-mlb-Sm-f-d-s-V-a-S/BA__Projekt/"'

# Überprüfe, ob die Zeile bereits existiert
if ! grep -Fxq "$export_line" ~/.bashrc; then
  # Wenn nicht, füge sie ans Ende der Datei an
  echo "$export_line" >> ~/.bashrc
  echo "Zeile hinzugefügt: $export_line"
else
  echo "Zeile bereits vorhanden: $export_line"
fi