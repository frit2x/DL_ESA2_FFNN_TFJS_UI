# ESA 2
# FFNN Regression mit TensorFlow.js

Dieses Projekt ist eine saubere Web-Anwendung für die Uni-Aufgabe:
- Regression der Funktion y(x) = 0.5*(x+0.8)*(x+1.8)*(x-0.2)*(x-0.3)*(x-1.9)+1
- TensorFlow.js für Modelltraining und Evaluation im Browser
- Plotly für interaktive Diagramme

## Dateien
- `index.html` — Struktur und UI
- `style.css` — Layout und Design
- `script.js` — Datengenerierung, FFNN-Modell, Training, Plotting

## Installation
Die App ist statisch und benötigt keinen Paketmanager. Du kannst sie lokal mit einem Webserver öffnen.

## Lokaler Test
```bash
cd /Users/heikefritz/Documents/DL/DL_ESA2_FFNN_TFJS_UI
python3 -m http.server 8000
```
Dann im Browser: `http://localhost:8000`

## Git
Initialisiere ein neues Git-Repository im Ordner:
```bash
git init
git add .
git commit -m "Initial commit: TFJS FFNN Regression UI"
```
