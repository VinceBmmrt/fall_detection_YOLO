# G.E.R.O — Vision / Fall Detection

> **EN** — Real-time human fall detection module for the Unitree G1 EDU security robot.  
> **FR** — Module de détection de chute en temps réel pour le robot de sécurité Unitree G1 EDU.

---

## Context / Contexte

### EN
This module is part of **G.E.R.O** (General Emergency Response Operations), an autonomous security robot built on the **Unitree G1 EDU** platform.

G.E.R.O is designed for a live demonstration at **Paris Charles de Gaulle Airport** as a next-generation security robot capable of detecting and responding to critical incidents in real time. Fall detection is the first incident type handled — when a person is detected as fallen, the robot raises an alert and the G.E.R.O agent system takes over to manage the response.

### FR
Ce module fait partie de **G.E.R.O** (General Emergency Response Operations), un robot de sécurité autonome construit sur la plateforme **Unitree G1 EDU**.

G.E.R.O est conçu pour une démonstration en conditions réelles à l'**Aéroport Paris Charles de Gaulle** en tant que robot de sécurité de nouvelle génération, capable de détecter et de répondre à des incidents critiques en temps réel. La détection de chute est le premier type d'incident pris en charge — lorsqu'une personne est détectée au sol, le robot déclenche une alerte et le système agent de G.E.R.O prend le relais pour gérer la réponse.

---

## Model / Modèle

### EN
The detection model is a **pretrained YOLOv11** sourced from HuggingFace:
[`melihuzunoglu/human-fall-detection`](https://huggingface.co/melihuzunoglu/human-fall-detection)

We did not train this model ourselves — it was trained on a human posture dataset and published openly on HuggingFace. The weights (`best.pt`) are downloaded automatically at first run and cached locally. They are never committed to the repository.

### FR
Le modèle de détection est un **YOLOv11 pré-entraîné** récupéré sur HuggingFace :
[`melihuzunoglu/human-fall-detection`](https://huggingface.co/melihuzunoglu/human-fall-detection)

Nous n'avons pas entraîné ce modèle — il a été entraîné sur un jeu de données de postures humaines et publié librement sur HuggingFace. Les poids (`best.pt`) sont téléchargés automatiquement au premier lancement et mis en cache localement. Ils ne sont jamais commités dans le dépôt.

---

## How it works / Fonctionnement

```
Camera feed
    │
    ▼
YOLOv11 (melihuzunoglu/human-fall-detection)
    │
    ├── Standing  ──► no action
    ├── Sitting   ──► no action
    └── Fallen    ──► AlertManager
                          │
                          ├── sliding window (N consecutive frames)
                          ├── cooldown check
                          └── AlertHandler ──► [ALERT] + agent trigger
```

**3 classes detected:** `standing` (green) · `sitting` (orange) · `fallen` (red)  
**Alert fires** after `min_fall_frames` consecutive fallen detections, with cooldown between alerts.

---

## Project Structure / Structure du projet

```
fall_detection/
├── main.py                    # Entrypoint
├── requirements.txt           # Plain pip install
├── config/
│   ├── base.yaml              # Default config (laptop + webcam)
│   └── g1.yaml                # Robot overrides (headless, CUDA, G1 cam index)
├── detection/
│   ├── camera.py              # OpenCVCamera, RealSenseCamera, CameraFactory
│   ├── model.py               # YOLOv11 inference + frame annotation
│   ├── alert.py               # AlertManager + pluggable handlers
│   └── detector.py            # Main detection loop
└── scripts/
    ├── find_cameras.py        # Scan camera indexes and device names
    └── export_tensorrt.py     # Export best.pt → best.engine (Jetson only)
```

---

## Quickstart / Démarrage rapide

### With uv (recommended)
```bash
uv sync
uv run python main.py
```

### With pip
```bash
pip install -r requirements.txt
python main.py
```

Press `q` to quit the detection window.

---

## Configuration

All parameters live in `config/base.yaml`. No magic numbers in code.

| Key | Default | Description |
|-----|---------|-------------|
| `camera.source` | `1` | Camera index, file path, or RTSP URL |
| `camera.backend` | `opencv` | `opencv` or `realsense` |
| `model.confidence` | `0.5` | Detection confidence threshold |
| `alert.min_fall_frames` | `3` | Consecutive fallen frames before alert fires |
| `alert.cooldown_seconds` | `10` | Minimum delay between two alerts |
| `alert.save_image` | `false` | Save a snapshot on alert |
| `display.show_window` | `true` | Show live video window |

### CLI overrides
```bash
python main.py -c config/g1.yaml          # use G1 robot config
python main.py --source 2                 # override camera index
python main.py --no-display               # headless mode
```

---

## Deployment on Unitree G1 / Déploiement sur le Unitree G1

See [G1_DEPLOY.md](G1_DEPLOY.md) for the full step-by-step guide.

**TL;DR:**
```bash
git clone <repo> && cd fall_detection
uv sync
uv run python scripts/find_cameras.py        # find UGREEN cam index
uv run python scripts/export_tensorrt.py     # convert to TensorRT (once)
uv run python main.py -c config/g1.yaml
```

---

## Alert System / Système d'alertes

The alert system is pluggable. Three handlers are defined:

| Handler | Status | Description |
|---------|--------|-------------|
| `ConsoleAlertHandler` | Ready | Prints alert to stdout, optionally saves snapshot |
| `WebhookAlertHandler` | Stub | HTTP webhook — not yet implemented |
| `AgentToolHandler` | Stub | **G.E.R.O agent integration point** — wire up here only |

The `AgentToolHandler` is the future bridge between fall detection and the G.E.R.O agent response system. No other code needs to change when it is wired up.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Detection model | YOLOv11 via `ultralytics` |
| Model weights | `huggingface_hub` — auto-downloaded, never committed |
| Camera | `opencv-python` / `pyrealsense2` (lazy) |
| Config | `pyyaml` |
| Robot platform | Unitree G1 EDU — Jetson Orin NX |

---

## Phase Roadmap

- [x] **Phase 1** — Laptop + USB webcam, fall detection + console alerts
- [ ] **Phase 2** — Unitree G1 EDU robot, TensorRT inference, G.E.R.O agent integration
- [ ] **Phase 3** — CDG Airport live demonstration
