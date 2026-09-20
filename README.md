# FaceCam

FaceCam is a Windows virtual-camera app that watches your webcam with MediaPipe, detects facial expressions and hand/body gestures, then overlays meme reactions on your face in real time. The clean meme feed is sent to a virtual camera you can select in Google Meet, Zoom, Discord, and similar apps.

Inspired by [gazijarin/itsgiving](https://github.com/gazijarin/itsgiving).

```text
Webcam → MediaPipe (face / hands / pose) → calibration + reactions
      → meme overlay → virtual camera → Meet / Zoom / Discord
```

Local debug window shows landmarks and HUD text. The virtual camera feed stays clean (memes only).

---

## Features

- Live face, hand, and pose tracking (MediaPipe Tasks)
- Neutral-face calibration so expressions are relative to *you*
- 14 meme reactions (still images + GIFs) that track your face
- Virtual camera output via [pyvirtualcam](https://github.com/letmaik/pyvirtualcam) (OBS backend on Windows)
- Local preview with FPS / reaction HUD (`--no-vcam` for preview-only)

---

## Requirements

- **Windows** 10/11
- **Python** 3.11 (tested with 3.11.9)
- A webcam
- **[OBS Studio](https://obsproject.com/)** — needed once so the OBS Virtual Camera backend exists (you do not need to leave OBS running while FaceCam runs)

---

## Install

```powershell
git clone https://github.com/Diasnk/facememe.git
cd facememe

python -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install [OBS Studio](https://obsproject.com/) if you have not already. Open OBS once, optionally click **Start Virtual Camera** then stop it, and quit OBS. That registers the Windows device FaceCam uses.

> **Dependency pins matter.** This stack uses `mediapipe==0.10.21` with `numpy<2` and `opencv-python<5`. Do not casually unpin them.

---

## Use

### 1. Start FaceCam

Close other apps that are using your physical webcam, and leave OBS’s own Virtual Camera **stopped**.

```powershell
.\venv\Scripts\Activate.ps1
python app.py
```

On first run, MediaPipe models download into `models/`, reaction assets load from `assets/`, and calibration may start automatically (hold a neutral face for a few seconds).

When the virtual camera opens, the console prints a device name, for example:

```text
Virtual camera: 'OBS Virtual Camera' <- pick this camera in Zoom / Meet / Discord
```

Keep FaceCam running for the whole call.

**Preview only** (no virtual camera):

```powershell
python app.py --no-vcam
```

### 2. Pick the camera in your call app

| App | Where |
|-----|--------|
| **Google Meet** | Settings → Video → Camera → OBS Virtual Camera |
| **Zoom** | Settings → Video → Camera → OBS Virtual Camera |
| **Discord** | User Settings → Voice & Video → Camera → OBS Virtual Camera |

You should see yourself with memes, **without** the green box / z-score HUD (that stays in the local FaceCam window only).

### 3. Keys

| Key | Action |
|-----|--------|
| `q` | Quit |
| `c` | Recalibrate neutral face |

---

## Reactions

First matching gesture wins each frame. Examples:

| Reaction | How to trigger |
|----------|----------------|
| `open_mouth` | Open mouth wide |
| `hand_up` | Raise an open palm beside/above your face |
| `dance` | Elbows up |
| `disgusted` | Nose sneer / disgust face |
| `suspicious` | Turn head + squint |
| `tongue_out` | Stick tongue out (jaw open) |
| `heart` | Heart hands |
| `time_out` | T-shape with both hands |
| `spin` | Leave the frame (no face) |

Assets live in `assets/` and map by filename (e.g. `open_mouth.jpeg`).

---

## Troubleshooting

| Problem | What to try |
|---------|-------------|
| `Virtual camera unavailable` / could not be started | Install OBS; quit OBS and Windows Camera; start FaceCam alone |
| Black feed in Meet/Zoom | FaceCam must be running first; select the printed device name |
| Raw webcam, no memes | Call app selected the physical camera — switch to OBS Virtual Camera |
| Webcam won’t open | Close Teams / browser / Zoom tabs using the camera |
| Reactions feel wrong | Press `c` and recalibrate under the same lighting |
| Odd red/blue colors | Report it — FaceCam sends OpenCV **BGR** frames on purpose |

---

## Project layout

```text
facememe/
├── app.py                 # main loop
├── assets/                # meme overlays
├── camera/                # webcam + virtual camera
├── calibration/           # neutral baseline
├── vision/                # MediaPipe detectors + features
├── reactions/             # classifier + arm/hold state
├── rendering/             # asset load + compositor
├── models/                # auto-downloaded .task files
├── data/                  # calibration.json (local)
└── requirements.txt
```

---

## Credits

- Behavior and assets inspired by [itsgiving](https://github.com/gazijarin/itsgiving)
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [pyvirtualcam](https://github.com/letmaik/pyvirtualcam)
- [OBS Studio](https://obsproject.com/) / [Virtual Camera](https://obsproject.com/kb/virtual-camera)
