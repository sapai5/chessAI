# Chess.com AI Engine & Overlay

An end-to-end computer vision and AI chess engine designed to play alongside you on Chess.com. It automatically detects the board from your screen, slices the pieces, classifies them using a custom PyTorch ResNet-18 model, builds the FEN state, and queries a local Stockfish instance to provide the optimal move.

The project comes with a sleek, transparent Electron overlay that stays on top of your board and automatically polls the screen for the best move.

## Features

- **No API Hooks**: Fully vision-based. Uses screen capturing to read the board exactly like a human does.
- **Custom CNN**: Includes a full offline PyTorch pipeline to generate synthetic training data using real Chess.com pieces and colors to train a ResNet-18 classifier.
- **Stockfish Integration**: Bundled with a UCI Stockfish client to calculate best moves.
- **Transparent Overlay**: A glassmorphic Electron UI that hovers over your screen and automatically updates as the game progresses.

## Project Structure

- `data/` — Synthetic training dataset (generated via script).
- `weights/` — Saved PyTorch model weights.
- `train/` — Scripts to generate datasets, train the CNN, and evaluate performance.
- `vision/` — Computer vision modules for screen capture, board detection, square slicing, and PyTorch inference.
- `engine/` — Orchestrates vision + FEN generation + Stockfish.
- `ui/` — The Electron frontend overlay.
- `server.py` — A Flask API bridging the PyTorch/Stockfish backend to the Electron UI.
- `stockfish.exe` — The actual Stockfish binary.

---

## Getting Started

### 1. Requirements

- Python 3.10+
- Node.js (for the Electron UI)
- Windows OS (using `mss` for Windows screen capture)

Install the backend Python dependencies:
```bash
venv\Scripts\activate
pip install -r requirements.txt
pip install Flask Flask-CORS
```

Install the frontend Electron dependencies:
```bash
cd ui
npm install
```

### 2. Running the UI Overlay (Daily Use)

If the model is already trained and Stockfish is present, you can launch the AI overlay immediately:

1. Open your terminal in the `ui/` directory:
   ```bash
   cd ui
   ```
2. Start the app:
   ```bash
   npm start
   ```

The Electron app will automatically boot up `server.py` in the background and present a transparent overlay window. Keep a Chess.com green-themed board open and visible on your screen. The UI will auto-scan and show you the best move!

### 3. Testing the Backend Components

You can test individual pieces of the AI pipeline from the CLI:

- **Test Board Detection:**
  ```bash
  python -m vision.board_detector
  ```
  *(Saves a `debug_detection.png` showing the bounding box it found).*

- **Test Full AI Engine (CLI):**
  ```bash
  python -m engine.engine --color w
  ```
  *(Outputs JSON with the FEN string and the calculated Stockfish move).*

---

## Advanced: Training the Model from Scratch

If you want to train the vision model yourself (for example, to support a different board color or piece theme), follow these steps:

### 1. Generate the Dataset
You need a massive PGN file (e.g., Lichess database). The script will parse the PGN, render the board states using Chess.com assets, and save thousands of 64x64 piece crops.
```bash
python -m train.generate_dataset --pgn lichess_db_standard_rated_2015-08.pgn --output data --max-positions 5000
```

### 2. Train the CNN
The pipeline uses a two-stage training loop (frozen backbone -> full fine-tune) using a `WeightedRandomSampler` to handle massive class imbalances (lots of empty squares, few queens).
```bash
python -m train.train --data data --epochs 20 --batch 128 --out weights/model.pt --max-per-class 20000
```
*(Tip: For a fast prototype, you can drop `--epochs` to 3 and `--max-per-class` to 5000).*

### 3. Evaluate
Check the accuracy per piece and generate a confusion matrix.
```bash
python -m train.evaluate --data data --weights weights/model.pt
```

## Credits
Built by Antigravity.
