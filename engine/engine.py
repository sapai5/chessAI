"""
engine/engine.py
─────────────────
Top-level orchestration: screenshot → best move.

This is the entry point called by the Electron UI.

Usage (CLI):
    python engine/engine.py --color w --flipped false

Usage (from Electron via subprocess):
    python engine/engine.py --color w
    → stdout: JSON with keys: fen, move, san, score, mate, confidence_avg
"""

import argparse
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path

from vision.board_detector import screenshot_board
from vision.square_slicer import slice_board
from vision.classifier import PieceClassifier
from engine.fen_builder import build_fen, validate_fen, repair_labels
from engine.stockfish_client import StockfishClient

WEIGHTS_PATH = Path(__file__).parent.parent / "weights" / "model.pt"
LOG_DIR = Path(__file__).parent.parent / "logs"

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("chessai")
logger.setLevel(logging.DEBUG)

# File handler — appends to logs/chessai.log
_fh = logging.FileHandler(LOG_DIR / "chessai.log", encoding="utf-8")
_fh.setFormatter(logging.Formatter(
    "%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S"
))
logger.addHandler(_fh)

# Also log to stderr so Electron can see it
_sh = logging.StreamHandler(sys.stderr)
_sh.setFormatter(logging.Formatter("[engine] %(message)s"))
_sh.setLevel(logging.INFO)
logger.addHandler(_sh)

# ── Singletons ────────────────────────────────────────────────────────────────
_classifier: PieceClassifier | None = None
_stockfish: StockfishClient | None = None


def _get_classifier() -> PieceClassifier:
    global _classifier
    if _classifier is None:
        if not WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"Model weights not found at {WEIGHTS_PATH}. "
                "Run train/train.py first."
            )
        _classifier = PieceClassifier(str(WEIGHTS_PATH))
    return _classifier


def _get_stockfish() -> StockfishClient:
    """Keep a single Stockfish process alive across requests."""
    global _stockfish
    if _stockfish is None:
        _stockfish = StockfishClient(depth=20, time_limit=2.0)
        _stockfish._ensure_engine()
        logger.info("Stockfish engine started (persistent)")
    return _stockfish


def _format_board_labels(labels: list[str]) -> str:
    """Pretty-print the 64 labels as an 8×8 grid for the log."""
    lines = []
    for rank in range(8):
        row = labels[rank * 8 : (rank + 1) * 8]
        lines.append("  " + " ".join(f"{c:5s}" for c in row))
    return "\n" + "\n".join(lines)


def analyze(active_color: str = "w", flipped: bool = False, platform: str = "chess.com") -> dict:
    """
    Full pipeline: screenshot → FEN → Stockfish best move.

    Args:
        active_color: 'w' or 'b'
        flipped:      True if board is shown from Black's perspective
        platform:     'chess.com' or 'lichess'

    Returns:
        dict with: fen, move, san, score, mate, confidence_avg, labels
    """
    logger.info("=" * 60)
    logger.info(f"Analyze request  color={active_color}  flipped={flipped}  platform={platform}")

    # 1. Screenshot board
    board_img = screenshot_board(platform=platform)
    logger.debug(f"Screenshot captured: {board_img.size}")

    # 2. Slice into 64 crops
    crops = slice_board(board_img)
    logger.debug(f"Sliced into {len(crops)} crops")

    # 3. Classify each square
    clf = _get_classifier()
    results = clf.predict_with_confidence(crops)
    labels = [r[0] for r in results]
    confidences = [r[1] for r in results]
    confidence_avg = sum(confidences) / len(confidences)

    # Log the board grid and low-confidence squares
    logger.info(f"Board vision (avg conf: {confidence_avg:.4f}):")
    logger.info(_format_board_labels(labels))

    low_conf = [(i, labels[i], confidences[i])
                for i in range(64) if confidences[i] < 0.90]
    if low_conf:
        logger.warning(f"Low-confidence squares ({len(low_conf)}):")
        for idx, lbl, conf in low_conf:
            r, c = divmod(idx, 8)
            files = "abcdefgh"
            sq_name = f"{files[c]}{8 - r}"
            logger.warning(f"  {sq_name} -> {lbl} (conf={conf:.4f})")

    # 3b. Confidence gate — reject frames that are too noisy to trust.
    # avg conf < 0.95 or > 6 uncertain squares means the board isn't fully visible
    # (e.g. game result overlay, animation, or partial occlusion).
    MAX_LOW_CONF_SQUARES = 6
    MIN_AVG_CONF = 0.95
    if confidence_avg < MIN_AVG_CONF or len(low_conf) > MAX_LOW_CONF_SQUARES:
        logger.warning(
            f"Board confidence too low to analyze "
            f"(avg={confidence_avg:.4f}, low_conf_squares={len(low_conf)}) — skipping."
        )
        return {"error": f"Board not clearly visible (avg conf {confidence_avg:.3f}, {len(low_conf)} uncertain squares)"}

    # 3c. Repair labels — enforce exactly one king per side before building FEN
    labels, repair_log = repair_labels(labels, confidences)
    if repair_log:
        for msg in repair_log:
            logger.warning(msg)

    # 4. Build FEN
    fen = build_fen(labels, active_color=active_color, flipped=flipped)
    logger.info(f"FEN: {fen}")

    if not validate_fen(fen):
        logger.error(f"INVALID FEN: {fen}")
        return {"error": f"Invalid FEN generated: {fen}", "labels": labels}


    # 5. Get best move — use persistent Stockfish
    try:
        sf = _get_stockfish()
        result = sf.get_best_move(fen)
    except Exception as e:
        logger.error(f"Stockfish error: {e}")
        logger.debug(traceback.format_exc())
        # Try to restart the engine
        global _stockfish
        try:
            if _stockfish:
                _stockfish.close()
        except Exception:
            pass
        _stockfish = None
        return {"error": f"Stockfish error: {e}", "labels": labels}

    move_str = result.get("move", "?")
    san_str = result.get("san", "?")
    score = result.get("score")
    mate = result.get("mate")

    score_str = f"M{mate}" if mate is not None else f"{score}cp"
    logger.info(f"Best move: {san_str} ({move_str})  eval: {score_str}")
    logger.info("=" * 60)

    return {
        "fen": fen,
        "move": move_str,
        "san": san_str,
        "score": score,
        "mate": mate,
        "confidence_avg": round(confidence_avg, 4),
        "labels": labels,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chess AI — analyze board and return best move")
    parser.add_argument("--color",   choices=["w", "b"], default="w",
                        help="Active color: w=white to move, b=black to move")
    parser.add_argument("--flipped", action="store_true",
                        help="Board is shown from Black's perspective")
    parser.add_argument("--platform", choices=["chess.com", "lichess"], default="chess.com",
                        help="Platform to target for board detection")
    args = parser.parse_args()

    try:
        result = analyze(active_color=args.color, flipped=args.flipped, platform=args.platform)
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)
