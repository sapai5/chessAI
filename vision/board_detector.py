"""
vision/board_detector.py
─────────────────────────
Detect the Chess.com board region from a screenshot.

Strategy:
  1. Take a full screenshot (or receive an image path)
  2. Scan for the Chess.com classic board colors:
       Light squares: #f0d9b5  →  RGB(240, 217, 181)
       Dark  squares: #b58863  →  RGB(181, 136,  99)
  3. Find the tight bounding box of the board region
  4. Snap it to a perfect square and return (x, y, size) in screen coords

Usage:
    from vision.board_detector import detect_board, screenshot_board
    x, y, size = detect_board()           # auto screenshot
    board_img  = screenshot_board()       # returns cropped PIL Image
"""

import numpy as np
import cv2
import mss
from PIL import Image


# ── Primary board color targets (Chess.com Walnut / classic, in BGR for OpenCV) ─
# These exactly match chess.com's default board. Only fall back to green if needed.
CHESS_COM_THEMES = [
    # Walnut (most common default)
    {"light": np.array([181, 217, 240], dtype=np.uint8), "dark": np.array([99, 136, 181], dtype=np.uint8)},
    # Green (second most common)
    {"light": np.array([210, 238, 238], dtype=np.uint8), "dark": np.array([86, 150, 118], dtype=np.uint8)},
]

LICHESS_THEMES = [
    {"light": np.array([232, 232, 232], dtype=np.uint8), "dark": np.array([106, 178, 147], dtype=np.uint8)},
    {"light": np.array([181, 217, 240], dtype=np.uint8), "dark": np.array([99, 136, 181], dtype=np.uint8)},
]

COLOR_TOLERANCE = 30   # ±tolerance in each channel


def _color_mask(img_bgr: np.ndarray, target_bgr: np.ndarray, tol: int) -> np.ndarray:
    """Return binary mask where pixels are within `tol` of target color."""
    diff = np.abs(img_bgr.astype(np.int32) - target_bgr.astype(np.int32))
    return (diff.max(axis=2) < tol).astype(np.uint8) * 255


def _try_detect(screenshot: np.ndarray, theme: dict) -> tuple[int, int, int] | None:
    """Try to detect a board using one specific color theme. Returns (x,y,size) or None."""
    mask_light = _color_mask(screenshot, theme["light"], COLOR_TOLERANCE)
    mask_dark  = _color_mask(screenshot, theme["dark"],  COLOR_TOLERANCE)
    mask = cv2.bitwise_or(mask_light, mask_dark)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    best_area = 0
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / h if h > 0 else 0
        area = w * h
        if 0.85 < aspect < 1.15 and min(w, h) > 200 and area > best_area:
            best = (x, y, w, h)
            best_area = area

    if best is None:
        return None
    x, y, w, h = best
    return x, y, min(w, h)


def detect_board(screenshot: np.ndarray | None = None, platform: str = "chess.com") -> tuple[int, int, int]:
    """
    Detect the chess board in a screenshot based on the selected platform.

    Args:
        screenshot: BGR numpy array. If None, captures the primary monitor.
        platform: 'chess.com' or 'lichess'

    Returns:
        (x, y, size) — top-left corner and side length of the board in pixels.

    Raises:
        RuntimeError if no board region is found.
    """
    if screenshot is None:
        screenshot = _take_screenshot()

    themes = CHESS_COM_THEMES if platform == "chess.com" else LICHESS_THEMES

    # Try each theme in priority order — stop at the first successful detection.
    # This avoids false positives from matching too many color ranges.
    for theme in themes:
        result = _try_detect(screenshot, theme)
        if result is not None:
            return result

    raise RuntimeError(f"Could not isolate board rectangle for {platform}. Is the board visible and using a supported theme?")




def screenshot_board(monitor_index: int = 1, platform: str = "chess.com") -> Image.Image:
    """
    Take a screenshot, detect the board, and return a cropped PIL Image
    of exactly the board region.
    """
    raw = _take_screenshot(monitor_index)
    x, y, size = detect_board(raw, platform=platform)
    # Crop using PIL for clean resize
    pil = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_BGR2RGB))
    board = pil.crop((x, y, x + size, y + size))
    return board


def _take_screenshot(monitor_index: int = 1) -> np.ndarray:
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        raw = sct.grab(monitor)
    img = np.array(raw)                          # BGRA
    return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


# ── Debug helper ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    print("Taking screenshot and detecting board...")
    try:
        raw = _take_screenshot()
        x, y, size = detect_board(raw)
        print(f"Board detected at x={x}, y={y}, size={size}px")

        # Draw rectangle on screenshot for visual confirmation
        debug = raw.copy()
        cv2.rectangle(debug, (x, y), (x + size, y + size), (0, 255, 0), 3)
        out = "vision/debug_detection.png"
        cv2.imwrite(out, debug)
        print(f"Debug image saved to {out}")
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
