"""
vision/square_slicer.py
────────────────────────
Slice a board image into 64 square crops (one per chess square).

Input:  PIL Image of the board (any size, assumed square).
Output: List of 64 PIL Images, 64×64px each, in visual order:
        index 0 = top-left square (a8 when white is at bottom),
        index 63 = bottom-right square (h1).

Usage:
    from vision.square_slicer import slice_board
    crops = slice_board(board_pil_image)   # → list of 64 PIL Images
"""

from PIL import Image

SQUARE_SIZE = 64   # Output size for each crop


def slice_board(board_img: Image.Image) -> list[Image.Image]:
    """
    Divide a square board image into 64 cell crops.

    Args:
        board_img: PIL Image of the full board. Should be square.

    Returns:
        List of 64 PIL Images (64×64px), row-major left-to-right top-to-bottom.
    """
    w, h = board_img.size
    # Snap to square in case of minor asymmetry
    side = min(w, h)
    board_img = board_img.crop((0, 0, side, side))

    cell = side // 8
    crops = []
    for row in range(8):
        for col in range(8):
            x0 = col * cell
            y0 = row * cell
            crop = board_img.crop((x0, y0, x0 + cell, y0 + cell))
            crop = crop.resize((SQUARE_SIZE, SQUARE_SIZE), Image.LANCZOS)
            crops.append(crop)
    return crops


def square_index(row: int, col: int) -> int:
    """Convert (row, col) → flat index 0–63."""
    return row * 8 + col


def index_to_square_name(idx: int, flipped: bool = False) -> str:
    """
    Convert flat index 0–63 to algebraic square name (e.g., 'a8', 'h1').

    Args:
        idx:     0 = top-left cell, 63 = bottom-right cell
        flipped: True if black is at bottom (board shown from black's perspective)
    """
    row = idx // 8
    col = idx % 8
    if not flipped:
        # Standard: top row = rank 8, left col = file a
        rank = 8 - row
        file = col
    else:
        rank = row + 1
        file = 7 - col
    return "abcdefgh"[file] + str(rank)
