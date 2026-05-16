from pathlib import Path
from train.generate_dataset import _load_piece_images, _draw_board, _slice_board, _get_labels, _PIECE_CACHE
import chess

print("Downloading piece images...")
Path("data").mkdir(exist_ok=True)
_load_piece_images(Path("data/.piece_cache"))
print(f"Pieces loaded: {list(_PIECE_CACHE.keys())}")

board = chess.Board()
img = _draw_board(board, flipped=False)
img.save("data/test_board.png")
print(f"Board rendered: {img.size}")

crops = _slice_board(img)
labels = _get_labels(board)
print(f"Crops: {len(crops)}, Labels sample: {labels[:8]}")
print("Smoke test PASSED")
