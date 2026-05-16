"""Test Chess.com CDN piece download."""
import requests
from pathlib import Path
from PIL import Image
import io

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.chess.com/",
}

urls_to_try = [
    ("neo/150", "https://images.chesscomfiles.com/chess-themes/pieces/neo/150/wp.png"),
    ("neo/200", "https://images.chesscomfiles.com/chess-themes/pieces/neo/200/wp.png"),
    ("classic/150", "https://images.chesscomfiles.com/chess-themes/pieces/classic/150/wp.png"),
    ("wood/150", "https://images.chesscomfiles.com/chess-themes/pieces/wood/150/wp.png"),
    ("lolz", "https://images.chesscomfiles.com/chess-themes/pieces/lolz/150/wp.png"),
]

for label, url in urls_to_try:
    try:
        r = requests.get(url, headers=headers, timeout=10)
        size = len(r.content)
        ct = r.headers.get("Content-Type", "?")
        print(f"  [{label}] status={r.status_code} size={size}B type={ct}")
        if r.status_code == 200 and size > 1000:
            img = Image.open(io.BytesIO(r.content))
            img.save(f"data/.piece_cache/test_wp_{label.replace('/', '_')}.png")
            print(f"    -> Saved! Size: {img.size}")
            break
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
