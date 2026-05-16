"""
train/generate_dataset.py
─────────────────────────
Generates a labeled dataset of 64×64px chess square crops from PGN game data.

Piece images are downloaded directly from Chess.com's CDN as PNG files —
the EXACT same pixels the CNN will see during inference. No SVG/Cairo needed.

Board colors match Chess.com's "Green" classic theme by default.

Each rendered board position is sliced into 64 squares and saved to:
    data/{class_name}/{uuid}.jpg

13 classes:
    empty, wP, wN, wB, wR, wQ, wK, bP, bN, bB, bR, bQ, bK

Usage:
    python -m train.generate_dataset --pgn games.pgn --output data/ --max-positions 5000
"""

import argparse
import io
import uuid
import random
from pathlib import Path

import chess
import chess.pgn
import requests
from PIL import Image, ImageEnhance
from tqdm import tqdm

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = [
    "empty",
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
]

PIECE_TO_CLASS = {
    (chess.PAWN,   chess.WHITE): "wP",
    (chess.KNIGHT, chess.WHITE): "wN",
    (chess.BISHOP, chess.WHITE): "wB",
    (chess.ROOK,   chess.WHITE): "wR",
    (chess.QUEEN,  chess.WHITE): "wQ",
    (chess.KING,   chess.WHITE): "wK",
    (chess.PAWN,   chess.BLACK): "bP",
    (chess.KNIGHT, chess.BLACK): "bN",
    (chess.BISHOP, chess.BLACK): "bB",
    (chess.ROOK,   chess.BLACK): "bR",
    (chess.QUEEN,  chess.BLACK): "bQ",
    (chess.KING,   chess.BLACK): "bK",
}

# Chess.com piece themes available on their CDN
PIECE_THEMES = ["neo", "alpha", "bases", "book", "classic", "club", "condal", "gothic", "icy_sea", "maya"]

def _get_piece_urls(theme: str) -> dict[str, str]:
    base = f"https://images.chesscomfiles.com/chess-themes/pieces/{theme}/150"
    return {
        "wP": f"{base}/wp.png", "wN": f"{base}/wn.png", "wB": f"{base}/wb.png",
        "wR": f"{base}/wr.png", "wQ": f"{base}/wq.png", "wK": f"{base}/wk.png",
        "bP": f"{base}/bp.png", "bN": f"{base}/bn.png", "bB": f"{base}/bb.png",
        "bR": f"{base}/br.png", "bQ": f"{base}/bq.png", "bK": f"{base}/bk.png",
    }

# Chess.com board color themes only (no lichess)
BOARD_THEMES = [
    ((238, 238, 210), (118, 150,  86)),  # Chess.com Green (default)
    ((240, 217, 181), (181, 136,  99)),  # Walnut / Classic Brown
    ((222, 227, 230), (140, 162, 173)),  # Blue
    ((234, 220, 230), (176, 139, 163)),  # Purple
    ((207, 207, 207), (140, 140, 140)),  # Newspaper / Grey
]

# Chess.com last-move highlight colors (applied as overlay to simulate board highlights)
HIGHLIGHT_COLORS = [
    (244, 246, 128, 100),  # Yellow-green (default)
    (255, 255,   0,  80),  # Bright yellow (some themes)
    (130, 151, 105, 110),  # Dark green
]

BOARD_PX  = 512
CELL_PX   = BOARD_PX // 8      # 64px per square at render time
SQUARE_PX = 64                  # Final output crop size

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ── Piece image cache ─────────────────────────────────────────────────────────

_PIECE_CACHE: dict[tuple[str, str], Image.Image] = {}


def _load_piece_images(cache_dir: Path):
    """Download and cache piece PNG images for all themes."""
    cache_dir.mkdir(parents=True, exist_ok=True)

    for theme in PIECE_THEMES:
        urls = _get_piece_urls(theme)
        for code, url in urls.items():
            if (theme, code) in _PIECE_CACHE:
                continue

            png_path = cache_dir / f"{theme}_{code}.png"

            if not png_path.exists():
                print(f"  Downloading {theme} {code}...", end=" ", flush=True)
                try:
                    resp = requests.get(url, headers=_HEADERS, timeout=15)
                    resp.raise_for_status()
                    with open(png_path, "wb") as f:
                        f.write(resp.content)
                    print("OK")
                except Exception as e:
                    print(f"FAILED ({e}) — using placeholder")
                    _make_placeholder(code, png_path)

            try:
                _PIECE_CACHE[(theme, code)] = Image.open(png_path).convert("RGBA")
            except Exception:
                _make_placeholder(code, png_path)
                _PIECE_CACHE[(theme, code)] = Image.open(png_path).convert("RGBA")


def _make_placeholder(code: str, out_path: Path):
    """Labeled circle fallback if download fails."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGBA", (CELL_PX, CELL_PX), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    fill = (255, 255, 255, 220) if code[0] == "w" else (40, 40, 40, 220)
    draw.ellipse([4, 4, CELL_PX - 4, CELL_PX - 4], fill=fill, outline=(0, 0, 0, 255), width=2)
    try:
        font = ImageFont.truetype("arial.ttf", 26)
    except Exception:
        font = ImageFont.load_default()
    text_color = (30, 30, 30, 255) if code[0] == "w" else (220, 220, 220, 255)
    bbox = draw.textbbox((0, 0), code[1], font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(((CELL_PX - tw) // 2, (CELL_PX - th) // 2 - 1), code[1],
              fill=text_color, font=font)
    img.save(out_path)


# ── Board rendering ───────────────────────────────────────────────────────────

def _draw_board(
    board: chess.Board,
    flipped: bool = False,
    piece_theme: str = "neo",
    board_theme: int = 0,
    highlight_squares: list[int] | None = None,
) -> Image.Image:
    """
    Render a chess.Board to a BOARD_PX×BOARD_PX PIL Image.
    Optionally highlights squares (list of chess.Square ints) to simulate
    chess.com's last-move indicator.
    """
    from PIL import ImageDraw
    img = Image.new("RGB", (BOARD_PX, BOARD_PX))
    draw = ImageDraw.Draw(img)

    light_color, dark_color = BOARD_THEMES[board_theme]

    for rank in range(8):
        for file in range(8):
            visual_row = (7 - rank) if not flipped else rank
            visual_col = file if not flipped else (7 - file)

            is_light = (rank + file) % 2 == 0
            sq_color = light_color if is_light else dark_color

            x0 = visual_col * CELL_PX
            y0 = visual_row * CELL_PX

            draw.rectangle([x0, y0, x0 + CELL_PX - 1, y0 + CELL_PX - 1], fill=sq_color)

            # Apply highlight overlay if this square is highlighted
            sq = chess.square(file, rank)
            if highlight_squares and sq in highlight_squares:
                hl_color = random.choice(HIGHLIGHT_COLORS)
                overlay = Image.new("RGBA", (CELL_PX, CELL_PX), hl_color)
                base = Image.new("RGBA", (CELL_PX, CELL_PX), sq_color + (255,))
                base.paste(overlay, (0, 0), overlay)
                img.paste(base.convert("RGB"), (x0, y0))
                sq_color = base.convert("RGB").getpixel((CELL_PX // 2, CELL_PX // 2))

            piece = board.piece_at(sq)
            if piece is not None:
                code = PIECE_TO_CLASS[(piece.piece_type, piece.color)]
                if (piece_theme, code) in _PIECE_CACHE:
                    piece_img = _PIECE_CACHE[(piece_theme, code)].resize((CELL_PX, CELL_PX), Image.LANCZOS)
                    bg = Image.new("RGBA", (CELL_PX, CELL_PX), sq_color + (255,))
                    bg.paste(piece_img, (0, 0), piece_img)
                    img.paste(bg.convert("RGB"), (x0, y0))

    return img


def _get_labels(board: chess.Board, flipped: bool = False) -> list[str]:
    """Return 64 class labels in visual order (top-left → bottom-right)."""
    labels = []
    for visual_row in range(8):
        for visual_col in range(8):
            if not flipped:
                rank = 7 - visual_row
                file = visual_col
            else:
                rank = visual_row
                file = 7 - visual_col
            sq = chess.square(file, rank)
            piece = board.piece_at(sq)
            labels.append("empty" if piece is None
                          else PIECE_TO_CLASS[(piece.piece_type, piece.color)])
    return labels


def _slice_board(img: Image.Image) -> list[Image.Image]:
    """Slice BOARD_PX×BOARD_PX image into 64 SQUARE_PX×SQUARE_PX crops."""
    cell = img.width // 8
    crops = []
    for row in range(8):
        for col in range(8):
            x0, y0 = col * cell, row * cell
            crop = img.crop((x0, y0, x0 + cell, y0 + cell))
            crop = crop.resize((SQUARE_PX, SQUARE_PX), Image.LANCZOS)
            crops.append(crop)
    return crops


# ── Augmentation ─────────────────────────────────────────────────────────────

def _augment(img: Image.Image) -> Image.Image:
    """Mild augmentation: brightness, contrast jitter + JPEG compression."""
    img = ImageEnhance.Brightness(img).enhance(random.uniform(0.80, 1.20))
    img = ImageEnhance.Contrast(img).enhance(random.uniform(0.85, 1.15))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=random.randint(75, 100))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ── Main ─────────────────────────────────────────────────────────────────────

def generate(pgn_path: str, output_dir: str, max_positions: int, augment_count: int,
             endgame_bias: float = 0.7, king_queen_boost: int = 3):
    """
    Generate a chess.com-specific labeled dataset.

    Args:
        endgame_bias:     Fraction of samples drawn from the LAST third of each game (0-1).
                          0 = uniform random sampling, 1 = only endgame positions.
        king_queen_boost: Extra augmented copies saved for every King/Queen square crop.
                          Combats the K/Q visual confusion that causes INVALID FEN errors.
    """
    output = Path(output_dir)
    cache_dir = output / ".piece_cache"

    print("Loading piece images for chess.com themes...")
    _load_piece_images(cache_dir)
    print("  All piece types loaded successfully.")

    for cls in CLASSES:
        (output / cls).mkdir(parents=True, exist_ok=True)

    counts = {cls: 0 for cls in CLASSES}
    total_positions = 0

    with open(pgn_path, encoding="utf-8", errors="ignore") as f:
        pbar = tqdm(total=max_positions, desc="Positions processed")
        while total_positions < max_positions:
            game = chess.pgn.read_game(f)
            if game is None:
                break

            board = game.board()
            positions = [board.copy()]
            for move in game.mainline_moves():
                board.push(move)
                positions.append(board.copy())

            if len(positions) < 4:
                continue

            # --- Endgame-biased sampling ---
            # Split the game into early and late halves.
            # Draw `endgame_bias` fraction of samples from the last third.
            cutoff = max(1, int(len(positions) * 0.65))  # last 35% = endgame
            early  = positions[:cutoff]
            late   = positions[cutoff:]

            n_total = min(10, len(positions))
            n_late  = max(1, int(n_total * endgame_bias)) if late else 0
            n_early = n_total - n_late

            sampled = []
            if n_late > 0 and late:
                sampled += random.sample(late,  min(n_late,  len(late)))
            if n_early > 0 and early:
                sampled += random.sample(early, min(n_early, len(early)))

            for pos in sampled:
                if total_positions >= max_positions:
                    break
                flipped = random.random() < 0.5
                piece_theme = random.choice(PIECE_THEMES)
                board_theme = random.randrange(len(BOARD_THEMES))

                # Simulate chess.com's last-move highlight on 0-2 random squares
                highlight = None
                if random.random() < 0.6 and pos.move_stack:
                    last = pos.peek()
                    highlight = [last.from_square, last.to_square]

                img    = _draw_board(pos, flipped=flipped, piece_theme=piece_theme,
                                     board_theme=board_theme, highlight_squares=highlight)
                crops  = _slice_board(img)
                labels = _get_labels(pos, flipped=flipped)

                for crop, label in zip(crops, labels):
                    # Standard augmented copies
                    for _ in range(augment_count):
                        aug = _augment(crop)
                        fname = output / label / f"{uuid.uuid4().hex}.jpg"
                        aug.save(fname, format="JPEG", quality=95)
                        counts[label] += 1

                    # Extra copies for Kings and Queens to combat K/Q confusion
                    if label in ("wK", "bK", "wQ", "bQ") and king_queen_boost > 0:
                        for _ in range(king_queen_boost):
                            aug = _augment(crop)
                            fname = output / label / f"{uuid.uuid4().hex}.jpg"
                            aug.save(fname, format="JPEG", quality=95)
                            counts[label] += 1

                total_positions += 1
                pbar.update(1)
        pbar.close()

    print("\n── Dataset Summary ──")
    for cls in CLASSES:
        print(f"  {cls:6s}: {counts[cls]:,}")
    print(f"\n  Total: {sum(counts.values()):,} crops from {total_positions} positions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate chess.com-specific training data")
    parser.add_argument("--pgn",              required=True,  help="Path to .pgn file with games")
    parser.add_argument("--output",           default="data", help="Output data directory")
    parser.add_argument("--max-positions",    type=int,   default=5000)
    parser.add_argument("--augment-count",    type=int,   default=3)
    parser.add_argument("--endgame-bias",     type=float, default=0.7,
                        help="Fraction of samples from endgame (last 35%% of game). Default 0.7")
    parser.add_argument("--king-queen-boost", type=int,   default=3,
                        help="Extra K/Q crop copies per position to combat K/Q confusion")
    args = parser.parse_args()
    generate(args.pgn, args.output, args.max_positions, args.augment_count,
             endgame_bias=args.endgame_bias, king_queen_boost=args.king_queen_boost)
