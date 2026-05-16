"""
engine/fen_builder.py
──────────────────────
Convert 64 square class labels into a valid FEN string.

FEN format:  <piece placement>/<rank7>/.../<rank1> <active color> <castling> <en passant> <halfmove> <fullmove>

Example:
    labels = ['empty', 'empty', ..., 'wK', ..., 'bK', ...]   # 64 strings
    fen = build_fen(labels, active_color='w')
    # → 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'

Usage:
    from engine.fen_builder import build_fen
"""

# Maps class label → FEN character
LABEL_TO_FEN = {
    "empty": None,
    "wP": "P", "wN": "N", "wB": "B", "wR": "R", "wQ": "Q", "wK": "K",
    "bP": "p", "bN": "n", "bB": "b", "bR": "r", "bQ": "q", "bK": "k",
}


def build_fen(
    labels: list[str],
    active_color: str = "w",
    flipped: bool = False,
) -> str:
    """
    Build a FEN string from 64 class labels.

    Args:
        labels:       64 class labels in visual order (row 0 = top of screen).
        active_color: 'w' or 'b' — whose turn it is.
        flipped:      True if the board was captured from Black's perspective.

    Returns:
        FEN position string (piece placement + active color + castling + etc).

    Note:
        Castling rights are inferred from whether kings and rooks are on their
        starting squares. En passant cannot be inferred from a single image.
    """
    if len(labels) != 64:
        raise ValueError(f"Expected 64 labels, got {len(labels)}")

    # Convert visual labels to a board[rank][file] grid
    # board[0] = rank 8 (top), board[7] = rank 1 (bottom)
    if not flipped:
        # Normal orientation: labels[0]=a8, labels[7]=h8, ..., labels[63]=h1
        board = []
        for row in range(8):
            rank = []
            for col in range(8):
                rank.append(labels[row * 8 + col])
            board.append(rank)
    else:
        # Flipped: labels[0]=h1, labels[7]=a1, ..., labels[63]=a8
        # We need to reverse both row and column order
        board = []
        for row in range(7, -1, -1):
            rank = []
            for col in range(7, -1, -1):
                rank.append(labels[row * 8 + col])
            board.append(rank)

    # Build piece placement string (rank 8 first, rank 1 last)
    ranks = []
    for rank_labels in board:
        empty_count = 0
        rank_str = ""
        for label in rank_labels:
            ch = LABEL_TO_FEN.get(label)
            if ch is None:
                empty_count += 1
            else:
                if empty_count:
                    rank_str += str(empty_count)
                    empty_count = 0
                rank_str += ch
        if empty_count:
            rank_str += str(empty_count)
        ranks.append(rank_str)

    piece_placement = "/".join(ranks)

    # Infer castling rights from piece positions
    # board[7] = rank 1 (white's back rank), board[0] = rank 8 (black's back rank)
    castling = ""
    # White kingside: king on e1 (board[7][4]) and rook on h1 (board[7][7])
    if board[7][4] == "wK" and board[7][7] == "wR":
        castling += "K"
    # White queenside: king on e1 and rook on a1 (board[7][0])
    if board[7][4] == "wK" and board[7][0] == "wR":
        castling += "Q"
    # Black kingside: king on e8 (board[0][4]) and rook on h8 (board[0][7])
    if board[0][4] == "bK" and board[0][7] == "bR":
        castling += "k"
    # Black queenside: king on e8 and rook on a8 (board[0][0])
    if board[0][4] == "bK" and board[0][0] == "bR":
        castling += "q"

    if not castling:
        castling = "-"

    fen = f"{piece_placement} {active_color} {castling} - 0 1"
    return fen


def repair_labels(labels: list[str], confidences: list[float]) -> tuple[list[str], list[str]]:
    """
    Sanity-check and repair the 64 square labels to ensure exactly one king
    per side exists. The most common vision error is K/Q confusion.

    Strategy:
      - Missing king + has queen(s) → promote the highest-confidence queen to king.
      - Extra kings (>1) → demote the lowest-confidence extras back to queens.
      - Missing king + no queens → board is unrecoverable; return as-is with a warning.

    Returns:
        (repaired_labels, repair_log) — modified label list and list of log messages.
    """
    labels = list(labels)  # don't mutate original
    log = []

    for color, king_label, queen_label in [
        ("White", "wK", "wQ"),
        ("Black", "bK", "bQ"),
    ]:
        king_indices = [i for i, l in enumerate(labels) if l == king_label]
        queen_indices = [i for i, l in enumerate(labels) if l == queen_label]

        if len(king_indices) == 1:
            continue  # perfect

        if len(king_indices) == 0:
            # Missing king — try to promote the highest-confidence queen
            if queen_indices:
                best_q = max(queen_indices, key=lambda i: confidences[i])
                labels[best_q] = king_label
                msg = (f"[Repair] {color} king missing — promoted {queen_label} "
                       f"at index {best_q} (conf={confidences[best_q]:.3f}) to {king_label}")
                log.append(msg)
            else:
                log.append(f"[Repair] WARNING: {color} king missing and no queens to promote — FEN may still be invalid")

        elif len(king_indices) > 1:
            # Too many kings — keep the highest-confidence one, demote the rest
            best_k = max(king_indices, key=lambda i: confidences[i])
            extras = [i for i in king_indices if i != best_k]
            for idx in extras:
                labels[idx] = queen_label
                msg = (f"[Repair] {color} has {len(king_indices)} kings — demoted "
                       f"index {idx} (conf={confidences[idx]:.3f}) to {queen_label}")
                log.append(msg)

    return labels, log


def validate_fen(fen: str) -> bool:
    """Quick sanity check that the FEN parses correctly and is a legal chess position."""
    import chess
    try:
        board = chess.Board(fen)
        return board.is_valid()
    except ValueError:
        return False
