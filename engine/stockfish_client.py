"""
engine/stockfish_client.py
───────────────────────────
Thin wrapper around python-chess's Stockfish UCI engine.

The `stockfish` pip package auto-downloads the Stockfish binary on first use.
Alternatively, set STOCKFISH_PATH env var to point to a custom binary.

Usage:
    from engine.stockfish_client import StockfishClient
    sf = StockfishClient()
    move = sf.get_best_move("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
    print(move)   # e.g. "e7e5"
"""

import os
import sys
from pathlib import Path
import chess
import chess.engine


def _find_stockfish_path() -> str:
    """
    Resolve Stockfish binary path:
    1. STOCKFISH_PATH environment variable
    2. Check side-by-side in production (parent directory of sys.executable)
    3. Look in the local project root folder (for development)
    4. System PATH
    """
    # 1. Env var override
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        return env_path

    # 2. Check side-by-side in production
    if getattr(sys, 'frozen', False):
        production_sf = Path(sys.executable).parent / "stockfish.exe"
        if production_sf.exists():
            return str(production_sf)

    # 3. Look in the project root folder (for development)
    project_root = Path(__file__).parent.parent
    local_sf = project_root / "stockfish.exe"
    if local_sf.exists():
        return str(local_sf)

    # 4. System PATH
    return "stockfish"


class StockfishClient:
    def __init__(self, depth: int = 20, time_limit: float = 2.0):
        """
        Args:
            depth:      Search depth (higher = stronger / slower)
            time_limit: Max seconds per move
        """
        self.depth = depth
        self.time_limit = time_limit
        self._engine: chess.engine.SimpleEngine | None = None
        self._path = _find_stockfish_path()
        # Cache results by FEN so the same position always returns the same move.
        # This prevents the hint and coach feedback from showing different moves.
        self._cache: dict[str, dict] = {}
        self._CACHE_MAX = 128

    def _ensure_engine(self):
        if self._engine is None:
            self._engine = chess.engine.SimpleEngine.popen_uci(self._path)

    def get_best_move(self, fen: str) -> dict:
        """
        Analyze a FEN position and return the best move.
        Results are cached by FEN — repeated calls for the same position are instant
        and guaranteed to return the same move (no non-determinism between the hint
        button and the coach explanation).

        Returns:
            dict with keys:
                'move'  : UCI string  (e.g. 'e2e4')
                'san'   : SAN string  (e.g. 'e4')
                'score' : centipawn score (positive = white advantage)
                'mate'  : int or None — if not None, mate in N moves
        """
        # Return cached result if we've seen this position before
        if fen in self._cache:
            return self._cache[fen]

        self._ensure_engine()
        board = chess.Board(fen)

        limit = chess.engine.Limit(depth=self.depth, time=self.time_limit)
        info = self._engine.analyse(board, limit)

        best_move = info.get("pv", [None])[0]
        score_obj = info.get("score")

        cp_score = None
        mate = None
        if score_obj:
            pov = score_obj.white()
            if pov.is_mate():
                mate = pov.mate()
            else:
                cp_score = pov.score()

        if best_move is None:
            result = {"move": None, "san": None, "score": cp_score, "mate": mate}
        else:
            san = board.san(best_move)
            result = {
                "move": best_move.uci(),
                "san": san,
                "score": cp_score,
                "mate": mate,
            }

        # Evict oldest entry if cache is full
        if len(self._cache) >= self._CACHE_MAX:
            self._cache.pop(next(iter(self._cache)))
        self._cache[fen] = result
        return result


    def close(self):
        if self._engine:
            self._engine.quit()
            self._engine = None

    def __enter__(self):
        self._ensure_engine()
        return self

    def __exit__(self, *args):
        self.close()


if __name__ == "__main__":
    with StockfishClient(depth=15) as sf:
        result = sf.get_best_move("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        print(f"Best move: {result['move']} ({result['san']})")
        if result['score'] is not None:
            print(f"Score: {result['score']} cp")
        if result['mate'] is not None:
            print(f"Mate in: {result['mate']}")
