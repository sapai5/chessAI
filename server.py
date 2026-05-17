import os
import sys
sys.stdout.reconfigure(line_buffering=True)  # flush print() immediately when running as subprocess

# Custom zero-dependency .env loader to read GEMINI_API_KEY from the project root
if getattr(sys, 'frozen', False):
    app_dir = os.path.dirname(sys.executable)
else:
    app_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(app_dir, ".env")
if os.path.exists(env_path):
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    os.environ[k] = v
        print(f"[Coach] Loaded environment variables from .env")
    except Exception as e:
        print(f"[Coach] Error loading .env: {e}")

import logging
import urllib.request
import json
import chess
from flask import Flask, request, jsonify
from flask_cors import CORS

from engine.engine import analyze

def move_to_english(board: chess.Board, move: chess.Move) -> str:
    if not move:
        return "unknown move"
    
    if board.is_castling(move):
        from_file = chess.square_file(move.from_square)
        to_file = chess.square_file(move.to_square)
        if to_file > from_file:
            return "castle kingside"
        else:
            return "castle queenside"
            
    piece = board.piece_at(move.from_square)
    if not piece:
        return "unknown move"
        
    piece_names = {
        chess.PAWN: "pawn",
        chess.KNIGHT: "knight",
        chess.BISHOP: "bishop",
        chess.ROOK: "rook",
        chess.QUEEN: "queen",
        chess.KING: "king"
    }
    piece_name = piece_names.get(piece.piece_type, "piece")
    to_sq_name = chess.square_name(move.to_square)
    
    if board.is_capture(move):
        desc = f"{piece_name} takes on {to_sq_name}"
    else:
        desc = f"{piece_name} to {to_sq_name}"
        
    if move.promotion:
        promo_piece = piece_names.get(move.promotion, "queen")
        desc += f" promoting to {promo_piece}"
        
    return desc

def san_to_english_string(san: str) -> str:
    if not san or san == "unknown":
        return "unknown move"
        
    san = san.strip()
    if san in ["O-O", "O-O+", "O-O#"]:
        return "castle kingside"
    if san in ["O-O-O", "O-O-O+", "O-O-O#"]:
        return "castle queenside"
        
    # Strip check (+), checkmate (#), promote (=Q), and comments (!, ?)
    clean_san = san.replace("+", "").replace("#", "").split("=")[0].rstrip("!?")
    if not clean_san:
        return "unknown move"
    
    # Piece map
    piece_map = {
        'K': 'king',
        'Q': 'queen',
        'R': 'rook',
        'B': 'bishop',
        'N': 'knight'
    }
    
    # Check if first character is a piece
    first = clean_san[0]
    if first in piece_map:
        piece_name = piece_map[first]
    else:
        piece_name = 'pawn'
        
    # Destination square is always the last 2 characters of clean_san
    if len(clean_san) >= 2:
        dest_square = clean_san[-2:]
    else:
        dest_square = clean_san
        
    if "x" in clean_san:
        return f"{piece_name} takes on {dest_square}"
    else:
        return f"{piece_name} to {dest_square}"

# Global state for the coach session
game_history = []  # List of SAN moves played
eval_history = []  # List of (move_san, eval_cp) tracking positional trend
current_board = chess.Board()

# Disable Flask's default request logging for cleaner output
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)  # Allow our Electron frontend to call the API

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check so the UI knows the Python backend is up."""
    return jsonify({"status": "ok"})

@app.route('/analyze', methods=['POST'])
def analyze_endpoint():
    """
    Main endpoint for the UI to request an analysis.
    Expects JSON: { "color": "w"|"b", "flipped": true|false }
    """
    try:
        data = request.get_json() or {}
        color = data.get("color", "w")
        flipped = data.get("flipped", False)
        platform = data.get("platform", "chess.com")
        
        # Call the engine analysis pipeline
        result = analyze(active_color=color, flipped=flipped, platform=platform)
        
        # If the engine returns an error (e.g. board not found)
        if "error" in result:
            return jsonify({"status": "error", "message": result["error"]}), 400
            
        # Return success with the FEN, move, score, etc.
        return jsonify({
            "status": "success",
            "data": result
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

def call_gemini_api(prompt: str, api_key: str) -> str:
    """
    Standard library implementation to call Gemini 1.5 Flash API with zero dependencies.
    """
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 150,
            "temperature": 0.5
        }
    }
    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=10) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception as e:
        print(f"[Coach] Gemini API call failed: {e}")
    return ""

def set_fen_turn(fen: str, turn: str) -> str:
    if not fen:
        return fen
    parts = fen.split(" ")
    if len(parts) >= 2:
        parts[1] = turn
        return " ".join(parts)
    return fen

@app.route('/coach_explanation', methods=['POST'])
def coach_explanation():
    """
    Endpoint for getting natural language feedback from a local Ollama model.
    Expects JSON: { "fen_before": "...", "fen_after": "...", "best_move": "..." }
    """
    global game_history, eval_history, current_board
    try:
        data = request.get_json() or {}
        fen_before = data.get("fen_before")
        fen_after = data.get("fen_after")
        best_move = data.get("best_move")

        player_color = data.get("player_color", "w")

        print(f"\n[Coach] ====== NEW COACH REQUEST ======")
        print(f"[Coach] Player color: {player_color}")
        print(f"[Coach] FEN before : {fen_before}")
        print(f"[Coach] FEN after  : {fen_after}")
        print(f"[Coach] Best move  : {best_move}")

        if not fen_before or not fen_after:
            print(f"[Coach] ERROR: Missing required fields")
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

        best_move = best_move or "unknown"

        # Check if the game is over (checkmate, stalemate, draw)
        try:
            check_board = chess.Board(fen_after)
            if check_board.is_checkmate():
                if check_board.turn == (chess.WHITE if player_color == 'w' else chess.BLACK):
                    explanation = "Tough game! You fought hard, but your opponent has delivered checkmate on this turn. Let's reset the board and analyze where we can strengthen our play for the next round!"
                else:
                    explanation = "Incredible job! You've successfully delivered checkmate and won the game. That was an outstanding victory, well played!"
                
                print(f"[Coach] Game Over (Checkmate) explanation: {explanation}")
                return jsonify({
                    "status": "success",
                    "explanation": explanation,
                    "played_move": "unknown",
                    "move_quality": "game_over",
                    "history": game_history
                })
            elif check_board.is_stalemate():
                explanation = "The game has ended in a stalemate! It's a draw, with neither side able to make a legal move. Excellent resilience in defending your position!"
                print(f"[Coach] Game Over (Stalemate) explanation: {explanation}")
                return jsonify({
                    "status": "success",
                    "explanation": explanation,
                    "played_move": "unknown",
                    "move_quality": "game_over",
                    "history": game_history
                })
            elif check_board.is_insufficient_material():
                explanation = "The game is a draw due to insufficient mating material. Neither side has enough pieces left to force checkmate. Well played!"
                print(f"[Coach] Game Over (Draw) explanation: {explanation}")
                return jsonify({
                    "status": "success",
                    "explanation": explanation,
                    "played_move": "unknown",
                    "move_quality": "game_over",
                    "history": game_history
                })
        except Exception as e:
            print(f"[Coach] Game over check error: {e}")

        # 1. Determine what move was actually played and update history
        played_move_san = "unknown"
        try:
            if current_board.fen() != fen_before:
                print(f"[Coach] Board state mismatch — resyncing to FEN before")
                current_board = chess.Board(fen_before)

            # Force the board turn to match the player's color
            # (Vision system may not set this correctly)
            current_board.turn = chess.WHITE if player_color == 'w' else chess.BLACK
            print(f"[Coach] Forced board turn to: {'White' if current_board.turn == chess.WHITE else 'Black'}")

            target_placement = fen_after.split()[0]
            for move in current_board.legal_moves:
                move_san = current_board.san(move)
                current_board.push(move)
                if current_board.board_fen() == target_placement:
                    played_move_san = move_san
                    game_history.append(played_move_san)
                    print(f"[Coach] Detected played move : {played_move_san}")
                    break
                current_board.pop()

            if played_move_san == "unknown":
                print(f"[Coach] WARNING: Could not match any legal move to the board diff")
        except Exception as e:
            print(f"[Coach] ERROR detecting move: {e}")

        print(f"[Coach] Game history ({len(game_history)} moves): {', '.join(game_history)}")

        # ── Step 2: Stockfish evaluation of BOTH positions ──────────────────────
        # We evaluate fen_before to get the baseline, then compute what changed.
        # This gives us ground-truth facts Qwen can narrate without guessing.
        cp_before = None
        cp_after  = None
        mate_after = None
        cp_loss = None
        move_quality = "unknown"

        try:
            from engine.engine import _get_stockfish
            sf = _get_stockfish()

            # Correct active turn in both FENs to ensure Stockfish assesses the moves correctly
            opponent_color = 'b' if player_color == 'w' else 'w'
            fen_before_corrected = set_fen_turn(fen_before, player_color)
            
            if played_move_san != "unknown":
                fen_after_corrected = set_fen_turn(fen_after, opponent_color)
            else:
                fen_after_corrected = set_fen_turn(fen_after, player_color)

            # Eval before (from the perspective of the player who moved)
            r_before = sf.get_best_move(fen_before_corrected)
            r_after  = sf.get_best_move(fen_after_corrected)

            if player_color == 'w':
                cp_before = r_before.get("score")   # positive = good for White
                cp_after  = r_after.get("score")
                mate_after = r_after.get("mate")
            else:
                s_b = r_before.get("score")
                s_a = r_after.get("score")
                cp_before = -s_b if s_b is not None else None
                cp_after  = -s_a if s_a is not None else None
                mate_after = r_after.get("mate")
                if mate_after is not None:
                    mate_after = -mate_after

            if cp_before is not None and cp_after is not None:
                cp_loss = cp_after - cp_before  # positive = improved, negative = worsened
                if   cp_loss >=  50: move_quality = "excellent"
                elif cp_loss >=   0: move_quality = "good"
                elif cp_loss >= -50: move_quality = "inaccuracy"
                elif cp_loss >= -150: move_quality = "mistake"
                else:                move_quality = "blunder"

            # Dynamically resolve the true best move to eliminate UI lag/desync
            if played_move_san != "unknown":
                best_move = r_before.get("san") or best_move
            else:
                best_move = r_after.get("san") or best_move

            print(f"[Coach] Eval before: {cp_before}cp  after: {cp_after}cp  loss: {cp_loss}  quality: {move_quality}  true_best: {best_move}")
        except Exception as e:
            print(f"[Coach] Stockfish eval error: {e}")

        # ── Step 3: python-chess tactical facts ──────────────────────────────────
        captured_piece_name = None
        gives_check = False
        played_move_obj = None

        if played_move_san != "unknown":
            try:
                probe = chess.Board(fen_before)
                probe.turn = chess.WHITE if player_color == 'w' else chess.BLACK
                for move in probe.legal_moves:
                    if probe.san(move) == played_move_san:
                        played_move_obj = move
                        break
                if played_move_obj:
                    if probe.is_capture(played_move_obj):
                        cap_sq = played_move_obj.to_square
                        cap_piece = probe.piece_at(cap_sq)
                        if cap_piece:
                            captured_piece_name = chess.piece_name(cap_piece.piece_type)
                    probe.push(played_move_obj)
                    gives_check = probe.is_check()
            except Exception as e:
                print(f"[Coach] Tactical analysis error: {e}")

        # ── Step 4: Format eval string for display ───────────────────────────────
        if mate_after is not None:
            if mate_after > 0:
                eval_str = f"Mate in {mate_after} for {'White' if player_color == 'w' else 'Black'}"
                eval_friendly = f"White has mate in {mate_after}" if player_color == 'w' else f"Black has mate in {mate_after}"
            else:
                eval_str = f"Mate in {abs(mate_after)} for {'Black' if player_color == 'w' else 'White'}"
                eval_friendly = f"Black has mate in {abs(mate_after)}" if player_color == 'w' else f"White has mate in {abs(mate_after)}"
        elif cp_after is not None:
            sign = "+" if cp_after >= 0 else ""
            eval_str = f"{sign}{cp_after/100:.1f} ({'White' if cp_after >= 0 else 'Black'} is {'winning' if abs(cp_after) > 200 else 'better' if abs(cp_after) > 50 else 'roughly equal'})"
            eval_friendly = f"{'White' if cp_after >= 0 else 'Black'} is {'winning' if abs(cp_after) > 200 else 'better' if abs(cp_after) > 50 else 'roughly equal'}"
        else:
            eval_str = "unknown"
            eval_friendly = "unknown"

        # ── Step 5: Record eval history & build context ──────────────────────────
        player_side = "White" if player_color == 'w' else "Black"
        move_num = len(game_history)

        # Track eval per move so we can describe the trend to Qwen
        if cp_after is not None:
            eval_history.append((played_move_san, cp_after))
            if len(eval_history) > 15:
                eval_history.pop(0)

        # Game phase based on move count
        if move_num < 10:
            game_phase = "opening"
        elif move_num < 25:
            game_phase = "middlegame"
        else:
            game_phase = "endgame"

        # Eval trend over last 4 moves
        eval_trend_str = ""
        if len(eval_history) >= 3:
            recent_evals = [e for _, e in eval_history[-4:]]
            # From player's perspective
            if player_color == 'b':
                recent_evals = [-e for e in recent_evals]
            trend = recent_evals[-1] - recent_evals[0]
            if trend > 80:
                eval_trend_str = f"{player_side} has been gaining ground over the last few moves."
            elif trend < -80:
                eval_trend_str = f"{player_side} has been losing ground over the last few moves."
            else:
                eval_trend_str = "The position has been fairly stable recently."

        # Format recent game history (last 8 moves, paired as White/Black)
        history_context = ""
        if game_history:
            recent = game_history[-8:]
            # Pair moves into White/Black turns
            pairs = []
            start_move_num = (move_num - len(recent)) // 2 + 1
            i = 0
            turn = start_move_num
            while i < len(recent):
                w_move = recent[i] if i < len(recent) else "..."
                b_move = recent[i + 1] if i + 1 < len(recent) else "..."
                pairs.append(f"{turn}. {w_move} {b_move}")
                i += 2
                turn += 1
            history_context = "Recent moves: " + "  ".join(pairs)
        else:
            history_context = "Game just started (no moves recorded yet)."

        # Convert best_move and played_move to plain English descriptions
        best_move_english = best_move
        try:
            temp_board = chess.Board(fen_after)
            temp_board.turn = chess.WHITE if player_color == 'w' else chess.BLACK
            best_move_obj = temp_board.parse_san(best_move)
            best_move_english = move_to_english(temp_board, best_move_obj)
        except Exception as e:
            print(f"[Coach] Error parsing best_move to English: {e}")
            best_move_english = san_to_english_string(best_move)

        played_move_english = "unknown move"
        if played_move_san != "unknown":
            try:
                temp_board = chess.Board(fen_before)
                temp_board.turn = chess.WHITE if player_color == 'w' else chess.BLACK
                played_move_obj = temp_board.parse_san(played_move_san)
                played_move_english = move_to_english(temp_board, played_move_obj)
            except Exception as e:
                print(f"[Coach] Error parsing played_move to English: {e}")
                played_move_english = san_to_english_string(played_move_san)

        facts = []
        if played_move_san != "unknown":
            facts.append(f"Move played by {player_side}: {played_move_english}")
        if captured_piece_name:
            facts.append(f"This move captured a {captured_piece_name}")
        if gives_check:
            facts.append("This move gives check")
        facts.append(f"Move quality: {move_quality}")
        facts.append(f"Best move: {best_move_english}")
        facts.append(f"Current evaluation after the move: {eval_friendly}")
        if move_num > 0:
            facts.append(f"Move number in game: {move_num} ({game_phase})")
        if eval_trend_str:
            facts.append(f"Positional trend: {eval_trend_str}")

        facts_block = "\n".join(f"- {f}" for f in facts)

        # Determine the piece type to make explanations highly accurate
        piece_guideline = ""
        try:
            # Use played_move if known (on fen_before), otherwise best_move (on fen_after)
            if played_move_san != "unknown":
                temp_board = chess.Board(fen_before)
                temp_board.turn = chess.WHITE if player_color == 'w' else chess.BLACK
                ref_move_obj = temp_board.parse_san(played_move_san)
            else:
                temp_board = chess.Board(fen_after)
                temp_board.turn = chess.WHITE if player_color == 'w' else chess.BLACK
                ref_move_obj = temp_board.parse_san(best_move)
                
            moving_piece = temp_board.piece_at(ref_move_obj.from_square)
            if moving_piece:
                p_type = moving_piece.piece_type
                if p_type == chess.KING:
                    piece_guideline = "Since this is a king move, focus your explanation strictly on king safety, escaping threats, escaping check, or defensive positioning. Do NOT mention piece development, attacking opportunities, or activating other pieces."
                elif p_type in [chess.KNIGHT, chess.BISHOP]:
                    piece_guideline = "Since this is a minor piece move, focus your explanation on piece activity, development, minor piece coordination, or controlling key squares."
                elif p_type in [chess.ROOK, chess.QUEEN]:
                    piece_guideline = "Since this is a major piece move, focus on open files, controlling diagonals, major piece activity, or creating major threats."
                elif p_type == chess.PAWN:
                    piece_guideline = "Since this is a pawn move, focus on pawn structure, center control, space, or creating defensive chains."
        except Exception as e:
            print(f"[Coach] Error computing piece_guideline: {e}")

        if played_move_san == best_move:
            task = (f"Act as a supportive, warm, and professional chess coach. Write exactly 2 sentences in first-person ('I' or 'we'): "
                    f"(1) Warmly praise the player for finding the excellent move {played_move_english} (noting if it captured a piece or gave check) and explain why it was strategically the best choice, "
                    f"(2) explain in simple terms how this move improves their position, coordinates their pieces, or maintains their momentum in the {game_phase}. "
                    f"Do NOT mention standard chess notation (like Nxf3 or Be2) or raw numerical ratings. {piece_guideline}")
            example = (f"Great job playing {played_move_english}! It is the strongest continuation because it wins important material and coordinates your pieces, maintaining your strong momentum in the {game_phase}.")
        elif played_move_san != "unknown":
            task = (f"Act as a supportive, encouraging chess coach. Write exactly 2 sentences in first-person: "
                    f"(1) Comment on their played move {played_move_english}, explaining the key strategic weakness or drawback of playing it (e.g., why {played_move_english} was a {move_quality} in terms of piece safety, space, or development), "
                    f"(2) explain why {best_move_english} would have been a much stronger choice, detailing the advantages of {best_move_english} (like claiming the center or securing king safety) compared to what they played. "
                    f"Do NOT mention standard chess notation (like Nxf3 or Be2) or raw numerical ratings. {piece_guideline}")
            example = (f"You chose {played_move_english}, which is a {move_quality} because it leaves your queen exposed and slows down your development. "
                       f"Instead, playing {best_move_english} would have been much stronger, securing your king safety and claiming the center.")
        else:
            task = (f"Act as a friendly, expert chess coach. Write exactly 2 sentences in first-person: "
                    f"(1) Recommend {best_move_english} to the player as a very strong option to play in this position, "
                    f"(2) explain in simple strategic terms how {best_move_english} helps their position (such as improving development, securing king safety, or contesting the center) given the current {game_phase} and trend. "
                    f"Do NOT mention standard chess notation (like Nxf3 or Be2) or raw numerical ratings. {piece_guideline}")
            example = (f"You should consider playing {best_move_english} in this position, as it's a highly effective option here. "
                       f"Looking at this {game_phase}, {best_move_english} is critical for coordinating your pieces and keeping the center well-protected.")

        prompt = (
            f"You are a supportive chess coach analyzing a live game with your student. The player is {player_side}.\n"
            f"Game context: {history_context}\n\n"
            f"VERIFIED FACTS about the move just played:\n"
            f"{facts_block}\n\n"
            f"Your task: {task}\n\n"
            f"Rules:\n"
            f"- Speak in a friendly, encouraging, and highly conversational tone, like a real-life coach.\n"
            f"- Speak directly to the player in first-person (e.g. 'I recommend...', 'You played...', 'Let's look at...').\n"
            f"- Speak in plain English descriptions for all moves (e.g. say 'knight takes on f3' instead of 'Nxf3', and 'bishop to e2' instead of 'Be2'). Do NOT use standard algebraic chess notation (like Nxf3 or Be2) in your response.\n"
            f"- Avoid technical engine jargon like 'Stockfish recommends', 'eval', 'centipawns', '+1.4', etc.\n"
            f"- Instead of saying 'Stockfish recommends', say 'A better option is...', 'You should consider...', 'A strong move here is...', etc.\n"
            f"- Only reference moves and pieces explicitly named in the FACTS.\n"
            f"- Do NOT invent specific move names or square names not stated in the FACTS, but feel free to refer to general strategic themes (e.g. piece development, control of the center, king safety, defense).\n"
            f"- Be specific and direct. Output ONLY the 2 sentences, nothing else.\n"
            f"Example: '{example}'\n\n"
            f"Your response:"
        )

        print(f"[Coach] Prompt sent to Qwen:\n---\n{prompt}\n---")

        # 6. Call LLM (Prioritize Gemini Cloud, Fallback to Local Ollama)
        explanation = ""
        try:
            raw_response = None
            gemini_api_key = os.environ.get("GEMINI_API_KEY")
            
            if gemini_api_key:
                print(f"[Coach] GEMINI_API_KEY found! Calling Gemini 1.5 Flash Cloud...")
                raw_response = call_gemini_api(prompt, gemini_api_key)
                if raw_response:
                    print(f"[Coach] Raw Gemini response:\n---\n{raw_response}\n---")
                else:
                    print(f"[Coach] Gemini API returned empty, falling back to local Ollama...")
            
            if not raw_response:
                print(f"[Coach] Calling local Ollama...")
                req_data = json.dumps({
                    "model": "qwen2.5:1.5b",
                    "prompt": prompt,
                    "stream": False
                }).encode('utf-8')

                req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=req_data, headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req, timeout=30) as response:
                    result = json.loads(response.read().decode('utf-8'))
                    raw_response = result.get("response", "").strip()
                    print(f"[Coach] Raw Qwen response:\n---\n{raw_response}\n---")

            import re
            clean = re.sub(r'<[^>]+>', '', raw_response).strip()
            lines = [
                l.strip() for l in clean.split('\n')
                if l.strip() and not l.strip().startswith(('-', '*', '#'))
            ]
            explanation = lines[0] if lines else clean

            # Post-process: find any leftover raw algebraic moves and convert them to plain English
            for san, eng in [(played_move_san, played_move_english), (best_move, best_move_english)]:
                if san and san != "unknown":
                    pattern = (r'(?<!\bto\s)(?<!\bon\s)(?<!\btakes\s)(?<!\bpawn\s)(?<!\bknight\s)'
                               r'(?<!\bbishop\s)(?<!\brook\s)(?<!\bqueen\s)(?<!\bking\s)(?<![a-zA-Z0-9])'
                               + re.escape(san) + r'(?![a-zA-Z0-9])')
                    explanation = re.sub(pattern, eng, explanation, flags=re.IGNORECASE)

            print(f"[Coach] Final explanation: {explanation}")

        except Exception as e:
            print(f"[Coach] ERROR calling LLM (Gemini/Ollama): {e}")
            # Fallback: build a purely factual but human-friendly explanation without LLM
            if played_move_san != "unknown":
                cap_str = f", capturing their {captured_piece_name}" if captured_piece_name else ""
                check_str = " and putting them in check" if gives_check else ""
                
                if played_move_san == best_move:
                    explanation = f"Excellent move with {played_move_english}{cap_str}{check_str}! That was exactly the best continuation to keep your advantage."
                else:
                    quality_phrases = {
                        "excellent": "a very strong choice",
                        "good": "a decent move",
                        "inaccuracy": "a bit of an inaccuracy",
                        "mistake": "a mistake that loses some control",
                        "blunder": "a blunder that hurts your position"
                    }
                    phrase = quality_phrases.get(move_quality, f"a {move_quality}")
                    explanation = f"You played {played_move_english}{cap_str}{check_str}, which is {phrase}. Consider playing {best_move_english} instead, which would be much stronger here."
            else:
                explanation = f"You should look at playing {best_move_english} here, which is a very strong and stable choice for this position."

        return jsonify({
            "status": "success",
            "explanation": explanation,
            "played_move": played_move_san,
            "move_quality": move_quality,
            "history": game_history
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/reset_coach', methods=['POST'])
def reset_coach():
    global game_history, eval_history, current_board
    game_history = []
    eval_history = []
    current_board = chess.Board()
    return jsonify({"status": "success", "message": "Game history reset"})

if __name__ == '__main__':
    # Electron will spawn this process. Run on port 5050.
    print("[Python] Starting Chess AI server on port 5050...", flush=True)
    app.run(host='127.0.0.1', port=5050, debug=False)
