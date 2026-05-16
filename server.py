import os
import sys
sys.stdout.reconfigure(line_buffering=True)  # flush print() immediately when running as subprocess

import logging
import urllib.request
import json
import chess
from flask import Flask, request, jsonify
from flask_cors import CORS

from engine.engine import analyze

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

        if not fen_before or not fen_after or not best_move:
            print(f"[Coach] ERROR: Missing required fields")
            return jsonify({"status": "error", "message": "Missing required fields"}), 400

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
                current_board.push(move)
                if current_board.board_fen() == target_placement:
                    played_move_san = current_board.san(move)
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

            # Eval before (from the perspective of the player who moved)
            r_before = sf.get_best_move(fen_before)
            r_after  = sf.get_best_move(fen_after)

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

            print(f"[Coach] Eval before: {cp_before}cp  after: {cp_after}cp  loss: {cp_loss}  quality: {move_quality}")
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
            else:
                eval_str = f"Mate in {abs(mate_after)} for {'Black' if player_color == 'w' else 'White'}"
        elif cp_after is not None:
            sign = "+" if cp_after >= 0 else ""
            eval_str = f"{sign}{cp_after/100:.1f} ({'White' if cp_after >= 0 else 'Black'} is {'winning' if abs(cp_after) > 200 else 'better' if abs(cp_after) > 50 else 'roughly equal'})"
        else:
            eval_str = "unknown"

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

        facts = []
        if played_move_san != "unknown":
            facts.append(f"Move played by {player_side}: {played_move_san}")
        if captured_piece_name:
            facts.append(f"This move captured a {captured_piece_name}")
        if gives_check:
            facts.append("This move gives check")
        if cp_loss is not None:
            facts.append(f"Move quality: {move_quality} (centipawn change: {cp_loss:+d}cp)")
        else:
            facts.append(f"Move quality: {move_quality}")
        facts.append(f"Best move according to Stockfish: {best_move}")
        facts.append(f"Current evaluation after the move: {eval_str}")
        if move_num > 0:
            facts.append(f"Move number in game: {move_num} ({game_phase})")
        if eval_trend_str:
            facts.append(f"Positional trend: {eval_trend_str}")

        facts_block = "\n".join(f"- {f}" for f in facts)

        if played_move_san == best_move:
            task = (f"Write 2 sentences: (1) confirm the player found the best move ({played_move_san}) "
                    f"and it was {move_quality}, noting any captures or check, "
                    f"(2) tie it into the game context — reference the trend or game phase if relevant.")
        elif played_move_san != "unknown":
            cp_desc = f"losing {abs(cp_loss)} centipawns" if cp_loss is not None and cp_loss < 0 else "gaining ground"
            task = (f"Write 2 sentences: (1) state the player played {played_move_san} ({move_quality}, {cp_desc}), "
                    f"(2) state the best move was {best_move} and why it would have been stronger — "
                    f"reference the game context or trend if it adds insight.")
        else:
            task = (f"Write 2 sentences: (1) Stockfish recommends {best_move} for this position "
                    f"(eval: {eval_str}), (2) briefly explain why given the current {game_phase} situation.")

        prompt = (
            f"You are a chess coach analyzing a live game. The player is {player_side}.\n"
            f"Game context: {history_context}\n\n"
            f"VERIFIED FACTS about the move just played:\n"
            f"{facts_block}\n\n"
            f"Your task: {task}\n\n"
            f"Rules:\n"
            f"- Only reference moves and pieces explicitly named in the FACTS.\n"
            f"- Do NOT invent move names, square names, or tactics not stated above.\n"
            f"- You MAY reference the game history and trend when it adds insight.\n"
            f"- Be specific and direct. Output ONLY the 2 sentences, nothing else.\n"
            f"Example: 'You played Nxe5, winning a pawn — an excellent move that also gives check. "
            f"With White gaining ground over the last few moves, Nxe5 consolidates your advantage perfectly.'\n\n"
            f"Your response:"
        )

        print(f"[Coach] Prompt sent to Qwen:\n---\n{prompt}\n---")

        # 6. Call local Ollama
        req_data = json.dumps({
            "model": "qwen2.5:1.5b",
            "prompt": prompt,
            "stream": False
        }).encode('utf-8')

        req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=req_data, headers={'Content-Type': 'application/json'})

        try:
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
                explanation = ' '.join(lines[:2]) if lines else clean
                print(f"[Coach] Final explanation: {explanation}")

        except Exception as e:
            print(f"[Coach] ERROR calling Ollama: {e}")
            # Fallback: build a purely factual explanation without Qwen
            if played_move_san != "unknown":
                cap_str = f", capturing a {captured_piece_name}" if captured_piece_name else ""
                check_str = " and giving check" if gives_check else ""
                explanation = (
                    f"You played {played_move_san}{cap_str}{check_str} — a {move_quality} "
                    f"({cp_loss:+d}cp). "
                    f"{'Best move was ' + best_move + '.' if played_move_san != best_move else 'That was the best move!'}"
                )
            else:
                explanation = f"Stockfish recommends {best_move}. Current eval: {eval_str}."

        return jsonify({
            "status": "success",
            "explanation": explanation,
            "played_move": played_move_san,
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
