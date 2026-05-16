const { ipcRenderer } = require('electron');

const closeBtn = document.getElementById('close-btn');
const minBtn = document.getElementById('min-btn');
const btnWhite = document.getElementById('btn-white');
const btnBlack = document.getElementById('btn-black');
const autoPollToggle = document.getElementById('auto-poll');
const scanBtn = document.getElementById('scan-btn');
const evalScoreEl = document.getElementById('eval-score');
const bestMoveEl = document.getElementById('best-move');
const statusBar = document.getElementById('status-bar');
const platformSelect = document.getElementById('platform-select');

// Splash Screen Elements
const splashScreen = document.getElementById('splash-screen');
const mainApp = document.getElementById('main-app');
const btnCheat = document.getElementById('btn-cheat');
const btnCoach = document.getElementById('btn-coach');
const backBtn = document.getElementById('back-btn');
const splashVideo = document.querySelector('.splash-video');

if (splashVideo) {
    splashVideo.playbackRate = 1.7; // Speed up animation
}

let currentColor = 'w';
let isFlipped = false;
let pollingInterval = null;
let isAnalyzing = false;

// Coach State
let coachLastEval = null;
let coachLastBestMove = null;       // English display (e.g. 'Knight to c3')
let coachLastBestMoveSan = null;   // Raw SAN (e.g. 'Nc3') — sent to coach endpoint
let coachLastFen = null;
let coachRequestInFlight = false; // semaphore to prevent overlapping Qwen requests

// ── Splash Screen Logic ──
let appMode = null;

function enterApp(mode) {
    appMode = mode;
    splashScreen.classList.add('hidden');
    mainApp.classList.remove('hidden');
    backBtn.classList.remove('hidden'); // show back arrow
    
    if (mode === 'coach') {
        document.getElementById('best-move').classList.add('hidden');
        document.getElementById('coach-ui').classList.remove('hidden');
        document.getElementById('btn-hint').classList.remove('hidden'); // show hint to reveal best move
    } else {
        document.getElementById('best-move').classList.remove('hidden');
        document.getElementById('coach-ui').classList.add('hidden');
        document.getElementById('btn-hint').classList.add('hidden');
    }
    
    // Start polling once we enter the app
    statusBar.textContent = "Ready.";
    setupPolling();
}

function goBackToSplash() {
    // Stop polling
    if (pollingInterval) {
        clearInterval(pollingInterval);
        pollingInterval = null;
    }
    // Reset coach state
    coachLastFen = null;
    coachLastBestMove = null;
    coachLastBestMoveSan = null;
    coachLastEval = null;
    coachRequestInFlight = false;
    appMode = null;

    // Swap screens
    mainApp.classList.add('hidden');
    splashScreen.classList.remove('hidden');
    backBtn.classList.add('hidden'); // hide back arrow on splash

    // Reset coach UI for next session
    document.getElementById('coach-feedback').textContent = 'Waiting for your first move...';
    document.getElementById('coach-feedback').className = 'feedback-neutral';
}

btnCheat.addEventListener('click', () => enterApp('cheat'));
btnCoach.addEventListener('click', () => enterApp('coach'));
backBtn.addEventListener('click', goBackToSplash);

const btnHint = document.getElementById('btn-hint');
if (btnHint) {
    // Hint button only toggles the best move display — not coach feedback
    btnHint.addEventListener('click', () => {
        document.getElementById('best-move').classList.toggle('hidden');
    });
}

// ── Platform Selection ──
platformSelect.addEventListener('change', () => {
    triggerAnalysis();
});

// ── Window Controls ──
closeBtn.addEventListener('click', () => {
    ipcRenderer.send('quit-app');
});

minBtn.addEventListener('click', () => {
    ipcRenderer.send('minimize-app');
});

// ── Color Selection ──
btnWhite.addEventListener('click', () => {
    currentColor = 'w';
    isFlipped = false;
    btnWhite.classList.add('active');
    btnBlack.classList.remove('active');
    triggerAnalysis();
});

btnBlack.addEventListener('click', () => {
    currentColor = 'b';
    isFlipped = true;  // Usually, if you play black, the board is flipped
    btnBlack.classList.add('active');
    btnWhite.classList.remove('active');
    triggerAnalysis();
});

// ── API Interaction ──
async function triggerAnalysis() {
    if (isAnalyzing) return;
    
    isAnalyzing = true;
    bestMoveEl.classList.add('updating');
    statusBar.textContent = "Analyzing board...";

    try {
        const response = await fetch('http://127.0.0.1:5050/analyze', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                color: currentColor,
                flipped: isFlipped,
                platform: platformSelect.value
            })
        });

        const result = await response.json();

        if (result.status === 'success') {
            const data = result.data;
            
            // Format evaluation score
            let evalText = "";
            let scoreFloat = 0.0;
            
            if (data.mate !== null) {
                evalText = `M${data.mate}`;
                evalScoreEl.className = 'eval-score positive';
                // Assign an arbitrary large value for mate to avoid math breaking
                scoreFloat = data.mate > 0 ? 100.0 : -100.0; 
            } else {
                // Score is typically centipawns from White's perspective
                scoreFloat = data.score / 100.0;
                evalText = scoreFloat > 0 ? `+${scoreFloat.toFixed(2)}` : `${scoreFloat.toFixed(2)}`;
                evalScoreEl.className = scoreFloat > 0 ? 'eval-score positive' : 'eval-score negative';
            }
            
            evalScoreEl.textContent = evalText;
            
            // Format best move to plain English
            const san = data.san || "";
            let englishMove = "---";
            if (san) {
                if (san === "O-O") englishMove = "Castle Kingside";
                else if (san === "O-O-O") englishMove = "Castle Queenside";
                else {
                    let piece = "Pawn";
                    if (san[0] === 'N') piece = "Knight";
                    else if (san[0] === 'B') piece = "Bishop";
                    else if (san[0] === 'R') piece = "Rook";
                    else if (san[0] === 'Q') piece = "Queen";
                    else if (san[0] === 'K') piece = "King";
                    
                    const match = san.match(/([a-h][1-8])/);
                    const pos = match ? match[1] : "";
                    
                    englishMove = `${piece} to ${pos}`;
                    if (san.includes('x')) englishMove = `${piece} takes on ${pos}`;
                    if (san.includes('=')) englishMove = `Pawn to ${pos} (Promote)`;
                }
            }
            
            bestMoveEl.textContent = englishMove;
            // Store both forms for use by the coach endpoint
            coachLastBestMove = englishMove;       // English display
            coachLastBestMoveSan = san || null;    // Raw SAN for the API

            // --- Coach Mode Logic ---
            if (appMode === 'coach') {
                const coachFeedback = document.getElementById('coach-feedback');

                // Auto-reset Coach if board returns to starting position
                const startFen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR";
                if (data.fen.startsWith(startFen)) {
                    fetch('http://127.0.0.1:5050/reset_coach', { method: 'POST' });
                    coachLastFen = data.fen;
                    coachLastEval = 0.0;
                    coachLastBestMove = "---";
                }

                // If the FEN changed, a move was made — and no request is already running
                if (coachLastFen !== null && coachLastFen !== data.fen && !coachRequestInFlight) {
                    const drop = coachLastEval - scoreFloat;
                    const snapFenBefore = coachLastFen;
                    const snapFenAfter = data.fen;
                    const snapBestMove = coachLastBestMoveSan || coachLastBestMove; // prefer raw SAN
                    const snapDrop = drop;

                    // Only trigger when the PLAYER's own pieces moved.
                    // Mask out opponent pieces to 'x' and compare board strings —
                    // if the masked board changes, the player's pieces changed position.
                    const playerIsWhite = (currentColor === 'w');
                    const maskOpponent = (fen) => {
                        const board = fen.split(' ')[0];
                        return playerIsWhite
                            ? board.replace(/[a-z]/g, 'x')   // mask black pieces
                            : board.replace(/[A-Z]/g, 'X');   // mask white pieces
                    };
                    const playerPiecesMoved = maskOpponent(snapFenBefore) !== maskOpponent(snapFenAfter);

                    if (playerPiecesMoved) {
                        coachFeedback.textContent = `Analyzing your last move...`;
                        coachFeedback.className = 'feedback-neutral';
                        coachRequestInFlight = true;

                        fetch('http://127.0.0.1:5050/coach_explanation', {
                            method: 'POST',
                            headers: { 'Content-Type': 'application/json' },
                            body: JSON.stringify({
                                fen_before: snapFenBefore,
                                fen_after: snapFenAfter,
                                best_move: snapBestMove,
                                player_color: currentColor
                            })
                        })
                        .then(res => res.json())
                        .then(coachData => {
                            if (coachData.status === 'success') {
                                const moveName = coachData.played_move !== 'unknown' ? coachData.played_move : 'that move';
                                if (coachData.played_move === snapBestMove) {
                                    coachFeedback.textContent = `Move ${moveName}: ${coachData.explanation}`;
                                    coachFeedback.className = 'feedback-neutral';
                                } else {
                                    coachFeedback.textContent = `Move ${moveName}: ${coachData.explanation}`;
                                    coachFeedback.className = snapDrop > 1.0 ? 'feedback-negative' : 'feedback-neutral';
                                }
                            }
                        })
                        .catch(err => {
                            console.error(err);
                            coachFeedback.textContent = `Failed to connect to Coach API.`;
                        })
                        .finally(() => {
                            coachRequestInFlight = false; // always release lock
                        });
                    } else {
                        console.log('[Coach] Opponent moved — skipping');
                    }
                }

                coachLastEval = scoreFloat;
                coachLastBestMove = englishMove;
                coachLastFen = data.fen;
            }
            
            // Note the engine confidence
            const conf = (data.confidence_avg * 100).toFixed(1);
            statusBar.textContent = `Board found. Vision confidence: ${conf}%`;
        } else {
            statusBar.textContent = result.message || "Analysis failed.";
            evalScoreEl.textContent = "0.0";
            evalScoreEl.className = 'eval-score';
            bestMoveEl.textContent = "---";
        }
    } catch (err) {
        console.error(err);
        statusBar.textContent = "Failed to connect to Python server.";
    } finally {
        isAnalyzing = false;
        bestMoveEl.classList.remove('updating');
    }
}

// ── Manual Scan ──
scanBtn.addEventListener('click', () => {
    triggerAnalysis();
});

// ── Auto Polling ──
function setupPolling() {
    if (pollingInterval) clearInterval(pollingInterval);
    
    if (autoPollToggle.checked) {
        // Poll every 3 seconds
        pollingInterval = setInterval(triggerAnalysis, 3000);
        triggerAnalysis(); // run immediately
    }
}

autoPollToggle.addEventListener('change', setupPolling);

// Wait slightly for Python server to boot
setTimeout(() => {
    // The status bar will be updated when the user enters the app
    console.log("App loaded, waiting for mode selection.");
}, 2000);
