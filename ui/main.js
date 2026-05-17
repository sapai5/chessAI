const { app, BrowserWindow, ipcMain, net } = require('electron');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');

let mainWindow;
let loadingWindow;
let pythonServer;
let ollamaProcess;

// Default direct download URL for the classifier weights
const WEIGHTS_URL = "https://github.com/sapai5/chessAI/releases/download/v1.0.0/model.pt";

// Resolve clean persistent folder path: AppData/ChessAI/weights/model.pt
const appDataPath = path.join(app.getPath('appData'), 'ChessAI');
const weightsDir = path.join(appDataPath, 'weights');
const weightsPath = path.join(weightsDir, 'model.pt');

function createLoadingWindow() {
    loadingWindow = new BrowserWindow({
        width: 440,
        height: 250,
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    loadingWindow.loadFile(path.join(__dirname, 'renderer', 'loading.html'));
}

function downloadWeights() {
    if (!fs.existsSync(weightsDir)) {
        fs.mkdirSync(weightsDir, { recursive: true });
    }

    const fileStream = fs.createWriteStream(weightsPath);
    let downloadedBytes = 0;
    let totalBytes = 0;

    console.log(`Starting download from ${WEIGHTS_URL}...`);
    const request = net.request(WEIGHTS_URL);

    request.on('response', (response) => {
        totalBytes = parseInt(response.headers['content-length'], 10) || 0;
        console.log(`Download response received. Total size: ${totalBytes} bytes`);

        response.on('data', (chunk) => {
            downloadedBytes += chunk.length;
            fileStream.write(chunk);

            if (totalBytes > 0) {
                const percent = Math.round((downloadedBytes / totalBytes) * 100);
                if (loadingWindow && !loadingWindow.isDestroyed()) {
                    loadingWindow.webContents.send('download-progress', percent);
                }
            }
        });

        response.on('end', () => {
            fileStream.end();
            console.log("Weights downloaded successfully.");
            setTimeout(() => {
                if (loadingWindow) {
                    loadingWindow.close();
                }
                startPythonServer();
                createWindow();
            }, 1000);
        });
    });

    request.on('error', (err) => {
        fileStream.end();
        console.error("Failed to download weights:", err);
        if (loadingWindow && !loadingWindow.isDestroyed()) {
            loadingWindow.webContents.send('download-error', err.message);
        }
    });

    request.end();
}

function createWindow() {
    mainWindow = new BrowserWindow({
        width: 480,
        height: 600,
        x: 50,
        y: 50,
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    mainWindow.loadFile(path.join(__dirname, 'index.html'));

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function startPythonServer() {
    console.log("Starting Python API server...");

    let pythonExecutable;
    let args = [];

    if (app.isPackaged) {
        // Packaged production environment - spawn compiled server.exe sidecar
        pythonExecutable = path.join(process.resourcesPath, 'server.exe');
    } else {
        // Local development environment - spawn local python environment
        pythonExecutable = path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
        args.push(path.join(__dirname, '..', 'server.py'));
    }

    console.log(`Executing: ${pythonExecutable} ${args.join(' ')}`);
    pythonServer = spawn(pythonExecutable, args);

    pythonServer.stdout.on('data', (data) => {
        console.log(`[Python]: ${data}`);
    });

    pythonServer.stderr.on('data', (data) => {
        console.error(`[Python ERR]: ${data}`);
    });

    pythonServer.on('close', (code) => {
        console.log(`Python server process exited with code ${code}`);
    });
}

function startOllama() {
    console.log("Starting local Ollama Qwen model...");
    const ollamaPath = path.join(process.env.LOCALAPPDATA || process.env.APPDATA, 'Programs', 'Ollama', 'ollama.exe');
    try {
        ollamaProcess = spawn(ollamaPath, ['run', 'qwen2.5:1.5b'], { shell: false });
        
        ollamaProcess.stdout.on('data', (data) => {
            console.log(`[Ollama]: ${data}`);
        });

        ollamaProcess.stderr.on('data', (data) => {
            console.error(`[Ollama ERR]: ${data}`);
        });
    } catch (e) {
        console.error("Failed to start Ollama. Ensure it is installed.", e);
    }
}

app.whenReady().then(() => {
    startOllama();

    if (!fs.existsSync(weightsPath)) {
        createLoadingWindow();
        loadingWindow.once('show', () => {
            downloadWeights();
        });
    } else {
        startPythonServer();
        createWindow();
    }

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

app.on('window-all-closed', function () {
    if (pythonServer) pythonServer.kill();
    if (ollamaProcess) ollamaProcess.kill();
    if (process.platform !== 'darwin') app.quit();
});

ipcMain.on('quit-app', () => {
    if (mainWindow) {
        mainWindow.hide();
        mainWindow.setSkipTaskbar(true);
        setTimeout(() => {
            mainWindow.destroy();
            if (pythonServer) pythonServer.kill();
            if (ollamaProcess) ollamaProcess.kill();
            app.quit();
        }, 100);
    } else {
        if (pythonServer) pythonServer.kill();
        if (ollamaProcess) ollamaProcess.kill();
        app.quit();
    }
});
