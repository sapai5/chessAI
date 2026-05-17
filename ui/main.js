const { app, BrowserWindow, ipcMain, net } = require('electron');
const path = require('path');
const fs = require('fs');
const https = require('https');
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

function downloadWeightsHelper(url, dest, callback, progressCallback) {
    const fileStream = fs.createWriteStream(dest);
    
    // Use a standard browser user agent to ensure AWS S3 does not restrict or block the connection
    const options = {
        headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
        }
    };

    const request = https.get(url, options, (response) => {
        // Recursively handle standard HTTP Redirects (301, 302, 307, 308)
        if (response.statusCode >= 300 && response.statusCode < 400 && response.headers.location) {
            fileStream.close();
            try {
                fs.unlinkSync(dest);
            } catch (e) {}
            console.log(`Redirecting to: ${response.headers.location}`);
            return downloadWeightsHelper(response.headers.location, dest, callback, progressCallback);
        }

        if (response.statusCode !== 200) {
            fileStream.close();
            try {
                fs.unlinkSync(dest);
            } catch (e) {}
            return callback(new Error(`Server responded with status code ${response.statusCode}`));
        }

        const totalBytes = parseInt(response.headers['content-length'], 10) || 0;
        let downloadedBytes = 0;

        response.on('data', (chunk) => {
            downloadedBytes += chunk.length;
            fileStream.write(chunk);
            if (totalBytes > 0) {
                const percent = Math.round((downloadedBytes / totalBytes) * 100);
                progressCallback(percent);
            }
        });

        response.on('end', () => {
            fileStream.end();
            callback(null);
        });
    });

    request.on('error', (err) => {
        fileStream.close();
        try {
            if (fs.existsSync(dest)) {
                fs.unlinkSync(dest);
            }
        } catch (e) {}
        callback(err);
    });
}

function downloadWeights() {
    if (!fs.existsSync(weightsDir)) {
        fs.mkdirSync(weightsDir, { recursive: true });
    }

    console.log(`Starting download from ${WEIGHTS_URL}...`);
    
    downloadWeightsHelper(WEIGHTS_URL, weightsPath, (err) => {
        if (err) {
            console.error("Failed to download weights:", err);
            if (loadingWindow && !loadingWindow.isDestroyed()) {
                loadingWindow.webContents.send('download-error', err.message);
            }
        } else {
            console.log("Weights downloaded successfully.");
            setTimeout(() => {
                if (loadingWindow) {
                    loadingWindow.close();
                }
                startPythonServer();
                createWindow();
            }, 1000);
        }
    }, (percent) => {
        if (loadingWindow && !loadingWindow.isDestroyed()) {
            loadingWindow.webContents.send('download-progress', percent);
        }
    });
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
        // Start downloading immediately to bypass race conditions of the show event!
        downloadWeights();
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
