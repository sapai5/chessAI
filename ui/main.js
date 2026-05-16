const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonServer;
let ollamaProcess;

function createWindow() {
    // Create an always-on-top, transparent, borderless window
    mainWindow = new BrowserWindow({
        width: 480,
        height: 600,
        x: 50,  // Top left corner so it doesn't block the board
        y: 50,
        transparent: true,
        frame: false,
        alwaysOnTop: true,
        webPreferences: {
            nodeIntegration: true,
            contextIsolation: false
        }
    });

    // Make it so you can drag the window around
    mainWindow.loadFile('index.html');

    // Optional: open DevTools during development
    // mainWindow.webContents.openDevTools({ mode: 'detach' });

    mainWindow.on('closed', () => {
        mainWindow = null;
    });
}

function startPythonServer() {
    console.log("Starting Python API server...");

    // Path to the virtual environment's python and the server script
    const pythonExecutable = path.join(__dirname, '..', 'venv', 'Scripts', 'python.exe');
    const serverScript = path.join(__dirname, '..', 'server.py');

    pythonServer = spawn(pythonExecutable, [serverScript]);

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
    
    // Resolve absolute path to avoid Windows PATH cache issues
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
    startPythonServer();
    createWindow();

    app.on('activate', function () {
        if (BrowserWindow.getAllWindows().length === 0) createWindow();
    });
});

// Quit when all windows are closed, and kill the servers
app.on('window-all-closed', function () {
    if (pythonServer) pythonServer.kill();
    if (ollamaProcess) ollamaProcess.kill();
    
    if (process.platform !== 'darwin') app.quit();
});

ipcMain.on('quit-app', () => {
    if (mainWindow) {
        // Hiding the window first prevents the Windows DWM Alt-Tab ghosting bug
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
