const { contextBridge, ipcRenderer } = require('electron');

// Expose secure and selected APIs to the renderer context
contextBridge.exposeInMainWorld('electron', {
    ipcRenderer: {
        send: (channel, data) => ipcRenderer.send(channel, data),
        on: (channel, func) => ipcRenderer.on(channel, (event, ...args) => func(event, ...args)),
        removeListener: (channel, func) => ipcRenderer.removeListener(channel, func)
    }
});
