# How to ship a release your friend can install

## One-time setup

### 1. Upload your model weights to GitHub Releases
1. Go to your repo on GitHub → **Releases** → **Draft a new release**
2. Tag it `v1.0.0`
3. Drag and drop `weights/model.pt` into the assets section
4. Publish the release
5. Copy the direct download URL — it will look like:
   `https://github.com/sapai5/chessAI/releases/download/v1.0.0/model.pt`
6. Open `ui/main.js` and replace `WEIGHTS_URL` with this URL

### 2. Add the GitHub Actions workflow
Copy `.github/workflows/build-release.yml` into your repo at that path.

### 3. Make sure your repo has these files from this package:
- `server.spec` → in repo root
- `ui/main.js` → replace your existing one
- `ui/package.json` → replace your existing one
- `ui/preload.js` → add to ui/
- `ui/renderer/loading.html` → add to ui/renderer/

---

## Every time you want to ship a new version

```bash
git tag v1.0.0
git push origin v1.0.0
```

That's it. GitHub Actions will:
1. Install Python deps & build `server.exe` via PyInstaller
2. Install Node deps & build the Electron installer
3. Upload `ChessAI Setup 1.0.0.exe` to the GitHub Release automatically

---

## What your friend does

1. Go to: `https://github.com/sapai5/chessAI/releases/latest`
2. Download `ChessAI Setup 1.0.0.exe`
3. Double-click → click through the installer wizard
4. Launch **ChessAI** from the desktop shortcut
5. First launch: a progress bar downloads the model weights (~once, then cached)
6. Open Chess.com, start a game — the overlay appears automatically

---

## Troubleshooting

**PyInstaller missing modules**: Add them to `hiddenimports` in `server.spec`

**server.exe not found**: Make sure `pyinstaller server.spec` runs before `npm run build`
in your local workflow. The GitHub Actions workflow handles this order automatically.

**Weights not downloading**: Check that `WEIGHTS_URL` in `main.js` is a direct download
link (not a GitHub release page link). It should end in `.pt`.

**Flask not starting**: Check `electron-log` logs at:
`%APPDATA%\ChessAI\logs\main.log`
