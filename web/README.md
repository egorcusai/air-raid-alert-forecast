# web/ — Dashboard (deployable)

Self-contained static dashboard. The results data is embedded directly in
`index.html`, so it works offline and needs no backend or build step.

## Local test server
```bash
cd web
python3 -m http.server 8000   # http://localhost:8000
```

## Production (Vercel)
Import the repo at vercel.com and set **Root Directory = `web`**.
Framework Preset: Other · Build command: none · Output dir: `.`
Every push to `main` then auto-deploys.

## Regenerating the embedded data
The dashboard reflects a snapshot of pipeline output. To refresh it after
re-running the analysis:
```bash
python3 scripts/build_dashboard.py   # from repo root
```
