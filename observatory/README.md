# observatory — OMNIS OBSERVATORY FastAPI Backend & 3D Cognitive UI

Interactive 8-panel scientific Observatory featuring a Three.js 3D connectome viewer, 2D DIAMOND world canvas, live NEUROHIVE inspector with lineage tree visualization, GENESIS 8-stage developmental progress strip & checkpoint timeline, A–E benchmark Pareto frontier canvas, GHOST ENGINE hypothesis runner, and HARDWARE & SUPERMIX-AETHER control console.

## Architecture & Key Modules
- `api.py`: FastAPI application factory (`create_observatory_app`), optional `X-API-Key` header enforcement on `/api/*` routes (keeping `/` and `/api/health` public), hardware optimization/parallel sweep/tensor-pass routes, and developmental stage progression endpoint.
- `static/index.html`, `static/styles.css`, `static/app.js`: Semantic dark-mode scientific UI using CSS `@layer`, container queries, accessible keyboard controls, and zero inline styles.
- `interfaces.py`: `ObservatoryFactoryProtocol`.

## Standalone Execution
```bash
python -m observatory.example
```
