## Development & Run Instructions

- Create and activate a Python virtual environment (recommended):

	```powershell
	python -m venv .venv
	.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	```

- Run database migrations if this is the first time you start the project:

	```powershell
	python scripts/migrate_db.py
	```

- Register a trained model (example):

	```powershell
	python scripts/register_model.py --version v2026-01-01 --model-type RandomForest --artifact-path saved_models/ford_artifacts.joblib
	```

- Run the cron prediction script (local test):

	```powershell
	python scripts/cron_predict_new.py
	```


## Project Architecture (short)

- Backend: `backend/` contains the FastAPI server, DB engine, market calendar, and database services.
- Scripts: `scripts/` contains ETL, training, registration, migration and cron utilities.
- Models: `stock_models/` contains per-ticker training code (HCLA PyTorch or RandomForest scikit-learn).
- Artifacts: `saved_models/` holds joblib artifact files (`<ticker>_artifacts.joblib`).
- Data: `stock_data/` contains CSV fallbacks; in production data is persisted to PostgreSQL (`stock_prices`).


## Frontend Recommendation

This repository hosts the Python backend and ML artifacts. I recommend building the frontend as a separate project (separate repo or directory) using a JavaScript framework (Next.js, React, or Vue). Reasons:

- Different stack and tooling (Node.js/npm) — keeps dependency management and CI/CD distinct.
- Independent deployment lifecycle — frontend can be deployed to Vercel/Netlify while backend runs on Render/Heroku.
- Simpler CI: frontend builds (static or SSR) without affecting backend image size or runtime.

Integration pattern:

- Frontend calls the backend API: GET `/predictions/{ticker}` to fetch 365 points.
- Backend serves CORS-enabled API and authenticates if needed.

If you prefer a monorepo, keep frontend in a top-level `frontend/` folder with its own package.json and CI steps; otherwise create a separate repository.

# Project description 📰
This project investigates the use of machine learning and deep learning models for predicting stock prices of selected publicly traded companies in the United States. The work is inspired by the methodology proposed by Bhanujyothi et al. (2025), who introduced a Hybrid CNN–LSTM with an Attention (HCLA) model for forecasting Tesla’s stock prices. Building on this framework, I extend the approach to multiple companies to evaluate its generalizability and predictive performance across different market conditions.

The project leverages convolutional layers to extract meaningful patterns from historical price data, LSTM layers to capture long-term temporal dependencies, and an attention mechanism to emphasize the most relevant time steps influencing price movements. Model performance is assessed using standard regression metrics and visual comparisons between predicted and actual stock prices.

Currently developed and tested in Google Colab, this project is a personal initiative aimed at exploring data-driven strategies for informed and potentially profitable stock trading. As a next step, the project will be deployed as an interactive web application using Flask or FastAPI, allowing users to select companies, input time horizons, and generate real-time stock price predictions through a user-friendly interface.

This project serves both as a practical application of advanced deep learning techniques in financial time-series forecasting and as a foundation for a scalable prediction platform accessible to a broader audience. 💸 💰

# Data Source and chosen stocks
**Data Source**  
The dataset was obtained from **Yahoo Finance**, which provides free access for both personal and academic use.  

Here you go — a clean, professional **Markdown table** summarizing all six assets and their descriptions.

---

# 📘 **Selected Assets — Summary Table**

| **Ticker** | **Company / Asset** | **Description** |
|-----------|----------------------|-----------------|
| **TSLA** | Tesla, Inc. | A high‑volatility electric vehicle and clean‑energy company with substantial trading volume. Its large price swings make it ideal for studying short‑term market dynamics and risk behavior. |
| **F** | Ford Motor Company | A major U.S. automaker with cyclical performance tied to the broader economy. Lower‑priced and stable, useful for analyzing sector cycles and value‑oriented investing. |
| **PFE** | Pfizer Inc. | A global biopharmaceutical company known for defensive characteristics, lower volatility, and a strong dividend yield. Performance is driven by drug pipelines and healthcare trends. |
| **CMCSA** | Comcast Corporation | A diversified media and telecommunications conglomerate operating in broadband, cable, and entertainment. Offers stable cash flows and moderate volatility. |
| **T** | AT&T Inc. | A large telecommunications provider known for high dividend yield and defensive behavior. Influenced by subscription growth and infrastructure investments. |
| **ROIV** | Roivant Sciences Ltd. | A clinical‑stage biopharmaceutical company using a decentralized “Vant” model. Higher volatility and growth‑oriented, driven by trial results and biotech sentiment. |

---

