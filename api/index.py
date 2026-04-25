"""
Streaming Analytics Dashboard - FastAPI Backend
Endpoints:
  POST /api/predict  -- real sklearn GBR viewership prediction
  GET  /api/survey   -- live SurveyMonkey response data
  GET  /api/data     -- serves ratings + weekly share data
  GET  /api/health   -- health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os, requests, warnings, pickle
from functools import lru_cache

warnings.filterwarnings("ignore")

app = FastAPI(title="Streaming Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")


@lru_cache(maxsize=1)
def get_models():
    """Load pre-trained model from pickle, or train and save if not found."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score

    # Try loading pre-trained model first (fast path)
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)

    # Train model (slow path - only runs once then saves)
    shows = pd.read_csv(os.path.join(DATA_DIR, "show_ratings.csv"))
    le_genre    = LabelEncoder().fit(shows["genre"])
    le_timeslot = LabelEncoder().fit(shows["timeslot"])
    le_network  = LabelEncoder().fit(shows["network"])

    df = shows.copy()
    df["ge"] = le_genre.transform(df["genre"])
    df["te"] = le_timeslot.transform(df["timeslot"])
    df["ne"] = le_network.transform(df["network"])
    df["yo"] = df["season_year"] - 2019

    feats = ["ge", "te", "ne", "is_live", "is_streaming", "yo"]
    X, y  = df[feats], df["viewers_millions"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

    gbr = GradientBoostingRegressor(
        n_estimators=400, learning_rate=0.05, max_depth=4, random_state=42
    )
    gbr.fit(Xtr, ytr)
    yp  = gbr.predict(Xte)

    result = {
        "gbr":         gbr,
        "le_genre":    le_genre,
        "le_timeslot": le_timeslot,
        "le_network":  le_network,
        "mae":         round(float(mean_absolute_error(yte, yp)), 3),
        "r2":          round(float(r2_score(yte, yp)), 3),
        "n_samples":   len(X),
        "feature_importances": dict(zip(feats, gbr.feature_importances_.tolist())),
    }

    # Save for future cold starts
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        pass  # Read-only filesystem on some platforms - fine, just retrain next time

    return result


def simple_forecast(platform: str, weeks: int) -> float:
    """
    Fast linear trend forecast replacing Prophet.
    Uses last 2 years of weekly data to project forward.
    """
    weekly = pd.read_csv(os.path.join(DATA_DIR, "weekly_viewership.csv"))
    col = {"bc": "broadcast_share", "st": "streaming_total_share", "pk": "peacock_share"}.get(platform, "broadcast_share")
    recent = weekly[col].tail(104).values
    n = len(recent)
    xs = np.arange(n)
    slope, intercept = np.polyfit(xs, recent, 1)
    projected = intercept + slope * (n + weeks)
    return round(float(max(0, projected)), 2)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/data")
def get_data():
    shows  = pd.read_csv(os.path.join(DATA_DIR, "show_ratings.csv"))
    weekly = pd.read_csv(os.path.join(DATA_DIR, "weekly_viewership.csv"))
    subs   = pd.read_csv(os.path.join(DATA_DIR, "peacock_subscribers.csv"))
    weekly["week"] = weekly["week"].astype(str).str[:7]
    return JSONResponse({
        "shows":  shows.to_dict(orient="records"),
        "weekly": weekly.to_dict(orient="records"),
        "subs":   subs.rename(columns={"quarter": "q", "subscribers_millions": "subs"}).to_dict(orient="records"),
    })


class PredictRequest(BaseModel):
    genre:         str
    network:       str
    timeslot:      str
    is_live:       int   = 0
    target_year:   int   = 2026
    budget_mult:   float = 1.0
    tentpole_mult: float = 1.0
    reboot_mult:   float = 1.0
    critic_score:  int   = 72
    season_num:    int   = 1
    forecast_weeks:int   = 52


@app.post("/api/predict")
def predict(req: PredictRequest):
    m = get_models()

    try: ge = int(m["le_genre"].transform([req.genre])[0])
    except ValueError: ge = 0
    try: te = int(m["le_timeslot"].transform([req.timeslot])[0])
    except ValueError: te = 0
    try: ne = int(m["le_network"].transform([req.network])[0])
    except ValueError: ne = 0

    is_streaming = 1 if req.network == "Peacock" else 0
    yo = req.target_year - 2019

    X_in = pd.DataFrame(
        [[ge, te, ne, req.is_live, is_streaming, yo]],
        columns=["ge", "te", "ne", "is_live", "is_streaming", "yo"],
    )

    base = float(m["gbr"].predict(X_in)[0])
    critic_mult = 0.85 + (req.critic_score / 100) * 0.30
    season_mult = max(0.75, 1.0 - (req.season_num - 1) * 0.02)
    adjusted    = base * req.budget_mult * req.tentpole_mult * req.reboot_mult * critic_mult * season_mult

    platform_map = {"NBC": "bc", "Peacock": "pk"}
    broadcast_forecast = simple_forecast(platform_map.get(req.network, "bc"), req.forecast_weeks)

    return {
        "base_prediction":          round(base, 2),
        "adjusted_forecast":        round(adjusted, 2),
        "conservative":             round(adjusted * 0.80, 2),
        "optimistic":               round(adjusted * 1.20, 2),
        "broadcast_share_forecast": broadcast_forecast,
        "factors": {
            "budget":   round(req.budget_mult, 3),
            "tentpole": round(req.tentpole_mult, 3),
            "reboot":   round(req.reboot_mult, 3),
            "critic":   round(critic_mult, 3),
            "season":   round(season_mult, 3),
        },
        "model_meta": {
            "mae":       m["mae"],
            "r2":        m["r2"],
            "n_samples": m["n_samples"],
            "feature_importances": m["feature_importances"],
        },
    }


@app.get("/api/survey")
def get_survey():
    token     = os.getenv("SURVEYMONKEY_TOKEN", "")
    survey_id = os.getenv("SURVEYMONKEY_SURVEY_ID", "")

    if not token or not survey_id:
        return JSONResponse({
            "live": False,
            "message": "Set SURVEYMONKEY_TOKEN and SURVEYMONKEY_SURVEY_ID to enable live data.",
            "responses": [],
            "question_summaries": _placeholder_survey_data(),
        })

    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    try:
        resp = requests.get(
            f"https://api.surveymonkey.com/v3/surveys/{survey_id}/rollups",
            headers=headers, timeout=8
        )
        resp.raise_for_status()
        rollups = resp.json()

        summaries = []
        for page in rollups.get("pages", []):
            for q in page.get("questions", []):
                q_data = {
                    "question_id": q.get("id"),
                    "heading":     q.get("heading", ""),
                    "type":        q.get("family", ""),
                    "answers":     [],
                }
                for row in q.get("answers", {}).get("rows", []):
                    q_data["answers"].append({
                        "text":  row.get("text", ""),
                        "count": row.get("count", 0),
                    })
                summaries.append(q_data)

        det = requests.get(
            f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details",
            headers=headers, timeout=8
        ).json()

        return JSONResponse({
            "live":               True,
            "response_count":     det.get("response_count", 0),
            "question_summaries": summaries,
        })

    except requests.RequestException as e:
        return JSONResponse({
            "live": False,
            "message": str(e),
            "question_summaries": _placeholder_survey_data(),
        })


def _placeholder_survey_data():
    return [
        {"question_id": "q1", "heading": "How often do you watch content on Peacock?",
         "answers": [{"text":"Daily","count":3},{"text":"Several/week","count":7},{"text":"Weekly","count":9},{"text":"Monthly","count":4},{"text":"Rarely","count":2}]},
        {"question_id": "q2", "heading": "Primary reason you subscribe to Peacock?",
         "answers": [{"text":"Live sports","count":8},{"text":"Originals","count":6},{"text":"Bundle","count":4},{"text":"NFL","count":3},{"text":"News","count":2},{"text":"Other","count":2}]},
        {"question_id": "q3", "heading": "Likelihood to keep subscription?",
         "answers": [{"text":"Very likely","count":7},{"text":"Likely","count":9},{"text":"Neutral","count":5},{"text":"Unlikely","count":3},{"text":"Very unlikely","count":1}]},
        {"question_id": "q4", "heading": "What would most increase your usage?",
         "answers": [{"text":"More live sports","count":10},{"text":"Better originals","count":8},{"text":"Lower price","count":7},{"text":"Fewer ads","count":6},{"text":"More movies","count":4},{"text":"Better UI","count":3}]},
        {"question_id": "q5", "heading": "NPS: Likelihood to recommend (0-10)",
         "answers": [{"text":"0","count":0},{"text":"1","count":0},{"text":"2","count":1},{"text":"3","count":0},{"text":"4","count":1},{"text":"5","count":2},{"text":"6","count":4},{"text":"7","count":7},{"text":"8","count":5},{"text":"9","count":3},{"text":"10","count":2}]},
    ]
