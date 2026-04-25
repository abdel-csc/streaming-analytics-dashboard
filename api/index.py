"""
Streaming Analytics Dashboard - FastAPI Backend
Endpoints:
  POST /api/predict  -- real sklearn GBR + Prophet viewership prediction
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
import os, json, requests, warnings
from functools import lru_cache

warnings.filterwarnings("ignore")

app = FastAPI(title="Streaming Analytics API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Paths ──
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# ── Load and train models at startup ──
@lru_cache(maxsize=1)
def get_models():
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, r2_score
    from prophet import Prophet

    shows = pd.read_csv(os.path.join(DATA_DIR, "show_ratings.csv"))

    # Label encoders
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
    mae = mean_absolute_error(yte, yp)
    r2  = r2_score(yte, yp)

    # Prophet model on weekly broadcast share
    weekly = pd.read_csv(os.path.join(DATA_DIR, "weekly_viewership.csv"), parse_dates=["week"])
    pdf = weekly[["week", "broadcast_share"]].rename(
        columns={"week": "ds", "broadcast_share": "y"}
    )
    pdf["nfl"]      = weekly["nfl_week"].values
    pdf["olympics"] = weekly["olympics"].values

    prophet = Prophet(
        changepoint_prior_scale=0.1,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
    )
    prophet.add_regressor("nfl")
    prophet.add_regressor("olympics")
    prophet.fit(pdf)

    return {
        "gbr":        gbr,
        "prophet":    prophet,
        "le_genre":   le_genre,
        "le_timeslot":le_timeslot,
        "le_network": le_network,
        "mae":        round(mae, 3),
        "r2":         round(r2, 3),
        "n_samples":  len(X),
        "feature_importances": dict(zip(feats, gbr.feature_importances_.tolist())),
    }


# ── Health ──
@app.get("/api/health")
def health():
    return {"status": "ok", "models_loaded": True}


# ── Data endpoint ──
@app.get("/api/data")
def get_data():
    shows  = pd.read_csv(os.path.join(DATA_DIR, "show_ratings.csv"))
    weekly = pd.read_csv(os.path.join(DATA_DIR, "weekly_viewership.csv"))
    subs   = pd.read_csv(os.path.join(DATA_DIR, "peacock_subscribers.csv"))

    # Trim weekly to YYYY-MM for frontend
    weekly["week"] = weekly["week"].astype(str).str[:7]

    return JSONResponse({
        "shows":  shows.to_dict(orient="records"),
        "weekly": weekly.to_dict(orient="records"),
        "subs":   subs.rename(columns={"quarter": "q", "subscribers_millions": "subs"}).to_dict(orient="records"),
    })


# ── Prediction endpoint ──
class PredictRequest(BaseModel):
    genre:         str
    network:       str
    timeslot:      str
    is_live:       int = 0
    target_year:   int = 2026
    budget_mult:   float = 1.0
    tentpole_mult: float = 1.0
    reboot_mult:   float = 1.0
    critic_score:  int = 72
    season_num:    int = 1
    forecast_weeks:int = 52


@app.post("/api/predict")
def predict(req: PredictRequest):
    m = get_models()

    # Encode inputs — handle unseen labels gracefully
    try:
        ge = int(m["le_genre"].transform([req.genre])[0])
    except ValueError:
        ge = 0
    try:
        te = int(m["le_timeslot"].transform([req.timeslot])[0])
    except ValueError:
        te = 0
    try:
        ne = int(m["le_network"].transform([req.network])[0])
    except ValueError:
        ne = 0

    is_streaming = 1 if req.network == "Peacock" else 0
    yo = req.target_year - 2019

    X_in = pd.DataFrame(
        [[ge, te, ne, req.is_live, is_streaming, yo]],
        columns=["ge", "te", "ne", "is_live", "is_streaming", "yo"],
    )

    base = float(m["gbr"].predict(X_in)[0])

    # Qualitative multipliers
    critic_mult = 0.85 + (req.critic_score / 100) * 0.30
    season_mult = max(0.75, 1.0 - (req.season_num - 1) * 0.02)
    adjusted    = base * req.budget_mult * req.tentpole_mult * req.reboot_mult * critic_mult * season_mult

    # Prophet forecast for broadcast share (context signal)
    prophet = m["prophet"]
    fut = prophet.make_future_dataframe(req.forecast_weeks, freq="W")
    fut["nfl"]      = pd.to_datetime(fut["ds"]).dt.month.isin([9,10,11,12,1,2]).astype(int)
    fut["olympics"] = 0
    fut.loc[(fut["ds"] >= "2028-07-14") & (fut["ds"] <= "2028-08-11"), "olympics"] = 1
    fc  = prophet.predict(fut)
    broadcast_forecast = round(float(fc.iloc[-1]["yhat"]), 2)

    return {
        "base_prediction":       round(base, 2),
        "adjusted_forecast":     round(adjusted, 2),
        "conservative":          round(adjusted * 0.80, 2),
        "optimistic":            round(adjusted * 1.20, 2),
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


# ── SurveyMonkey endpoint ──
@app.get("/api/survey")
def get_survey():
    token   = os.getenv("SURVEYMONKEY_TOKEN", "")
    survey_id = os.getenv("SURVEYMONKEY_SURVEY_ID", "")

    if not token or not survey_id:
        # Return placeholder structure if no credentials set
        return JSONResponse({
            "live": False,
            "message": "Set SURVEYMONKEY_TOKEN and SURVEYMONKEY_SURVEY_ID env vars to enable live data.",
            "responses": [],
            "question_summaries": _placeholder_survey_data(),
        })

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type":  "application/json",
    }

    try:
        # Fetch survey details
        survey_url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/rollups"
        resp = requests.get(survey_url, headers=headers, timeout=10)
        resp.raise_for_status()
        rollups = resp.json()

        # Parse question summaries
        summaries = []
        for page in rollups.get("pages", []):
            for q in page.get("questions", []):
                q_data = {
                    "question_id":   q.get("id"),
                    "heading":       q.get("heading", ""),
                    "type":          q.get("family", ""),
                    "answers":       [],
                }
                for row in q.get("answers", {}).get("rows", []):
                    q_data["answers"].append({
                        "text":  row.get("text", ""),
                        "count": row.get("count", 0),
                    })
                summaries.append(q_data)

        # Fetch response count
        details_url = f"https://api.surveymonkey.com/v3/surveys/{survey_id}/details"
        det = requests.get(details_url, headers=headers, timeout=10).json()
        response_count = det.get("response_count", 0)

        return JSONResponse({
            "live":             True,
            "response_count":   response_count,
            "question_summaries": summaries,
        })

    except requests.RequestException as e:
        raise HTTPException(status_code=502, detail=f"SurveyMonkey API error: {str(e)}")


def _placeholder_survey_data():
    """Placeholder data matching the 10 survey questions."""
    return [
        {"question_id": "q1", "heading": "How often do you watch content on Peacock?",
         "answers": [{"text":"Daily","count":3},{"text":"Several/week","count":7},
                     {"text":"Weekly","count":9},{"text":"Monthly","count":4},{"text":"Rarely","count":2}]},
        {"question_id": "q2", "heading": "Primary reason you subscribe to Peacock?",
         "answers": [{"text":"Live sports/events","count":8},{"text":"Original shows","count":6},
                     {"text":"Free with Comcast","count":4},{"text":"NFL games","count":3},
                     {"text":"News","count":2},{"text":"Other","count":2}]},
        {"question_id": "q3", "heading": "How likely are you to keep your Peacock subscription?",
         "answers": [{"text":"Very likely","count":7},{"text":"Likely","count":9},
                     {"text":"Neutral","count":5},{"text":"Unlikely","count":3},{"text":"Very unlikely","count":1}]},
        {"question_id": "q4", "heading": "What would most increase your Peacock usage?",
         "answers": [{"text":"More live sports","count":10},{"text":"Better originals","count":8},
                     {"text":"Lower price","count":7},{"text":"Fewer ads","count":6},
                     {"text":"More movies","count":4},{"text":"Better UI","count":3}]},
        {"question_id": "q5", "heading": "NPS: How likely to recommend Peacock? (0-10)",
         "answers": [{"text":"0","count":0},{"text":"1","count":0},{"text":"2","count":1},
                     {"text":"3","count":0},{"text":"4","count":1},{"text":"5","count":2},
                     {"text":"6","count":4},{"text":"7","count":7},{"text":"8","count":5},
                     {"text":"9","count":3},{"text":"10","count":2}]},
        {"question_id": "q6", "heading": "Which NBC shows do you watch on broadcast? (select all)",
         "answers": [{"text":"Chicago franchise","count":14},{"text":"The Voice","count":18},
                     {"text":"Law & Order franchise","count":12},{"text":"AGT","count":10},
                     {"text":"SNL","count":8},{"text":"None","count":3}]},
        {"question_id": "q7", "heading": "How did you first discover Peacock?",
         "answers": [{"text":"Olympics/live event","count":9},{"text":"Comcast bundle","count":7},
                     {"text":"Friend recommendation","count":4},{"text":"Social media ad","count":3},
                     {"text":"NBC promotion","count":2}]},
        {"question_id": "q8", "heading": "Which tier do you use?",
         "answers": [{"text":"Free (ad-supported)","count":8},{"text":"Premium","count":10},
                     {"text":"Premium Plus","count":4},{"text":"Bundled","count":3}]},
        {"question_id": "q9", "heading": "Compared to 12 months ago, your usage has:",
         "answers": [{"text":"Much more","count":4},{"text":"Somewhat more","count":8},
                     {"text":"About the same","count":6},{"text":"Somewhat less","count":5},
                     {"text":"Much less","count":2}]},
        {"question_id": "q10", "heading": "Age range",
         "answers": [{"text":"18-24","count":6},{"text":"25-34","count":9},{"text":"35-44","count":5},
                     {"text":"45-54","count":3},{"text":"55-64","count":1},{"text":"65+","count":1}]},
    ]
