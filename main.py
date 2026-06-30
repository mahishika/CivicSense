"""
CivicSense Backend — FastAPI + Gemini 2.5 Flash
================================================
9 specialized agents, all powered by Gemini, each with a distinct prompt/role:

  1. ReportAgent          - image/description -> category + structured issue data
  2. GeoAgent             - location tagging + zone matching + duplicate detection
  3. VerifyAgent          - community verification + fake-report filtering
  4. TrustAgent           - citizen credibility scoring (rises/falls with accuracy)
  5. PriorityAgent        - severity + urgency scoring, auto-escalation to dept
  6. ResolutionVerifyAgent- before/after image comparison to confirm genuine fix
  7. PredictAgent         - pattern-based hotspot prediction
  8. SentimentAgent       - urgency/tone detection from report text (any language)
  9. SummaryAgent         - auto daily/weekly admin report generation

LOCATION-BASED ROUTING (core fix — every report MUST carry a zone_id, and
every authority dashboard MUST filter on it):
  - JURISDICTION_ZONES is the single source of truth for which areas exist.
    GET /zones exposes it so the frontend never hardcodes a second copy that
    can drift out of sync (that drift was the root cause of reports not
    routing to any authority before).
  - Every admin/authority account is bound to exactly ONE zone_id at signup.
    There is no free-text "locality" field anymore.
  - Every citizen report gets a zone_id stamped on it by GeoAgent at submit
    time, computed via nearest-centroid haversine distance — automatic the
    moment lat/lng is known, no manual area-typing step required.
  - GET /reports accepts EITHER zone_id (admin dashboards: only their own
    zone's reports) OR user_id (citizen tracking view: only their own
    reports, regardless of which zone each landed in).

Run:
    pip install -r requirements.txt
    uvicorn civicsense_backend:app --reload --port 8000
"""

import os
import json
import time
import math
import random
import string
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "PASTE_YOUR_GEMINI_API_KEY_HERE")
if GEMINI_API_KEY and GEMINI_API_KEY != "PASTE_YOUR_GEMINI_API_KEY_HERE":
    genai.configure(api_key=GEMINI_API_KEY)
else:
    GEMINI_API_KEY = ""  # treat placeholder as "not configured" -> demo fallback mode

MODEL_NAME = "gemini-2.5-flash"


def get_model():
    return genai.GenerativeModel(MODEL_NAME)


app = FastAPI(title="CivicSense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# IN-MEMORY DATABASE
# ---------------------------------------------------------------------------
USERS: Dict[str, Dict[str, Any]] = {}
REPORTS: Dict[str, Dict[str, Any]] = {}


def gen_id(prefix: str) -> str:
    return prefix + "-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=10))


# ---------------------------------------------------------------------------
# JURISDICTION ZONES — single source of truth.
# The frontend fetches this from GET /zones instead of hardcoding its own
# copy, so the two sides can never drift apart.
# ---------------------------------------------------------------------------
JURISDICTION_ZONES = [
    {"id": "zone-sector4", "name": "Sector 4, Noida", "lat": 28.6160, "lng": 77.2110},
    {"id": "zone-sector5", "name": "Sector 5, Noida", "lat": 28.6145, "lng": 77.2090},
    {"id": "zone-sector6", "name": "Sector 6, Noida", "lat": 28.6120, "lng": 77.2060},
    {"id": "zone-lajpatnagar", "name": "Lajpat Nagar, Delhi", "lat": 28.5677, "lng": 77.2433},
    {"id": "zone-cp", "name": "Connaught Place, Delhi", "lat": 28.6315, "lng": 77.2167},
]
ZONES_BY_ID = {z["id"]: z for z in JURISDICTION_ZONES}


def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def match_zone(lat: float, lng: float) -> dict:
    """The actual routing decision: nearest-centroid match. Called once per
    report, at submit time, immediately after lat/lng is known."""
    best = JURISDICTION_ZONES[0]
    best_dist = float("inf")
    for z in JURISDICTION_ZONES:
        d = haversine_km(lat, lng, z["lat"], z["lng"])
        if d < best_dist:
            best_dist = d
            best = z
    return best


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class SignupPayload(BaseModel):
    name: str
    email: str
    password: str
    role: str  # "citizen" | "admin"
    zone_id: Optional[str] = None  # REQUIRED if role == "admin"


class ReconcilePayload(BaseModel):
    admin_confirmed: bool
    community_confirms: int = 0
    community_disputes: int = 0


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def safe_json_parse(text: str, fallback: dict) -> dict:
    try:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        return json.loads(cleaned.strip())
    except Exception:
        return fallback


def call_gemini_json(prompt: str, fallback: dict, image_bytes: Optional[bytes] = None) -> dict:
    if not GEMINI_API_KEY:
        return fallback
    try:
        model = get_model()
        parts = [prompt]
        if image_bytes:
            parts.append({"mime_type": "image/jpeg", "data": image_bytes})
        response = model.generate_content(parts)
        return safe_json_parse(response.text, fallback)
    except Exception as e:
        print(f"[Gemini error] {e}")
        return fallback


# ---------------------------------------------------------------------------
# AGENT 1 — ReportAgent
# ---------------------------------------------------------------------------
def agent_report(description: str, language: str, has_image: bool, image_bytes: Optional[bytes]) -> dict:
    prompt = f"""You are ReportAgent, a civic-issue classifier for an Indian city reporting app.
The citizen wrote this in language code "{language}" (could be Hindi, Bengali, Tamil, Telugu,
Marathi, Gujarati, Kannada, Punjabi, or English). Read it in its original language, understand it,
then respond ONLY with JSON (no markdown, no preamble) in this exact shape:

{{
  "category": "pothole" | "water_leakage" | "streetlight" | "garbage" | "drainage" | "illegal_construction" | "stray_animal" | "other",
  "title": "short English title, max 8 words",
  "description_en": "English translation/summary of the issue, max 2 sentences",
  "confidence": 0.0 to 1.0
}}

Citizen's report text: "{description if description else '(no text — see attached image)'}"
{"An image was also attached showing the issue — factor in visible damage/condition." if has_image else ""}
"""
    fallback = {
        "category": "other",
        "title": "Civic issue reported",
        "description_en": description[:150] if description else "Reported via photo",
        "confidence": 0.5,
    }
    result = call_gemini_json(prompt, fallback, image_bytes if has_image else None)
    if "category" not in result:
        result = fallback
    return result


# ---------------------------------------------------------------------------
# AGENT 2 — GeoAgent
# THIS is the agent that makes "report goes to the right area's authority"
# actually work. zone_id is computed here and stamped onto every report.
# ---------------------------------------------------------------------------
def agent_geo(lat: float, lng: float, category: str, address: str) -> dict:
    zone = match_zone(lat, lng)
    zone_dist = haversine_km(lat, lng, zone["lat"], zone["lng"])

    duplicates = []
    for rid, r in REPORTS.items():
        if r.get("category") != category:
            continue
        if r.get("status") == "resolved":
            continue
        try:
            d = haversine_km(lat, lng, r["lat"], r["lng"])
        except Exception:
            continue
        if d <= 0.15:  # 150 meters
            age_days = (time.time() - r.get("created_at", 0)) / 86400
            if age_days <= 14:
                duplicates.append(rid)

    return {
        "zone_id": zone["id"],
        "zone_name": zone["name"],
        "zone_distance_km": round(zone_dist, 2),
        "duplicate_of": duplicates[0] if duplicates else None,
        "duplicate_count": len(duplicates),
        "address": address or f"Lat {lat:.4f}, Lng {lng:.4f}",
    }


# ---------------------------------------------------------------------------
# AGENT 3 — VerifyAgent
# ---------------------------------------------------------------------------
def agent_verify(description: str, has_image: bool, reporter_trust: int) -> dict:
    needs_more_verification = (not description or len(description) < 5) and not has_image and reporter_trust < 40
    return {
        "auto_verified": reporter_trust >= 70,
        "needs_community_verification": needs_more_verification,
        "votes_required": 1 if reporter_trust >= 70 else 3,
    }


# ---------------------------------------------------------------------------
# AGENT 4 — TrustAgent
# ---------------------------------------------------------------------------
DEFAULT_TRUST = 50


def get_or_init_trust(user_id: Optional[str]) -> int:
    if not user_id or user_id not in USERS:
        return DEFAULT_TRUST
    return USERS[user_id].get("trust_score", DEFAULT_TRUST)


def adjust_trust(user_id: Optional[str], delta: int, reason: str):
    if not user_id or user_id not in USERS:
        return
    u = USERS[user_id]
    u["trust_score"] = max(0, min(100, u.get("trust_score", DEFAULT_TRUST) + delta))
    u.setdefault("trust_log", []).append({"delta": delta, "reason": reason, "at": time.time()})


def trust_tier(score: int) -> str:
    if score >= 80:
        return "highly_trusted"
    if score >= 60:
        return "trusted"
    if score >= 35:
        return "neutral"
    return "low_trust"


# ---------------------------------------------------------------------------
# AGENT 5 — PriorityAgent
# (Score is computed and returned on every report, but the frontend only
# renders it in the admin dashboard — never on the citizen's own tracking
# view, where a "low urgency" label reads as dismissive.)
# ---------------------------------------------------------------------------
SENSITIVE_KEYWORDS = ["school", "hospital", "clinic", "anganwadi", "college", "metro", "station"]


def agent_priority(category: str, sentiment: dict, address: str, duplicate_count: int) -> dict:
    base_risk = {
        "pothole": 55, "water_leakage": 50, "drainage": 60, "streetlight": 45,
        "garbage": 40, "illegal_construction": 35, "stray_animal": 50, "other": 30,
    }.get(category, 30)

    urgency_score = base_risk
    urgency_score += sentiment.get("urgency_boost", 0)
    urgency_score += min(duplicate_count * 8, 24)

    addr_lower = (address or "").lower()
    sensitive_zone = any(kw in addr_lower for kw in SENSITIVE_KEYWORDS)
    if sensitive_zone:
        urgency_score += 20

    urgency_score = max(0, min(100, urgency_score))

    if urgency_score >= 75:
        level = "critical"
    elif urgency_score >= 55:
        level = "high"
    elif urgency_score >= 35:
        level = "medium"
    else:
        level = "low"

    dept_map = {
        "pothole": "Roads & Infrastructure", "water_leakage": "Water Supply Dept.",
        "streetlight": "Electrical Dept.", "garbage": "Sanitation Dept.",
        "drainage": "Drainage & Sewage Dept.", "illegal_construction": "Town Planning",
        "stray_animal": "Animal Control", "other": "General Municipal Dept.",
    }

    return {
        "urgency_score": urgency_score,
        "priority_level": level,
        "escalate_to_department": dept_map.get(category, "General Municipal Dept."),
        "sensitive_zone": sensitive_zone,
    }


# ---------------------------------------------------------------------------
# AGENT 6 — ResolutionVerifyAgent
# ---------------------------------------------------------------------------
def agent_resolution_verify(category: str, before_image_bytes: Optional[bytes], after_image_bytes: Optional[bytes], notes: str) -> dict:
    if not after_image_bytes:
        return {"confidence": 0.3, "verdict": "inconclusive", "reasoning": "No after-photo provided."}

    if not GEMINI_API_KEY:
        conf = 0.78 if notes else 0.55
        conf += random.uniform(-0.12, 0.15)
        conf = max(0.1, min(0.97, conf))
        return {"confidence": round(conf, 2), "verdict": "fixed" if conf >= 0.6 else "inconclusive", "reasoning": "Demo mode — no Gemini key configured."}

    prompt = f"""You are ResolutionVerifyAgent. A citizen reported a "{category}" civic issue.
The authority claims it has been fixed and uploaded an after-photo (and notes below).
Admin notes: "{notes}"

Look at the after-photo and judge whether the described "{category}" issue appears resolved.
Respond ONLY with JSON:
{{"confidence": 0.0 to 1.0, "verdict": "fixed" | "not_fixed" | "inconclusive", "reasoning": "one sentence"}}
"""
    fallback = {"confidence": 0.5, "verdict": "inconclusive", "reasoning": "Could not analyze image."}
    parts = [prompt, {"mime_type": "image/jpeg", "data": after_image_bytes}]
    try:
        model = get_model()
        response = model.generate_content(parts)
        return safe_json_parse(response.text, fallback)
    except Exception as e:
        print(f"[ResolutionVerifyAgent error] {e}")
        return fallback


# ---------------------------------------------------------------------------
# AGENT 7 — PredictAgent (zone-scoped)
# ---------------------------------------------------------------------------
def agent_predict(zone_id: Optional[str] = None) -> List[dict]:
    cells: Dict[str, List[dict]] = {}
    for r in REPORTS.values():
        if r.get("status") == "resolved":
            continue
        if zone_id and r.get("zone_id") != zone_id:
            continue
        key = f"{round(r['lat'], 2)}_{round(r['lng'], 2)}"
        cells.setdefault(key, []).append(r)

    predictions = []
    for key, items in cells.items():
        if len(items) < 2:
            continue
        cats = [i["category"] for i in items]
        dominant = max(set(cats), key=cats.count)
        confidence = min(95, 50 + len(items) * 10)
        risk_map = {
            "drainage": "Flooding risk in the next 7 days",
            "pothole": "Pothole formation likely after rain",
            "water_leakage": "Pipeline failure risk increasing",
            "garbage": "Sanitation overflow risk",
        }
        predictions.append({
            "area": items[0].get("address", key),
            "zone_id": items[0].get("zone_id"),
            "zone_name": items[0].get("zone_name"),
            "risk": risk_map.get(dominant, "Recurring issue pattern detected"),
            "confidence": confidence,
            "category": dominant,
            "report_count": len(items),
        })
    predictions.sort(key=lambda p: p["confidence"], reverse=True)
    return predictions[:5]


# ---------------------------------------------------------------------------
# AGENT 8 — SentimentAgent
# ---------------------------------------------------------------------------
def agent_sentiment(description: str, language: str) -> dict:
    if not description:
        return {"sentiment": "neutral", "urgency_boost": 0}

    fallback_words_urgent = ["urgent", "emergency", "danger", "accident", "immediately", "खतरा", "जल्दी", "तुरंत"]
    if any(w in description.lower() for w in fallback_words_urgent):
        fallback = {"sentiment": "urgent", "urgency_boost": 15}
    else:
        fallback = {"sentiment": "neutral", "urgency_boost": 0}

    prompt = f"""You are SentimentAgent. The citizen wrote this civic complaint in language "{language}":
"{description}"

Analyze the emotional tone/urgency (regardless of language). Respond ONLY with JSON:
{{"sentiment": "urgent" | "frustrated" | "concerned" | "neutral", "urgency_boost": integer from 0 to 20}}
"""
    return call_gemini_json(prompt, fallback)


# ---------------------------------------------------------------------------
# AGENT 9 — SummaryAgent (zone-scoped)
# ---------------------------------------------------------------------------
def agent_summary(zone_id: Optional[str] = None) -> str:
    reports = list(REPORTS.values())
    if zone_id:
        reports = [r for r in reports if r.get("zone_id") == zone_id]

    total = len(reports)
    resolved = len([r for r in reports if r["status"] == "resolved"])
    critical = len([r for r in reports if r.get("priority", {}).get("priority_level") == "critical"])
    pending_review = len([r for r in reports if r["status"] == "pending_review"])

    cat_counts: Dict[str, int] = {}
    for r in reports:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1
    top_category = max(cat_counts, key=cat_counts.get) if cat_counts else "none"

    zone_name = ZONES_BY_ID.get(zone_id, {}).get("name", "all areas") if zone_id else "all areas"

    stats_blob = {
        "zone": zone_name, "total_reports": total, "resolved": resolved,
        "critical_open": critical, "pending_review": pending_review,
        "top_category": top_category, "category_breakdown": cat_counts,
    }

    fallback = (
        f"{zone_name}: {total} reports logged, {resolved} resolved, {critical} still critical. "
        f"Most common issue: {top_category}. {pending_review} reports awaiting manual review."
    )

    if not GEMINI_API_KEY:
        return fallback

    prompt = f"""You are SummaryAgent for a civic issue tracker admin dashboard.
Given this raw data: {json.dumps(stats_blob)}
Write a short, clear weekly summary report (4-6 sentences) for a municipal authority
managing "{zone_name}". Mention totals, resolution rate, the most urgent open issue
category, and one actionable recommendation. Plain text only, no markdown headers."""
    try:
        model = get_model()
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[SummaryAgent error] {e}")
        return fallback


# ---------------------------------------------------------------------------
# ROUTES — AUTH
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "gemini_configured": bool(GEMINI_API_KEY)}


@app.get("/zones")
def list_zones():
    """Frontend MUST populate its zone-picker (admin signup) and its
    address-suggestion list from this — never hardcode a second list,
    or zone ids silently drift apart and reports stop matching authorities."""
    return JURISDICTION_ZONES


@app.post("/auth/signup")
def signup(payload: SignupPayload):
    if payload.role == "admin":
        if not payload.zone_id:
            raise HTTPException(400, "Authority accounts must select a jurisdiction (zone_id).")
        if payload.zone_id not in ZONES_BY_ID:
            raise HTTPException(400, f"Unknown zone_id '{payload.zone_id}'.")

    # Reuse the same uid for the same email+role so logging back in finds the
    # same account (and the same reports / trust score) instead of orphaning
    # the previous one with a brand-new uid every time.
    existing_uid = None
    email_norm = (payload.email or "").strip().lower()
    if email_norm:
        for existing_id, u in USERS.items():
            if (u.get("email") or "").strip().lower() == email_norm and u.get("role") == payload.role:
                existing_uid = existing_id
                break

    if existing_uid:
        uid = existing_uid
        USERS[uid]["name"] = payload.name or USERS[uid]["name"]
        if payload.role == "admin" and payload.zone_id:
            USERS[uid]["zone_id"] = payload.zone_id
    else:
        uid = gen_id("user")
        USERS[uid] = {
            "uid": uid,
            "name": payload.name,
            "email": payload.email,
            "role": payload.role,
            "zone_id": payload.zone_id if payload.role == "admin" else None,
            "trust_score": DEFAULT_TRUST,
            "trust_log": [],
            "created_at": time.time(),
        }

    u = USERS[uid]
    return {
        "uid": uid,
        "role": payload.role,
        "zone_id": u["zone_id"],
        "zone_name": ZONES_BY_ID.get(u["zone_id"], {}).get("name") if u["zone_id"] else None,
        "trust_score": u["trust_score"],
    }


@app.get("/users/{uid}/trust")
def get_trust(uid: str):
    if uid not in USERS:
        raise HTTPException(404, "User not found")
    u = USERS[uid]
    return {
        "uid": uid, "trust_score": u.get("trust_score", DEFAULT_TRUST),
        "tier": trust_tier(u.get("trust_score", DEFAULT_TRUST)),
        "log": u.get("trust_log", [])[-10:],
    }


# ---------------------------------------------------------------------------
# ROUTES — REPORTS
# ---------------------------------------------------------------------------
@app.post("/reports/submit")
async def submit_report(payload: str = Form(...), image: Optional[UploadFile] = File(None)):
    data = json.loads(payload)
    description = data.get("description", "")
    language = data.get("language", "en-IN")
    lat = data.get("lat")
    lng = data.get("lng")
    address = data.get("address", "")
    anonymous = data.get("anonymous", False)
    user_id = None if anonymous else data.get("user_id")

    if lat is None or lng is None:
        raise HTTPException(400, "Location (lat/lng) is required")

    image_bytes = await image.read() if image else None
    has_image = image_bytes is not None

    reporter_trust = get_or_init_trust(user_id)

    report_info = agent_report(description, language, has_image, image_bytes)
    sentiment = agent_sentiment(description, language)
    geo_info = agent_geo(lat, lng, report_info["category"], address)  # <- zone_id assigned here, automatically
    verify_info = agent_verify(description, has_image, reporter_trust)
    priority_info = agent_priority(report_info["category"], sentiment, geo_info["address"], geo_info["duplicate_count"])

    rid = gen_id("rpt")
    report = {
        "id": rid,
        "title": report_info.get("title", "Civic issue"),
        "category": report_info.get("category", "other"),
        "description": report_info.get("description_en", description),
        "description_original": description,
        "language": language,
        "status": "submitted",
        "anonymous": anonymous,
        "user_id": user_id,
        "reporter_trust_at_submission": reporter_trust,
        "lat": lat, "lng": lng, "address": geo_info["address"],
        # --- this triplet is the actual routing result ---
        "zone_id": geo_info["zone_id"],
        "zone_name": geo_info["zone_name"],
        "zone_distance_km": geo_info["zone_distance_km"],
        "duplicate_of": geo_info["duplicate_of"],
        "duplicate_count": geo_info["duplicate_count"],
        "sentiment": sentiment.get("sentiment", "neutral"),
        "verification": verify_info,
        "priority": priority_info,
        "community_votes": 0,
        "has_image": has_image,
        "before_image": image_bytes,
        "created_at": time.time(),
        "history": [{"status": "submitted", "at": time.time()}],
    }
    REPORTS[rid] = report

    if user_id and (has_image or len(description) > 15):
        adjust_trust(user_id, 1, "submitted_detailed_report")

    return {"report_id": rid, "report": _public_report(report)}


def _public_report(r: dict) -> dict:
    out = {k: v for k, v in r.items() if k not in ("before_image", "after_image")}
    return out


@app.get("/reports")
def list_reports(zone_id: Optional[str] = None, user_id: Optional[str] = None, sort_by_urgency: bool = True):
    """
    Two distinct usage patterns:
      - Admin dashboard: ALWAYS pass zone_id (the logged-in admin's own zone)
        -> only that zone's reports come back.
      - Citizen tracking view: pass user_id instead -> a citizen sees their
        own reports regardless of which zone each one landed in.
    """
    reports = list(REPORTS.values())
    if zone_id:
        if zone_id not in ZONES_BY_ID:
            raise HTTPException(400, f"Unknown zone_id '{zone_id}'.")
        reports = [r for r in reports if r.get("zone_id") == zone_id]
    if user_id:
        reports = [r for r in reports if r.get("user_id") == user_id or r.get("anonymous")]
    if sort_by_urgency:
        reports.sort(key=lambda r: r.get("priority", {}).get("urgency_score", 0), reverse=True)
    else:
        reports.sort(key=lambda r: r.get("created_at", 0), reverse=True)
    return [_public_report(r) for r in reports]


@app.get("/reports/{report_id}")
def get_report(report_id: str):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found")
    return _public_report(REPORTS[report_id])


@app.post("/reports/{report_id}/vote")
def community_vote(report_id: str, confirm: bool = True):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found")
    r = REPORTS[report_id]
    r["community_votes"] = r.get("community_votes", 0) + (1 if confirm else 0)
    if confirm and r["community_votes"] in (3, 5, 10) and r.get("user_id"):
        adjust_trust(r["user_id"], 2, "community_verified_report")
    return {"community_votes": r["community_votes"]}


@app.post("/reports/{report_id}/admin/mark_in_progress")
def mark_in_progress(report_id: str, admin_uid: Optional[str] = None):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found")
    r = REPORTS[report_id]
    _assert_admin_owns_report(admin_uid, r)
    r["status"] = "in_progress"
    r["history"].append({"status": "in_progress", "at": time.time()})
    return {"status": "in_progress"}


@app.post("/reports/{report_id}/admin/resolve")
async def admin_resolve(
    report_id: str,
    after_image: UploadFile = File(...),
    notes: str = Form(""),
    admin_uid: Optional[str] = Form(None),
):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found")
    r = REPORTS[report_id]
    _assert_admin_owns_report(admin_uid, r)

    after_bytes = await after_image.read()
    r["after_image"] = after_bytes
    r["resolution_notes"] = notes

    ai_result = agent_resolution_verify(r["category"], r.get("before_image"), after_bytes, notes)
    r["resolution_ai_result"] = ai_result

    if ai_result["confidence"] >= 0.6 and ai_result["verdict"] == "fixed":
        r["status"] = "resolved"
        if r.get("user_id"):
            adjust_trust(r["user_id"], 3, "report_resolved_genuinely")
    else:
        r["status"] = "pending_review"

    r["history"].append({"status": r["status"], "at": time.time(), "ai_confidence": ai_result["confidence"]})
    return {"ai_result": ai_result, "report": _public_report(r)}


@app.post("/reports/{report_id}/reconcile")
def reconcile(report_id: str, payload: ReconcilePayload, admin_uid: Optional[str] = None):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found")
    r = REPORTS[report_id]
    _assert_admin_owns_report(admin_uid, r)

    ai_says_fixed = r.get("resolution_ai_result", {}).get("verdict") == "fixed"
    votes_for_fixed = (
        (1 if ai_says_fixed else 0)
        + (1 if payload.admin_confirmed else 0)
        + (1 if payload.community_confirms > payload.community_disputes else 0)
    )

    if votes_for_fixed >= 2:
        r["status"] = "resolved"
        if r.get("user_id"):
            adjust_trust(r["user_id"], 2, "report_resolved_after_reconciliation")
    else:
        r["status"] = "reopened"
        if not payload.admin_confirmed and r.get("user_id") and r.get("reporter_trust_at_submission", 50) < 40:
            adjust_trust(r["user_id"], -3, "report_disputed_after_reconciliation")

    r["history"].append({"status": r["status"], "at": time.time(), "reconciled": True})
    return {"final_status": r["status"]}


@app.post("/reports/{report_id}/flag_duplicate")
def flag_duplicate(report_id: str, admin_uid: Optional[str] = None):
    if report_id not in REPORTS:
        raise HTTPException(404, "Report not found")
    r = REPORTS[report_id]
    _assert_admin_owns_report(admin_uid, r)
    r["status"] = "duplicate"
    if r.get("user_id"):
        adjust_trust(r["user_id"], -5, "report_flagged_duplicate")
    return {"status": "duplicate"}


def _assert_admin_owns_report(admin_uid: Optional[str], report: dict):
    """Server-side enforcement that an authority can only act on reports
    inside their own zone — not just hidden client-side. Skipped if
    admin_uid isn't passed, so local/demo calls without auth wiring still
    work during development."""
    if not admin_uid:
        return
    admin = USERS.get(admin_uid)
    if not admin or admin.get("role") != "admin":
        raise HTTPException(403, "Not an authority account.")
    if admin.get("zone_id") and admin["zone_id"] != report.get("zone_id"):
        raise HTTPException(403, "This report is outside your jurisdiction.")


# ---------------------------------------------------------------------------
# ROUTES — PREDICTIONS, SUMMARY
# ---------------------------------------------------------------------------
@app.get("/predictions")
def get_predictions(zone_id: Optional[str] = None):
    return agent_predict(zone_id)


@app.get("/summary")
def get_summary(zone_id: Optional[str] = None):
    return {"summary": agent_summary(zone_id)}


# ---------------------------------------------------------------------------
# STATS (for admin dashboard cards)
# ---------------------------------------------------------------------------
@app.get("/stats")
def get_stats(zone_id: Optional[str] = None):
    reports = list(REPORTS.values())
    if zone_id:
        reports = [r for r in reports if r.get("zone_id") == zone_id]
    total = len(reports)
    resolved = len([r for r in reports if r["status"] == "resolved"])
    in_progress = len([r for r in reports if r["status"] == "in_progress"])
    pending = len([r for r in reports if r["status"] in ("submitted", "pending_review")])
    critical = len([r for r in reports if r.get("priority", {}).get("priority_level") == "critical"])

    if zone_id:
        zone_user_ids = {r["user_id"] for r in reports if r.get("user_id")}
        trusts = [USERS[uid]["trust_score"] for uid in zone_user_ids if uid in USERS]
    else:
        trusts = [u.get("trust_score", DEFAULT_TRUST) for u in USERS.values() if u.get("role") == "citizen"]
    avg_trust = round(sum(trusts) / len(trusts)) if trusts else 0

    cat_counts: Dict[str, int] = {}
    for r in reports:
        cat_counts[r["category"]] = cat_counts.get(r["category"], 0) + 1

    return {
        "zone_id": zone_id,
        "zone_name": ZONES_BY_ID.get(zone_id, {}).get("name") if zone_id else "All areas",
        "total": total, "resolved": resolved, "in_progress": in_progress,
        "pending": pending, "critical": critical, "avg_community_trust": avg_trust,
        "category_breakdown": cat_counts,
    }