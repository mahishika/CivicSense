"""
CivicSense Backend — COMPLETE INTEGRATION
===========================================
FastAPI + Gemini 2.5 Flash + Firestore + Google Cloud Storage + Google Maps API
9 specialized agents with real data persistence, media storage, and geo-tagging.

Supports all deployment phases (local, Firebase, GCS, Maps, Cloud Run)

Run:
    pip install -r requirements.txt
    python civicsense_backend_complete.py
"""

import os
import io
import json
import time
import math
import random
import string
from typing import Optional, List, Dict, Any
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import storage as gcs_storage
from google.maps import client as gmaps
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG & KEYS
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
FIREBASE_CREDS_PATH = os.environ.get("FIREBASE_CREDENTIALS_PATH", "./firebase-key.json")
GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME", "civicsense-media")
GOOGLE_MAPS_API_KEY = os.environ.get("GOOGLE_MAPS_API_KEY", "")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "gemini-2.5-flash"

# ---------------------------------------------------------------------------
# FIREBASE INIT
# ---------------------------------------------------------------------------
db = None
if os.path.exists(FIREBASE_CREDS_PATH):
    try:
        cred = credentials.Certificate(FIREBASE_CREDS_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase Firestore initialized")
    except Exception as e:
        print(f"⚠️ Firebase init failed: {e}")

# ---------------------------------------------------------------------------
# GOOGLE CLOUD STORAGE INIT
# ---------------------------------------------------------------------------
gcs_client = None
gcs_bucket = None
try:
    if os.path.exists(FIREBASE_CREDS_PATH):
        gcs_client = gcs_storage.Client.from_service_account_json(FIREBASE_CREDS_PATH)
        try:
            gcs_bucket = gcs_client.bucket(GCS_BUCKET_NAME)
            gcs_bucket.reload()
            print(f"✅ Google Cloud Storage connected to bucket: {GCS_BUCKET_NAME}")
        except Exception as e:
            print(f"⚠️ GCS bucket '{GCS_BUCKET_NAME}' not found or not accessible: {e}")
            print(f"   → Create it manually: gsutil mb gs://{GCS_BUCKET_NAME}")
            gcs_bucket = None
except Exception as e:
    print(f"⚠️ GCS initialization failed: {e}")

# ---------------------------------------------------------------------------
# GOOGLE MAPS INIT
# ---------------------------------------------------------------------------
gmaps_client = None
if GOOGLE_MAPS_API_KEY:
    try:
        gmaps_client = gmaps.Client(key=GOOGLE_MAPS_API_KEY)
        print("✅ Google Maps API initialized")
    except Exception as e:
        print(f"⚠️ Google Maps init failed: {e}")

# ---------------------------------------------------------------------------
# FastAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(title="CivicSense API — Complete")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def gen_id(prefix: str) -> str:
    return prefix + "-" + "".join(random.choices(string.ascii_lowercase + string.digits, k=10))


# ---------------------------------------------------------------------------
# JURISDICTION ZONES
# ---------------------------------------------------------------------------
JURISDICTION_ZONES = [
    {"id": "zone-sector4", "name": "Sector 4, Noida", "lat": 28.6160, "lng": 77.2110},
    {"id": "zone-sector5", "name": "Sector 5, Noida", "lat": 28.6145, "lng": 77.2090},
    {"id": "zone-sector6", "name": "Sector 6, Noida", "lat": 28.6120, "lng": 77.2060},
    {"id": "zone-lajpatnagar", "name": "Lajpat Nagar, Delhi", "lat": 28.5677, "lng": 77.2433},
    {"id": "zone-cp", "name": "Connaught Place, Delhi", "lat": 28.6315, "lng": 77.2167},
]
ZONES_BY_ID = {z["id"]: z for z in JURISDICTION_ZONES}


# ---------------------------------------------------------------------------
# SCHEMAS
# ---------------------------------------------------------------------------
class SignupPayload(BaseModel):
    name: str
    email: str
    password: str
    role: str
    zone_id: Optional[str] = None


class ReconcilePayload(BaseModel):
    admin_confirmed: bool
    community_confirms: int = 0
    community_disputes: int = 0


# ---------------------------------------------------------------------------
# UTILITIES
# ---------------------------------------------------------------------------
def haversine_km(lat1, lng1, lat2, lng2):
    R = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def match_zone(lat: float, lng: float) -> dict:
    best = JURISDICTION_ZONES[0]
    best_dist = float("inf")
    for z in JURISDICTION_ZONES:
        d = haversine_km(lat, lng, z["lat"], z["lng"])
        if d < best_dist:
            best_dist = d
            best = z
    return best


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
        model = genai.GenerativeModel(MODEL_NAME)
        parts = [prompt]
        if image_bytes:
            parts.append({"mime_type": "image/jpeg", "data": image_bytes})
        response = model.generate_content(parts)
        return safe_json_parse(response.text, fallback)
    except Exception as e:
        print(f"[Gemini error] {e}")
        return fallback


# ---------------------------------------------------------------------------
# PHASE 3: GOOGLE CLOUD STORAGE
# ---------------------------------------------------------------------------
def upload_to_gcs(report_id: str, file_bytes: bytes, file_type: str = "before") -> Optional[str]:
    """Upload image to GCS, return public URL"""
    if not gcs_bucket or not file_bytes:
        return None
    try:
        timestamp = int(time.time())
        blob_name = f"reports/{report_id}/{file_type}-{timestamp}.jpg"
        blob = gcs_bucket.blob(blob_name)
        blob.upload_from_string(file_bytes, content_type="image/jpeg")
        # Make public
        blob.make_public()
        url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{blob_name}"
        print(f"✅ Uploaded to GCS: {url}")
        return url
    except Exception as e:
        print(f"⚠️ GCS upload failed: {e}")
        return None


# ---------------------------------------------------------------------------
# PHASE 4: GOOGLE MAPS API
# ---------------------------------------------------------------------------
def get_real_address(lat: float, lng: float) -> str:
    """Get real address from Google Maps Reverse Geocoding"""
    if not gmaps_client:
        return f"Lat {lat:.4f}, Lng {lng:.4f}"
    try:
        result = gmaps_client.reverse_geocode((lat, lng))
        if result:
            return result[0]["formatted_address"]
    except Exception as e:
        print(f"⚠️ Maps reverse geocode failed: {e}")
    return f"Lat {lat:.4f}, Lng {lng:.4f}"


def get_nearby_landmarks(lat: float, lng: float) -> List[str]:
    """Get nearby landmarks from Google Maps Places API"""
    if not gmaps_client:
        return []
    try:
        places = gmaps_client.places_nearby(location=(lat, lng), radius=500, type="landmark")
        landmarks = [p["name"] for p in places.get("results", [])][:3]
        return landmarks
    except Exception as e:
        print(f"⚠️ Maps places nearby failed: {e}")
    return []


# ---------------------------------------------------------------------------
# FIRESTORE HELPERS
# ---------------------------------------------------------------------------
DEFAULT_TRUST = 50


def get_user_from_firestore(uid: str) -> Optional[Dict]:
    if not db:
        return None
    try:
        doc = db.collection("users").document(uid).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"[Firestore error] {e}")
        return None


def save_user_to_firestore(uid: str, data: Dict):
    if not db:
        return
    try:
        db.collection("users").document(uid).set(data, merge=True)
    except Exception as e:
        print(f"[Firestore error] {e}")


def get_report_from_firestore(rid: str) -> Optional[Dict]:
    if not db:
        return None
    try:
        doc = db.collection("reports").document(rid).get()
        return doc.to_dict() if doc.exists else None
    except Exception as e:
        print(f"[Firestore error] {e}")
        return None


def save_report_to_firestore(rid: str, data: Dict):
    if not db:
        return
    try:
        # Don't store binary in Firestore
        data_to_save = {k: v for k, v in data.items() if k not in ("before_image", "after_image")}
        data_to_save["has_before_image"] = "before_image_url" in data and data.get("before_image_url")
        data_to_save["has_after_image"] = "after_image_url" in data and data.get("after_image_url")
        db.collection("reports").document(rid).set(data_to_save, merge=True)
    except Exception as e:
        print(f"[Firestore error] {e}")


def get_all_reports_from_firestore(zone_id: Optional[str] = None) -> List[Dict]:
    if not db:
        return []
    try:
        query = db.collection("reports")
        if zone_id:
            query = query.where("zone_id", "==", zone_id)
        docs = query.stream()
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        print(f"[Firestore error] {e}")
        return []


# ---------------------------------------------------------------------------
# AI AGENTS (1-9)
# ---------------------------------------------------------------------------
def agent_report(description: str, language: str, has_image: bool, image_bytes: Optional[bytes]) -> dict:
    prompt = f"""You are ReportAgent. Classify this civic issue (in language "{language}"):
"{description if description else '(reported via photo)'}"
Respond ONLY with JSON: {{"category": "pothole"|"water_leakage"|"streetlight"|"garbage"|"drainage"|"illegal_construction"|"stray_animal"|"other", "title": "short title", "description_en": "English summary", "confidence": 0.0-1.0}}"""
    fallback = {"category": "other", "title": "Civic issue", "description_en": description[:150] if description else "Reported via photo", "confidence": 0.5}
    return call_gemini_json(prompt, fallback, image_bytes if has_image else None)


def agent_geo(lat: float, lng: float, category: str, address: str) -> dict:
    # Use real Google Maps if available
    real_address = get_real_address(lat, lng) if gmaps_client else address
    nearby = get_nearby_landmarks(lat, lng) if gmaps_client else []
    
    zone = match_zone(lat, lng)
    zone_dist = haversine_km(lat, lng, zone["lat"], zone["lng"])

    duplicates = []
    reports = get_all_reports_from_firestore()
    for r in reports:
        if r.get("category") != category or r.get("status") == "resolved":
            continue
        try:
            d = haversine_km(lat, lng, r["lat"], r["lng"])
            if d <= 0.15:
                age_days = (time.time() - r.get("created_at", 0)) / 86400
                if age_days <= 14:
                    duplicates.append(r["id"])
        except:
            pass

    return {
        "zone_id": zone["id"],
        "zone_name": zone["name"],
        "zone_distance_km": round(zone_dist, 2),
        "address": real_address,
        "nearby_landmarks": nearby,
        "duplicate_of": duplicates[0] if duplicates else None,
        "duplicate_count": len(duplicates),
    }


def agent_verify(description: str, has_image: bool, reporter_trust: int) -> dict:
    return {
        "auto_verified": reporter_trust >= 70,
        "needs_community_verification": (not description or len(description) < 5) and not has_image and reporter_trust < 40,
        "votes_required": 1 if reporter_trust >= 70 else 3,
    }


def agent_priority(category: str, sentiment: dict, address: str, duplicate_count: int) -> dict:
    base_risk = {"pothole": 55, "water_leakage": 50, "drainage": 60, "streetlight": 45, "garbage": 40, "illegal_construction": 35, "stray_animal": 50, "other": 30}.get(category, 30)
    urgency_score = base_risk + sentiment.get("urgency_boost", 0) + min(duplicate_count * 8, 24)
    
    sensitive_keywords = ["school", "hospital", "clinic", "college", "metro", "station"]
    if any(kw in (address or "").lower() for kw in sensitive_keywords):
        urgency_score += 20
    
    urgency_score = max(0, min(100, urgency_score))
    level = "critical" if urgency_score >= 75 else "high" if urgency_score >= 55 else "medium" if urgency_score >= 35 else "low"
    
    dept_map = {
        "pothole": "Roads & Infrastructure", "water_leakage": "Water Supply",
        "streetlight": "Electrical", "garbage": "Sanitation",
        "drainage": "Drainage & Sewage", "illegal_construction": "Town Planning",
        "stray_animal": "Animal Control", "other": "General Municipal",
    }
    
    return {"urgency_score": urgency_score, "priority_level": level, "escalate_to_department": dept_map.get(category, "General")}


def agent_resolution_verify(category: str, after_image_bytes: Optional[bytes], notes: str) -> dict:
    if not after_image_bytes:
        return {"confidence": 0.3, "verdict": "inconclusive", "reasoning": "No after-photo."}
    
    if not GEMINI_API_KEY:
        conf = round(random.uniform(0.5, 0.9), 2)
        return {"confidence": conf, "verdict": "fixed" if conf >= 0.6 else "inconclusive", "reasoning": "Demo mode"}
    
    prompt = f"""ResolutionVerifyAgent: Did the admin fix this "{category}" issue?
Admin notes: "{notes}"
Look at the after-photo and respond ONLY with JSON: {{"confidence": 0.0-1.0, "verdict": "fixed"|"not_fixed"|"inconclusive", "reasoning": "one sentence"}}"""
    fallback = {"confidence": 0.5, "verdict": "inconclusive", "reasoning": "Could not analyze"}
    
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content([prompt, {"mime_type": "image/jpeg", "data": after_image_bytes}])
        return safe_json_parse(response.text, fallback)
    except Exception as e:
        print(f"[ResolutionVerifyAgent error] {e}")
        return fallback


def agent_sentiment(description: str, language: str) -> dict:
    fallback = {"sentiment": "neutral", "urgency_boost": 0}
    if not description:
        return fallback
    
    urgent_words = ["urgent", "emergency", "danger", "immediately", "critical", "खतरा", "জরুরি"]
    if any(w in description.lower() for w in urgent_words):
        return {"sentiment": "urgent", "urgency_boost": 15}
    
    prompt = f"""Analyze tone: "{description[:200]}"
Respond ONLY with JSON: {{"sentiment": "urgent"|"frustrated"|"concerned"|"neutral", "urgency_boost": 0-20}}"""
    return call_gemini_json(prompt, fallback)


def agent_predict(zone_id: Optional[str] = None) -> List[dict]:
    cells = {}
    for r in get_all_reports_from_firestore(zone_id):
        if r.get("status") != "resolved":
            key = f"{round(r['lat'], 2)}_{round(r['lng'], 2)}"
            cells.setdefault(key, []).append(r)
    
    predictions = []
    for key, items in cells.items():
        if len(items) < 2:
            continue
        cats = [i["category"] for i in items]
        dominant = max(set(cats), key=cats.count)
        confidence = min(95, 50 + len(items) * 10)
        risk_map = {"drainage": "Flooding risk", "pothole": "Pothole formation likely", "water_leakage": "Pipeline failure risk", "garbage": "Sanitation overflow"}
        predictions.append({
            "area": items[0].get("address"),
            "risk": risk_map.get(dominant, "Recurring issue"),
            "confidence": confidence,
            "category": dominant,
            "report_count": len(items),
        })
    return sorted(predictions, key=lambda p: p["confidence"], reverse=True)[:5]


def agent_summary(zone_id: Optional[str] = None) -> str:
    reports = get_all_reports_from_firestore(zone_id)
    total, resolved, critical = len(reports), len([r for r in reports if r["status"] == "resolved"]), len([r for r in reports if r.get("priority", {}).get("priority_level") == "critical"])
    zone_name = ZONES_BY_ID.get(zone_id, {}).get("name", "all areas") if zone_id else "all areas"
    
    if not GEMINI_API_KEY:
        return f"{zone_name}: {total} reports, {resolved} resolved, {critical} critical open."
    
    prompt = f"""Summary for {zone_name}: {total} total, {resolved} resolved, {critical} critical open.
Write 3-4 sentence summary for a municipal authority. Plain text."""
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text.strip()
    except:
        return f"{zone_name}: {total} reports, {resolved} resolved, {critical} critical."


def get_or_init_trust(user_id: Optional[str]) -> int:
    if not user_id:
        return DEFAULT_TRUST
    user = get_user_from_firestore(user_id)
    return user.get("trust_score", DEFAULT_TRUST) if user else DEFAULT_TRUST


def adjust_trust(user_id: Optional[str], delta: int, reason: str):
    if not user_id:
        return
    user = get_user_from_firestore(user_id)
    if not user:
        return
    user["trust_score"] = max(0, min(100, user.get("trust_score", DEFAULT_TRUST) + delta))
    user.setdefault("trust_log", []).append({"delta": delta, "reason": reason, "at": time.time()})
    save_user_to_firestore(user_id, user)


# ---------------------------------------------------------------------------
# ROUTES
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "gemini": bool(GEMINI_API_KEY),
        "firestore": db is not None,
        "gcs": gcs_bucket is not None,
        "maps": gmaps_client is not None,
    }


@app.post("/auth/signup")
def signup(payload: SignupPayload):
    if payload.role == "admin":
        if not payload.zone_id or payload.zone_id not in ZONES_BY_ID:
            raise HTTPException(400, "Authority must select valid zone_id")
    
    uid = gen_id("user")
    user = {
        "uid": uid,
        "name": payload.name,
        "email": payload.email,
        "role": payload.role,
        "zone_id": payload.zone_id if payload.role == "admin" else None,
        "trust_score": DEFAULT_TRUST,
        "created_at": time.time(),
    }
    save_user_to_firestore(uid, user)
    return {"uid": uid, "role": payload.role, "zone_id": payload.zone_id, "trust_score": DEFAULT_TRUST}


@app.post("/reports/submit")
async def submit_report(payload: str = Form(...), image: Optional[UploadFile] = File(None)):
    data = json.loads(payload)
    description, language = data.get("description", ""), data.get("language", "en-IN")
    lat, lng, address = data.get("lat"), data.get("lng"), data.get("address", "")
    anonymous, user_id = data.get("anonymous", False), None if data.get("anonymous") else data.get("user_id")

    if lat is None or lng is None:
        raise HTTPException(400, "lat/lng required")

    image_bytes = await image.read() if image else None
    has_image = image_bytes is not None

    # Run agents
    report_info = agent_report(description, language, has_image, image_bytes)
    sentiment = agent_sentiment(description, language)
    reporter_trust = get_or_init_trust(user_id)
    geo_info = agent_geo(lat, lng, report_info["category"], address)
    priority_info = agent_priority(report_info["category"], sentiment, geo_info["address"], geo_info["duplicate_count"])

    # Upload to GCS if available
    image_url = None
    rid = gen_id("rpt")
    if has_image:
        image_url = upload_to_gcs(rid, image_bytes, "before")

    report = {
        "id": rid,
        "title": report_info.get("title", "Civic issue"),
        "category": report_info.get("category", "other"),
        "description": report_info.get("description_en", description),
        "status": "submitted",
        "anonymous": anonymous,
        "user_id": user_id,
        "reporter_trust": reporter_trust,
        "lat": lat,
        "lng": lng,
        "address": geo_info["address"],
        "nearby_landmarks": geo_info.get("nearby_landmarks", []),
        "zone_id": geo_info["zone_id"],
        "zone_name": geo_info["zone_name"],
        "sentiment": sentiment.get("sentiment", "neutral"),
        "priority": priority_info,
        "before_image_url": image_url,
        "community_votes": 0,
        "created_at": time.time(),
    }
    save_report_to_firestore(rid, report)

    if user_id and (has_image or len(description) > 15):
        adjust_trust(user_id, 1, "submitted_detailed_report")

    return {"report_id": rid, "report": {k: v for k, v in report.items() if k not in ("before_image", "after_image")}}


@app.get("/reports")
def list_reports(zone_id: Optional[str] = None):
    reports = get_all_reports_from_firestore(zone_id)
    reports.sort(key=lambda r: r.get("priority", {}).get("urgency_score", 0), reverse=True)
    return [{"id": r["id"], "title": r["title"], "zone": r["zone_name"], "priority": r.get("priority", {})} for r in reports]


@app.get("/predictions")
def predictions(zone_id: Optional[str] = None):
    return agent_predict(zone_id)


@app.get("/summary")
def summary(zone_id: Optional[str] = None):
    return {"summary": agent_summary(zone_id)}


@app.post("/reports/{report_id}/admin/resolve")
async def resolve(report_id: str, after_image: UploadFile = File(...), notes: str = Form("")):
    r = get_report_from_firestore(report_id)
    if not r:
        raise HTTPException(404, "Report not found")

    after_bytes = await after_image.read()
    
    # Upload to GCS
    after_url = upload_to_gcs(report_id, after_bytes, "after")
    
    ai_result = agent_resolution_verify(r["category"], after_bytes, notes)
    
    final_status = "resolved" if ai_result["confidence"] >= 0.6 and ai_result["verdict"] == "fixed" else "pending_review"
    
    r["status"] = final_status
    r["after_image_url"] = after_url
    r["resolution_notes"] = notes
    r["resolution_ai_result"] = ai_result
    
    if final_status == "resolved" and r.get("user_id"):
        adjust_trust(r["user_id"], 3, "report_resolved")
    
    save_report_to_firestore(report_id, r)
    return {"status": final_status, "ai_result": ai_result}


@app.get("/stats")
def stats(zone_id: Optional[str] = None):
    reports = get_all_reports_from_firestore(zone_id)
    return {
        "total": len(reports),
        "resolved": len([r for r in reports if r["status"] == "resolved"]),
        "in_progress": len([r for r in reports if r["status"] == "in_progress"]),
        "pending": len([r for r in reports if r["status"] in ("submitted", "pending_review")]),
        "critical": len([r for r in reports if r.get("priority", {}).get("priority_level") == "critical"]),
    }


# Run: uvicorn civicsense_backend_complete:app --reload --port 8000