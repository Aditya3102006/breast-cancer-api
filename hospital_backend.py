"""
hospital_backend.py
Advanced backend for finding nearby hospitals, clinics and cancer centers.
Features:
  - Government vs Private classification
  - Panel hospital detection (CGHS, ECHS, Army, ESI, PM-JAY)
  - Hospital tier grading (Apex/Super-Speciality, Multi-Speciality, General, Clinic)
  - Oncology speciality detection
  - OpenStreetMap Overpass API + Nominatim (no API key required)
"""

import requests
import math
from typing import Optional

# ─── Constants ───────────────────────────────────────────────────────────────
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
OVERPASS_URLS = [
    "https://overpass-api.de/api/interpreter",
    "https://lz4.overpass-api.de/api/interpreter",
    "https://z.overpass-api.de/api/interpreter",
]
USER_AGENT = "BreastCancerDiagnosticApp/2.0 (educational project)"
HEADERS = {"User-Agent": USER_AGENT}

# ─── Panel / Scheme Keywords ─────────────────────────────────────────────────
PANEL_KEYWORDS = [
    "cghs", "echs", "esis", "esi ", "esic", "pm-jay", "ayushman", "pmjay",
    "panel", "empanelled", "empaneled", "govt panel", "government panel",
    "railway", "armed forces", "army", "navy", "air force", "paramilitary",
    "cisf", "bsf", "crpf", "itbp", "ssb", "ex-servicemen", "sainik",
    "central govt", "state govt", "municipal", "psu panel",
    "employee state insurance", "central government health"
]

GOVT_KEYWORDS = [
    "aiims", "government", "govt", "sarkari", "municipal", "nmc ", "pmc ",
    "bmc", "corporation", "district hospital", "civil hospital",
    "community health centre", "chc", "primary health centre", "phc",
    "esic", "esis", "esi hospital", "railway hospital",
    "armed forces medical", "military hospital", "army hospital",
    "naval hospital", "air force hospital", "central govt hospital",
    "national cancer institute", "nci", "regional cancer centre", "rcc",
    "tata memorial", "state cancer", "government medical college",
    "medical college hospital", "gmc", "safdarjung", "lnjp", "lok nayak",
    "deen dayal", "lady hardinge", "ram manohar lohia", "rml hospital",
    "post graduate institute", "pgimer", "pgi", "jipmer"
]

APEX_KEYWORDS = [
    "aiims", "tata memorial", "pgimer", "pgi", "jipmer", "nimhans",
    "national cancer", "regional cancer centre", "rcc", "cancer institute",
    "cancer hospital", "oncology centre", "oncology center",
    "super speciality", "super-speciality", "super specialty", "apex hospital",
    "advanced cancer", "nanavati", "kokilaben", "medanta", "fortis",
    "apollo", "max hospital", "manipal hospital", "wockhardt", "aster",
    "global hospital", "care hospital", "columbia asia", "cloud nine",
    "jaslok", "lilavati", "breach candy", "bombay hospital",
    "christian medical college", "cmc", "st john's", "sjmch"
]

ONCOLOGY_KEYWORDS = [
    "cancer", "oncol", "tumor", "tumour", "breast", "mammolog",
    "radiation", "chemotherapy", "oncology", "onco ",
    "malignant", "radiology & oncology", "surgical oncology"
]


# ─── 1. Geocoding ─────────────────────────────────────────────────────────────
def geocode_location(query: str) -> Optional[dict]:
    """Convert a city name / address string to lat/lng via Nominatim."""
    try:
        params = {
            "q": query,
            "format": "json",
            "limit": 1,
            "addressdetails": 1,
        }
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            return None
        result = data[0]
        return {
            "lat": float(result["lat"]),
            "lng": float(result["lon"]),
            "display_name": result.get("display_name", query),
        }
    except Exception as e:
        return {"error": str(e)}


def reverse_geocode(lat: float, lng: float) -> str:
    """Convert lat/lng to a readable place name."""
    try:
        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat, "lon": lng, "format": "json"},
            headers=HEADERS,
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("display_name", f"{lat:.4f}, {lng:.4f}")
    except Exception:
        return f"{lat:.4f}, {lng:.4f}"


# ─── 2. Haversine Distance & Bounding Box ────────────────────────────────────
def haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def get_bounding_box(lat, lng, radius_km) -> tuple:
    lat_delta = radius_km / 111.32
    lng_delta = radius_km / (111.32 * math.cos(math.radians(lat)))
    return (lat - lat_delta, lng - lng_delta, lat + lat_delta, lng + lng_delta)


# ─── 3. Build Overpass Query ─────────────────────────────────────────────────
def _build_overpass_query(lat: float, lng: float, radius_km: float) -> str:
    s, w, n, e = get_bounding_box(lat, lng, radius_km)
    bbox = f"({s},{w},{n},{e})"

    selectors = [
        f'node["amenity"="hospital"]{bbox};',
        f'way["amenity"="hospital"]{bbox};',
        f'node["amenity"="clinic"]{bbox};',
        f'way["amenity"="clinic"]{bbox};',
        f'node["amenity"="doctors"]{bbox};',
        f'way["amenity"="doctors"]{bbox};',
        f'node["healthcare"="centre"]{bbox};',
        f'way["healthcare"="centre"]{bbox};',
        f'node["healthcare"="hospital"]{bbox};',
        f'way["healthcare"="hospital"]{bbox};',
        f'node["healthcare"="clinic"]{bbox};',
        f'way["healthcare"="clinic"]{bbox};',
    ]
    body = "\n".join(selectors)
    return f"[out:json][timeout:30];\n(\n{body}\n);\nout center tags;"


# ─── 4. Classification Helpers ───────────────────────────────────────────────
def _check_keywords(text: str, keywords: list) -> bool:
    text_l = text.lower()
    return any(k in text_l for k in keywords)


def _is_government(tags: dict) -> bool:
    """Return True if the facility is government-operated."""
    op_type   = tags.get("operator:type", "").lower()
    ownership = tags.get("ownership", "").lower()
    operator  = tags.get("operator", "").lower()
    name      = tags.get("name", "").lower()
    desc      = tags.get("description", "").lower()

    if op_type in ("government", "public", "municipal", "national", "state"):
        return True
    if ownership in ("public", "government"):
        return True
    combined_text = " ".join([operator, name, desc])
    return _check_keywords(combined_text, GOVT_KEYWORDS)


def _detect_panels(tags: dict) -> list:
    """Return a list of detected panel schemes for this facility."""
    insurance = tags.get("healthcare:insurance", "").lower()
    desc      = tags.get("description", "").lower()
    note      = tags.get("note", "").lower()
    name      = tags.get("name", "").lower()
    operator  = tags.get("operator", "").lower()
    text      = " ".join([insurance, desc, note, name, operator])

    panels_found = []
    panel_map = {
        "CGHS": ["cghs", "central government health"],
        "ECHS": ["echs", "ex-servicemen", "armed forces", "sainik"],
        "ESI / ESIC": ["esi ", "esic", "esis", "employee state insurance"],
        "Ayushman Bharat (PM-JAY)": ["ayushman", "pm-jay", "pmjay", "abdm"],
        "Army / Defence": ["army", "navy", "air force", "military", "paramilitary",
                           "cisf", "bsf", "crpf", "itbp", "ssb"],
        "Railway Panel": ["railway"],
        "PSU / Corporate Panel": ["psu panel", "corporate panel"],
    }
    for panel_name, keywords in panel_map.items():
        if _check_keywords(text, keywords):
            panels_found.append(panel_name)

    # Government hospitals are typically on CGHS/Ayushman by default if detected govt
    # (inferred heuristic — not tagged in OSM, but real-world knowledge)
    return panels_found


def _get_tier(tags: dict, is_govt: bool) -> str:
    """
    Grade the facility:
      Apex / Cancer Institute → for AIIMS, Tata Memorial, NCI, RCC type
      Super-Speciality        → large hospitals with known speciality
      Multi-Speciality        → general big hospitals
      General Hospital        → standard govt/private hospital
      Clinic / Health Centre  → smaller outpatient only
    """
    name    = tags.get("name", "").lower()
    beds    = tags.get("beds", "")
    amenity = tags.get("amenity", "").lower()
    hlth    = tags.get("healthcare", "").lower()
    spec    = tags.get("healthcare:speciality", "").lower()
    desc    = tags.get("description", "").lower()
    text    = " ".join([name, desc, spec])

    if _check_keywords(text, ONCOLOGY_KEYWORDS) and _check_keywords(text, APEX_KEYWORDS):
        return "Apex Cancer Institute"
    if _check_keywords(text, APEX_KEYWORDS):
        return "Super-Speciality / Apex"
    if _check_keywords(text, ONCOLOGY_KEYWORDS):
        return "Oncology Centre"
    try:
        bed_count = int(beds)
        if bed_count >= 300:
            return "Multi-Speciality (Large)"
        if bed_count >= 100:
            return "Multi-Speciality (Medium)"
        if bed_count >= 30:
            return "General Hospital"
    except (ValueError, TypeError):
        pass
    if amenity == "hospital" or hlth == "hospital":
        return "General Hospital"
    if amenity == "clinic" or hlth in ("clinic", "centre"):
        return "Clinic / Health Centre"
    return "Medical Facility"


def _classify_specialty(tags: dict) -> str:
    """Return primary specialty string if detectable."""
    spec    = tags.get("healthcare:speciality", "")
    name    = tags.get("name", "").lower()
    desc    = tags.get("description", "").lower()
    text    = name + " " + desc

    if _check_keywords(text, ONCOLOGY_KEYWORDS):
        return "Oncology / Cancer"
    if spec and spec not in ("general", ""):
        return spec.replace(";", ", ").title()
    return "General"


# ─── 5. Parse a Single OSM Element ───────────────────────────────────────────
def _parse_element(el: dict, user_lat: float, user_lng: float) -> Optional[dict]:
    """Extract useful fields from an OSM node/way element."""
    try:
        tags = el.get("tags", {})
        name = tags.get("name", "").strip()
        if not name:
            return None

        lat, lng = None, None
        if el["type"] == "node":
            lat, lng = el.get("lat"), el.get("lon")
        elif el["type"] == "way":
            center = el.get("center", {})
            lat, lng = center.get("lat"), center.get("lon")
        if lat is None or lng is None:
            return None

        dist = haversine_km(user_lat, user_lng, lat, lng)
        is_govt  = _is_government(tags)
        panels   = _detect_panels(tags)
        tier     = _get_tier(tags, is_govt)
        specialty = _classify_specialty(tags)

        # Build address
        addr_parts = [
            tags.get("addr:housenumber", ""),
            tags.get("addr:street", ""),
            tags.get("addr:city", ""),
            tags.get("addr:state", ""),
            tags.get("addr:postcode", ""),
        ]
        address = ", ".join(p for p in addr_parts if p) or "Address not available"

        phone   = tags.get("phone", tags.get("contact:phone", "Not available"))
        website = tags.get("website", tags.get("contact:website", tags.get("url", "")))
        email   = tags.get("email", tags.get("contact:email", ""))
        hours   = tags.get("opening_hours", "Not available")
        beds    = tags.get("beds", "")
        emergency = tags.get("emergency", "")
        operator  = tags.get("operator", "")

        # Google Maps URLs
        gmaps_url = f"https://www.google.com/maps?q={lat},{lng}"
        directions_url = (
            f"https://www.google.com/maps/dir/?api=1"
            f"&origin={user_lat},{user_lng}"
            f"&destination={lat},{lng}"
            f"&travelmode=driving"
        )

        return {
            "name":          name,
            "tier":          tier,
            "specialty":     specialty,
            "is_govt":       is_govt,
            "panels":        panels,
            "lat":           lat,
            "lng":           lng,
            "distance_km":   round(dist, 2),
            "address":       address,
            "phone":         phone,
            "website":       website,
            "email":         email,
            "opening_hours": hours,
            "beds":          beds,
            "emergency":     emergency,
            "operator":      operator,
            "gmaps_url":     gmaps_url,
            "directions_url": directions_url,
            "osm_id":        el.get("id"),
        }
    except Exception:
        return None


# ─── 6. Nominatim Fallback ────────────────────────────────────────────────────
def _search_nominatim_fallback(lat, lng, radius_km, filter_type, filter_ownership, max_results) -> dict:
    """Fallback when Overpass API is unavailable."""
    try:
        s, w, n, e = get_bounding_box(lat, lng, radius_km)
        params = {
            "q": "hospital",
            "format": "json",
            "viewbox": f"{w},{n},{e},{s}",
            "bounded": 1,
            "limit": max_results * 2,
        }
        resp = requests.get(NOMINATIM_URL, params=params, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        facilities = []
        seen = set()
        for item in data:
            name = item.get("display_name", "").split(",")[0].strip()
            if not name or name.lower() in seen:
                continue
            seen.add(name.lower())

            i_lat, i_lng = float(item["lat"]), float(item["lon"])
            name_l = name.lower()
            disp   = item.get("display_name", "").lower()
            text   = name_l + " " + disp

            is_govt   = _check_keywords(text, GOVT_KEYWORDS)
            panels    = []
            for kw, scheme in [
                (["cghs"], "CGHS"), (["echs", "armed forces"], "ECHS"),
                (["esi ","esic"], "ESI / ESIC"), (["ayushman", "pmjay"], "Ayushman Bharat"),
                (["railway"], "Railway Panel"), (["army","navy"], "Army / Defence"),
            ]:
                if _check_keywords(text, kw):
                    panels.append(scheme)

            tier = "General Hospital"
            if _check_keywords(text, ONCOLOGY_KEYWORDS) and _check_keywords(text, APEX_KEYWORDS):
                tier = "Apex Cancer Institute"
            elif _check_keywords(text, APEX_KEYWORDS):
                tier = "Super-Speciality / Apex"
            elif _check_keywords(text, ONCOLOGY_KEYWORDS):
                tier = "Oncology Centre"

            specialty = "Oncology / Cancer" if _check_keywords(text, ONCOLOGY_KEYWORDS) else "General"

            gmaps_url = f"https://www.google.com/maps?q={i_lat},{i_lng}"
            directions_url = (
                f"https://www.google.com/maps/dir/?api=1"
                f"&origin={lat},{lng}&destination={i_lat},{i_lng}&travelmode=driving"
            )
            facilities.append({
                "name": name, "tier": tier, "specialty": specialty,
                "is_govt": is_govt, "panels": panels,
                "lat": i_lat, "lng": i_lng,
                "distance_km": round(haversine_km(lat, lng, i_lat, i_lng), 2),
                "address": item.get("display_name", "Address not available"),
                "phone": "Not available", "website": "", "email": "",
                "opening_hours": "Not available", "beds": "", "emergency": "",
                "operator": "", "gmaps_url": gmaps_url, "directions_url": directions_url,
                "osm_id": item.get("osm_id"),
            })

        # Apply filters
        if filter_type != "All":
            if filter_type == "Cancer Center / Oncology":
                facilities = [f for f in facilities if "Onco" in f["tier"] or "Cancer" in f["tier"]]
            elif filter_type == "Hospital":
                facilities = [f for f in facilities if "Hospital" in f["tier"] or "Apex" in f["tier"] or "Speciality" in f["tier"]]
            elif filter_type == "Clinic":
                facilities = [f for f in facilities if "Clinic" in f["tier"]]

        if filter_ownership == "Government Only":
            facilities = [f for f in facilities if f["is_govt"]]
        elif filter_ownership == "Private Only":
            facilities = [f for f in facilities if not f["is_govt"]]
        elif filter_ownership == "Panel / Empanelled":
            facilities = [f for f in facilities if f["panels"]]

        facilities.sort(key=lambda x: x["distance_km"])
        return {
            "success": True,
            "results": facilities[:max_results],
            "count": len(facilities[:max_results]),
            "fallback": True,
        }
    except Exception as e:
        return {"success": False, "error": str(e), "results": [], "count": 0}


# ─── 7. Main Search Function ─────────────────────────────────────────────────
def search_medical_facilities(
    lat: float,
    lng: float,
    radius_km: int = 10,
    filter_type: str = "All",
    filter_ownership: str = "All",
    max_results: int = 40,
) -> dict:
    """
    Search Overpass / Nominatim for medical facilities near (lat, lng).
    Returns enriched dicts with govt/panel/tier classification.
    """
    query = _build_overpass_query(lat, lng, radius_km)
    data = None
    last_error = "Unknown"

    for url in OVERPASS_URLS:
        try:
            resp = requests.post(url, data={"data": query}, headers=HEADERS, timeout=35)
            if resp.status_code in (429,) or resp.status_code >= 500:
                last_error = f"HTTP {resp.status_code} on {url}"
                continue
            resp.raise_for_status()
            data = resp.json()
            break
        except requests.exceptions.Timeout:
            last_error = f"Timeout on {url}"
        except requests.exceptions.ConnectionError:
            last_error = f"ConnectionError on {url}"
        except Exception as e:
            last_error = f"{type(e).__name__} on {url}: {e}"

    if not data or "elements" not in data:
        return _search_nominatim_fallback(lat, lng, radius_km, filter_type, filter_ownership, max_results)

    facilities = []
    seen = set()
    for el in data.get("elements", []):
        parsed = _parse_element(el, lat, lng)
        if parsed is None:
            continue
        key = parsed["name"].lower().strip()
        if key in seen:
            continue
        seen.add(key)
        facilities.append(parsed)

    # Apply filters
    if filter_type == "Cancer Center / Oncology":
        facilities = [f for f in facilities if "Onco" in f["tier"] or "Cancer" in f["tier"] or f["specialty"] == "Oncology / Cancer"]
    elif filter_type == "Hospital":
        facilities = [f for f in facilities if "Hospital" in f["tier"] or "Apex" in f["tier"] or "Speciality" in f["tier"]]
    elif filter_type == "Clinic":
        facilities = [f for f in facilities if "Clinic" in f["tier"]]

    if filter_ownership == "Government Only":
        facilities = [f for f in facilities if f["is_govt"]]
    elif filter_ownership == "Private Only":
        facilities = [f for f in facilities if not f["is_govt"]]
    elif filter_ownership == "Panel / Empanelled":
        facilities = [f for f in facilities if f["panels"]]

    facilities.sort(key=lambda f: f["distance_km"])
    facilities = facilities[:max_results]

    return {"success": True, "results": facilities, "count": len(facilities)}


# ─── 8. Build Folium Map ─────────────────────────────────────────────────────
def build_map(user_lat: float, user_lng: float, facilities: list, radius_km: int):
    import folium

    TIER_COLORS = {
        "Apex Cancer Institute":     "darkred",
        "Super-Speciality / Apex":   "red",
        "Oncology Centre":           "orange",
        "Multi-Speciality (Large)":  "darkblue",
        "Multi-Speciality (Medium)": "blue",
        "General Hospital":          "cadetblue",
        "Clinic / Health Centre":    "green",
        "Medical Facility":          "gray",
    }
    TIER_ICONS = {
        "Apex Cancer Institute":     "star",
        "Super-Speciality / Apex":   "plus-sign",
        "Oncology Centre":           "certificate",
        "Multi-Speciality (Large)":  "hospital",
        "Multi-Speciality (Medium)": "plus-sign",
        "General Hospital":          "plus-sign",
        "Clinic / Health Centre":    "heart",
        "Medical Facility":          "info-sign",
    }

    fmap = folium.Map(location=[user_lat, user_lng], zoom_start=13, tiles="CartoDB dark_matter")

    folium.Circle(
        location=[user_lat, user_lng],
        radius=radius_km * 1000,
        color="#a78bfa", fill=True, fill_opacity=0.06, weight=1.5,
        popup="Search radius",
    ).add_to(fmap)

    folium.Marker(
        location=[user_lat, user_lng],
        popup=folium.Popup("<b>📍 Your Location</b>", max_width=200),
        tooltip="Your Location",
        icon=folium.Icon(color="green", icon="user", prefix="glyphicon"),
    ).add_to(fmap)

    for f in facilities:
        govt_tag  = "🏛️ GOVT" if f["is_govt"] else "🏥 PRIVATE"
        panel_str = " | ".join(f["panels"]) if f["panels"] else "None detected"
        beds_str  = f"{f['beds']} beds" if f["beds"] else ""

        popup_html = f"""
        <div style="font-family:Arial,sans-serif;min-width:240px;max-width:300px;">
            <h4 style="margin:0 0 4px;color:#1a1a2e;">{f['name']}</h4>
            <span style="background:#6c3483;color:white;border-radius:10px;padding:2px 8px;font-size:11px;">{f['tier']}</span>
            &nbsp;<span style="background:{'#2980b9' if f['is_govt'] else '#c0392b'};color:white;border-radius:10px;padding:2px 8px;font-size:11px;">{govt_tag}</span>
            <hr style="margin:6px 0;"/>
            <p style="margin:2px 0;font-size:12px;">📍 {f['address']}</p>
            <p style="margin:2px 0;font-size:12px;">📏 {f['distance_km']} km away</p>
            {f"<p style='margin:2px 0;font-size:12px;'>🛏️ {beds_str}</p>" if beds_str else ""}
            {f"<p style='margin:2px 0;font-size:12px;'>📞 {f['phone']}</p>" if f['phone'] != 'Not available' else ""}
            <p style="margin:4px 0;font-size:11px;color:#27ae60;"><b>🎫 Panels: {panel_str}</b></p>
            <br/>
            <a href="{f['directions_url']}" target="_blank"
               style="background:#e74c3c;color:white;padding:4px 10px;border-radius:6px;text-decoration:none;font-size:12px;">
               🗺️ Get Directions
            </a>
            {f'&nbsp;<a href="{f["website"]}" target="_blank" style="background:#2980b9;color:white;padding:4px 10px;border-radius:6px;text-decoration:none;font-size:12px;">🌐 Website</a>' if f.get("website") else ""}
        </div>
        """
        color = TIER_COLORS.get(f["tier"], "gray")
        icon  = TIER_ICONS.get(f["tier"], "info-sign")
        # Government hospitals get a special marker outline
        folium.Marker(
            location=[f["lat"], f["lng"]],
            popup=folium.Popup(popup_html, max_width=320),
            tooltip=f"{'🏛️' if f['is_govt'] else '🏥'} {f['name']} ({f['distance_km']} km)",
            icon=folium.Icon(color=color, icon=icon, prefix="glyphicon"),
        ).add_to(fmap)

    return fmap


# ─── 9. CLI Self-Test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Testing geocode: New Delhi")
    loc = geocode_location("New Delhi, India")
    print(loc)
    if loc and "lat" in loc:
        res = search_medical_facilities(loc["lat"], loc["lng"], radius_km=5)
        print(f"Found {res['count']} facilities")
        for f in res["results"][:5]:
            print(f"  [{f['tier']}] {'GOVT' if f['is_govt'] else 'PRIVATE'} | {f['name']} — {f['distance_km']} km | Panels: {f['panels']}")
