"""Home Craving v2.
Input : recipe as photo / voice memo / URL / text  +  your location
Output: the recipe rewritten with local substitutes, and for each swap the nearest shops
        (OpenStreetMap), a navigation link, and a reference price (SerpApi, optional).

Gemma    - reads the photo, extracts the recipe, proposes + rewrites swaps (any OpenAI-compatible endpoint)
TabPFN   - scores each swap, trained on substitutions.csv
Whisper  - local speech-to-text for voice memos
OSM      - Nominatim (geocoding) + Overpass (shops): open data, no key
SerpApi  - optional Google Shopping price lookup
"""
import base64
import csv
import ipaddress
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import math
import os
import pathlib
import re
import socket
import tempfile
import threading
from typing import Optional
from urllib.parse import urlparse

import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

load_dotenv()  # reads keys from a local .env file


def env_float(name, default):
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        print(f"Warning: {name} is not a number, using {default}", flush=True)
        return default


def env_int(name, default):
    """A typo like LLM_TIMEOUT=180x should not break the app: warn and use the default."""
    v = os.getenv(name, "").strip()
    try:
        return int(v) if v else default
    except ValueError:
        print(f"Warning: {name}={v!r} is not a number, using {default}", flush=True)
        return default


BASE = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "gemma3:4b")  # must be a vision-capable Gemma for photos
SWAP_MODEL = os.getenv("SWAP_MODEL") or MODEL  # a faster model for the big swap-analysis call
REWRITE_MODEL = os.getenv("REWRITE_MODEL") or MODEL  # model for the final recipe rewrite
KEY = os.getenv("LLM_API_KEY", "ollama")
SERP = os.getenv("SERPAPI_KEY")
DATA = pathlib.Path(__file__).with_name("substitutions.csv")
FEATS = ["flavor", "texture", "moisture", "acid", "heat"]
RATINGS = pathlib.Path(os.getenv("RATINGS_PATH", str(DATA.with_name("ratings.csv"))))
DEMO = os.getenv("DEMO") == "1"
REWRITE_RECIPE = os.getenv("REWRITE_RECIPE", "1") == "1"  # set to 0 to skip the final rewrite call
COLS = ["pair"] + FEATS + ["importance", "rating"]
UA = {"User-Agent": "home-craving/0.2 (hackathon demo)"}
OVERPASS = "https://overpass-api.de/api/interpreter"
OVERPASS_URLS = [OVERPASS, "https://overpass.kumi.systems/api/interpreter",
                 "https://overpass.private.coffee/api/interpreter"]

app = FastAPI(title="Home Craving")
clf = None


def load_rows():
    """Seed swaps + the ratings real users have given through the UI."""
    rows = list(csv.DictReader(open(DATA)))
    if RATINGS.exists():
        rows += list(csv.DictReader(open(RATINGS)))
    return rows


# ---------- TabPFN scoring ----------
@app.on_event("startup")
def train():
    global clf
    if DEMO:  # demo needs no scorer; avoids TabPFN's login prompt blocking startup
        return
    try:
        if os.getenv("TABPFN_TOKEN"):  # hosted TabPFN: no local weights, light on RAM
            import tabpfn_client
            tabpfn_client.set_access_token(os.getenv("TABPFN_TOKEN"))
            from tabpfn_client import TabPFNClassifier
        else:  # local weights
            from tabpfn import TabPFNClassifier

        rows = load_rows()
        X = np.array([[float(r[f]) for f in FEATS + ["importance"]] for r in rows], dtype=float)
        y = np.array([int(r["rating"]) for r in rows])
        clf = TabPFNClassifier()
        clf.fit(X, y)
        print(f"TabPFN ready on {len(rows)} rated swaps")
    except Exception as e:
        print("TabPFN unavailable, using fallback scorer:", e)


def clamp(v):
    try:
        return min(1.0, max(0.0, float(v)))
    except (TypeError, ValueError):
        return 0.5


def score(x):
    if clf is not None:
        try:
            # TabPFN needs a numpy array (a plain list has no .shape)
            p = clf.predict_proba(np.array([x], dtype=float))[0]
            return round((sum(c * q for c, q in zip(clf.classes_, p)) - 1) / 2 * 100), "tabpfn"
        except Exception as e:
            print("TabPFN predict failed:", e)
    return round(sum(x[:5]) / 5 * 100 * (1 - 0.1 * (x[5] - 1))), "fallback"


# ---------- LLM ----------
SWAP_SYS = """You help people recreate home recipes after moving abroad.
List every ingredient. "hard_to_find": true if an ordinary supermarket in the new country is unlikely to stock it.
For hard ones give up to 2 "substitutes" that shops there DO sell. "importance" is 1-5 (5 = dish isn't itself without it).
For each substitute rate 0-1 how well it matches the original on flavor, texture, moisture, acid, heat. Be honest.
"store_kind" is where to buy it: supermarket, greengrocer, butcher, seafood, bakery, health_food or asian_grocery.
"tip" is at most 12 words. Keep the whole reply compact.
If the user message lists foods to avoid, mark any ingredient containing them hard_to_find true with reason "avoid" and give substitutes that are safe; otherwise reason is "local" or null.
"qty" is the amount as written.
Reply with ONE valid JSON object filled with real values, never the format description. Example:
{"title":"Dish","ingredients":[{"name":"Kashmiri chilli powder","qty":"2 tsp","reason":"local","hard_to_find":true,"importance":3,"substitutes":[{"name":"Sweet paprika","store_kind":"supermarket","tip":"Add a pinch of cayenne for heat.","flavor":0.7,"texture":0.9,"moisture":0.9,"acid":0.8,"heat":0.5}]},{"name":"Onion","qty":"2","reason":null,"hard_to_find":false,"importance":2,"substitutes":[]}]}"""

ADAPT_SYS = """Rewrite the recipe using the chosen substitutions. Keep quantities from the original; where a swap
changes the ratio, say so. Do not invent steps.
Reply with ONE valid JSON object with real values. Example:
{"title":"Dish","ingredients":["500 g beef"],"steps":["Do this first."],"notes":"One short note."}"""


def parse_json(txt):
    a, b = txt.find("{"), txt.rfind("}")
    return json.loads(txt[a:b + 1])


def chat(system, user, as_json=True, model=None, timeout=None):
    """One chat-completions call. Retries transient provider errors (429/5xx) with backoff,
    drops JSON mode if the provider rejects it, and asks once more if the JSON is malformed.
    Timeouts are not retried: that would only double the wait."""
    use_model = model or MODEL
    timeout = timeout or env_int("LLM_TIMEOUT", 180)
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    body = {"model": use_model, "messages": msgs, "temperature": env_float("LLM_TEMPERATURE", 0.2)}
    if as_json:
        body["response_format"] = {"type": "json_object"}
    headers = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"}

    def call():
        print(f"[llm] calling {use_model} (timeout={timeout}s)", flush=True)
        r = None
        for wait in (0, 2, 5, 10):
            if wait:
                time.sleep(wait)
            try:
                r = requests.post(f"{BASE}/chat/completions", headers=headers, json=body, timeout=timeout)
            except requests.exceptions.Timeout:
                raise RuntimeError(f"LLM request timed out after {timeout}s")
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"LLM connection failed: {e}")
            if r.status_code == 400 and "response_format" in body:
                body.pop("response_format")  # some providers choke on JSON mode
                continue
            if r.status_code in (429, 500, 502, 503, 504):
                continue
            break
        if not r.ok:
            raise RuntimeError(f"LLM provider said {r.status_code}: {r.text[:500]}")
        try:
            return r.json()["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError, TypeError, ValueError, AttributeError) as e:
            raise RuntimeError(f"Unexpected LLM response: {r.text[:500]}") from e

    txt = call()
    if not as_json:
        return txt
    try:
        return parse_json(txt)
    except ValueError:  # includes json.JSONDecodeError
        print("[llm] invalid JSON; asking model to correct it", flush=True)
        msgs.extend([
            {"role": "assistant", "content": txt[:3000]},
            {"role": "user", "content": "Your previous response was not valid JSON. Return ONLY one valid JSON "
                                        "object filled with real values. No markdown fences, no explanation."},
        ])
        body["messages"] = msgs
        txt = call()
        try:
            return parse_json(txt)
        except ValueError as e:
            raise RuntimeError(f"LLM returned invalid JSON twice: {txt[:500]}") from e


# ---------- input -> recipe text ----------
def from_image(data, mime):
    url = f"data:{mime or 'image/jpeg'};base64,{base64.b64encode(data).decode()}"
    return chat("Transcribe the recipe in this image exactly, ingredients and steps. Plain text only.",
                [{"type": "text", "text": "Transcribe this recipe."}, {"type": "image_url", "image_url": {"url": url}}],
                as_json=False)


def from_audio(data, name):
    k = os.getenv("ELEVENLABS_API_KEY")
    if k:  # hosted transcription
        r = requests.post("https://api.elevenlabs.io/v1/speech-to-text", headers={"xi-api-key": k},
                          data={"model_id": "scribe_v1"}, files={"file": (name or "audio.webm", data)}, timeout=120)
        r.raise_for_status()
        return r.json().get("text", "")
    from faster_whisper import WhisperModel

    with tempfile.NamedTemporaryFile(suffix=pathlib.Path(name or "a.m4a").suffix) as f:
        f.write(data)
        f.flush()
        segs, _ = WhisperModel(os.getenv("WHISPER_SIZE", "base"), compute_type="int8").transcribe(f.name, vad_filter=True)
        return " ".join(s.text.strip() for s in segs)


def check_url(u):
    p = urlparse(u)
    if p.scheme not in ("http", "https") or not p.hostname:
        raise HTTPException(400, "Bad URL")
    for info in socket.getaddrinfo(p.hostname, None):
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise HTTPException(400, "That URL isn't allowed")


def from_url(u):
    check_url(u)
    h = requests.get(u, headers=UA, timeout=15).text
    for m in re.finditer(r'<script[^>]+ld\+json[^>]*>(.*?)</script>', h, re.S):
        try:
            d = json.loads(m.group(1))
        except ValueError:
            continue
        for it in (d if isinstance(d, list) else d.get("@graph", [d])):
            t = it.get("@type") if isinstance(it, dict) else None
            if t == "Recipe" or (isinstance(t, list) and "Recipe" in t):
                return json.dumps({k: it.get(k) for k in ("name", "recipeIngredient", "recipeInstructions")})
    h = re.sub(r"<(script|style)[\s\S]*?</\1>", " ", h)
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", h))[:8000]


# ---------- location, shops, prices ----------
# ---------- location (replace your old `def locate(...)` with everything in this block) ----------
# Nominatim (OpenStreetMap's geocoder) often blocks shared cloud IPs such as Render's, and then answers with an
# HTML/plain-text page. The old code called .json() on that page and crashed with "Expecting value".
# This version identifies itself properly, never trusts the reply to be JSON, caches results, and falls back to
# two other free geocoders (Photon and Open-Meteo) before giving up with a clear message.
GEO_UA = {"User-Agent": f"home-craving/0.2 (contact: {os.getenv('GEO_CONTACT', 'set GEO_CONTACT in your environment')})"}
_geo_cache = {}


def _geo_json(url, params):
    r = requests.get(url, params=params, headers=GEO_UA, timeout=12)
    try:
        return r.json()
    except ValueError:
        raise RuntimeError(f"HTTP {r.status_code}, not JSON: {r.text[:80]!r}")


def _forward(place):
    def nominatim():
        r = _geo_json("https://nominatim.openstreetmap.org/search",
                      {"q": place, "format": "json", "limit": 1, "addressdetails": 1})
        if not r:
            return None
        a = r[0].get("address", {})
        return float(r[0]["lat"]), float(r[0]["lon"]), a.get("country"), a.get("country_code")

    def photon():
        r = _geo_json("https://photon.komoot.io/api/", {"q": place, "limit": 1})
        f = (r.get("features") or [None])[0]
        if not f:
            return None
        lon, lat = f["geometry"]["coordinates"]
        p = f.get("properties", {})
        return lat, lon, p.get("country"), p.get("countrycode")

    def open_meteo():
        r = _geo_json("https://geocoding-api.open-meteo.com/v1/search", {"name": place, "count": 1})
        f = (r.get("results") or [None])[0]
        if not f:
            return None
        return f["latitude"], f["longitude"], f.get("country"), f.get("country_code")

    return [("nominatim", nominatim), ("photon", photon), ("open-meteo", open_meteo)]


def _reverse(lat, lon):
    """Country name and code for a GPS point. Never blocks the request: unknown is fine."""
    try:
        a = _geo_json("https://nominatim.openstreetmap.org/reverse",
                      {"lat": lat, "lon": lon, "format": "json", "zoom": 10}).get("address", {})
        if a.get("country"):
            return a["country"], (a.get("country_code") or "").lower() or None
    except Exception as e:
        print(f"[geo] reverse via nominatim failed: {e}", flush=True)
    try:
        r = _geo_json("https://photon.komoot.io/reverse", {"lat": lat, "lon": lon})
        p = ((r.get("features") or [{}])[0]).get("properties", {})
        if p.get("country"):
            return p["country"], (p.get("countrycode") or "").lower() or None
    except Exception as e:
        print(f"[geo] reverse via photon failed: {e}", flush=True)
    return "the local country", None


def locate(lat, lon, place):
    if lat is not None:
        country, cc = _reverse(lat, lon)
        return lat, lon, country, cc
    key = place.strip().lower()
    if key in _geo_cache:
        return _geo_cache[key]
    no_match, failures = False, []
    for name, fn in _forward(place):
        try:
            res = fn()
        except Exception as e:
            failures.append(name)
            print(f"[geo] {name} failed: {e}", flush=True)
            continue
        if res:
            la, lo, country, cc = res
            out = (float(la), float(lo), country or "the local country", (cc or "").lower() or None)
            _geo_cache[key] = out
            print(f"[geo] '{place}' found via {name}", flush=True)
            return out
        no_match = True
    if no_match:
        raise HTTPException(400, "Couldn't find that place. Try a nearby city name, or tap Use my location.")
    raise HTTPException(502, "The place lookup services are not responding right now "
                             f"({', '.join(failures)}). Try again in a minute, or tap Use my location.")

SHOP = '["shop"~"^(supermarket|convenience)$"]'
KINDS = {
    "supermarket": SHOP,
    "greengrocer": '["shop"="greengrocer"]',
    "butcher": '["shop"="butcher"]',
    "seafood": '["shop"="seafood"]',
    "bakery": '["shop"="bakery"]',
    "health_food": '["shop"~"^(health_food|organic)$"]',
    "asian_grocery": '["shop"~"^(supermarket|convenience|grocery|greengrocer)$"]["name"~"asia|indian|oriental|halal|spice|ethnic|bazaar|world",i]',
}


def km(a, b, c, d):
    p = math.pi / 180
    h = math.sin((c - a) * p / 2) ** 2 + math.cos(a * p) * math.cos(c * p) * math.sin((d - b) * p / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


SHOPS_FILE = pathlib.Path(os.getenv("SHOPS_CACHE_PATH", "shops_cache.json"))
SHOPS_CACHE_DAYS = env_int("SHOPS_CACHE_DAYS", 14)
_shops_lock = threading.Lock()


def _shops_key(kind, lat, lon):
    return f"{kind}|{round(lat, 2)}|{round(lon, 2)}"  # ~1 km grid: nearby users share an entry


def _shops_load():
    try:
        return json.load(open(SHOPS_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _overpass(url, q):
    r = requests.post(url, data={"data": q}, headers=UA, timeout=15)
    try:
        j = r.json()
    except ValueError:
        raise RuntimeError(f"HTTP {r.status_code}, not JSON: {r.text[:80]!r}")
    # a busy server can answer 200 with no elements and a "remark" explaining why
    if not r.ok or (j.get("remark") and not j.get("elements")):
        raise RuntimeError(f"HTTP {r.status_code} {str(j.get('remark', ''))[:100]}")
    return j.get("elements", [])


def overpass_fetch(q, kind):
    """Ask every Overpass server at once; the first good answer wins. None means all failed."""
    ex = ThreadPoolExecutor(len(OVERPASS_URLS))
    futs = {ex.submit(_overpass, u, q): u for u in OVERPASS_URLS}
    try:
        for f in as_completed(futs):
            host = urlparse(futs[f]).hostname
            try:
                els = f.result()
                print(f"[shops] {kind}: {len(els)} found via {host}", flush=True)
                return els
            except Exception as e:
                print(f"[shops] {kind}: {host} failed: {e}", flush=True)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)
    return None


def shops(kind, lat, lon, cache):
    kind = kind if kind in KINDS else "supermarket"
    if kind not in cache:
        key = _shops_key(kind, lat, lon)
        with _shops_lock:
            hit = _shops_load().get(key)
        if hit and time.time() - hit["t"] < SHOPS_CACHE_DAYS * 86400:
            raw = hit["v"]
            print(f"[shops] {kind}: cache hit, no Overpass call", flush=True)
        else:
            q = f"[out:json][timeout:10];nwr{KINDS[kind]}(around:5000,{lat},{lon});out center 30;"
            els = overpass_fetch(q, kind)
            raw = []
            for e in els or []:
                c = e.get("center") or e
                if c.get("lat") is not None:
                    raw.append({"name": e.get("tags", {}).get("name", "(unnamed shop)"), "lat": c["lat"], "lon": c["lon"]})
            if raw:  # never cache a failure or an empty answer
                with _shops_lock:
                    d = _shops_load()
                    d[key] = {"t": time.time(), "v": raw}
                    try:
                        json.dump(d, open(SHOPS_FILE, "w", encoding="utf-8"))
                    except OSError:
                        pass
        out = [{"name": s["name"], "km": round(km(lat, lon, s["lat"], s["lon"]), 1),
                "nav": f"https://www.google.com/maps/dir/?api=1&destination={s['lat']},{s['lon']}"} for s in raw]
        cache[kind] = sorted(out, key=lambda s: s["km"])[:2]
    return cache[kind]


# SerpApi's free plan allows about 250 searches a month. Every search is counted before it is made,
# repeat lookups are served from a 7-day cache, and a hard monthly cap stops the app from overspending.
SERP_FILE = pathlib.Path(os.getenv("SERP_USAGE_PATH", "serp_usage.json"))
SERP_LIMIT = env_int("SERPAPI_MONTHLY_LIMIT", 240)  # a small reserve under 250
SERP_PER_RECIPE = env_int("SERPAPI_PER_RECIPE", 4)
_serp_lock = threading.Lock()


def _serp_state():
    try:
        d = json.load(open(SERP_FILE, encoding="utf-8"))
    except (OSError, ValueError):
        d = {}
    month = time.strftime("%Y-%m")
    if d.get("month") != month:
        d = {"month": month, "count": 0, "cache": d.get("cache", {})}
    d.setdefault("cache", {})
    return d


def _serp_save(d):
    try:
        json.dump(d, open(SERP_FILE, "w", encoding="utf-8"))
    except OSError:
        pass


def serp_left():
    with _serp_lock:
        return max(0, SERP_LIMIT - _serp_state()["count"])


SERP_CACHE_DAYS = env_int("SERPAPI_CACHE_DAYS", 30)  # 0 = cached prices never expire


def _norm(s):
    """'Sweet Paprika!' and 'sweet  paprika' share one cache entry."""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def _serp_key(item, cc):
    return f"{cc}|{_norm(item)}"


def _fresh(hit):
    return bool(hit) and (SERP_CACHE_DAYS <= 0 or time.time() - hit["t"] < SERP_CACHE_DAYS * 86400)


def serp_cached(item, cc):
    """(found, value) from the local cache only. Never makes a network call."""
    if not SERP:
        return True, None
    with _serp_lock:
        hit = _serp_state()["cache"].get(_serp_key(item, cc))
    return (True, hit["v"]) if _fresh(hit) else (False, None)


def _serp_refund():
    """A failed call is not charged by SerpApi, so give the slot back."""
    with _serp_lock:
        d = _serp_state()
        d["count"] = max(0, d["count"] - 1)
        _serp_save(d)


def price(item, cc):
    """Online reference price, not the local shelf price. Served from the cache whenever possible;
    the API is only called for an item that has never been looked up (or whose entry expired)."""
    if not SERP:
        return None
    key = _serp_key(item, cc)
    with _serp_lock:
        d = _serp_state()
        hit = d["cache"].get(key)
        if _fresh(hit):
            print(f"[price] {item}: cache hit, no API call", flush=True)
            return hit["v"]
        if d["count"] >= SERP_LIMIT:
            print(f"[price] {item}: monthly limit reached, skipping", flush=True)
            return None
        d["count"] += 1  # counted before the call, so parallel lookups cannot overshoot
        _serp_save(d)
    try:
        r = requests.get("https://serpapi.com/search.json", timeout=15,
                         params={"engine": "google_shopping", "q": item, "gl": cc, "api_key": SERP}).json()
    except Exception as e:
        print(f"[price] {item} failed: {e}", flush=True)
        _serp_refund()
        return None
    err = str(r.get("error") or "")
    if err and "hasn't returned any results" not in err:  # real error: do not cache it
        print(f"[price] {item}: SerpApi error: {err[:150]}", flush=True)
        _serp_refund()
        return None
    s = (r.get("shopping_results") or [None])[0]
    v = {"price": s.get("price"), "seller": s.get("source")} if s else None  # "no result" is cached too
    print(f"[price] {item}: API call made -> {v or 'no shopping results'}", flush=True)
    with _serp_lock:
        d = _serp_state()
        d["cache"][key] = {"v": v, "t": time.time()}
        _serp_save(d)
    return v


# ---------- endpoint ----------
@app.post("/api/adapt")
def adapt(text: str = Form(""), url: str = Form(""), home: str = Form(""), place: str = Form(""), avoid: str = Form(""),
          lat: Optional[float] = Form(None), lon: Optional[float] = Form(None),
          image: Optional[UploadFile] = File(None), audio: Optional[UploadFile] = File(None)):
    if DEMO:
        return demo_recipe(avoid)
    t0 = time.time()
    if lat is None and not place.strip():
        raise HTTPException(400, "Add a location")
    try:
        if image and image.filename:
            text = from_image(image.file.read(), image.content_type)
        elif audio and audio.filename:
            text = from_audio(audio.file.read(), audio.filename)
        elif url.strip():
            text = from_url(url.strip())
        if not text.strip():
            raise HTTPException(400, "No recipe found in the input")
        lat, lon, country, cc = locate(lat, lon, place)
        print(f"[adapt] location found; asking {SWAP_MODEL} for swaps", flush=True)
        data = chat(SWAP_SYS, f"Home cuisine: {home or 'infer from the recipe'}\nNow living in: {country}\nFoods to avoid: {avoid or 'none'}\nRecipe:\n{text}",
                    model=SWAP_MODEL)
    except HTTPException:
        raise
    except Exception as e:
        print("[adapt] failed:", e, flush=True)
        raise HTTPException(502, f"Processing failed: {e}")

    cache, items, chosen = {}, [], []
    for ing in data.get("ingredients", []):
        try:
            imp = min(5, max(1, int(ing.get("importance", 3))))
        except (TypeError, ValueError):
            imp = 3
        subs = []
        for s in (ing.get("substitutes") or [])[:3] if ing.get("hard_to_find") else []:
            x = [clamp(s.get(f)) for f in FEATS] + [imp]
            sc, how = score(x)
            subs.append({"name": s.get("name"), "tip": s.get("tip", ""), "score": sc, "scored_by": how,
                         "store_kind": s.get("store_kind", "supermarket"), "x": x})
        subs.sort(key=lambda s: -s["score"])
        if subs:  # shops + price only for the swap we'll actually use
            chosen.append(f"{ing['name']} -> {subs[0]['name']} ({subs[0]['tip']})")
        items.append({"name": ing.get("name"), "qty": ing.get("qty"), "reason": ing.get("reason"), "hard": bool(ing.get("hard_to_find")), "subs": subs})

    print(f"[adapt] swaps scored in {time.time() - t0:.0f}s total so far; looking up shops", flush=True)
    tops = [it for it in items if it["subs"]]
    with ThreadPoolExecutor(6) as ex:  # shop and price lookups run side by side
        kf = {k: ex.submit(shops, k, lat, lon, cache) for k in {it["subs"][0]["store_kind"] for it in tops}}
        pending, pf, new_calls = {}, [], 0
        for it in tops:  # cache hits are free; only genuinely new items use an API call
            nm = it["subs"][0]["name"]
            k = _norm(nm)
            if k not in pending:  # the same item twice in one recipe is looked up once
                found, val = serp_cached(nm, cc)
                if found:
                    pending[k] = val
                elif new_calls < SERP_PER_RECIPE:
                    pending[k] = ex.submit(price, nm, cc)
                    new_calls += 1
                else:
                    pending[k] = None
            pf.append(pending[k])
        for it, p in zip(tops, pf):
            it["subs"][0]["shops"] = kf[it["subs"][0]["store_kind"]].result()
            it["subs"][0]["price"] = p.result() if hasattr(p, "result") else p
    print(f"[adapt] shops done at {time.time() - t0:.0f}s", flush=True)
    recipe = None
    if chosen and REWRITE_RECIPE:
        print("[adapt] rewriting recipe", flush=True)
        try:
            recipe = chat(ADAPT_SYS, f"Original recipe:\n{text}\n\nSubstitutions:\n" + "\n".join(chosen), model=REWRITE_MODEL)
        except Exception as e:
            print("[adapt] rewrite failed:", e, flush=True)
    print(f"[adapt] finished in {time.time() - t0:.0f}s", flush=True)
    return {"country": country, "recipe": recipe, "ingredients": items}


class Rate(BaseModel):
    pair: str
    x: list
    rating: int


@app.post("/api/rate")
def rate(r: Rate):
    """Every thumbs up/down becomes a training row for TabPFN."""
    if len(r.x) != 6 or r.rating not in (1, 2, 3):
        raise HTTPException(400, "Bad rating")
    new = not RATINGS.exists()
    with open(RATINGS, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(COLS)
        w.writerow([r.pair[:80].lstrip("=+-@")] + [round(clamp(v), 2) for v in r.x[:5]] + [min(5, max(1, int(r.x[5]))), r.rating])
    return {"your_ratings": sum(1 for _ in open(RATINGS)) - 1}


@app.post("/api/retrain")
def retrain():
    train()
    return status()


@app.get("/api/status")
def status():
    mine = sum(1 for _ in open(RATINGS)) - 1 if RATINGS.exists() else 0
    return {"scorer": "tabpfn" if clf else "fallback", "rows": len(load_rows()), "your_ratings": mine, "demo": DEMO, "stt": "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "whisper-local", "prices": bool(SERP), "serp_left": serp_left() if SERP else None, "voice": bool(os.getenv("ELEVENLABS_API_KEY")), "llm_key_set": KEY != "ollama"}


def demo():
    shop = lambda n, k: [{"name": n, "km": k, "nav": "https://www.openstreetmap.org/"}]
    return {"demo": True, "country": "Demo land", "recipe": {
        "title": "Paneer Butter Masala (demo)",
        "ingredients": ["250 g halloumi, cubed", "1 tbsp lime juice + 1 tsp brown sugar", "1 tbsp clarified butter", "1 tsp garam masala"],
        "steps": ["Sear the halloumi until golden.", "Simmer the tomato base with the lime and sugar.", "Fold the halloumi in and finish with butter."],
        "notes": "Demo data. Halloumi is saltier than paneer, so skip added salt."},
        "ingredients": [
        {"name": "paneer", "hard": True, "subs": [{"name": "halloumi", "tip": "Salty: skip extra salt.", "score": 71, "scored_by": "demo",
            "store_kind": "supermarket", "x": [.7, .8, .7, .6, .6, 4], "shops": shop("Demo Supermarket", 0.6), "price": {"price": "3.49", "seller": "demo"}}]},
        {"name": "tamarind", "hard": True, "subs": [{"name": "lime juice + brown sugar", "tip": "About 1 tbsp lime to 1 tsp sugar.", "score": 64, "scored_by": "demo",
            "store_kind": "supermarket", "x": [.7, .9, .9, .8, .5, 3], "shops": shop("Demo Supermarket", 0.6), "price": None}]},
        {"name": "garam masala", "hard": True, "subs": [{"name": "allspice + cumin", "tip": "Half and half.", "score": 55, "scored_by": "demo",
            "store_kind": "asian_grocery", "x": [.6, 1, 1, 1, .7, 3], "shops": shop("Demo Spice Bazaar", 1.8), "price": None}]},
        {"name": "onion", "hard": False, "subs": []}]}


RECIPE_SYS = """You are an experienced home cook. Write a traditional, practical recipe for the dish the user names.
Give real quantities for the requested servings. Do not invent history or origin claims: "story" is one plain sentence about the dish.
Reply with ONE valid JSON object with real values. Example:
{"title":"Dish name","story":"One plain sentence.","prep_min":20,"cook_min":35,"servings":4,"ingredients":["500 g beef, cubed"],"steps":["Marinate the beef."]}"""


class RecipeReq(BaseModel):
    dish: str
    home: str = ""
    place: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    avoid: list = []
    servings: int = 4


def demo_recipe(avoid):
    shop = lambda n, k: [{"name": n, "km": k, "nav": "https://www.openstreetmap.org/"}]
    nut = any(a in avoid.lower() for a in ("coconut", "nut"))
    sw = lambda name, tip, sc, kind, sh, p=None: {"name": name, "tip": tip, "score": sc, "scored_by": "demo", "store_kind": kind, "x": [.7, .7, .7, .7, .6, 3], "shops": sh, "price": p}
    ings = [
        {"name": "Beef, cubed", "qty": "500 g", "hard": False, "reason": None, "subs": []},
        {"name": "Ginger-garlic paste", "qty": "2 tbsp", "hard": False, "reason": None, "subs": []},
        {"name": "Kashmiri chilli powder", "qty": "2 tsp", "hard": True, "reason": "local",
         "subs": [sw("Sweet paprika + a pinch of cayenne", "Gives the red colour with gentle heat.", 68, "supermarket", shop("Demo Supermarket", 0.6))]},
        {"name": "Coconut slices", "qty": "1/2 cup", "hard": True, "reason": "avoid" if nut else "local",
         "subs": [sw("Toasted sunflower seeds + garlic flakes", "Same crunch, no coconut.", 61, "supermarket", shop("Demo Supermarket", 0.6)) if nut
                  else sw("Unsweetened coconut chips", "Toast them dry until golden.", 82, "health_food", shop("Demo Health Store", 1.1), {"price": "2.99", "seller": "demo"})]},
        {"name": "Curry leaves", "qty": "2 sprigs", "hard": True, "reason": "local",
         "subs": [sw("Fresh basil + a little lime zest", "Not the same flavour: use it as a fresh, aromatic finish.", 38, "greengrocer", shop("Demo Greengrocer", 0.9))]},
        {"name": "Malabar tamarind (kudampuli)", "qty": "2 pieces", "hard": True, "reason": "local",
         "subs": [sw("Lime juice + brown sugar", "About 1 tbsp lime to 1 tsp sugar, added at the end.", 64, "supermarket", shop("Demo Supermarket", 0.6))]}]
    dish = {"title": "Kerala Beef Fry (sample)", "story": "A dry-fried, deeply spiced beef dish that is cooked slowly until the masala clings.",
            "prep_min": 20, "cook_min": 35, "servings": 4, "ingredients": [], "steps": []}
    steps = ["Marinate the beef with ginger-garlic paste, chilli powder and salt for 20 minutes.",
             "Pressure-cook or simmer the beef until tender, keeping the stock.",
             "Fry the crunchy topping in oil until golden, then set aside.",
             "Fry the aromatics, add the beef and reduce the stock until the masala clings, then finish with the topping."]
    return {"demo": True, "country": "Demo land", "dish": dish, "recipe": {"title": dish["title"], "ingredients": [f"{i['qty']} {i['name']}" for i in ings],
            "steps": steps, "notes": "Demo data. Sample swaps only; no model or shop lookups were used."}, "ingredients": ings}


def gen_response(dish, lines, steps):
    dish = {**dish, "ingredients": lines, "steps": steps}
    text = (f"{dish.get('title', '')}\nIngredients:\n" + "\n".join(lines) + "\nSteps:\n"
            + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(steps)))
    return {"generated": True, "dish": dish, "text": text,
            "recipe": {"title": dish.get("title"), "ingredients": lines, "steps": steps, "notes": ""},
            "ingredients": [{"name": ln, "qty": "", "reason": None, "hard": False, "subs": []} for ln in lines]}


@app.post("/api/recipe")
def recipe(r: RecipeReq):
    """Dish name -> recipe only. Substitutes are a separate, explicit step (/api/adapt)."""
    if DEMO:
        full = demo_recipe("")
        return gen_response(full["dish"], full["recipe"]["ingredients"], full["recipe"]["steps"])
    if not r.dish.strip():
        raise HTTPException(400, "Name a dish")
    try:
        print("[recipe] asking the model for the recipe", flush=True)
        dish = chat(RECIPE_SYS, f"Dish: {r.dish}\nServings: {r.servings}\nThe cook's home cuisine: {r.home or 'unknown'}")
    except Exception as e:
        print("[recipe] model call failed:", e, flush=True)
        raise HTTPException(502, f"Recipe generation failed: {e}")
    lines = [str(x) for x in dish.get("ingredients", [])]
    steps = [str(x) for x in dish.get("steps", [])]
    if not lines or not steps:
        raise HTTPException(502, "The model did not return a full recipe. Try again.")
    return gen_response(dish, lines, steps)


class Speak(BaseModel):
    text: str


@app.post("/api/speak")
def speak(s: Speak):
    """Hands-free kitchen: read one step aloud with ElevenLabs. The key stays on the server."""
    key = os.getenv("ELEVENLABS_API_KEY")
    if not key:
        raise HTTPException(503, "No ElevenLabs key set")
    text = s.text.strip()[:600]  # one step at a time keeps cost and abuse down
    if not text:
        raise HTTPException(400, "Nothing to read")
    voice = os.getenv("ELEVENLABS_VOICE_ID") or "21m00Tcm4TlvDq8ikWAM"
    model = os.getenv("ELEVENLABS_TTS_MODEL") or "eleven_multilingual_v2"
    try:
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}", timeout=30,
                          headers={"xi-api-key": key, "accept": "audio/mpeg"}, json={"text": text, "model_id": model})
    except requests.RequestException as e:
        raise HTTPException(502, f"ElevenLabs unreachable: {e}")
    if not r.ok:
        print("[speak] ElevenLabs said", r.status_code, r.text[:200], flush=True)
        raise HTTPException(502, f"ElevenLabs said {r.status_code}")
    return Response(content=r.content, media_type="audio/mpeg")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def index():
    here = pathlib.Path(__file__).parent
    for p in (here / "static" / "index.html", here / "index.html"):  # works in either location
        if p.exists():
            return p.read_text(encoding="utf-8")
    raise HTTPException(500, "index.html not found next to app.py or in static/")