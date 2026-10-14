"""
Crude Intel — Backend Server
FastAPI server that scrapes live crude oil news, prices, and feeds Claude AI for analysis.
Run: uvicorn server:app --reload --port 8000
"""

import asyncio
import httpx
import feedparser
import json
import time
from datetime import datetime
from typing import Optional
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import anthropic

# ── CONFIG ──────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = "YOUR_ANTHROPIC_API_KEY_HERE"   # ← paste your key

RSS_FEEDS = {
    "Reuters Energy":    "https://feeds.reuters.com/reuters/businessNews",
    "OilPrice.com":      "https://oilprice.com/rss/main",
    "EIA News":          "https://www.eia.gov/rss/news.xml",
    "Rigzone":           "https://www.rigzone.com/news/rss/rigzone_latest.aspx",
    "OPEC Newsroom":     "https://www.opec.org/opec_web/en/press_room/rss.htm",
    "Nasdaq Commodities":"https://www.nasdaq.com/feed/rssoutbound?category=Commodities",
}

OIL_KEYWORDS = [
    "crude","oil","brent","wti","opec","barrel","petroleum","refinery",
    "gasoline","energy","natural gas","lng","pipeline","iran","saudi",
    "russia","venezuela","libya","nigeria","iraq","shale","rig count",
    "inventory","eia","api","production cut","supply","demand","tanker",
    "geopolit","middle east","houthi","straits of hormuz","nato","sanctions",
]

PRICE_SOURCES = {
    "WTI":   "https://query1.finance.yahoo.com/v8/finance/chart/CL=F?interval=1m&range=1d",
    "Brent": "https://query1.finance.yahoo.com/v8/finance/chart/BZ=F?interval=1m&range=1d",
    "DXY":   "https://query1.finance.yahoo.com/v8/finance/chart/DX-Y.NYB?interval=5m&range=1d",
    "NatGas":"https://query1.finance.yahoo.com/v8/finance/chart/NG=F?interval=5m&range=1d",
    "SPX":   "https://query1.finance.yahoo.com/v8/finance/chart/%5EGSPC?interval=5m&range=1d",
}

REFRESH_INTERVAL_SECS = 300   # 5 minutes

# ── APP SETUP ───────────────────────────────────────────────────────────────

app = FastAPI(title="Crude Intel API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# ── IN-MEMORY CACHE ─────────────────────────────────────────────────────────

cache = {
    "news":       [],
    "prices":     {},
    "analysis":   "",
    "last_update": None,
    "status":     "initializing",
}

# ── PRICE FETCHER ────────────────────────────────────────────────────────────

async def fetch_prices() -> dict:
    prices = {}
    async with httpx.AsyncClient(timeout=10) as http:
        for symbol, url in PRICE_SOURCES.items():
            try:
                r = await http.get(url, headers={"User-Agent": "Mozilla/5.0"})
                data = r.json()
                meta = data["chart"]["result"][0]["meta"]
                prices[symbol] = {
                    "price":   round(meta.get("regularMarketPrice", 0), 2),
                    "prev":    round(meta.get("previousClose", 0), 2),
                    "change":  round(meta.get("regularMarketPrice", 0) - meta.get("previousClose", 0), 2),
                    "pct":     round(((meta.get("regularMarketPrice", 0) - meta.get("previousClose", 0))
                                     / max(meta.get("previousClose", 1), 1)) * 100, 2),
                    "high":    round(meta.get("regularMarketDayHigh", 0), 2),
                    "low":     round(meta.get("regularMarketDayLow", 0), 2),
                    "currency": meta.get("currency", "USD"),
                }
            except Exception as e:
                prices[symbol] = {"price": 0, "change": 0, "pct": 0, "error": str(e)}
    return prices


# ── NEWS FETCHER ─────────────────────────────────────────────────────────────

def is_oil_related(text: str) -> bool:
    text_lower = text.lower()
    return any(kw in text_lower for kw in OIL_KEYWORDS)

async def fetch_news() -> list:
    all_articles = []
    async with httpx.AsyncClient(timeout=15) as http:
        for source_name, feed_url in RSS_FEEDS.items():
            try:
                r = await http.get(feed_url, headers={
                    "User-Agent": "Mozilla/5.0 (compatible; CrudeIntel/1.0)"
                })
                feed = feedparser.parse(r.text)
                for entry in feed.entries[:20]:
                    title   = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    link    = entry.get("link", "")
                    pub     = entry.get("published", entry.get("updated", ""))

                    if is_oil_related(title + " " + summary):
                        all_articles.append({
                            "source":   source_name,
                            "title":    title,
                            "summary":  summary[:300],
                            "link":     link,
                            "published": pub,
                            "fetched_at": datetime.utcnow().isoformat(),
                        })
            except Exception as e:
                print(f"[NEWS] Error fetching {source_name}: {e}")

    # Deduplicate by title similarity and sort by recency
    seen = set()
    unique = []
    for a in all_articles:
        key = a["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            unique.append(a)

    return unique[:40]   # top 40 oil-relevant headlines


# ── AI ANALYSIS ──────────────────────────────────────────────────────────────

def build_analysis_prompt(news: list, prices: dict) -> str:
    price_block = "\n".join([
        f"- {sym}: ${d.get('price','?')} ({'+' if d.get('change',0)>=0 else ''}{d.get('change','?')} | {'+' if d.get('pct',0)>=0 else ''}{d.get('pct','?')}%)"
        for sym, d in prices.items()
    ])

    headlines = "\n".join([
        f"[{a['source']}] {a['title']}"
        for a in news[:25]
    ])

    return f"""You are a senior crude oil market analyst. Based on the live market data below, provide a structured market intelligence briefing.

CURRENT PRICES:
{price_block}

LATEST OIL-RELATED HEADLINES ({datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}):
{headlines}

Provide a market briefing with these sections:
## Overall Market Bias
State clearly: BULLISH ▲ / BEARISH ▼ / NEUTRAL ◆ with confidence level (%)

## Price Outlook (Next 4-8 Hours)
Short-term WTI and Brent direction with key levels

## Top 3 Catalysts Right Now
The 3 most market-moving developments from today's headlines

## Key Levels to Watch
WTI and Brent support/resistance levels

## Risk Events This Session
What could cause a sudden reversal

## Entity Signals
One-line signal for each: OPEC+, USD/DXY, Geopolitics, China, U.S. Macro

Keep it tight, professional, and actionable. Traders need signal, not prose."""


async def run_ai_analysis(news: list, prices: dict) -> str:
    prompt = build_analysis_prompt(news, prices)
    try:
        message = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1200,
            system="You are a senior crude oil market analyst. Be sharp, specific, and actionable. Use ▲ for bullish, ▼ for bearish, ◆ for neutral.",
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"AI analysis error: {e}"


# ── BACKGROUND REFRESH LOOP ──────────────────────────────────────────────────

async def refresh_data():
    """Runs every REFRESH_INTERVAL_SECS seconds to update cache."""
    while True:
        try:
            cache["status"] = "refreshing"
            print(f"[{datetime.utcnow().isoformat()}] Refreshing data...")

            prices, news = await asyncio.gather(
                fetch_prices(),
                fetch_news(),
            )

            cache["prices"]      = prices
            cache["news"]        = news
            cache["analysis"]    = await run_ai_analysis(news, prices)
            cache["last_update"] = datetime.utcnow().isoformat()
            cache["status"]      = "live"

            print(f"[{datetime.utcnow().isoformat()}] Refresh complete. {len(news)} headlines, analysis ready.")

        except Exception as e:
            cache["status"] = "error"
            print(f"[REFRESH ERROR] {e}")

        await asyncio.sleep(REFRESH_INTERVAL_SECS)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(refresh_data())


# ── API ENDPOINTS ─────────────────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return {
        "status":      cache["status"],
        "last_update": cache["last_update"],
        "news_count":  len(cache["news"]),
        "prices_count": len(cache["prices"]),
    }

@app.get("/api/prices")
async def get_prices():
    return {"prices": cache["prices"], "last_update": cache["last_update"]}

@app.get("/api/news")
async def get_news(limit: int = 20, source: Optional[str] = None):
    news = cache["news"]
    if source:
        news = [n for n in news if source.lower() in n["source"].lower()]
    return {"news": news[:limit], "total": len(cache["news"]), "last_update": cache["last_update"]}

@app.get("/api/analysis")
async def get_analysis():
    return {"analysis": cache["analysis"], "last_update": cache["last_update"]}

@app.get("/api/all")
async def get_all():
    """Single endpoint for dashboard to fetch everything at once."""
    return {
        "prices":    cache["prices"],
        "news":      cache["news"][:30],
        "analysis":  cache["analysis"],
        "last_update": cache["last_update"],
        "status":    cache["status"],
    }

@app.post("/api/analyze-news")
async def analyze_custom_news(body: dict):
    """Analyze a custom headline on demand."""
    headline = body.get("headline", "")
    if not headline:
        return {"error": "No headline provided"}
    try:
        msg = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=800,
            system="You are a crude oil market analyst. Analyze news headlines for crude oil market impact. Be sharp and specific.",
            messages=[{"role": "user", "content": f"""Analyze this for crude oil impact:
"{headline}"

Current WTI: ${cache['prices'].get('WTI', {}).get('price', 'N/A')}
Current Brent: ${cache['prices'].get('Brent', {}).get('price', 'N/A')}

Provide: 1) Bullish▲/Bearish▼/Neutral◆ signal (strength 1-10), 2) Expected price move, 3) Duration, 4) WTI vs Brent impact, 5) Related assets."""}],
        )
        return {"analysis": msg.content[0].text}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/stream-analysis")
async def stream_analysis():
    """Stream AI analysis via SSE for real-time feel."""
    news    = cache["news"][:25]
    prices  = cache["prices"]
    prompt  = build_analysis_prompt(news, prices)

    async def event_generator():
        try:
            with client.messages.stream(
                model="claude-opus-4-5",
                max_tokens=1200,
                system="You are a senior crude oil market analyst. Be sharp, specific, and actionable.",
                messages=[{"role": "user", "content": prompt}],
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'chunk': text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
