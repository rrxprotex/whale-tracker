"""FastAPI analysis server and browser UI.

Run with::

    uvicorn src.analytics.api:app --reload --port 8010

Endpoints
---------
GET  /                       browser UI
GET  /api/health             service status
GET  /api/catalog            curated symbol catalog
GET  /api/symbols?q=...      symbol search
GET  /api/analyze/{symbol}   full JSON report + SVG charts
GET  /api/summary/{symbol}   compact JSON summary
GET  /api/report/{symbol}    standalone HTML report
POST /api/analyze            JSON body variant
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from . import __version__
from .engine import _json_safe, analysis_summary, analyze
from .market_data import CACHE_DIR, DataError
from .symbols import catalog, search_symbols, supported_intervals

logger = logging.getLogger(__name__)

app = FastAPI(
    title="WhaleTracker Analytics",
    version=__version__,
    description=(
        "Çoklu metodlu piyasa analizi: teknik göstergeler, Fourier/SSA, "
        "istatistik, formasyonlar, rejim ve walk-forward doğrulamalı tahmin."
    ),
)


class AnalysisRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=32)
    market: str = Field("auto", pattern="^(auto|crypto|us|bist)$")
    interval: str = Field("1d", pattern="^(1h|4h|1d|1w)$")
    limit: int = Field(500, ge=80, le=5000)
    horizon: int = Field(30, ge=1, le=365)
    use_cache: bool = True
    save: bool = False


async def _run(request: AnalysisRequest):
    try:
        return await run_in_threadpool(
            analyze,
            request.symbol,
            request.market,
            request.interval,
            request.limit,
            request.horizon,
            request.use_cache,
            request.save,
            None,
        )
    except DataError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Analysis failed for %s", request.symbol)
        raise HTTPException(status_code=500, detail=f"Analiz hatası: {exc}") from exc


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "version": __version__,
        "cache_dir": str(CACHE_DIR),
        "intervals": {
            "crypto": supported_intervals("crypto"),
            "us": supported_intervals("us"),
            "bist": supported_intervals("bist"),
        },
    }


@app.get("/api/catalog")
async def catalog_endpoint():
    return catalog()


@app.get("/api/symbols")
async def symbols_endpoint(
    q: str = Query("", max_length=64),
    market: str = Query("auto"),
    limit: int = Query(15, ge=1, le=50),
):
    return {"query": q, "results": search_symbols(q, market, limit)}


@app.get("/api/analyze/{symbol:path}")
async def analyze_endpoint(
    symbol: str,
    market: str = Query("auto", pattern="^(auto|crypto|us|bist)$"),
    interval: str = Query("1d", pattern="^(1h|4h|1d|1w)$"),
    limit: int = Query(500, ge=80, le=5000),
    horizon: int = Query(30, ge=1, le=365),
    use_cache: bool = True,
    save: bool = False,
):
    request = AnalysisRequest(
        symbol=symbol, market=market, interval=interval,
        limit=limit, horizon=horizon, use_cache=use_cache, save=save,
    )
    result = await _run(request)
    return JSONResponse(
        _json_safe(
            {
                "summary": analysis_summary(result),
                "report": result.report,
                "charts": result.charts,
                "paths": {
                    "html": str(result.html_path) if result.html_path else None,
                    "json": str(result.json_path) if result.json_path else None,
                },
            }
        )
    )


@app.get("/api/summary/{symbol:path}")
async def summary_endpoint(
    symbol: str,
    market: str = Query("auto", pattern="^(auto|crypto|us|bist)$"),
    interval: str = Query("1d", pattern="^(1h|4h|1d|1w)$"),
    limit: int = Query(500, ge=80, le=5000),
    horizon: int = Query(30, ge=1, le=365),
    use_cache: bool = True,
):
    request = AnalysisRequest(
        symbol=symbol, market=market, interval=interval,
        limit=limit, horizon=horizon, use_cache=use_cache,
    )
    result = await _run(request)
    return JSONResponse(_json_safe(analysis_summary(result)))


@app.get("/api/report/{symbol:path}", response_class=HTMLResponse)
async def report_endpoint(
    symbol: str,
    market: str = Query("auto", pattern="^(auto|crypto|us|bist)$"),
    interval: str = Query("1d", pattern="^(1h|4h|1d|1w)$"),
    limit: int = Query(500, ge=80, le=5000),
    horizon: int = Query(30, ge=1, le=365),
    use_cache: bool = True,
    save: bool = False,
):
    request = AnalysisRequest(
        symbol=symbol, market=market, interval=interval,
        limit=limit, horizon=horizon, use_cache=use_cache, save=save,
    )
    result = await _run(request)
    return HTMLResponse(result.html)


@app.post("/api/analyze")
async def analyze_post(request: AnalysisRequest):
    result = await _run(request)
    return JSONResponse(
        _json_safe(
            {
                "summary": analysis_summary(result),
                "report": result.report,
                "charts": result.charts,
                "paths": {
                    "html": str(result.html_path) if result.html_path else None,
                    "json": str(result.json_path) if result.json_path else None,
                },
            }
        )
    )


UI_HTML = """<!DOCTYPE html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>WhaleTracker Analytics</title>
<style>
:root{--bg:#080b10;--panel:#11161d;--line:#222c38;--text:#dce6f2;--muted:#8b9bb0;
--blue:#58a6ff;--green:#3fb950;--red:#f85149;--yellow:#d29922}
*{box-sizing:border-box}html{scroll-behavior:smooth}
body{margin:0;background:radial-gradient(1100px 620px at 12% -10%,#172234,var(--bg) 58%) fixed;
color:var(--text);font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:34px 22px 60px}
header{display:flex;justify-content:space-between;align-items:flex-start;gap:20px;flex-wrap:wrap;margin-bottom:24px}
.brand{font-size:26px;font-weight:850;letter-spacing:-.02em}
.brand span{color:var(--blue)}
.tag{color:var(--muted);font-size:12.5px;margin-top:4px;max-width:680px}
.links{display:flex;gap:9px;flex-wrap:wrap}
.links a{color:var(--muted);font-size:12px;text-decoration:none;border:1px solid var(--line);
padding:7px 10px;border-radius:8px;background:rgba(17,22,29,.7)}
.links a:hover{color:var(--text);border-color:var(--blue)}
.panel{background:linear-gradient(180deg,rgba(21,28,37,.95),rgba(17,22,29,.96));
border:1px solid var(--line);border-radius:18px;padding:22px;box-shadow:0 20px 60px rgba(0,0,0,.28)}
form{display:grid;grid-template-columns:minmax(200px,2fr) repeat(4,minmax(120px,1fr)) auto;gap:10px;align-items:end}
label{display:block;font-size:10px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-bottom:5px}
input,select{width:100%;background:#0a0f15;border:1px solid var(--line);border-radius:9px;color:var(--text);
padding:11px 12px;font:inherit;font-size:13px;outline:none}
input:focus,select:focus{border-color:var(--blue);box-shadow:0 0 0 3px rgba(88,166,255,.12)}
button{border:0;border-radius:9px;padding:12px 18px;background:var(--blue);color:#04101f;font:inherit;
font-size:13px;font-weight:800;cursor:pointer;white-space:nowrap}
button:hover{filter:brightness(1.08)}button:disabled{opacity:.55;cursor:wait}
.chips{display:flex;gap:7px;flex-wrap:wrap;margin-top:14px}
.chip{border:1px solid var(--line);background:rgba(8,11,16,.45);color:var(--muted);border-radius:999px;
padding:6px 10px;font-size:11.5px;cursor:pointer}.chip:hover{color:var(--text);border-color:var(--blue)}
.status{margin-top:16px;font-size:12px;color:var(--muted);display:flex;align-items:center;gap:9px}
.spinner{width:14px;height:14px;border:2px solid var(--line);border-top-color:var(--blue);border-radius:50%;
animation:spin .8s linear infinite;display:none}.loading .spinner{display:block}
@keyframes spin{to{transform:rotate(360deg)}}
.frame-wrap{margin-top:18px;border:1px solid var(--line);border-radius:16px;overflow:hidden;background:#0d1117;
display:none;min-height:420px}
.frame-wrap.visible{display:block}
iframe{width:100%;height:82vh;border:0;display:block;background:#0d1117}
.frame-bar{display:flex;justify-content:space-between;align-items:center;gap:12px;padding:9px 12px;
background:#0a0f15;border-bottom:1px solid var(--line);font-size:11.5px;color:var(--muted)}
.frame-bar a{color:var(--blue);text-decoration:none}
.err{display:none;margin-top:14px;padding:12px 14px;border:1px solid rgba(248,81,73,.35);
background:rgba(248,81,73,.08);color:#ffb4b0;border-radius:10px;font-size:12.5px;white-space:pre-wrap}
.err.visible{display:block}
.quick-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:16px}
.quick{border:1px solid var(--line);border-radius:12px;padding:12px 13px;background:rgba(8,11,16,.35);cursor:pointer}
.quick:hover{border-color:var(--blue)}
.quick b{display:block;font-size:13px}.quick span{font-size:11px;color:var(--muted)}
.info-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;margin-top:20px}
.info{border:1px solid var(--line);border-radius:12px;padding:14px 15px;background:rgba(8,11,16,.28)}
.info b{display:block;font-size:12px;color:var(--blue);margin-bottom:5px}
.info span{font-size:11.5px;color:var(--muted);line-height:1.55}
footer{color:var(--muted);font-size:11px;text-align:center;margin-top:30px;line-height:1.7}
@media(max-width:1000px){form{grid-template-columns:repeat(2,minmax(0,1fr))}button{grid-column:1/-1}
.quick-grid,.info-grid{grid-template-columns:1fr}}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div>
    <div class="brand">Whale<span>Tracker</span> Analytics</div>
    <div class="tag">Teknik göstergeler · Fourier / SSA · istatistik · formasyonlar · rejim ·
    walk-forward doğrulamalı çoklu model tahmini. Tüm grafikler rapora gömülü SVG olarak üretilir.</div>
  </div>
  <div class="links">
    <a href="/docs" target="_blank">API docs</a>
    <a href="/api/catalog" target="_blank">Sembol kataloğu</a>
    <a href="/api/health" target="_blank">Durum</a>
  </div>
</header>

<section class="panel">
<form id="analysis-form" autocomplete="off">
  <div>
    <label for="symbol">Sembol</label>
    <input id="symbol" name="symbol" list="symbol-list" placeholder="BTCUSDT, AAPL, THYAO..." value="BTCUSDT" required>
    <datalist id="symbol-list"></datalist>
  </div>
  <div>
    <label for="market">Piyasa</label>
    <select id="market" name="market">
      <option value="auto">Otomatik</option>
      <option value="crypto">Kripto</option>
      <option value="us">ABD</option>
      <option value="bist">BIST</option>
    </select>
  </div>
  <div>
    <label for="interval">Aralık</label>
    <select id="interval" name="interval">
      <option value="1d">1 gün</option>
      <option value="4h">4 saat</option>
      <option value="1h">1 saat</option>
      <option value="1w">1 hafta</option>
    </select>
  </div>
  <div>
    <label for="horizon">Ufuk (bar)</label>
    <input id="horizon" name="horizon" type="number" min="1" max="365" value="30">
  </div>
  <div>
    <label for="limit">Bar</label>
    <input id="limit" name="limit" type="number" min="80" max="5000" value="500">
  </div>
  <button id="run" type="submit">Analiz et</button>
</form>
<div class="chips">
  <span class="chip" data-symbol="BTCUSDT">BTCUSDT</span>
  <span class="chip" data-symbol="ETHUSDT">ETHUSDT</span>
  <span class="chip" data-symbol="SOLUSDT">SOLUSDT</span>
  <span class="chip" data-symbol="AAPL">AAPL</span>
  <span class="chip" data-symbol="NVDA">NVDA</span>
  <span class="chip" data-symbol="THYAO">THYAO</span>
  <span class="chip" data-symbol="ASELS">ASELS</span>
</div>
<div class="status" id="status"><span class="spinner"></span><span id="status-text">Hazır.</span></div>
<div class="err" id="error"></div>
<div class="frame-wrap" id="frame-wrap">
  <div class="frame-bar"><span id="frame-title">Rapor</span><a id="open-new" href="#" target="_blank">Yeni sekmede aç ↗</a></div>
  <iframe id="frame" title="Analiz raporu" loading="lazy"></iframe>
</div>
<div class="quick-grid">
  <div class="quick" data-symbol="BTCUSDT"><b>BTCUSDT</b><span>Kripto · 1d · 30 bar</span></div>
  <div class="quick" data-symbol="AAPL"><b>AAPL</b><span>ABD · 1d · 30 bar</span></div>
  <div class="quick" data-symbol="THYAO"><b>THYAO</b><span>BIST · 1d · 30 bar</span></div>
</div>
<div class="info-grid">
  <div class="info"><b>Kapsam</b><span>Kripto, ABD ve BIST sembolleri; 1s/4s/1g/1h aralıkları. Veri Binance, Bybit ve Yahoo Finance üzerinden anahtarsız çekilir.</span></div>
  <div class="info"><b>Yöntem</b><span>Teknik göstergeler, Fourier/SSA, Hurst/ADF, formasyonlar, rejim ve walk-forward doğrulamalı 8 modelli olasılıksal tahmin.</span></div>
  <div class="info"><b>Sınırlama</b><span>Yalnızca fiyat/hacim geçmişi; haber, order-book ve temel veri yok. Çıktı yatırım tavsiyesi değildir.</span></div>
</div>
</section>
<footer>WhaleTracker Analytics · yatırım tavsiyesi değildir · veri kaynakları: Binance, Bybit, Yahoo Finance</footer>
</div>
<script>
const form=document.getElementById('analysis-form');
const statusEl=document.getElementById('status');
const statusText=document.getElementById('status-text');
const errEl=document.getElementById('error');
const frameWrap=document.getElementById('frame-wrap');
const frame=document.getElementById('frame');
const openNew=document.getElementById('open-new');
const frameTitle=document.getElementById('frame-title');
const symbolInput=document.getElementById('symbol');

function showError(message){
  errEl.textContent=message;errEl.classList.add('visible');
}
function clearError(){errEl.textContent='';errEl.classList.remove('visible');}
function setLoading(on,text){
  statusEl.classList.toggle('loading',on);
  statusText.textContent=text||(on?'Analiz çalışıyor...':'Hazır.');
  document.getElementById('run').disabled=on;
}
function buildUrl(symbol){
  const market=document.getElementById('market').value;
  const interval=document.getElementById('interval').value;
  const horizon=document.getElementById('horizon').value;
  const limit=document.getElementById('limit').value;
  return `/api/report/${encodeURIComponent(symbol)}?market=${market}&interval=${interval}&horizon=${horizon}&limit=${limit}`;
}
function runAnalysis(symbol){
  clearError();
  if(!symbol){showError('Sembol girin.');return;}
  symbolInput.value=symbol;
  const url=buildUrl(symbol);
  frameTitle.textContent=symbol+' · '+document.getElementById('interval').value;
  openNew.href=url;
  frameWrap.classList.add('visible');
  frame.src=url;
  setLoading(true,'Veri çekiliyor ve modeller çalıştırılıyor...');
  frame.onload=()=>{setLoading(false,'Rapor hazır.');};
  setTimeout(()=>{if(statusEl.classList.contains('loading')) setLoading(false,'Rapor yükleniyor...');},12000);
}
form.addEventListener('submit',(event)=>{
  event.preventDefault();
  runAnalysis(symbolInput.value.trim().toUpperCase());
});
document.querySelectorAll('.chip,.quick').forEach((node)=>{
  node.addEventListener('click',()=>runAnalysis(node.dataset.symbol));
});

let timer=null;
symbolInput.addEventListener('input',()=>{
  const q=symbolInput.value.trim();
  if(timer) clearTimeout(timer);
  if(q.length<1) return;
  timer=setTimeout(async()=>{
    try{
      const market=document.getElementById('market').value;
      const response=await fetch(`/api/symbols?q=${encodeURIComponent(q)}&market=${market}&limit=12`);
      const data=await response.json();
      const list=document.getElementById('symbol-list');
      list.innerHTML='';
      for(const item of data.results||[]){
        const option=document.createElement('option');
        option.value=item.symbol;
        option.label=`${item.name} · ${item.market}`;
        list.appendChild(option);
      }
    }catch(e){/* suggestions are optional */}
  },180);
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(UI_HTML)
