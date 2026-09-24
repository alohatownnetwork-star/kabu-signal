# -*- coding: utf-8 -*-
"""
毎朝の売買シグナル一覧を作るプログラム。
東証の全上場銘柄（プライム・スタンダード・グロース）の株価（前日までの日足）を取得して
シグナルを判定し、シグナルが出た銘柄は業績・割安度（ファンダメンタル）も調べて、
docs/index.html（スマホで見るページ）に書き出します。
watchlist.txt に書いた銘柄は「お気に入り」として目立たせます。
"""
import datetime as dt
import html
import io
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# ===== 見張る範囲 =====
MARKETS = ["プライム", "スタンダード", "グロース"]  # 見張る市場（消すとその市場を見なくなる）
MIN_TURNOVER = 1e8     # 1日の売買代金（20日平均）がこれ未満の銘柄は除外（1e8 = 1億円）。お気に入りは除外しない

# ===== テクニカルの設定（数字を変えるとルールが変わります） =====
SHORT_MA = 5           # 短期の移動平均（日）
LONG_MA = 25           # 長期の移動平均（日）
RSI_DAYS = 14          # RSIの期間（日）
RSI_LOW = 30           # これ以下で「売られすぎ」
RSI_HIGH = 75          # これ以上で「買われすぎ」
HIGH_DAYS = 245        # 高値更新を見る期間（約1年の営業日）
VOLUME_RATIO = 1.5     # 出来高が20日平均の何倍以上なら「出来高を伴う」とみなすか
UNIT = 100             # 売買単位（株）

# ===== ファンダメンタルの合格ライン（満たすと ○ が付き、★が1つ増える） =====
PER_MAX = 15           # PER（株価÷1株利益）がこれ以下なら割安
PBR_MAX = 1.5          # PBR（株価÷1株純資産）がこれ以下なら割安
ROE_MIN = 8            # ROE（%）がこれ以上なら稼ぐ力がある
DIV_MIN = 3            # 配当利回り（%）がこれ以上なら高配当
GROWTH_MIN = 0         # 売上の伸び（前年比%）がこれより上なら増収
# =====================================================

JPX_PAGE = "https://www.jpx.co.jp/markets/statistics-equities/misc/01.html"
JPX_URLS = [
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx",
    "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xls",
]
JST = dt.timezone(dt.timedelta(hours=9))
HERE = os.path.dirname(os.path.abspath(__file__))


def log(msg):
    print(msg, flush=True)


def load_watchlist(path):
    items = []
    if not os.path.exists(path):
        return items
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(",", 1)
            code = parts[0].strip().upper()
            name = parts[1].strip() if len(parts) > 1 else code
            items.append((code, name))
    return items


def find_jpx_urls():
    """東証のページから一覧ファイルのリンクを探す（ファイル名や形式が変わっても追従するため）。"""
    import re
    import requests
    urls = []
    try:
        page = requests.get(JPX_PAGE, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
        page.encoding = page.apparent_encoding
        for href in re.findall(r'href="([^"]+data_j\.xlsx?)"', page.text):
            urls.append(href if href.startswith("http") else "https://www.jpx.co.jp" + href)
    except Exception as e:
        log(f"  東証のページを読めませんでした（{e}）")
    for u in JPX_URLS:
        if u not in urls:
            urls.append(u)
    return urls


def load_universe():
    """東証（JPX）の上場銘柄一覧から、見張る市場の国内株式を取り出す。"""
    need = []
    for mod in ("openpyxl", "xlrd"):
        try:
            __import__(mod)
        except ImportError:
            need.append(mod)
    if need:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *need], check=False)
    import requests
    df, last_err = None, None
    for url in find_jpx_urls():
        try:
            r = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = pd.read_excel(io.BytesIO(r.content), dtype=str)
            log(f"  一覧ファイル：{url}")
            break
        except Exception as e:
            last_err = e
    if df is None:
        raise RuntimeError(last_err)
    col_mkt = [c for c in df.columns if "市場" in c][0]
    col_sec = [c for c in df.columns if "33業種区分" in c]
    col_code = [c for c in df.columns if "コード" in c and "業種" not in c and "規模" not in c][0]
    col_name = [c for c in df.columns if "銘柄名" in c][0]
    out = {}
    for _, row in df.iterrows():
        mkt = str(row[col_mkt])
        if "内国株式" not in mkt:
            continue
        m = next((m for m in MARKETS if mkt.startswith(m)), None)
        if not m:
            continue
        code = str(row[col_code]).strip().upper()
        out[code] = {"name": str(row[col_name]).strip(), "market": m,
                     "sector": str(row[col_sec[0]]).strip() if col_sec else ""}
    return out


def rsi(close, days):
    diff = close.diff()
    up = diff.clip(lower=0).ewm(alpha=1 / days, adjust=False).mean()
    down = (-diff.clip(upper=0)).ewm(alpha=1 / days, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def judge(df):
    """1銘柄の日足データからシグナルのリストを返す。[(種類, 理由), ...]"""
    df = df.dropna(subset=["Close"])
    if len(df) < LONG_MA + 2:
        return [], None
    close, high, vol = df["Close"], df["High"], df["Volume"].fillna(0)
    s = close.rolling(SHORT_MA).mean()
    l = close.rolling(LONG_MA).mean()
    r = rsi(close, RSI_DAYS)
    vol_avg = vol.shift(1).rolling(20).mean()

    out = []
    # 1) ゴールデンクロス／デッドクロス
    if s.iloc[-2] <= l.iloc[-2] and s.iloc[-1] > l.iloc[-1]:
        out.append(("buy", f"ゴールデンクロス（{SHORT_MA}日平均が{LONG_MA}日平均を上抜け）"))
    if s.iloc[-2] >= l.iloc[-2] and s.iloc[-1] < l.iloc[-1]:
        out.append(("sell", f"デッドクロス（{SHORT_MA}日平均が{LONG_MA}日平均を下抜け）"))

    # 2) 出来高を伴う高値ブレイク（約1年の高値を更新）
    past_high = high.iloc[-HIGH_DAYS - 1:-1].max()
    va = vol_avg.iloc[-1]
    vr = vol.iloc[-1] / va if va and va > 0 else 0
    if close.iloc[-1] > past_high and vr >= VOLUME_RATIO:
        out.append(("buy", f"出来高{vr:.1f}倍で1年来高値を更新"))

    # 3) RSI（売られすぎ・買われすぎ）
    rv = r.iloc[-1]
    if not np.isnan(rv):
        if rv <= RSI_LOW:
            out.append(("watch", f"売られすぎ（RSI {rv:.0f}）"))
        elif rv >= RSI_HIGH:
            out.append(("caution", f"買われすぎ（RSI {rv:.0f}）"))

    info = {
        "date": df.index[-1].strftime("%Y-%m-%d"),
        "close": float(close.iloc[-1]),
        "change": float((close.iloc[-1] / close.iloc[-2] - 1) * 100),
        "turnover": float((close * vol).tail(20).mean()),
    }
    return out, info


def download(codes, chunk=100):
    """株価を100銘柄ずつまとめて取得する（一度に頼みすぎると断られるため）。"""
    import yfinance as yf
    result = {}
    for i in range(0, len(codes), chunk):
        part = codes[i:i + chunk]
        tickers = [f"{c}.T" for c in part]
        data = None
        for attempt in range(3):
            try:
                data = yf.download(tickers, period="15mo", interval="1d", group_by="ticker",
                                   auto_adjust=False, threads=True, progress=False)
                break
            except Exception as e:
                log(f"  取得失敗（{attempt + 1}回目）: {e}")
                time.sleep(10)
        if data is None or data.empty:
            continue
        for c, t in zip(part, tickers):
            try:
                d = data[t] if isinstance(data.columns, pd.MultiIndex) else data
                d = d[["Close", "High", "Volume"]].dropna(subset=["Close"])
                if len(d):
                    result[c] = d
            except KeyError:
                pass
        log(f"  株価取得 {min(i + chunk, len(codes))}/{len(codes)}")
        time.sleep(1)
    return result


def _num(v):
    try:
        v = float(v)
        return None if math.isnan(v) or math.isinf(v) else v
    except (TypeError, ValueError):
        return None


def fetch_fundamentals(code, close):
    """PER・PBR・ROE・配当利回り・売上の伸び・時価総額を取得。取れない項目は None。"""
    import yfinance as yf
    info = {}
    for attempt in range(3):
        try:
            info = yf.Ticker(f"{code}.T").info or {}
            break
        except Exception:
            time.sleep(3 * (attempt + 1))
    per = _num(info.get("trailingPE"))
    pbr = _num(info.get("priceToBook"))
    roe = _num(info.get("returnOnEquity"))
    growth = _num(info.get("revenueGrowth"))
    rate = _num(info.get("dividendRate")) or _num(info.get("trailingAnnualDividendRate"))
    cap = _num(info.get("marketCap"))
    return {
        "per": per if per and per > 0 else None,
        "pbr": pbr if pbr and pbr > 0 else None,
        "roe": roe * 100 if roe is not None else None,
        "div": rate / close * 100 if rate and close else None,
        "growth": growth * 100 if growth is not None else None,
        "cap": cap,
    }


def fund_checks(f):
    """各項目が合格ラインを満たすか。True/False/None（データなし）"""
    def chk(v, ok):
        return None if v is None else ok(v)
    return {
        "per": chk(f["per"], lambda v: v <= PER_MAX),
        "pbr": chk(f["pbr"], lambda v: v <= PBR_MAX),
        "roe": chk(f["roe"], lambda v: v >= ROE_MIN),
        "div": chk(f["div"], lambda v: v >= DIV_MIN),
        "growth": chk(f["growth"], lambda v: v > GROWTH_MIN),
    }


LABEL = {
    "buy": "🟢 買い",
    "sell": "🔴 売り",
    "watch": "🟡 注目",
    "caution": "🟠 注意",
}
ORDER = ["buy", "sell", "watch", "caution"]


def fmt(v, suffix="", digits=1):
    return "-" if v is None else f"{v:,.{digits}f}{suffix}"


def fmt_cap(v):
    if v is None:
        return "-"
    return f"{v / 1e12:,.1f}兆円" if v >= 1e12 else f"{v / 1e8:,.0f}億円"


def fund_html(f, checks):
    items = [
        ("PER", fmt(f["per"], "倍"), checks["per"]),
        ("PBR", fmt(f["pbr"], "倍", 2), checks["pbr"]),
        ("ROE", fmt(f["roe"], "%"), checks["roe"]),
        ("配当", fmt(f["div"], "%", 2), checks["div"]),
        ("売上の伸び", fmt(f["growth"], "%"), checks["growth"]),
    ]
    cells = "".join(
        f'<span class="f{" good" if ok else ""}">{k} <b>{v}</b>{" ○" if ok else ""}</span>'
        for k, v, ok in items)
    return f'<div class="fund">{cells}<span class="f">時価総額 <b>{fmt_cap(f["cap"])}</b></span></div>'


def card_html(r):
    chg = r["change"]
    e = html.escape
    fav = '<span class="favmark">⭐お気に入り</span>' if r["fav"] else ""
    return f"""
      <a class="card" data-market="{e(r['market'])}" data-score="{r['score']}" data-fav="{1 if r['fav'] else 0}"
         href="https://kabutan.jp/stock/?code={e(r['code'])}" target="_blank" rel="noopener">
        <div class="top"><span class="name">{e(r['name'])}</span><span class="code">{e(r['code'])}</span></div>
        <div class="tags"><span class="mkt">{e(r['market'])}</span><span class="sec">{e(r['sector'])}</span>{fav}</div>
        <div class="reason">{e(r['reason'])}</div>
        <div class="nums">
          <span>終値 <b>{r['close']:,.0f}円</b> <span class="{'up' if chg >= 0 else 'down'}">{chg:+.1f}%</span></span>
          <span>{UNIT}株 {r['cost']:,.0f}円</span>
        </div>
        <div class="score">ファンダ {'★' * r['score']}{'☆' * (5 - r['score'])} <small>{r['score']}/5</small></div>
        {fund_html(r['fund'], r['checks'])}
      </a>"""


def build_html(rows, stats, updated):
    sections = []
    for k in ORDER:
        items = [r for r in rows if r["kind"] == k]
        # お気に入りを先頭に。買い・注目はファンダの点数が高い順、売り・注意は低い順
        rev = k in ("buy", "watch")
        items.sort(key=lambda x: (not x["fav"], -x["score"] if rev else x["score"], -x["turnover"]))
        body = "".join(card_html(r) for r in items)
        sections.append(
            f'<section class="{k}"><h2>{LABEL[k]} <small class="cnt">{len(items)}件</small></h2>'
            f'{body}<p class="empty"{"" if not items else " hidden"}>なし</p></section>')

    mkt_btns = "".join(f'<button data-m="{m}">{m}</button>' for m in MARKETS)
    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>朝の売買シグナル</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --text:#1c1f24; --sub:#6b7280; --line:#e5e7eb; --chip:#f1f3f5;
        --buy:#16a34a; --sell:#dc2626; --watch:#ca8a04; --caution:#ea580c; --up:#dc2626; --down:#2563eb;
        --good:#15803d; --goodbg:#dcfce7; --star:#b45309; --accent:#2563eb; --accentbg:#dbeafe; --fav:#a16207; --favbg:#fef3c7; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#111418; --card:#1b1f25; --text:#e8eaed; --sub:#9aa0a6; --line:#2c323a; --chip:#242a31;
          --up:#f87171; --down:#60a5fa; --good:#4ade80; --goodbg:#14301f; --star:#fbbf24;
          --accent:#93c5fd; --accentbg:#1e3a5f; --fav:#fcd34d; --favbg:#3a2e0b; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic UI",sans-serif; }}
main {{ max-width:640px; margin:0 auto; padding:16px; }}
h1 {{ font-size:20px; margin:4px 0; }}
.meta {{ color:var(--sub); font-size:13px; margin-bottom:12px; line-height:1.5; }}
.panel {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; margin:12px 0 4px;
         position:sticky; top:0; z-index:5; }}
.seg {{ display:flex; gap:6px; flex-wrap:wrap; margin-bottom:8px; }}
.seg button {{ font:inherit; font-size:13px; border:1px solid var(--line); background:var(--chip); color:var(--text);
              border-radius:999px; padding:4px 12px; cursor:pointer; }}
.seg button.on {{ background:var(--accentbg); color:var(--accent); border-color:var(--accent); font-weight:600; }}
.chk {{ display:flex; gap:14px; flex-wrap:wrap; font-size:14px; }}
.chk label {{ display:flex; gap:6px; align-items:center; }}
h2 {{ font-size:16px; margin:20px 0 8px; }}
h2 small {{ color:var(--sub); font-weight:normal; }}
.card {{ display:block; text-decoration:none; color:inherit; background:var(--card);
        border:1px solid var(--line); border-left:4px solid var(--line); border-radius:10px;
        padding:10px 12px; margin-bottom:8px; }}
.card[data-fav="1"] {{ outline:2px solid var(--fav); outline-offset:-1px; }}
.buy .card {{ border-left-color:var(--buy); }} .sell .card {{ border-left-color:var(--sell); }}
.watch .card {{ border-left-color:var(--watch); }} .caution .card {{ border-left-color:var(--caution); }}
.top {{ display:flex; justify-content:space-between; gap:8px; }}
.name {{ font-weight:600; }} .code {{ color:var(--sub); font-size:13px; }}
.tags {{ display:flex; gap:4px; flex-wrap:wrap; margin-top:4px; }}
.tags span {{ font-size:11px; border-radius:4px; padding:1px 6px; background:var(--chip); color:var(--sub); }}
.tags .favmark {{ background:var(--favbg); color:var(--fav); font-weight:600; }}
.reason {{ font-size:14px; margin:6px 0 4px; }}
.nums {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:4px 12px; font-size:13px; color:var(--sub); }}
.nums b {{ color:var(--text); }} .up {{ color:var(--up); }} .down {{ color:var(--down); }}
.score {{ font-size:13px; color:var(--star); margin-top:6px; }}
.score small {{ color:var(--sub); }}
.fund {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }}
.f {{ font-size:12px; color:var(--sub); background:var(--chip); border-radius:6px; padding:2px 6px; }}
.f b {{ color:var(--text); font-weight:600; }}
.f.good {{ background:var(--goodbg); color:var(--good); }} .f.good b {{ color:var(--good); }}
.empty {{ color:var(--sub); font-size:14px; }}
.card.hide {{ display:none; }}
.note {{ color:var(--sub); font-size:12px; line-height:1.6; margin-top:24px; }}
</style>
</head>
<body>
<main>
  <h1>朝の売買シグナル</h1>
  <div class="meta">株価の日付：{html.escape(updated['data_date'])} ／ 更新：{html.escape(updated['run_at'])}<br>
  東証 {stats['universe']:,}銘柄 → 株価取得 {stats['downloaded']:,}銘柄 → 売買代金{MIN_TURNOVER / 1e8:g}億円以上 {stats['liquid']:,}銘柄を判定</div>
  <div class="panel">
    <div class="seg" id="mkt"><button data-m="" class="on">すべて</button>{mkt_btns}</div>
    <div class="chk">
      <label><input type="checkbox" id="star"> ★3つ以上</label>
      <label><input type="checkbox" id="fav"> お気に入りだけ</label>
    </div>
  </div>
  {''.join(sections)}
  <p class="note">銘柄をタップすると株探の銘柄ページが開きます。⭐はwatchlist.txtに書いたお気に入り銘柄です。<br>
  <b>テクニカル</b>：{SHORT_MA}日・{LONG_MA}日移動平均のクロス／出来高{VOLUME_RATIO}倍以上での1年来高値更新／RSI（{RSI_DAYS}日）{RSI_LOW}以下・{RSI_HIGH}以上<br>
  <b>ファンダ★</b>：PER {PER_MAX}倍以下／PBR {PBR_MAX}倍以下／ROE {ROE_MIN}%以上／配当 {DIV_MIN}%以上／増収　を満たすごとに★1つ（○印）。「-」はデータが取れなかった項目です。<br>
  売買代金が少ない銘柄は、値動きが荒く売買しにくいため除外しています（お気に入りは除外しません）。<br>
  業績データはYahoo Financeのもので、最新の決算とずれることがあります。シグナルは決めたルールに機械的に当てはまった銘柄の一覧で、値上がりを保証するものではありません。売買はご自身の判断で行ってください。</p>
</main>
<script>
var state = {{ m: "", star: false, fav: false }};
function apply() {{
  document.querySelectorAll("section").forEach(function (sec) {{
    var n = 0;
    sec.querySelectorAll(".card").forEach(function (c) {{
      var show = (!state.m || c.dataset.market === state.m)
        && (!state.star || +c.dataset.score >= 3)
        && (!state.fav || c.dataset.fav === "1");
      c.classList.toggle("hide", !show);
      if (show) n++;
    }});
    sec.querySelector(".cnt").textContent = n + "件";
    sec.querySelector(".empty").hidden = n > 0;
  }});
}}
document.querySelectorAll("#mkt button").forEach(function (b) {{
  b.addEventListener("click", function () {{
    document.querySelectorAll("#mkt button").forEach(function (x) {{ x.classList.remove("on"); }});
    b.classList.add("on"); state.m = b.dataset.m; apply();
  }});
}});
document.getElementById("star").addEventListener("change", function (e) {{ state.star = e.target.checked; apply(); }});
document.getElementById("fav").addEventListener("change", function (e) {{ state.fav = e.target.checked; apply(); }});
</script>
</body>
</html>
"""


def main():
    favs = dict(load_watchlist(os.path.join(HERE, "watchlist.txt")))
    try:
        universe = load_universe()
        log(f"東証の銘柄一覧：{len(universe)}銘柄")
    except Exception as e:
        log(f"東証の銘柄一覧を取得できませんでした（{e}）。watchlist.txt の銘柄だけで判定します。")
        universe = {}
    for code, name in favs.items():
        universe.setdefault(code, {"name": name, "market": "その他", "sector": ""})
    if not universe:
        log("見張る銘柄がありません。")
        sys.exit(1)

    prices = download(list(universe.keys()))
    if not prices:
        log("株価を1件も取得できませんでした。")
        sys.exit(1)

    rows, dates, liquid = [], [], 0
    for code, df in prices.items():
        sigs, info = judge(df)
        if info is None:
            continue
        dates.append(info["date"])
        fav = code in favs
        if info["turnover"] < MIN_TURNOVER and not fav:
            continue
        liquid += 1
        if not sigs:
            continue
        f = fetch_fundamentals(code, info["close"])
        time.sleep(0.2)
        checks = fund_checks(f)
        score = sum(1 for v in checks.values() if v)
        merged = {}
        for kind, reason in sigs:
            merged.setdefault(kind, []).append(reason)
        u = universe[code]
        for kind, reasons in merged.items():
            rows.append({"code": code, "name": u["name"], "market": u["market"], "sector": u["sector"],
                         "fav": fav, "kind": kind, "reason": " ＋ ".join(reasons),
                         "close": info["close"], "change": info["change"], "turnover": info["turnover"],
                         "cost": info["close"] * UNIT,
                         "fund": f, "checks": checks, "score": score})

    # 最新の日付のデータが無い銘柄（売買停止など）は日付の表示に影響させない
    updated = {
        "data_date": max(dates) if dates else "-",
        "run_at": dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
    }
    stats = {"universe": len(universe), "downloaded": len(prices), "liquid": liquid}
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    with open(os.path.join(HERE, "docs", "index.html"), "w", encoding="utf-8") as fp:
        fp.write(build_html(rows, stats, updated))
    with open(os.path.join(HERE, "docs", "signals.json"), "w", encoding="utf-8") as fp:
        json.dump({"updated": updated, "stats": stats, "signals": rows}, fp, ensure_ascii=False, indent=1)
    log(f"完了：{stats}、シグナル{len(rows)}件")


if __name__ == "__main__":
    main()
