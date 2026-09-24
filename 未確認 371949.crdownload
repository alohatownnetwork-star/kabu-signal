# -*- coding: utf-8 -*-
"""
毎朝の売買シグナル一覧を作るプログラム。
watchlist.txt の銘柄の株価（前日までの日足）を取得してシグナルを判定し、
シグナルが出た銘柄は業績・割安度（ファンダメンタル）も調べて、
docs/index.html（スマホで見るページ）に書き出します。
"""
import datetime as dt
import html
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

# ===== テクニカルの設定（数字を変えるとルールが変わります） =====
SHORT_MA = 5           # 短期の移動平均（日）
LONG_MA = 25           # 長期の移動平均（日）
RSI_DAYS = 14          # RSIの期間（日）
RSI_LOW = 30           # これ以下で「売られすぎ」
RSI_HIGH = 75          # これ以上で「買われすぎ」
HIGH_DAYS = 245        # 高値更新を見る期間（約1年の営業日）
VOLUME_RATIO = 1.5     # 出来高が20日平均の何倍以上なら「出来高を伴う」とみなすか
UNIT = 100             # 売買単位（株）

# ===== ファンダメンタルの合格ライン（満たすと ○ が付き、点数が上がる） =====
PER_MAX = 15           # PER（株価÷1株利益）がこれ以下なら割安
PBR_MAX = 1.5          # PBR（株価÷1株純資産）がこれ以下なら割安
ROE_MIN = 8            # ROE（%）がこれ以上なら稼ぐ力がある
DIV_MIN = 3            # 配当利回り（%）がこれ以上なら高配当
GROWTH_MIN = 0         # 売上の伸び（前年比%）がこれより上なら増収
# =====================================================

JST = dt.timezone(dt.timedelta(hours=9))
HERE = os.path.dirname(os.path.abspath(__file__))


def load_watchlist(path):
    items = []
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
    close, high, vol = df["Close"], df["High"], df["Volume"]
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
    }
    return out, info


def download(codes):
    import yfinance as yf
    tickers = [f"{c}.T" for c in codes]
    data = yf.download(tickers, period="15mo", interval="1d", group_by="ticker",
                       auto_adjust=False, threads=True, progress=False)
    result = {}
    for c, t in zip(codes, tickers):
        try:
            d = data[t] if isinstance(data.columns, pd.MultiIndex) else data
            d = d[["Close", "High", "Volume"]].dropna(subset=["Close"])
            if len(d):
                result[c] = d
        except KeyError:
            pass
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
    for _ in range(2):
        try:
            info = yf.Ticker(f"{code}.T").info or {}
            break
        except Exception:
            time.sleep(2)
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
    "buy": ("🟢 買い", "buy"),
    "sell": ("🔴 売り", "sell"),
    "watch": ("🟡 注目", "watch"),
    "caution": ("🟠 注意", "caution"),
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


def build_html(rows, n_total, n_ok, updated):
    groups = {k: [r for r in rows if r["kind"] == k] for k in ORDER}
    sections = []
    for k in ORDER:
        title, cls = LABEL[k]
        items = groups[k]
        if not items:
            body = '<p class="empty">なし</p>'
        else:
            cards = []
            # 買い・注目はファンダの点数が高い順、売り・注意は点数が低い順
            rev = k in ("buy", "watch")
            for r in sorted(items, key=lambda x: (x["score"] if rev else -x["score"]), reverse=True):
                chg = r["change"]
                star = r["score"] >= 3
                cards.append(f"""
      <a class="card{' star' if star else ''}" href="https://kabutan.jp/stock/?code={html.escape(r['code'])}" target="_blank" rel="noopener">
        <div class="top"><span class="name">{html.escape(r['name'])}</span><span class="code">{html.escape(r['code'])}</span></div>
        <div class="reason">{html.escape(r['reason'])}</div>
        <div class="nums">
          <span>終値 <b>{r['close']:,.0f}円</b> <span class="{'up' if chg >= 0 else 'down'}">{chg:+.1f}%</span></span>
          <span>{UNIT}株 {r['cost']:,.0f}円</span>
        </div>
        <div class="score">ファンダ {'★' * r['score']}{'☆' * (5 - r['score'])} <small>{r['score']}/5</small></div>
        {fund_html(r['fund'], r['checks'])}
      </a>""")
            body = "".join(cards)
        sections.append(f'<section class="{cls}"><h2>{title} <small>{len(items)}件</small></h2>{body}</section>')

    return f"""<!doctype html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>朝の売買シグナル</title>
<style>
:root {{ --bg:#f6f7f9; --card:#fff; --text:#1c1f24; --sub:#6b7280; --line:#e5e7eb; --chip:#f1f3f5;
        --buy:#16a34a; --sell:#dc2626; --watch:#ca8a04; --caution:#ea580c; --up:#dc2626; --down:#2563eb;
        --good:#15803d; --goodbg:#dcfce7; --star:#b45309; }}
@media (prefers-color-scheme: dark) {{
  :root {{ --bg:#111418; --card:#1b1f25; --text:#e8eaed; --sub:#9aa0a6; --line:#2c323a; --chip:#242a31;
          --up:#f87171; --down:#60a5fa; --good:#4ade80; --goodbg:#14301f; --star:#fbbf24; }}
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text);
       font-family:-apple-system,BlinkMacSystemFont,"Hiragino Sans","Yu Gothic UI",sans-serif; }}
main {{ max-width:640px; margin:0 auto; padding:16px; }}
h1 {{ font-size:20px; margin:4px 0; }}
.meta {{ color:var(--sub); font-size:13px; margin-bottom:12px; }}
.filter {{ display:flex; gap:8px; align-items:center; font-size:14px; margin:12px 0 4px;
          background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 12px; }}
h2 {{ font-size:16px; margin:20px 0 8px; }}
h2 small {{ color:var(--sub); font-weight:normal; }}
.card {{ display:block; text-decoration:none; color:inherit; background:var(--card);
        border:1px solid var(--line); border-left:4px solid var(--line); border-radius:10px;
        padding:10px 12px; margin-bottom:8px; }}
.buy .card {{ border-left-color:var(--buy); }} .sell .card {{ border-left-color:var(--sell); }}
.watch .card {{ border-left-color:var(--watch); }} .caution .card {{ border-left-color:var(--caution); }}
.top {{ display:flex; justify-content:space-between; gap:8px; }}
.name {{ font-weight:600; }} .code {{ color:var(--sub); font-size:13px; }}
.reason {{ font-size:14px; margin:4px 0; }}
.nums {{ display:flex; justify-content:space-between; flex-wrap:wrap; gap:4px 12px; font-size:13px; color:var(--sub); }}
.nums b {{ color:var(--text); }} .up {{ color:var(--up); }} .down {{ color:var(--down); }}
.score {{ font-size:13px; color:var(--star); margin-top:6px; }}
.score small {{ color:var(--sub); }}
.fund {{ display:flex; flex-wrap:wrap; gap:4px; margin-top:6px; }}
.f {{ font-size:12px; color:var(--sub); background:var(--chip); border-radius:6px; padding:2px 6px; }}
.f b {{ color:var(--text); font-weight:600; }}
.f.good {{ background:var(--goodbg); color:var(--good); }} .f.good b {{ color:var(--good); }}
.empty {{ color:var(--sub); font-size:14px; }}
body.only-star .card:not(.star) {{ display:none; }}
.note {{ color:var(--sub); font-size:12px; line-height:1.6; margin-top:24px; }}
</style>
</head>
<body>
<main>
  <h1>朝の売買シグナル</h1>
  <div class="meta">株価の日付：{html.escape(updated['data_date'])} ／ 更新：{html.escape(updated['run_at'])}<br>
  見張り {n_total}銘柄（取得できた {n_ok}銘柄）</div>
  <label class="filter"><input type="checkbox" id="star"> ファンダ★3つ以上の銘柄だけ表示</label>
  {''.join(sections)}
  <p class="note">銘柄をタップすると株探の銘柄ページが開きます。<br>
  <b>テクニカル</b>：{SHORT_MA}日・{LONG_MA}日移動平均のクロス／出来高{VOLUME_RATIO}倍以上での1年来高値更新／RSI（{RSI_DAYS}日）{RSI_LOW}以下・{RSI_HIGH}以上<br>
  <b>ファンダ★</b>：PER {PER_MAX}倍以下／PBR {PBR_MAX}倍以下／ROE {ROE_MIN}%以上／配当 {DIV_MIN}%以上／増収　を満たすごとに★1つ（○印）。「-」はデータが取れなかった項目です。<br>
  業績データはYahoo Financeのもので、最新の決算とずれることがあります。シグナルは決めたルールに機械的に当てはまった銘柄の一覧で、値上がりを保証するものではありません。売買はご自身の判断で行ってください。</p>
</main>
<script>
var cb = document.getElementById('star');
cb.addEventListener('change', function () {{ document.body.classList.toggle('only-star', cb.checked); }});
</script>
</body>
</html>
"""


def main():
    watch = load_watchlist(os.path.join(HERE, "watchlist.txt"))
    names = dict(watch)
    prices = download([c for c, _ in watch])
    if not prices:
        print("株価を1件も取得できませんでした。", file=sys.stderr)
        sys.exit(1)

    rows, dates, fund_cache = [], [], {}
    for code, df in prices.items():
        sigs, info = judge(df)
        if info is None:
            continue
        dates.append(info["date"])
        if not sigs:
            continue
        # シグナルが出た銘柄だけ、ファンダを調べる
        if code not in fund_cache:
            fund_cache[code] = fetch_fundamentals(code, info["close"])
        f = fund_cache[code]
        checks = fund_checks(f)
        score = sum(1 for v in checks.values() if v)
        merged = {}
        for kind, reason in sigs:
            merged.setdefault(kind, []).append(reason)
        for kind, reasons in merged.items():
            rows.append({"code": code, "name": names.get(code, code), "kind": kind,
                         "reason": " ＋ ".join(reasons),
                         "close": info["close"], "change": info["change"],
                         "cost": info["close"] * UNIT,
                         "fund": f, "checks": checks, "score": score})

    updated = {
        "data_date": max(dates) if dates else "-",
        "run_at": dt.datetime.now(JST).strftime("%Y-%m-%d %H:%M"),
    }
    os.makedirs(os.path.join(HERE, "docs"), exist_ok=True)
    with open(os.path.join(HERE, "docs", "index.html"), "w", encoding="utf-8") as fp:
        fp.write(build_html(rows, len(watch), len(prices), updated))
    with open(os.path.join(HERE, "docs", "signals.json"), "w", encoding="utf-8") as fp:
        json.dump({"updated": updated, "signals": rows}, fp, ensure_ascii=False, indent=1)
    print(f"完了：{len(prices)}/{len(watch)}銘柄を判定、シグナル{len(rows)}件")


if __name__ == "__main__":
    main()
