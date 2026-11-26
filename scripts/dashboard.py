"""Live training dashboard: http://127.0.0.1:8765 (refreshes every 10 s, read-only).

    .venv/bin/python scripts/dashboard.py
"""
import csv
import html
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


def history(rid):
    p = OUT / rid / "history.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return [{k: float(v) for k, v in r.items()} for r in csv.DictReader(f)]


def chart(rows, keys, colors, title, ymax=None):
    w, h, pad = 460, 200, 34
    if not rows:
        return f"<div class=card><b>{title}</b><p class=muted>waiting for the first epoch…</p></div>"
    vals = [r[k] for r in rows for k in keys]
    top = ymax or max(vals) * 1.1 or 1
    n = max(len(rows) - 1, 1)
    x = lambda i: pad + i * (w - 2 * pad) / n
    y = lambda v: h - pad - v / top * (h - 2 * pad)
    lines = ""
    for k, c in zip(keys, colors):
        pts = " ".join(f"{x(i):.1f},{y(r[k]):.1f}" for i, r in enumerate(rows))
        lines += f'<polyline fill=none stroke="{c}" stroke-width=2.5 points="{pts}"/>'
        lines += "".join(f'<circle cx={x(i):.1f} cy={y(r[k]):.1f} r=3 fill="{c}"><title>epoch {int(r["epoch"])}: {r[k]:.4f}</title></circle>' for i, r in enumerate(rows))
    legend = " ".join(f'<span style="color:{c}">■ {k}</span>' for k, c in zip(keys, colors))
    axis = f'<text x=4 y={pad} class=ax>{top:.2f}</text><text x=4 y={h-pad}>0</text><text x={w-pad} y={h-8} class=ax>epoch {int(rows[-1]["epoch"])}</text>'
    return f'<div class=card><b>{title}</b> <small>{legend}</small><svg viewBox="0 0 {w} {h}">{axis}<line x1={pad} y1={h-pad} x2={w-pad} y2={h-pad} stroke="#ccc"/>{lines}</svg></div>'


def page():
    sp = OUT / "grid_status.json"
    if not sp.exists():
        return "<p>No grid_status.json yet: start training with bash scripts/mac_train_all.sh</p>"
    s = json.loads(sp.read_text())
    runs = s.get("runs", {})
    done = [(k, v) for k, v in runs.items() if v.get("state") == "completed"]
    failed = [(k, v) for k, v in runs.items() if v.get("state") == "failed"]
    running = [k for k, v in runs.items() if v.get("state") == "running"]
    total = s.get("total", 65)
    pct = 100 * len(done) / total
    eta = ""
    if done:
        avg = sum(v["minutes"] for _, v in done) / len(done)
        left = (total - len(done)) * avg
        eta = f"rough ETA ≈ {left/60:.1f} h (average of finished runs: {avg:.1f} min/run; later models are heavier)"
    cur = running[0] if running else None
    rows = history(cur) if cur else []
    last = rows[-1] if rows else {}
    body = f"""<h1>RetinaLens · training live</h1>
<div class=bar><div style="width:{pct:.1f}%"></div></div>
<p><b>{len(done)}/{total}</b> runs completed · {len(failed)} failed · updated {s.get('updated','')} · {eta}</p>"""
    if cur:
        body += f"""<h2>Now training: <code>{html.escape(cur)}</code></h2>
<p>epoch {int(last.get('epoch',0))} · val macro F1 <b>{last.get('val_f1_macro',0):.4f}</b> · val loss {last.get('val_loss',0):.4f} · {last.get('seconds',0):.0f} s/epoch</p>
<div class=row>{chart(rows, ['train_loss','val_loss'], ['#3b6ea5','#d1495b'], 'Loss')}{chart(rows, ['val_f1_macro','val_accuracy'], ['#2a9d8f','#e9a03b'], 'Validation', 1.0)}</div>"""
    trs = "".join(f"<tr><td><code>{html.escape(k)}</code></td><td>{v['val_f1_macro']:.4f}</td><td>{v['epochs']}</td><td>{v['minutes']}</td></tr>" for k, v in done[::-1])
    body += f"<h2>Finished runs</h2><table><tr><th>run</th><th>val macro F1</th><th>epochs</th><th>minutes</th></tr>{trs or '<tr><td colspan=4 class=muted>none yet</td></tr>'}</table>"
    if failed:
        body += "<h2>Failed</h2>" + "".join(f"<p><code>{html.escape(k)}</code>: {html.escape(v.get('error',''))}</p>" for k, v in failed)
    return body


CSS = """body{font:15px -apple-system,system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1d2433}
h1{font-size:22px}h2{font-size:17px;margin-top:26px}.bar{height:12px;background:#eceef2;border-radius:6px;overflow:hidden}
.bar div{height:100%;background:#3b6ea5}.row{display:flex;gap:14px;flex-wrap:wrap}.card{flex:1;min-width:300px;border:1px solid #e3e6eb;border-radius:10px;padding:10px}
svg{width:100%;height:auto}text{font-size:10px;fill:#888}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #eee;padding:5px;text-align:left}
.muted{color:#999}small span{margin-right:8px}@media(prefers-color-scheme:dark){body{background:#15181e;color:#e6e6e6}.card{border-color:#333}.bar{background:#2a2e36}td,th{border-color:#2a2e36}}"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            content = page()
        except Exception as err:  # files are being rewritten: show the error, retry next refresh
            content = f"<p>reading… ({html.escape(str(err))})</p>"
        doc = f"<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=10><title>RetinaLens training</title><style>{CSS}</style>{content}"
        data = doc.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    print("Dashboard: http://127.0.0.1:8765  (Ctrl+C to stop; training is not affected)")
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
