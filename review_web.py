"""Review leftover rows: Edit -> edit/edit_web.csv, Pass -> group/leftover/pass.csv.

Run:  python review_web.py   then open http://127.0.0.1:8000
"""
import csv
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "group", "leftover", "leftover_rows.parquet")
EDIT_CSV = os.path.join(BASE, "edit", "edit_web.csv")
PASS_CSV = os.path.join(BASE, "group", "leftover", "pass.csv")

FIELDS = ["no", "moo", "mooban", "building", "floor", "room",
          "soi", "road", "tambol", "amphoe", "province", "leftover"]
COLS = ["addr_input"] + FIELDS

df = pd.read_parquet(SRC).fillna("")
ROWS = df[COLS].to_dict("records")


def done_inputs():
    seen = set()
    for path in (EDIT_CSV, PASS_CSV):
        if os.path.exists(path):
            with open(path, encoding="utf-8-sig", newline="") as f:
                seen.update(r["addr_input"] for r in csv.DictReader(f))
    return seen


def append_row(path, row):
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow({c: row.get(c, "") for c in COLS})


def next_row():
    seen = done_inputs()
    remaining = [r for r in ROWS if r["addr_input"] not in seen]
    return {"row": remaining[0] if remaining else None,
            "remaining": len(remaining), "total": len(ROWS)}


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Leftover Review</title>
<style>
body{background:#111;color:#ddd;font-family:sans-serif;max-width:700px;margin:20px auto;padding:0 16px}
input{width:100%;box-sizing:border-box;background:#222;color:#eee;border:1px solid #444;padding:6px}
label{display:block;margin-top:8px;font-size:13px;color:#999}
.addr{background:#1b1b1b;padding:10px;border:1px solid #333;font-size:18px}
button{margin-top:16px;padding:10px 24px;border:0;color:#fff;cursor:pointer;font-size:15px}
#edit{background:#2a5db0}#pass{background:#2e7d32}
</style></head><body>
<div id="count"></div>
<div id="box"></div>
<script>
const F=__FIELDS__;let cur=null;
async function load(){
  const d=await (await fetch('/next')).json();cur=d.row;
  document.getElementById('count').textContent=`เหลือ ${d.remaining} / ${d.total}`;
  const b=document.getElementById('box');
  if(!cur){b.innerHTML='<p>เสร็จหมดแล้ว</p>';return}
  b.innerHTML='<label>addr_input</label><div class="addr"></div>'+
    F.map(f=>`<label>${f}</label><input id="f_${f}">`).join('')+
    '<button id="edit">Edit</button> <button id="pass">Pass</button>';
  b.querySelector('.addr').textContent=cur.addr_input;
  F.forEach(f=>document.getElementById('f_'+f).value=cur[f]);
  document.getElementById('edit').onclick=()=>save('edit');
  document.getElementById('pass').onclick=()=>save('pass');
}
async function save(action){
  const row={addr_input:cur.addr_input};
  F.forEach(f=>row[f]=document.getElementById('f_'+f).value);
  await fetch('/save',{method:'POST',body:JSON.stringify({action,row})});
  load();
}
load();
</script></body></html>""".replace("__FIELDS__", json.dumps(FIELDS))


class Handler(BaseHTTPRequestHandler):
    def send(self, body, ctype):
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", ctype + "; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/next":
            self.send(json.dumps(next_row(), ensure_ascii=False), "application/json")
        else:
            self.send(PAGE, "text/html")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        append_row(EDIT_CSV if body["action"] == "edit" else PASS_CSV, body["row"])
        self.send("{}", "application/json")


if __name__ == "__main__":
    print("http://127.0.0.1:8000")
    ThreadingHTTPServer(("127.0.0.1", 8000), Handler).serve_forever()
