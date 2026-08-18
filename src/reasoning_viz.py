"""
Reasoning visualization for the HGR-FARM navigator.

A flag-gated, zero-overhead-when-off "insight" recorder for the per-step
navigation reasoning: open-vocab frontier room predictions, per-frontier
exploration scores/uncertainty, the VLM's chosen action + free-text reason,
and nearby FARM captions.

Two independent, additive outputs (both driven by the SAME record structure):

  OFFLINE  (always, when enabled): append each step record as a JSON line to
      ``hgr_farm/viz/<exp>/<subtask_id>.jsonl`` and (re)render a self-contained
      ``hgr_farm/viz/<exp>/index.html`` timeline viewer (pure HTML+inline
      JS/CSS, no external deps) that scrubs steps and shows the reasoning panel.

  ONLINE   (when ``HGR_REASONING_VIZ_SERVE=1`` or a cfg port): start a tiny
      Flask server in a background thread that serves a live page (auto-poll)
      showing the LATEST step's reasoning panel.

Enable via ``cfg["reasoning_viz"] = True`` or env ``HGR_REASONING_VIZ=1``.
When disabled, every method is a cheap no-op (a single boolean check).

This module NEVER raises into the caller: all I/O is wrapped so a viz failure
can never break the benchmark path.
"""

import json
import os
import threading
import time

# Root under the FARM-Frontier umbrella repo, as required by the task.
_VIZ_ROOT = "/data/erwinpi/FARM-Frontier/hgr_farm/viz"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class ReasoningViz:
    """Per-step reasoning recorder + offline HTML + optional live server."""

    def __init__(self, cfg=None, exp_name=None):
        cfg = cfg or {}
        try:
            cfg_enabled = bool(cfg.get("reasoning_viz", False))
        except Exception:
            cfg_enabled = False
        self.enabled = cfg_enabled or _env_truthy("HGR_REASONING_VIZ")

        # Everything below is only touched when enabled -> zero overhead when off.
        if not self.enabled:
            return

        # Resolve experiment name for the output directory.
        if exp_name is None:
            try:
                exp_name = cfg.get("exp_name", None)
            except Exception:
                exp_name = None
        self.exp_name = str(exp_name or os.environ.get("HGR_REASONING_VIZ_EXP", "exp_viz"))

        self.out_dir = os.path.join(_VIZ_ROOT, self.exp_name)
        self._latest = None  # last record, served by the live page
        self._lock = threading.Lock()
        self._html_written = False
        self._serve_started = False

        try:
            os.makedirs(self.out_dir, exist_ok=True)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[REASONING-VIZ] could not create {self.out_dir}: {exc}")
            self.enabled = False
            return

        # Live server (optional).
        want_serve = _env_truthy("HGR_REASONING_VIZ_SERVE")
        try:
            cfg_serve = bool(cfg.get("reasoning_viz_serve", False))
        except Exception:
            cfg_serve = False
        if want_serve or cfg_serve:
            port = os.environ.get("HGR_REASONING_VIZ_PORT")
            if port is None:
                try:
                    port = cfg.get("reasoning_viz_port", None)
                except Exception:
                    port = None
            try:
                self.port = int(port) if port else 8095
            except Exception:
                self.port = 8095
            self._start_server()

        print(
            f"[REASONING-VIZ] ENABLED exp='{self.exp_name}' -> {self.out_dir}"
            + (f" (live http://localhost:{getattr(self, 'port', '?')})" if self._serve_started else "")
        )

    # ------------------------------------------------------------------ API
    def log_step(self, record: dict):
        """Append a structured per-step record (no-op when disabled)."""
        if not self.enabled:
            return
        try:
            subtask_id = str(record.get("subtask_id", "unknown")).replace("/", "_")
            record.setdefault("wall_time", time.time())
            line = json.dumps(record, default=_json_default)
            path = os.path.join(self.out_dir, f"{subtask_id}.jsonl")
            with open(path, "a") as f:
                f.write(line + "\n")
            with self._lock:
                self._latest = record
            # Refresh the offline timeline viewer.
            self._write_index_html()
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[REASONING-VIZ] log_step failed (ignored): {exc}")

    # ------------------------------------------------------------- OFFLINE
    def _write_index_html(self):
        """(Re)write the self-contained timeline viewer index.html."""
        try:
            subtasks = sorted(
                fn[:-6]
                for fn in os.listdir(self.out_dir)
                if fn.endswith(".jsonl")
            )
            html = _INDEX_HTML.replace("__EXP__", _js_str(self.exp_name)).replace(
                "__SUBTASKS__", json.dumps(subtasks)
            )
            with open(os.path.join(self.out_dir, "index.html"), "w") as f:
                f.write(html)
            self._html_written = True
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[REASONING-VIZ] index.html write failed (ignored): {exc}")

    # -------------------------------------------------------------- ONLINE
    def _start_server(self):
        try:
            from flask import Flask, jsonify, Response
        except Exception as exc:
            print(f"[REASONING-VIZ] Flask unavailable, live server disabled: {exc}")
            return

        app = Flask("hgr_reasoning_viz")
        # Silence Flask/werkzeug request logging so it doesn't spam the run log.
        import logging as _logging

        _logging.getLogger("werkzeug").setLevel(_logging.ERROR)

        viz = self

        @app.route("/")
        def _index():  # noqa: ANN202
            return Response(_LIVE_HTML, mimetype="text/html")

        @app.route("/latest.json")
        def _latest():  # noqa: ANN202
            with viz._lock:
                rec = viz._latest
            return jsonify(rec or {})

        def _run():
            try:
                app.run(
                    host="0.0.0.0",
                    port=self.port,
                    threaded=True,
                    use_reloader=False,
                    debug=False,
                )
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[REASONING-VIZ] live server crashed: {exc}")

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        self._serve_started = True


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    return str(o)


def _js_str(s: str) -> str:
    return json.dumps(str(s))


# --------------------------------------------------------------------------
# Shared rendering JS: turns one record into an insight panel. Used by BOTH
# the offline timeline and the live page so their content is identical.
# --------------------------------------------------------------------------
_RENDER_JS = r"""
function esc(s){return String(s==null?'':s).replace(/[&<>]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;'}[c];});}
function renderPanel(r){
  if(!r||!r.step && r.step!==0){return '<div class="empty">No step yet…</div>';}
  var h='';
  h+='<div class="hdr"><span class="tag">'+esc(r.goal_type)+'</span>'
    +'<span class="goal">'+esc(r.goal_text||'')+'</span>'
    +'<span class="step">step '+esc(r.step)+'</span></div>';
  // decision
  var d=r.decision||{};
  h+='<div class="card decision"><div class="ct">Chosen action</div>'
    +'<div class="dchoice">'+esc(d.type||'?')+' '+esc(d.target==null?'':d.target)+'</div>'
    +'<div class="reason">'+esc(d.reason||'(no reason)')+'</div></div>';
  // frontiers with room beliefs
  var fs=r.frontiers||[];
  h+='<div class="card"><div class="ct">Frontiers &middot; room beliefs &amp; scores</div>';
  if(!fs.length){h+='<div class="empty">no frontiers</div>';}
  fs.forEach(function(f){
    var chosen=(d.type==='frontier' && String(d.target)===String(f.id));
    h+='<div class="fr'+(chosen?' chosen':'')+'">';
    h+='<div class="frtop"><span class="fid">F'+esc(f.id)+'</span>';
    if(f.score!=null){h+='<span class="score">score '+Number(f.score).toFixed(3)+'</span>';}
    if(f.uncertainty!=null){h+='<span class="unc">H '+Number(f.uncertainty).toFixed(2)+'</span>';}
    h+='</div>';
    var rp=f.room_prediction||[];
    if(rp.length){
      h+='<div class="rooms">';
      rp.forEach(function(pr){
        var room=pr[0], prob=pr[1];
        var pct=Math.max(2,Math.round((prob||0)*100));
        h+='<div class="room"><span class="rname">'+esc(room)+'</span>'
          +'<span class="bar"><span class="fill" style="width:'+pct+'%"></span></span>'
          +'<span class="rp">'+(prob!=null?(prob*100).toFixed(0)+'%':'')+'</span></div>';
      });
      h+='</div>';
    }
    h+='</div>';
  });
  h+='</div>';
  // nearby FARM captions
  var nc=r.nearby_captions||[];
  h+='<div class="card"><div class="ct">Nearby FARM captions ('+nc.length+')</div>';
  if(!nc.length){h+='<div class="empty">none</div>';}
  else{h+='<div class="caps">'+nc.map(function(c){return '<span class="cap">'+esc(c)+'</span>';}).join('')+'</div>';}
  h+='</div>';
  // agent position + images
  if(r.agent_position){h+='<div class="pos">pos ['+r.agent_position.map(function(x){return Number(x).toFixed(2);}).join(', ')+']</div>';}
  var imgs=r.images||{};
  var ih='';
  ['topdown','egocentric'].forEach(function(k){ if(imgs[k]){ih+='<figure><img src="'+esc(imgs[k])+'"><figcaption>'+k+'</figcaption></figure>';}});
  if(ih){h+='<div class="imgs">'+ih+'</div>';}
  return h;
}
"""

_STYLE = r"""
:root{--bg:#0f1419;--panel:#171d26;--card:#1e2733;--fg:#dfe6ee;--mut:#8b9aad;--acc:#4fb0ff;--good:#39d98a;--chosen:#2b3a2f;}
*{box-sizing:border-box}
body{margin:0;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:var(--bg);color:var(--fg)}
.wrap{max-width:760px;margin:0 auto;padding:16px}
h1{font-size:16px;margin:0 0 4px}
.sub{color:var(--mut);font-size:12px;margin-bottom:12px}
.hdr{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:10px}
.tag{background:var(--acc);color:#04121f;font-weight:700;font-size:11px;padding:2px 8px;border-radius:10px;text-transform:uppercase}
.goal{font-weight:600;flex:1;min-width:120px}
.step{color:var(--mut);font-size:12px}
.card{background:var(--card);border:1px solid #26313f;border-radius:8px;padding:10px 12px;margin-bottom:10px}
.ct{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px}
.decision{border-color:var(--acc)}
.dchoice{font-weight:700;font-size:15px;color:var(--acc)}
.reason{margin-top:4px;color:var(--fg)}
.fr{border-top:1px solid #26313f;padding:6px 0}
.fr:first-of-type{border-top:none}
.fr.chosen{background:var(--chosen);border-radius:6px;padding:6px 8px;margin:2px -4px}
.frtop{display:flex;gap:10px;align-items:center}
.fid{font-weight:700}
.score{color:var(--good)}.unc{color:var(--mut);font-size:12px}
.rooms{margin-top:4px}
.room{display:flex;align-items:center;gap:8px;font-size:12px;margin:2px 0}
.rname{width:130px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--mut)}
.bar{flex:1;height:8px;background:#0c1118;border-radius:4px;overflow:hidden}
.fill{display:block;height:100%;background:var(--acc)}
.rp{width:38px;text-align:right;color:var(--mut)}
.caps{display:flex;flex-wrap:wrap;gap:6px}
.cap{background:#0c1118;border:1px solid #26313f;border-radius:4px;padding:2px 8px;font-size:12px}
.pos{color:var(--mut);font-size:12px;margin:4px 0}
.empty{color:var(--mut);font-style:italic}
.imgs{display:flex;gap:8px;flex-wrap:wrap}
.imgs img{max-width:220px;border-radius:6px;border:1px solid #26313f}
figcaption{color:var(--mut);font-size:11px;text-align:center}
.ctrl{display:flex;align-items:center;gap:10px;margin:10px 0}
select,input[type=range]{accent-color:var(--acc)}
button{background:var(--card);color:var(--fg);border:1px solid #26313f;border-radius:6px;padding:4px 10px;cursor:pointer}
.live{color:var(--good);font-size:12px}
"""

# -------- OFFLINE timeline viewer (scrub steps within a subtask) ----------
_INDEX_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>HGR-FARM reasoning &mdash; __EXP__</title><style>" + _STYLE + "</style></head><body><div class='wrap'>"
    "<h1>HGR-FARM reasoning timeline</h1><div class='sub'>exp <b id='exp'></b> &middot; offline viewer</div>"
    "<div class='ctrl'>subtask <select id='sub'></select>"
    "<button id='prev'>&larr;</button><input type='range' id='rng' min='0' value='0'>"
    "<button id='next'>&rarr;</button><span id='idx' class='sub'></span></div>"
    "<div id='panel'></div>"
    "<script>" + _RENDER_JS + "\n"
    "var EXP=__EXP__; var SUBTASKS=__SUBTASKS__; var recs=[];\n"
    "document.getElementById('exp').textContent=EXP;\n"
    "var sel=document.getElementById('sub'), rng=document.getElementById('rng'), panel=document.getElementById('panel'), idxL=document.getElementById('idx');\n"
    "SUBTASKS.forEach(function(s){var o=document.createElement('option');o.value=s;o.textContent=s;sel.appendChild(o);});\n"
    "function draw(){var i=+rng.value; idxL.textContent=recs.length?('step '+(i+1)+'/'+recs.length):''; panel.innerHTML=recs.length?renderPanel(recs[i]):'<div class=\"empty\">no records</div>';}\n"
    "function loadSub(s){ if(!s){return;} fetch(s+'.jsonl?ts='+Date.now()).then(function(r){return r.text();}).then(function(t){recs=t.trim().split('\\n').filter(Boolean).map(function(l){try{return JSON.parse(l);}catch(e){return null;}}).filter(Boolean); rng.max=Math.max(0,recs.length-1); rng.value=recs.length-1; draw();}).catch(function(){panel.innerHTML='<div class=\"empty\">could not load '+s+'.jsonl</div>';}); }\n"
    "sel.onchange=function(){loadSub(sel.value);}; rng.oninput=draw;\n"
    "document.getElementById('prev').onclick=function(){rng.value=Math.max(0,+rng.value-1);draw();};\n"
    "document.getElementById('next').onclick=function(){rng.value=Math.min(+rng.max,+rng.value+1);draw();};\n"
    "if(SUBTASKS.length){sel.value=SUBTASKS[SUBTASKS.length-1];loadSub(sel.value);}\n"
    "</script></div></body></html>"
)

# -------- ONLINE live page (polls latest.json) ----------------------------
_LIVE_HTML = (
    "<!doctype html><html><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>HGR-FARM reasoning (live)</title><style>" + _STYLE + "</style></head><body><div class='wrap'>"
    "<h1>HGR-FARM reasoning <span class='live'>&#9679; live</span></h1>"
    "<div class='sub'>latest step &middot; auto-refresh 1s</div><div id='panel'></div>"
    "<script>" + _RENDER_JS + "\n"
    "var panel=document.getElementById('panel');\n"
    "function tick(){fetch('latest.json?ts='+Date.now()).then(function(r){return r.json();}).then(function(rec){panel.innerHTML=renderPanel(rec);}).catch(function(){});}\n"
    "tick(); setInterval(tick,1000);\n"
    "</script></div></body></html>"
)
