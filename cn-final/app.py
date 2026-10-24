"""
Dijkstra Shortest Path Simulator
Computer Networks project: shortest path between routers using Dijkstra's algorithm.
Stack: Python + Streamlit + PyVis

Run:  streamlit run app.py
"""

import csv
import heapq
import io
import json
import glob
import math
import os
from copy import deepcopy
from pathlib import Path

import pandas as pd
import streamlit as st
import pyvis
import streamlit.components.v1 as components
from pyvis.network import Network

# ----------------------------------------------------------------------------
# Page + theme
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Dijkstra Shortest Path Simulator", page_icon="🛰️", layout="wide")

BG = "#101826"
SURFACE = "#172235"
LINE = "#26364f"
TEXT = "#e8eef7"
MUTED = "#8fa3bf"
ROUTER = "#4a79d8"
PATH = "#ffb547"
SRC = "#35d0c0"
DST = "#ff6b6b"
VISITED = "#7c6cf0"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'IBM Plex Sans', sans-serif; }}
.stApp {{ background: {BG}; color: {TEXT}; }}
section[data-testid="stSidebar"] {{ background: {SURFACE}; border-right: 1px solid {LINE}; }}
.block-container {{ padding-top: 1.6rem; max-width: 1400px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

.hero {{ padding: 6px 0 18px 0; }}
.hero h1 {{ font-size: 2.3rem; font-weight: 700; margin: 0; color: {TEXT}; }}
.hero p {{ color: {MUTED}; margin: 6px 0 0 0; font-size: 1.02rem; max-width: 760px; }}

.route {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px;
          padding: 16px 20px; margin-bottom: 14px; }}
.route .label {{ color: {MUTED}; font-size: .82rem; margin-bottom: 8px; }}
.route .hops {{ display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
                font-family: 'IBM Plex Mono', monospace; font-size: 1.05rem; }}
.chip {{ padding: 4px 12px; border-radius: 999px; border: 1px solid {LINE};
         background: {BG}; color: {TEXT}; }}
.chip.src {{ border-color: {SRC}; color: {SRC}; }}
.chip.dst {{ border-color: {DST}; color: {DST}; }}
.chip.mid {{ border-color: {PATH}; color: {PATH}; }}
.arrow {{ color: {MUTED}; }}

.stat {{ background: {SURFACE}; border: 1px solid {LINE}; border-radius: 14px; padding: 14px 18px; }}
.stat .v {{ font-size: 1.9rem; font-weight: 700; color: {TEXT}; line-height: 1.1; }}
.stat .k {{ color: {MUTED}; font-size: .82rem; margin-top: 4px; }}

.legend {{ display: flex; flex-wrap: wrap; gap: 16px; color: {MUTED}; font-size: .86rem; margin: 4px 0 10px 0; }}
.dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}

.dtable {{ border-collapse: collapse; width: 100%; font-family: 'IBM Plex Mono', monospace; }}
.dtable th, .dtable td {{ text-align: center; padding: 10px 12px; border-bottom: 1px solid {LINE}; font-size: 1rem; }}
.dtable thead th {{ color: {MUTED}; font-weight: 500; }}
.dtable tbody th {{ color: {TEXT}; }}
.dtable sub {{ color: {MUTED}; font-size: .7em; }}
.dtable .box {{ display: inline-block; border: 2px solid {PATH}; border-radius: 6px; padding: 2px 9px; color: {PATH}; }}
.dtable .box sub {{ color: {PATH}; }}
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: transparent; }}
.hero .bar {{ width: 44px; height: 4px; background: {PATH}; border-radius: 4px; margin: 12px 0 0 0; }}
.stat.hl {{ border-color: {PATH}; }}
.stat.hl .v {{ color: {PATH}; }}
section[data-testid="stSidebar"] h3 {{ font-size: 1rem; font-weight: 600; color: {MUTED}; margin: 18px 0 6px 0; }}
div[data-baseweb="tab-list"] {{ gap: 6px; border-bottom: 1px solid {LINE}; }}
button[data-baseweb="tab"] {{ font-weight: 600; padding: 10px 18px; border-radius: 10px 10px 0 0; color: {MUTED}; }}
button[data-baseweb="tab"][aria-selected="true"] {{ color: {PATH}; background: {SURFACE}; }}
div[data-baseweb="tab-highlight"] {{ background: {PATH}; height: 3px; }}
div[data-baseweb="tab-panel"] {{ padding-top: 1.2rem; }}
h4 {{ margin-top: 1.4rem; }}
div[data-testid="stDataFrame"] {{ border: 1px solid {LINE}; border-radius: 12px; overflow: hidden; }}
.stButton > button {{ border-radius: 10px; border: 1px solid {LINE}; }}
.stButton > button[kind="primary"] {{ background: {PATH}; color: #1a1205; border: none; font-weight: 600; }}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Graph state
# ----------------------------------------------------------------------------
SAMPLE = {
    "nodes": ["R1", "R2", "R3", "R4", "R5", "R6", "R7"],
    "edges": [
        ["R1", "R2", 7],
        ["R1", "R3", 9],
        ["R1", "R6", 14],
        ["R2", "R3", 10],
        ["R2", "R4", 15],
        ["R3", "R4", 11],
        ["R3", "R6", 2],
        ["R4", "R5", 6],
        ["R5", "R6", 9],
        ["R5", "R7", 5],
        ["R6", "R7", 12],
    ],
}

if "g" not in st.session_state:
    st.session_state.g = deepcopy(SAMPLE)

g = st.session_state.g


def fmt(w):
    return str(int(w)) if float(w) == int(w) else f"{w:g}"


def apply_graph(data):
    nodes = [str(n) for n in data["nodes"]]
    edges = {}
    for u, v, w in data["edges"]:
        u, v, w = str(u), str(v), float(w)
        if u not in nodes or v not in nodes or u == v or w <= 0:
            raise ValueError(f"Invalid link {u} - {v} ({w})")
        edges[tuple(sorted((u, v)))] = [u, v, w]
    pos = data.get("pos") or {}
    g["nodes"] = nodes
    g["edges"] = list(edges.values())
    g["pos"] = {n: {"x": float(pos[n]["x"]), "y": float(pos[n]["y"])} for n in nodes if n in pos}


# ----------------------------------------------------------------------------
# Dijkstra (with a full trace so every step can be shown)
# ----------------------------------------------------------------------------
def dijkstra(nodes, edges, src, directed=False):
    adj = {n: [] for n in nodes}
    for u, v, w in edges:
        adj[u].append((v, w))
        if not directed:
            adj[v].append((u, w))

    dist = {n: math.inf for n in nodes}
    prev = {}
    dist[src] = 0
    pq = [(0, src)]
    visited = []
    steps = [{"visit": None, "relaxed": [], "changed": [], "dist": dict(dist), "prev": {}}]

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.append(u)
        relaxed = []
        changed = []
        for v, w in adj[u]:
            if v not in visited and d + w < dist[v]:
                dist[v] = d + w
                prev[v] = u
                heapq.heappush(pq, (dist[v], v))
                relaxed.append(f"{v} → {fmt(dist[v])}")
                changed.append(v)
        steps.append({"visit": u, "relaxed": relaxed, "changed": changed, "dist": dict(dist), "prev": dict(prev)})

    return dist, prev, steps


def path_to(prev, src, dst):
    if src == dst:
        return [src]
    if dst not in prev:
        return []
    out = [dst]
    while out[-1] != src:
        out.append(prev[out[-1]])
    return out[::-1]


def dijkstra_table_html(steps, nodes, src, only_updates=True):
    """Hand-written style Dijkstra table: one row per settled router, cells show cost with the
    previous router as a subscript, and the router settled in that row is boxed."""
    head = "<th>Settled</th>" + "".join(f"<th>{n}</th>" for n in nodes)
    rows = []
    for i, stp in enumerate(steps[1:], 1):
        u = stp["visit"]
        cells = []
        for n in nodes:
            d = stp["dist"][n]
            show = (i == 1) or n == u or n in stp["changed"] or not only_updates
            if not show:
                cells.append("<td></td>")
                continue
            if math.isinf(d):
                txt = "∞"
            else:
                pred = src if n == src else stp["prev"].get(n, "")
                txt = f"{fmt(d)}<sub>{pred}</sub>"
            cls = ' class="box"' if n == u else ""
            cells.append(f"<td><span{cls}>{txt}</span></td>")
        rows.append(f"<tr><th>{u}</th>{''.join(cells)}</tr>")
    return (f'<div style="overflow-x:auto"><table class="dtable"><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>')


# ----------------------------------------------------------------------------
# PyVis rendering
# ----------------------------------------------------------------------------
CANVAS_HTML = r'''<!doctype html>
<html><head><meta charset="utf-8">
<style>
  html, body { margin: 0; background: #101826; color: #e8eef7; font-family: 'IBM Plex Sans', Arial, sans-serif; }
  .vis-manipulation, .vis-edit-mode { display: none !important; }
  #tb { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; padding: 2px 0 10px 0; }
  #tb button { background: #172235; color: #e8eef7; border: 1px solid #26364f; border-radius: 10px;
               padding: 9px 16px; font-size: 14px; font-weight: 600; cursor: pointer; font-family: inherit; }
  #tb button:hover { border-color: #ffb547; }
  #tb button.primary { background: #ffb547; color: #1a1205; border-color: #ffb547; }
  #tb button.on { background: #35d0c0; color: #06201d; border-color: #35d0c0; }
  #hint { color: #8fa3bf; font-size: 13px; margin-left: 6px; }
  #net { height: 540px; border: 1px solid #26364f; border-radius: 14px; background: #101826; }
</style>
__VIS__
</head><body>
<div id="tb">
  <button id="b_add" class="primary">+ Add router</button>
  <button id="b_link">Draw links</button>
  <button id="b_del">Delete selected</button>
  <button id="b_zin" title="Zoom in">Zoom in</button>
  <button id="b_zout" title="Zoom out">Zoom out</button>
  <button id="b_fit">Center graph</button>
  <span id="hint"></span>
</div>
<div id="net"></div>
<script>
(function () {
  var TEXT = '#e8eef7';
  var HINT = 'Tip: double-click empty space to drop a router there. Double-click a link to change its cost.';
  var nodes = null, edges = null, network = null, linkOn = false, directed = false, seq = Date.now(), timer = null;
  var hint = document.getElementById('hint');
  hint.innerText = HINT;

  function post(msg) { msg.isStreamlitMessage = true; window.parent.postMessage(msg, '*'); }
  function snapshot() {
    return {nodes: nodes.getIds(),
            edges: edges.get().map(function (e) { return [e.from, e.to, parseFloat(e.label) || 1]; }),
            pos: network ? network.getPositions() : {}};
  }
  function canon(g, dir) {
    var es = g.edges.map(function (e) {
      var a = String(e[0]), b = String(e[1]);
      if (!dir && a > b) { var t = a; a = b; b = t; }
      return a + '|' + b + '|' + Number(e[2]);
    }).sort();
    return JSON.stringify([g.nodes.map(String).sort(), es]);
  }
  function emit() {
    clearTimeout(timer);
    timer = setTimeout(function () {
      var s = snapshot(); s.seq = ++seq;
      post({type: 'streamlit:setComponentValue', value: s, dataType: 'json'});
    }, 150);
  }
  function nodeObj(id) {
    return {id: id, label: id, shape: 'dot', size: 22,
            color: {background: '#4a79d8', border: '#9db8f2'}, font: {color: TEXT, size: 16}};
  }
  function askCost(def) {
    var c = prompt('Link cost (positive number):', def || '1');
    if (c === null) return null;
    var n = parseFloat(c);
    if (!(n > 0)) { alert('Cost must be a number greater than 0.'); return null; }
    return n;
  }
  function addRouter(x, y) {
    var i = 1;
    while (nodes.get('R' + i)) i++;
    var o = nodeObj('R' + i); o.x = x; o.y = y;
    nodes.add(o);
    setTimeout(function () { network.fit({animation: true}); }, 400);
  }
  function setLink(on) {
    linkOn = on;
    var b = document.getElementById('b_link');
    if (on) {
      network.addEdgeMode();
      b.className = 'on'; b.innerText = 'Drawing links (click to stop)';
      hint.innerText = 'Press on a router and drag to another router.';
    } else {
      network.disableEditMode();
      b.className = ''; b.innerText = 'Draw links';
      hint.innerText = HINT;
    }
  }
  function options() {
    return {
      nodes: {font: {face: 'arial'}},
      edges: {
        color: '#4b5f80', width: 2, smooth: false,
        font: {size: 14, color: TEXT, strokeWidth: 5, strokeColor: '#101826', align: 'middle'},
        arrows: {to: {enabled: directed, scaleFactor: 0.7}}
      },
      physics: {barnesHut: {gravitationalConstant: -9000, springLength: 150, springConstant: 0.04},
                stabilization: {iterations: 200}},
      interaction: {hover: true},
      manipulation: {
        enabled: true,
        addEdge: function (d, cb) {
          if (d.from === d.to) return;
          var dup = edges.get().some(function (x) {
            return (x.from === d.from && x.to === d.to) || (!directed && x.from === d.to && x.to === d.from);
          });
          if (dup) { hint.innerText = 'Those routers are already linked. Double-click the link to change its cost.'; return; }
          d.label = '1';
          cb(d);
          if (linkOn) setTimeout(function () { network.addEdgeMode(); }, 60);
        },
        editEdge: false
      }
    };
  }
  function build(g, pos) {
    pos = pos || {};
    var allPos = g.nodes.length > 0 && g.nodes.every(function (id) { return pos[String(id)]; });
    if (network) network.destroy();
    linkOn = false;
    var b = document.getElementById('b_link'); b.className = ''; b.innerText = 'Draw links';
    nodes = new vis.DataSet(g.nodes.map(function (id) {
      var o = nodeObj(String(id));
      var p = pos[String(id)];
      if (p) { o.x = p.x; o.y = p.y; }
      return o;
    }));
    edges = new vis.DataSet(g.edges.map(function (e) { return {from: String(e[0]), to: String(e[1]), label: String(e[2])}; }));
    var opts = options();
    if (allPos) opts.physics = {enabled: false};
    network = new vis.Network(document.getElementById('net'), {nodes: nodes, edges: edges}, opts);
    nodes.on('*', emit);
    edges.on('*', emit);
    network.on('doubleClick', function (p) {
      if (p.nodes.length) return;
      if (p.edges.length) {
        var id = p.edges[0];
        var c = askCost(edges.get(id).label);
        if (c !== null) edges.update({id: id, label: String(c)});
      } else if (!linkOn) {
        addRouter(p.pointer.canvas.x, p.pointer.canvas.y);
      }
    });
    network.on('dragEnd', emit);
    if (allPos) {
      setTimeout(function () { network.fit(); }, 100);
      setTimeout(function () { network.fit(); }, 600);
    } else {
      network.once('stabilizationIterationsDone', function () {
        network.setOptions({physics: false}); network.fit({animation: true});
        emit();
      });
      setTimeout(function () { network.setOptions({physics: false}); }, 2500);
    }
  }

  document.getElementById('b_add').onclick = function () {
    if (!network) return;
    var c = network.getViewPosition();
    addRouter(c.x + (Math.random() - 0.5) * 120, c.y + (Math.random() - 0.5) * 120);
  };
  document.getElementById('b_link').onclick = function () { if (network) setLink(!linkOn); };
  document.getElementById('b_del').onclick = function () {
    if (!network) return;
    if (!network.getSelectedNodes().length && !network.getSelectedEdges().length) {
      hint.innerText = 'Click a router or link first, then press Delete selected.';
      return;
    }
    network.deleteSelected();
  };
  document.getElementById('b_fit').onclick = function () { if (network) network.fit({animation: true}); };
  function zoom(f) {
    if (!network) return;
    var sc = Math.min(3, Math.max(0.3, network.getScale() * f));
    network.moveTo({position: network.getViewPosition(), scale: sc,
                    animation: {duration: 250, easingFunction: 'easeInOutQuad'}});
  }
  document.getElementById('b_zin').onclick = function () { zoom(1.25); };
  document.getElementById('b_zout').onclick = function () { zoom(0.8); };
  document.addEventListener('keydown', function (ev) {
    if (!network) return;
    if (ev.key === 'Escape' && linkOn) setLink(false);
    if (ev.key === 'Delete') network.deleteSelected();
  });

  window.addEventListener('message', function (ev) {
    if (!ev.data || ev.data.type !== 'streamlit:render') return;
    var a = ev.data.args || {};
    directed = !!a.directed;
    var g = {nodes: a.nodes || [], edges: a.edges || []};
    if (!network || canon(g, directed) !== canon(snapshot(), directed)) build(g, a.pos);
    else network.setOptions({edges: {arrows: {to: {enabled: directed}}}});
    post({type: 'streamlit:setFrameHeight', height: 620});
  });
  post({type: 'streamlit:componentReady', apiVersion: 1});
})();
</script>
</body></html>'''


def _vis_script():
    base = os.path.dirname(pyvis.__file__)
    hits = glob.glob(os.path.join(base, "**", "vis-network.min.js"), recursive=True)
    if hits:
        with open(hits[0], encoding="utf-8") as f:
            return "<script>" + f.read() + "</script>"
    return ('<script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.9/dist/'
            'vis-network.min.js"></script>')


def _canvas_component():
    folder = Path(__file__).parent / "_canvas_component"
    folder.mkdir(exist_ok=True)
    html = CANVAS_HTML.replace("__VIS__", _vis_script())
    target = folder / "index.html"
    if not target.exists() or target.read_text(encoding="utf-8") != html:
        target.write_text(html, encoding="utf-8")
    return components.declare_component("dijkstra_canvas", path=str(folder))


canvas = _canvas_component()


VIEW_TOP = """
<style>
  #vtb { font-family: 'IBM Plex Sans', Arial, sans-serif; display: flex; flex-wrap: wrap; align-items: center;
         gap: 8px; padding: 2px 0 10px 0; }
  #vtb button { background: #172235; color: #e8eef7; border: 1px solid #26364f; border-radius: 10px;
                padding: 8px 14px; font-size: 14px; font-weight: 600; cursor: pointer; font-family: inherit; }
  #vtb button:hover { border-color: #ffb547; }
  #vtb button.on { background: #35d0c0; color: #06201d; border-color: #35d0c0; }
  #vhint { color: #8fa3bf; font-size: 13px; margin-left: 6px; }
</style>
<div id="vtb">
  <button id="v_in" title="Zoom in">Zoom in</button>
  <button id="v_out" title="Zoom out">Zoom out</button>
  <button id="v_fit">Fit</button>
  <button id="v_lock" class="on">Locked</button>
  <span id="vhint">Locked: the diagram stays fixed. Unlock to drag or scroll-zoom.</span>
</div>
"""

VIEW_JS = """
<script>
(function () {
  var locked = true;
  var MIN = 0.3, MAX = 3;
  function applyLock() {
    network.setOptions({interaction: {dragNodes: !locked, dragView: !locked, zoomView: !locked}});
    var b = document.getElementById('v_lock');
    b.className = locked ? 'on' : '';
    b.innerText = locked ? 'Locked' : 'Unlocked';
    document.getElementById('vhint').innerText = locked
      ? 'Locked: the diagram stays fixed. Unlock to drag or scroll-zoom.'
      : 'Unlocked: drag routers or the background, scroll to zoom.';
  }
  function zoom(f) {
    var sc = Math.min(MAX, Math.max(MIN, network.getScale() * f));
    network.moveTo({position: network.getViewPosition(), scale: sc,
                    animation: {duration: 250, easingFunction: 'easeInOutQuad'}});
  }
  document.getElementById('v_in').onclick = function () { zoom(1.25); };
  document.getElementById('v_out').onclick = function () { zoom(0.8); };
  document.getElementById('v_fit').onclick = function () { network.fit({animation: true}); };
  document.getElementById('v_lock').onclick = function () { locked = !locked; applyLock(); };
  applyLock();
})();
</script>
"""


def build_html(directed, path=None, visited=None, current=None, labels=None,
               src=None, dst=None, tree=None, height=540):
    path = path or []
    visited = set(visited or [])
    tree = tree or []
    tree_nodes = {x for e in tree for x in e}
    path_edges = set()
    for a, b in tree:
        path_edges.add((a, b))
        if not directed:
            path_edges.add((b, a))
    for a, b in zip(path, path[1:]):
        path_edges.add((a, b))
        if not directed:
            path_edges.add((b, a))

    pos = g.get("pos", {})
    fixed = bool(g["nodes"]) and all(n in pos for n in g["nodes"])
    net = Network(height=f"{height}px", width="100%", bgcolor=BG, font_color=TEXT,
                  directed=directed, cdn_resources="in_line")

    for n in g["nodes"]:
        bg, border, size = ROUTER, "#9db8f2", 22
        if n in visited:
            bg, border = VISITED, "#b9b0ff"
        if n in path or n in tree_nodes:
            bg, border, size = PATH, "#ffe0a8", 26
        if n == src:
            bg, border, size = SRC, "#b8f5ee", 28
        if n == dst:
            bg, border, size = DST, "#ffc1c1", 28
        if n == current:
            border = "#ffffff"
        label = labels.get(n, n) if labels else n
        xy = {"x": pos[n]["x"], "y": pos[n]["y"]} if n in pos else {}
        net.add_node(n, label=label, title=f"Router {n}", shape="dot", size=size,
                     color={"background": bg, "border": border},
                     borderWidth=4 if n == current else 2,
                     font={"size": 16, "color": TEXT}, **xy)

    for u, v, w in g["edges"]:
        on_path = (u, v) in path_edges
        net.add_edge(u, v, label=fmt(w), title=f"{u} - {v}: cost {fmt(w)}",
                     color=PATH if on_path else "#4b5f80",
                     width=5 if on_path else 2)

    options = {
        "nodes": {"font": {"face": "arial"}},
        "edges": {
            "smooth": False,
            "font": {"size": 14, "color": TEXT, "strokeWidth": 5, "strokeColor": BG, "align": "middle"},
            "arrows": {"to": {"enabled": bool(directed), "scaleFactor": 0.7}},
        },
        "physics": ({"enabled": False} if fixed else {
            "barnesHut": {"gravitationalConstant": -9000, "springLength": 150, "springConstant": 0.04},
            "stabilization": {"iterations": 200},
        }),
        "interaction": {"hover": True, "tooltipDelay": 120},
    }
    net.set_options(json.dumps(options))
    html = net.generate_html()

    css = (
        "<style>body{background:%s;margin:0;} .card{border:none!important;background:transparent!important;}"
        "#mynetwork{border:1px solid %s!important;border-radius:14px;background:%s!important;}"
        "h1{display:none;}</style></head>" % (BG, LINE, BG)
    )
    html = html.replace("</head>", css, 1)
    freeze = (
        "<script>network.once('stabilizationIterationsDone', function () {"
        " network.setOptions({physics: false}); network.fit({animation: true}); });"
        " setTimeout(function () { network.setOptions({physics: false}); network.fit(); }, 400);</script>"
    )
    html = html.replace('<div id="mynetwork"', VIEW_TOP + '<div id="mynetwork"', 1)
    html = html.replace("</body>", freeze + VIEW_JS + "</body>", 1)
    return html


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### Route")
    nodes = g["nodes"]
    src = st.selectbox("Source router", nodes, index=0 if nodes else None, key="src")
    dst_default = len(nodes) - 1 if nodes else None
    dst = st.selectbox("Destination router", nodes, index=dst_default, key="dst")
    directed = st.toggle("Directed links", value=False,
                         help="Off: links work both ways. On: links go only from router A to router B.")

    st.markdown("### Build the network")
    st.caption("Add routers and links by dragging on the Draw on canvas tab.")

    st.markdown("### Topology")
    c1, c2 = st.columns(2)
    if c1.button("Load sample", use_container_width=True):
        st.session_state.g = deepcopy(SAMPLE)
        st.rerun()
    if c2.button("Clear all", use_container_width=True):
        g["nodes"], g["edges"], g["pos"] = [], [], {}
        st.rerun()

# ----------------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
  <h1>Dijkstra Shortest Path Simulator</h1>
  <p>Build a network of routers, set link costs, and watch Dijkstra's algorithm find the cheapest path between any two of them.</p>
  <div class="bar"></div>
</div>
""",
    unsafe_allow_html=True,
)

if "flash" in st.session_state:
    kind, msg = st.session_state.pop("flash")
    st.error(msg) if kind == "error" else st.success(msg)

# ----------------------------------------------------------------------------
# Compute
# ----------------------------------------------------------------------------
ready = len(g["nodes"]) >= 2 and src in g["nodes"] and dst in g["nodes"]
dist, prev, steps, path = {}, {}, [], []
if ready:
    dist, prev, steps = dijkstra(g["nodes"], g["edges"], src, directed)
    path = path_to(prev, src, dst)

legend = f"""
<div class="legend">
  <span><span class="dot" style="background:{SRC}"></span>Source</span>
  <span><span class="dot" style="background:{DST}"></span>Destination</span>
  <span><span class="dot" style="background:{PATH}"></span>Shortest path</span>
  <span><span class="dot" style="background:{VISITED}"></span>Visited (step view)</span>
  <span><span class="dot" style="background:{ROUTER}"></span>Router</span>
</div>
"""

tab_topo, tab_draw, tab_steps, tab_table, tab_links = st.tabs(
    ["Topology", "Draw on canvas", "Algorithm steps", "Routing table", "Links"]
)

# ---- Topology ---------------------------------------------------------------
with tab_topo:
    if not ready:
        st.info("Open the Draw on canvas tab and drag in at least two routers to get started.")
    else:
        if path:
            chips = []
            for i, n in enumerate(path):
                cls = "src" if i == 0 else ("dst" if i == len(path) - 1 else "mid")
                chips.append(f'<span class="chip {cls}">{n}</span>')
            st.markdown(
                '<div class="route"><div class="label">Cheapest route</div><div class="hops">'
                + '<span class="arrow">→</span>'.join(chips)
                + "</div></div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            for col, val, key in [
                (m1, fmt(dist[dst]), "Total cost"),
                (m2, str(len(path) - 1), "Hops"),
                (m3, str(len(g["nodes"])), "Routers"),
                (m4, str(len(g["edges"])), "Links"),
            ]:
                hl = " hl" if key == "Total cost" else ""
                col.markdown(f'<div class="stat{hl}"><div class="v">{val}</div><div class="k">{key}</div></div>',
                             unsafe_allow_html=True)
            st.write("")
        else:
            st.warning(f"No route from {src} to {dst}. They are not connected"
                       + (" in this direction." if directed else "."))

        st.markdown(legend, unsafe_allow_html=True)
        components.html(build_html(directed, path=path, src=src, dst=dst), height=610)

# ---- Draw on canvas ------------------------------------------------------------
with tab_draw:
    st.markdown("Press **+ Add router** (or double-click empty space) to place routers, then **Draw links** "
                "and drag from one router to another. Changes appear in Source and Destination right away.")
    drawn = canvas(nodes=g["nodes"], edges=g["edges"], pos=g.get("pos", {}), directed=directed,
                   key="canvas", default=None)
    if drawn and drawn.get("seq") != st.session_state.get("canvas_seq"):
        st.session_state.canvas_seq = drawn.get("seq")
        try:
            apply_graph(drawn)
            st.rerun()
        except ValueError as ex:
            st.error(f"Could not read the drawn graph: {ex}")

# ---- Algorithm steps -----------------------------------------------------------
with tab_steps:
    if not ready:
        st.info("Pick a source and a destination first.")
    else:
        st.markdown("Move the slider to see which router Dijkstra settles at each step. "
                    "Numbers under the routers are the best known distance from the source.")
        k = st.slider("Step", 0, len(steps) - 1, len(steps) - 1)
        snap = steps[k]
        vis = [s["visit"] for s in steps[1:k + 1]]
        labels = {n: f"{n}\n{('∞' if math.isinf(snap['dist'][n]) else fmt(snap['dist'][n]))}" for n in g["nodes"]}
        if k == 0:
            st.caption(f"Start: distance to {src} is 0, everything else is ∞.")
        else:
            rel = ", ".join(snap["relaxed"]) if snap["relaxed"] else "nothing improved"
            st.caption(f"Step {k}: settle {snap['visit']} (distance {fmt(snap['dist'][snap['visit']])}). Updated: {rel}.")
        components.html(
            build_html(directed, path=path if k == len(steps) - 1 else None, visited=vis,
                       current=snap["visit"], labels=labels, src=src, dst=dst if k == len(steps) - 1 else None),
            height=610,
        )
        st.markdown("#### Dijkstra table")
        only = st.checkbox("Show only updated values (like a hand-written table)", value=True)
        if k == 0:
            st.caption("Move the slider to start filling the table.")
        else:
            st.markdown(dijkstra_table_html(steps[:k + 1], g["nodes"], src, only), unsafe_allow_html=True)
            st.caption("Small letter = router the cost came through. Boxed = router settled in that row.")

# ---- Routing table -------------------------------------------------------------
with tab_table:
    if not ready:
        st.info("Pick a source router first.")
    else:
        st.markdown(f"Routing table for **{src}**, built from Dijkstra's result.")
        table = []
        for n in g["nodes"]:
            p = path_to(prev, src, n)
            table.append({
                "Destination": n,
                "Cost": "unreachable" if not p else fmt(dist[n]),
                "Next hop": "-" if len(p) < 2 else p[1],
                "Hops": "-" if not p else len(p) - 1,
                "Path": " → ".join(p) if p else "-",
            })
        df = pd.DataFrame(table)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("Download CSV", df.to_csv(index=False), file_name=f"routing_table_{src}.csv",
                           mime="text/csv")

        st.markdown("#### Final diagram")
        st.caption(f"Shortest-path tree from {src}: amber links are the routes in the table above, "
                   "and the number under each router is its total cost from the source.")
        tree_edges = [(prev[n], n) for n in g["nodes"] if n in prev]
        final_labels = {n: f"{n}\n{'∞' if math.isinf(dist[n]) else fmt(dist[n])}" for n in g["nodes"]}
        st.markdown(legend, unsafe_allow_html=True)
        components.html(build_html(directed, labels=final_labels, src=src, tree=tree_edges), height=610)

# ---- Links ------------------------------------------------------------------
with tab_links:
    st.markdown("Edit link costs here. Add or remove routers and links on the Draw on canvas tab.")
    edf = pd.DataFrame(g["edges"], columns=["Router A", "Router B", "Cost"])
    edited = st.data_editor(
        edf, num_rows="fixed", use_container_width=True, hide_index=True, key="editor",
        disabled=["Router A", "Router B"],
        column_config={
            "Cost": st.column_config.NumberColumn(min_value=0.1, required=True),
        },
    )
    if st.button("Apply link changes", type="primary"):
        clean = {}
        bad = None
        for _, r in edited.dropna().iterrows():
            a, b, w = r["Router A"], r["Router B"], float(r["Cost"])
            if a == b or w <= 0:
                bad = f"Invalid link {a} - {b}"
                break
            clean[tuple(sorted((a, b)))] = [a, b, w]
        if bad:
            st.error(bad)
        else:
            g["edges"] = list(clean.values())
            st.rerun()

    with st.expander("Adjacency matrix"):
        idx = {n: i for i, n in enumerate(g["nodes"])}
        mat = [["-" for _ in g["nodes"]] for _ in g["nodes"]]
        for u, v, w in g["edges"]:
            mat[idx[u]][idx[v]] = fmt(w)
            if not directed:
                mat[idx[v]][idx[u]] = fmt(w)
        st.dataframe(pd.DataFrame(mat, index=g["nodes"], columns=g["nodes"]), use_container_width=True)

# ----------------------------------------------------------------------------
# Footer
# ----------------------------------------------------------------------------
st.markdown(
    f"""
<div style="margin-top:48px;padding:18px 0 6px 0;border-top:1px solid {LINE};
            text-align:center;color:{MUTED};font-size:.9rem;">
  Developed by <span style="color:{TEXT};font-weight:600;">Hemil Patel</span>
</div>
""",
    unsafe_allow_html=True,
)