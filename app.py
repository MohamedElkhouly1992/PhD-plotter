import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import curve_fit
import io

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhDPlot — Scientific Visualization",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        color: white; padding: 18px 24px; border-radius: 12px;
        margin-bottom: 20px; border-left: 5px solid #1d4ed8;
    }
    .main-header h1 { margin:0; font-size:26px; font-weight:800; color:white; }
    .main-header p  { margin:4px 0 0; font-size:13px; color:#94a3b8; }
    .metric-box {
        background:#f0f9ff; border:1px solid #bae6fd;
        border-radius:8px; padding:10px 14px; margin-bottom:8px;
        font-size:12px; color:#0369a1;
    }
    .fit-box {
        background:#f0fdf4; border:1px solid #86efac;
        border-radius:8px; padding:10px 14px; margin:4px 0;
        font-size:12px;
    }
    .fit-box b { color:#15803d; }
    div[data-testid="stExpander"] > div { padding:8px 12px; }
    .stTabs [data-baseweb="tab"] { font-weight:600; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = [
    "#1d4ed8","#dc2626","#16a34a","#d97706","#7c3aed",
    "#0891b2","#db2777","#65a30d","#ea580c","#0f766e",
]

CHART_TYPES  = ["Line", "Scatter", "Bar", "Area"]
TREND_TYPES  = ["None", "Linear", "Quadratic", "Exponential", "Power Law"]
LEGEND_LOCS  = ["Top", "Bottom", "Hidden"]

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
def init_plot(n):
    return dict(
        id=n, title=f"Plot {n}",
        dataset=None, x_col=None, y_cols=[],
        series={},
        chart_type="Line", smooth=True, grid=True,
        legend_pos="Top", dual_y=False,
        x_label="X Axis", y_label="Y Axis", y2_label="Y2 Axis",
        x_min="", x_max="", y_min="", y_max="", y2_min="", y2_max="",
    )

if "datasets" not in st.session_state:
    st.session_state.datasets = {}          # {filename: DataFrame}
if "plots" not in st.session_state:
    st.session_state.plots = [init_plot(1)]
    st.session_state.next_id = 2

def default_series(idx):
    return dict(
        color=PALETTE[idx % len(PALETTE)],
        y_axis="Left", error_col="", trend="None",
    )

# ─────────────────────────────────────────────────────────────────────────────
# MATH — REGRESSION
# ─────────────────────────────────────────────────────────────────────────────
def r_squared(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(np.clip(1 - ss_res / ss_tot, 0, 1)) if ss_tot > 1e-14 else 1.0

def _clean(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m]

def fit_trend(x_raw, y_raw, trend_type):
    x, y = _clean(x_raw, y_raw)
    if len(x) < 3:
        return None
    try:
        if trend_type == "Linear":
            c = np.polyfit(x, y, 1)
            fn = np.poly1d(c)
            r2 = r_squared(y, fn(x))
            return dict(fn=fn, r2=r2,
                label=f"y = {c[0]:.4g}·x + {c[1]:.4g}")

        if trend_type == "Quadratic":
            c = np.polyfit(x, y, 2)
            fn = np.poly1d(c)
            r2 = r_squared(y, fn(x))
            return dict(fn=fn, r2=r2,
                label=f"y = {c[0]:.4g}·x² + {c[1]:.4g}·x + {c[2]:.4g}")

        if trend_type == "Exponential":
            pos = y > 0
            if pos.sum() < 3: return None
            def exp_fn(xi, a, b): return a * np.exp(b * xi)
            p0 = [np.mean(y[pos]), 0.01]
            popt, _ = curve_fit(exp_fn, x[pos], y[pos], p0=p0, maxfev=10000)
            fn = lambda xi, p=popt: exp_fn(xi, *p)
            r2 = r_squared(y[pos], fn(x[pos]))
            return dict(fn=fn, r2=r2,
                label=f"y = {popt[0]:.4g}·e^({popt[1]:.4g}·x)")

        if trend_type == "Power Law":
            pos = (x > 0) & (y > 0)
            if pos.sum() < 3: return None
            c = np.polyfit(np.log(x[pos]), np.log(y[pos]), 1)
            a, b = float(np.exp(c[1])), float(c[0])
            fn = lambda xi, a=a, b=b: a * np.abs(xi) ** b
            r2 = r_squared(y[pos], fn(x[pos]))
            return dict(fn=fn, r2=r2,
                label=f"y = {a:.4g}·x^{b:.4g}")

    except Exception:
        return None
    return None

# ─────────────────────────────────────────────────────────────────────────────
# FILE LOADING
# ─────────────────────────────────────────────────────────────────────────────
def load_file(f):
    ext = f.name.rsplit(".", 1)[-1].lower()
    try:
        if ext in ("csv", "txt"):
            df = pd.read_csv(f)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(f)
        else:
            st.sidebar.error(f"Unsupported: .{ext}"); return
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="ignore")
        st.session_state.datasets[f.name] = df
    except Exception as e:
        st.sidebar.error(f"Cannot read {f.name}: {e}")

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY FIGURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
def parse_range(lo_str, hi_str):
    """Return [lo, hi] or None if either is empty."""
    try:
        lo = float(lo_str) if str(lo_str).strip() else None
        hi = float(hi_str) if str(hi_str).strip() else None
        return [lo, hi] if (lo is not None and hi is not None) else None
    except Exception:
        return None

def build_figure(cfg):
    ds_name = cfg["dataset"]
    if not ds_name or ds_name not in st.session_state.datasets:
        return None
    df   = st.session_state.datasets[ds_name]
    xcol = cfg["x_col"]
    ycols = cfg["y_cols"]
    if not xcol or not ycols:
        return None

    dual = cfg["dual_y"] and any(
        cfg["series"].get(c, {}).get("y_axis", "Left") == "Right"
        for c in ycols
    )
    fig = make_subplots(specs=[[{"secondary_y": dual}]]) if dual else go.Figure()

    x_ser = pd.to_numeric(df[xcol], errors="coerce")

    for i, col in enumerate(ycols):
        if col not in df.columns:
            continue
        s       = cfg["series"].get(col, default_series(i))
        color   = s.get("color", PALETTE[i % len(PALETTE)])
        y_axis  = s.get("y_axis", "Left")
        errcol  = s.get("error_col", "")
        trend   = s.get("trend", "None")
        is_sec  = dual and y_axis == "Right"
        ctype   = cfg["chart_type"]
        smooth  = cfg["smooth"]
        shape   = "spline" if smooth else "linear"

        y_ser  = pd.to_numeric(df[col], errors="coerce")
        mask   = x_ser.notna() & y_ser.notna()
        xv     = x_ser[mask].values
        yv     = y_ser[mask].values

        # Error bars
        error_y = None
        if errcol and errcol in df.columns:
            ev = pd.to_numeric(df[errcol], errors="coerce")[mask].values
            error_y = dict(type="data", array=np.abs(ev), visible=True,
                           color=color, thickness=1.5, width=5)

        # ── Trace
        common = dict(name=col, x=xv, y=yv, legendgroup=col)

        if ctype == "Line":
            tr = go.Scatter(**common, mode="lines+markers" if len(xv) < 100 else "lines",
                line=dict(color=color, width=2, shape=shape),
                marker=dict(color=color, size=5),
                error_y=error_y)

        elif ctype == "Scatter":
            tr = go.Scatter(**common, mode="markers",
                marker=dict(color=color, size=7, symbol="circle",
                            line=dict(width=1, color="white")),
                error_y=error_y)

        elif ctype == "Bar":
            tr = go.Bar(**common, marker_color=color,
                marker_line=dict(width=0.5, color="white"),
                error_y=error_y)

        elif ctype == "Area":
            tr = go.Scatter(**common, mode="lines",
                fill="tozeroy", fillcolor=color + "28",
                line=dict(color=color, width=2, shape=shape),
                error_y=error_y)
        else:
            tr = go.Scatter(**common, mode="lines", line=dict(color=color))

        if dual:
            fig.add_trace(tr, secondary_y=is_sec)
        else:
            fig.add_trace(tr)

        # ── Trend line
        if trend != "None" and len(xv) >= 3:
            fit = fit_trend(xv, yv, trend)
            if fit:
                x_fit = np.linspace(xv.min(), xv.max(), 300)
                y_fit = fit["fn"](x_fit)
                tr_trend = go.Scatter(
                    x=x_fit, y=y_fit,
                    mode="lines",
                    name=f"{col} ({trend}, R²={fit['r2']:.4f})",
                    line=dict(color=color, width=1.8, dash="dash"),
                    legendgroup=col, showlegend=True,
                )
                if dual:
                    fig.add_trace(tr_trend, secondary_y=is_sec)
                else:
                    fig.add_trace(tr_trend)

    # ── Axes & layout
    xrange = parse_range(cfg["x_min"], cfg["x_max"])
    yrange = parse_range(cfg["y_min"], cfg["y_max"])

    leg_pos = cfg.get("legend_pos", "Top")
    legend_cfg = dict(visible=True,
        orientation="h",
        yanchor="bottom" if leg_pos == "Top" else "top",
        y=1.02 if leg_pos == "Top" else -0.18,
        xanchor="right", x=1,
        font=dict(size=11),
        bgcolor="rgba(255,255,255,0.85)",
        bordercolor="#e2e8f0", borderwidth=1,
    ) if leg_pos != "Hidden" else dict(visible=False)

    fig.update_layout(
        title=dict(text=cfg["title"], x=0.5,
                   font=dict(size=17, family="sans-serif", color="#0f172a")),
        xaxis=dict(title=dict(text=cfg["x_label"], font=dict(size=13)),
                   showgrid=cfg["grid"], gridcolor="#e2e8f0", gridwidth=1,
                   zeroline=True, zerolinecolor="#cbd5e1",
                   range=xrange),
        yaxis=dict(title=dict(text=cfg["y_label"], font=dict(size=13)),
                   showgrid=cfg["grid"], gridcolor="#e2e8f0", gridwidth=1,
                   zeroline=True, zerolinecolor="#cbd5e1",
                   range=yrange),
        legend=legend_cfg,
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(family="sans-serif", size=12, color="#334155"),
        margin=dict(l=70, r=70, t=70, b=70),
        height=510,
        hovermode="x unified",
    )

    if dual:
        y2range = parse_range(cfg["y2_min"], cfg["y2_max"])
        fig.update_yaxes(
            title_text=cfg["y2_label"],
            secondary_y=True,
            showgrid=False,
            range=y2range,
        )

    return fig

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 PhDPlot")
    st.caption("New Mansoura University")
    st.divider()

    st.markdown("### 📂 Upload Data Files")
    uploaded = st.file_uploader(
        "files", type=["csv","xlsx","xls","txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.datasets:
                load_file(f)

    if st.session_state.datasets:
        st.markdown("### 🗂️ Loaded Datasets")
        for name, df in st.session_state.datasets.items():
            with st.container():
                st.markdown(f"**{name}**")
                st.caption(f"{len(df):,} rows · {len(df.columns)} columns")
        st.divider()

    if st.button("➕ Add New Plot", use_container_width=True, type="primary"):
        n = st.session_state.next_id
        st.session_state.plots.append(init_plot(n))
        st.session_state.next_id += 1
        st.rerun()

    if len(st.session_state.plots) > 1:
        st.markdown("### 🗑️ Remove Plot")
        titles = [p["title"] for p in st.session_state.plots]
        to_del = st.selectbox("Select plot to remove", ["—"] + titles,
                              label_visibility="collapsed")
        if to_del != "—":
            st.session_state.plots = [p for p in st.session_state.plots
                                      if p["title"] != to_del]
            st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>📊 PhDPlot — Scientific Data Visualization</h1>
  <p>Upload CSV / Excel results → Configure axes → Plot, fit curves, and export · New Mansoura University</p>
</div>
""", unsafe_allow_html=True)

if not st.session_state.datasets:
    st.info("👈  Upload a CSV or Excel file from the sidebar to begin.")
    with st.expander("📖 Quick Start Guide"):
        st.markdown("""
1. **Upload** your results file (CSV / Excel) from the sidebar
2. **Select X and Y columns** in the Data Configuration panel
3. **Choose chart type** — Line, Scatter, Bar, or Area
4. **Enable Curve Fitting** — Linear, Quadratic, Exponential, or Power Law with R²
5. **Add Error Bars** by selecting a ±std column per series
6. **Enable Dual Y-Axis** to overlay quantities with different units
7. **Export** as interactive HTML, high-res PNG, or publication-ready SVG
        """)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# PLOT TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_labels = [p["title"] for p in st.session_state.plots]
tabs = st.tabs(tab_labels)

for tab_i, (tab, cfg) in enumerate(zip(tabs, st.session_state.plots)):
    with tab:

        # ── Layout: settings | chart
        col_cfg, col_chart = st.columns([1, 2], gap="large")

        # ═══════════════════════════════════════════════
        # LEFT: CONFIGURATION
        # ═══════════════════════════════════════════════
        with col_cfg:

            # ── Plot title
            cfg["title"] = st.text_input("Plot Title", cfg["title"],
                                          key=f"title_{cfg['id']}")

            # ── Dataset
            ds_names = list(st.session_state.datasets.keys())
            sel_ds = st.selectbox("Dataset", ds_names,
                index=ds_names.index(cfg["dataset"]) if cfg["dataset"] in ds_names else 0,
                key=f"ds_{cfg['id']}")
            if sel_ds != cfg["dataset"]:
                cfg.update(dataset=sel_ds, x_col=None, y_cols=[], series={})
            cfg["dataset"] = sel_ds

            df    = st.session_state.datasets[sel_ds]
            cols  = df.columns.tolist()

            # ── X column
            x_default = cfg["x_col"] if cfg["x_col"] in cols else cols[0]
            cfg["x_col"] = st.selectbox("X Axis Column", cols,
                index=cols.index(x_default),
                key=f"x_{cfg['id']}")

            # ── Y columns
            y_opts = [c for c in cols if c != cfg["x_col"]]
            y_def  = [c for c in cfg["y_cols"] if c in y_opts] or y_opts[:1]
            cfg["y_cols"] = st.multiselect("Y Axis Columns (multi-select)",
                y_opts, default=y_def, key=f"y_{cfg['id']}")

            # ── Series config per Y column
            if cfg["y_cols"]:
                st.markdown("**⚙ Series Settings**")
                for i, col in enumerate(cfg["y_cols"]):
                    if col not in cfg["series"]:
                        cfg["series"][col] = default_series(i)
                    s = cfg["series"][col]
                    with st.expander(f"  {col}", expanded=(i == 0)):
                        ca, cb = st.columns(2)
                        with ca:
                            s["color"] = st.color_picker("Color", s["color"],
                                key=f"clr_{cfg['id']}_{col}")
                        with cb:
                            s["y_axis"] = st.radio("Y Axis Side",
                                ["Left","Right"],
                                index=0 if s["y_axis"]=="Left" else 1,
                                horizontal=True,
                                key=f"ya_{cfg['id']}_{col}")

                        err_opts = ["— None —"] + [c for c in cols if c != col]
                        err_idx  = (err_opts.index(s["error_col"])
                                    if s["error_col"] in err_opts else 0)
                        err_sel  = st.selectbox("Error Bar Column", err_opts,
                            index=err_idx, key=f"err_{cfg['id']}_{col}")
                        s["error_col"] = "" if err_sel == "— None —" else err_sel

                        t_idx = TREND_TYPES.index(s.get("trend","None"))
                        s["trend"] = st.selectbox("Curve Fit / Trend",
                            TREND_TYPES, index=t_idx,
                            key=f"tr_{cfg['id']}_{col}")

            st.divider()

            # ── Chart type & display options
            st.markdown("**📐 Chart Options**")
            c1, c2 = st.columns(2)
            with c1:
                ct_idx = CHART_TYPES.index(cfg["chart_type"]) if cfg["chart_type"] in CHART_TYPES else 0
                cfg["chart_type"] = st.selectbox("Chart Type", CHART_TYPES,
                    index=ct_idx, key=f"ct_{cfg['id']}")
            with c2:
                lp_idx = LEGEND_LOCS.index(cfg.get("legend_pos","Top"))
                cfg["legend_pos"] = st.selectbox("Legend", LEGEND_LOCS,
                    index=lp_idx, key=f"lp_{cfg['id']}")

            c3, c4 = st.columns(2)
            with c3:
                cfg["smooth"]  = st.checkbox("Smooth Curves", cfg["smooth"],
                                              key=f"sm_{cfg['id']}")
                cfg["grid"]    = st.checkbox("Show Grid",    cfg["grid"],
                                              key=f"gr_{cfg['id']}")
            with c4:
                cfg["dual_y"]  = st.checkbox("Dual Y Axis",  cfg["dual_y"],
                                              key=f"dy_{cfg['id']}")

            st.divider()

            # ── Axis labels
            st.markdown("**🏷 Axis Labels**")
            cfg["x_label"]  = st.text_input("X Label",       cfg["x_label"],  key=f"xl_{cfg['id']}")
            cfg["y_label"]  = st.text_input("Left Y Label",  cfg["y_label"],  key=f"yl_{cfg['id']}")
            if cfg["dual_y"]:
                cfg["y2_label"] = st.text_input("Right Y Label", cfg["y2_label"], key=f"y2l_{cfg['id']}")

            # ── Axis ranges
            with st.expander("🔍 Axis Ranges (leave blank for auto)"):
                r1, r2 = st.columns(2)
                with r1:
                    cfg["x_min"]  = st.text_input("X min",  cfg["x_min"],  placeholder="auto", key=f"xmn_{cfg['id']}")
                    cfg["y_min"]  = st.text_input("Y min",  cfg["y_min"],  placeholder="auto", key=f"ymn_{cfg['id']}")
                    if cfg["dual_y"]:
                        cfg["y2_min"] = st.text_input("Y2 min", cfg["y2_min"], placeholder="auto", key=f"y2mn_{cfg['id']}")
                with r2:
                    cfg["x_max"]  = st.text_input("X max",  cfg["x_max"],  placeholder="auto", key=f"xmx_{cfg['id']}")
                    cfg["y_max"]  = st.text_input("Y max",  cfg["y_max"],  placeholder="auto", key=f"ymx_{cfg['id']}")
                    if cfg["dual_y"]:
                        cfg["y2_max"] = st.text_input("Y2 max", cfg["y2_max"], placeholder="auto", key=f"y2mx_{cfg['id']}")

        # ═══════════════════════════════════════════════
        # RIGHT: CHART + EXPORT
        # ═══════════════════════════════════════════════
        with col_chart:
            if not cfg["x_col"] or not cfg["y_cols"]:
                st.info("Select X and Y columns on the left to render the chart.")
            else:
                fig = build_figure(cfg)
                if fig is None:
                    st.error("Could not build chart — check column selections.")
                else:
                    st.plotly_chart(fig, use_container_width=True)

                    # ── Fitted equations summary
                    df_cur = st.session_state.datasets[cfg["dataset"]]
                    x_num  = pd.to_numeric(df_cur[cfg["x_col"]], errors="coerce")
                    fits   = []
                    for col in cfg["y_cols"]:
                        s = cfg["series"].get(col, {})
                        if s.get("trend","None") == "None":
                            continue
                        y_num = pd.to_numeric(df_cur[col], errors="coerce")
                        mask  = x_num.notna() & y_num.notna()
                        fit   = fit_trend(x_num[mask].values,
                                          y_num[mask].values,
                                          s["trend"])
                        if fit:
                            fits.append((col, s["trend"], fit))

                    if fits:
                        st.markdown("#### 📐 Fitted Equations")
                        for col, ttype, fit in fits:
                            color = cfg["series"].get(col, {}).get("color","#1d4ed8")
                            st.markdown(
                                f'<div class="fit-box">'
                                f'<span style="color:{color};font-weight:700">■ {col}</span> '
                                f'({ttype}): &nbsp;<code>{fit["label"]}</code>'
                                f'&nbsp;&nbsp; <b>R² = {fit["r2"]:.6f}</b>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                    # ── Export buttons
                    st.markdown("#### ⬇ Export")
                    e1, e2, e3 = st.columns(3)

                    with e1:
                        try:
                            png = fig.to_image(format="png", width=1400,
                                               height=800, scale=2)
                            st.download_button(
                                "📷 PNG (High-Res)",
                                data=png,
                                file_name=f"{cfg['title'].replace(' ','_')}.png",
                                mime="image/png",
                                key=f"png_{cfg['id']}",
                                use_container_width=True,
                            )
                        except Exception:
                            st.caption("⚠ Install `kaleido` for PNG/SVG export")

                    with e2:
                        try:
                            svg = fig.to_image(format="svg", width=1400, height=800)
                            st.download_button(
                                "🖼 SVG (Vector)",
                                data=svg,
                                file_name=f"{cfg['title'].replace(' ','_')}.svg",
                                mime="image/svg+xml",
                                key=f"svg_{cfg['id']}",
                                use_container_width=True,
                            )
                        except Exception:
                            pass

                    with e3:
                        html_out = fig.to_html(include_plotlyjs="cdn")
                        st.download_button(
                            "🌐 HTML (Interactive)",
                            data=html_out.encode("utf-8"),
                            file_name=f"{cfg['title'].replace(' ','_')}.html",
                            mime="text/html",
                            key=f"html_{cfg['id']}",
                            use_container_width=True,
                        )

                    # ── Data table
                    with st.expander("📋 Data Table"):
                        show_cols = [cfg["x_col"]] + cfg["y_cols"]
                        show_cols = [c for c in show_cols if c in df_cur.columns]
                        st.dataframe(
                            df_cur[show_cols].reset_index(drop=True),
                            use_container_width=True,
                            height=260,
                        )
                        csv_dl = df_cur[show_cols].to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇ Download filtered CSV",
                            data=csv_dl,
                            file_name=f"{cfg['title'].replace(' ','_')}_data.csv",
                            mime="text/csv",
                            key=f"csv_{cfg['id']}",
                        )

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;font-size:12px;color:#94a3b8;'>"
    "PhDPlot · New Mansoura University · Built with Streamlit & Plotly"
    "</div>",
    unsafe_allow_html=True,
)
