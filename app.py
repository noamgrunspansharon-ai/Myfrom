"""
app.py - Streamlit front-end.   Run with:  streamlit run app.py
"""
import hashlib
import os
import tempfile

import pandas as pd
import streamlit as st

from biomechanics import EVENTS, analyze_event
from pose_engine import PoseEngine

st.set_page_config(page_title="Track & Field AI Form Analyzer", page_icon="🏃", layout="wide")

ICON = {"good": "🟢", "warning": "🟡", "poor": "🔴", "info": "🔵", "n/a": "⚪"}
EVENT_HELP = {
    "Long Jump": "Film side-on, covering the last ~5 strides, takeoff, flight and landing.",
    "High Jump": "Film side-on (ideally from the far side of the bar) covering the last 2-3 steps, takeoff and clearance.",
    "Hurdles": "Film side-on, tracking-free (static camera), covering at least one full hurdle clearance.",
}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🏃 Form Analyzer")
    event = st.selectbox("Event", EVENTS, help="Selects which biomechanical heuristics are applied.")
    st.caption(EVENT_HELP[event])
    height_cm = st.number_input("Athlete height (cm, optional)", min_value=0, max_value=250, value=0,
                                help="Used only to convert normalised distances (e.g. hurdle hip rise) to approximate cm.")
    with st.expander("Advanced settings"):
        complexity = st.select_slider("Pose model", options=[0, 1, 2], value=1,
                                      format_func=lambda v: {0: "Lite (fast)", 1: "Full", 2: "Heavy (accurate)"}[v])
        smooth = st.slider("Smoothing window (frames)", 1, 15, 5, step=2)
        fps_override = st.number_input("Real capture FPS (0 = use file FPS)", min_value=0, max_value=1000, value=0,
                                       help="For slow-motion clips: enter the TRUE capture rate (e.g. 240) so timings are correct.")
        max_width = st.select_slider("Processing width (px)", options=[640, 960, 1280, 1920], value=1280)
    with st.expander("Filming checklist"):
        st.markdown("- Static camera on a tripod, perpendicular to the runway\n"
                    "- Whole body in frame for the entire clip\n"
                    "- One athlete, plain background, good light\n"
                    "- 60 fps or higher (120-240 fps is ideal for takeoffs)")

st.title("AI Track & Field Form Analysis")
st.caption("MediaPipe BlazePose (33 keypoints) + biomechanical heuristics for Long Jump, High Jump and Hurdles.")

upload = st.file_uploader("Upload an athlete video", type=["mp4", "mov", "avi", "m4v", "mkv"])
go = st.button("Analyze", type="primary", disabled=upload is None)


def _save_temp(data: bytes, suffix: str) -> str:
    f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.write(data)
    f.close()
    return f.name


if go and upload is not None:
    data = upload.getvalue()
    digest = hashlib.md5(data).hexdigest()
    pose_key = (digest, complexity, int(fps_override), smooth, max_width)
    bar = st.progress(0.0, text="Starting...")

    def cb(frac, msg):
        bar.progress(float(frac), text=msg)

    try:
        cache = st.session_state.get("pose_cache")
        if not cache or cache["key"] != pose_key:
            src = _save_temp(data, os.path.splitext(upload.name)[1] or ".mp4")
            engine = PoseEngine(complexity, max_width=max_width)
            raw = engine.extract_poses(src, fps_override=float(fps_override),
                                       progress_cb=lambda f, m: cb(f * 0.6, m))
            cache = {"key": pose_key, "src": src, "seq": raw.preprocess(smooth_window=smooth)}
            st.session_state["pose_cache"] = cache
        seq, src = cache["seq"], cache["src"]

        analysis = analyze_event(event, seq, float(height_cm) or None)
        out_path = _save_temp(b"", ".mp4")
        engine = PoseEngine(complexity, max_width=max_width)
        engine.render_annotated_video(src, seq, out_path, analysis,
                                      progress_cb=lambda f, m: cb(0.6 + f * 0.4, m))
        with open(out_path, "rb") as fh:
            annotated = fh.read()
        st.session_state["result"] = dict(event=event, analysis=analysis, original=data,
                                          annotated=annotated, annotated_path=out_path, fps=seq.fps)
        bar.empty()
    except Exception as exc:  # surface friendly errors
        bar.empty()
        st.error(f"Analysis failed: {exc}")

res = st.session_state.get("result")
if res:
    analysis = res["analysis"]
    st.subheader(f"Results - {analysis.event}")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Original**")
        st.video(res["original"])
    with c2:
        st.markdown("**Annotated (skeleton, angles, hip trail, key events)**")
        st.video(res["annotated"])
        st.download_button("Download annotated video", res["annotated"], "annotated.mp4", "video/mp4")

    # ---- coaching cues
    st.markdown("### Coaching cues")
    flagged = [m for m in analysis.metrics if m.status in ("warning", "poor")]
    if flagged:
        for m in sorted(flagged, key=lambda x: x.status != "poor"):
            lbl = f"{m.group} - " if m.group else ""
            st.markdown(f"{ICON[m.status]} **{lbl}{m.name}** ({m.value:.0f} {m.unit}, target {m.target}): {m.cue}")
    else:
        st.success("No major technical flags detected in this attempt.")
    good = [m for m in analysis.metrics if m.status == "good"]
    if good:
        with st.expander("What's working well"):
            for m in good:
                lbl = f"{m.group} - " if m.group else ""
                st.markdown(f"{ICON['good']} **{lbl}{m.name}**: {m.value:.0f} {m.unit} - {m.cue}")

    # ---- metric cards + table
    st.markdown("### Measurements")
    groups = list(dict.fromkeys(m.group for m in analysis.metrics))
    for g in groups:
        if g:
            st.markdown(f"**{g}**")
        ms = [m for m in analysis.metrics if m.group == g]
        cols = st.columns(min(len(ms), 4))
        for i, m in enumerate(ms):
            with cols[i % len(cols)]:
                val = "n/a" if m.value is None else f"{m.value:.1f} {m.unit}"
                st.metric(f"{ICON[m.status]} {m.name}", val, help=f"Target: {m.target}")
    st.dataframe(pd.DataFrame([{
        "Group": m.group, "Metric": m.name, "Value": None if m.value is None else round(m.value, 1),
        "Unit": m.unit, "Target": m.target, "Status": f"{ICON[m.status]} {m.status}",
        "Frame": m.frame} for m in analysis.metrics]), use_container_width=True, hide_index=True)

    # ---- key frames
    if analysis.key_frames:
        from pose_engine import PoseEngine as _PE
        st.markdown("### Key frames")
        items = sorted(analysis.key_frames.items(), key=lambda kv: kv[1])
        cols = st.columns(min(len(items), 4))
        for i, (name, f) in enumerate(items):
            img = _PE.read_frame(res["annotated_path"], f)
            if img is not None:
                with cols[i % len(cols)]:
                    st.image(img, caption=f"{name} (t={f / res['fps']:.2f}s)", use_container_width=True)

    # ---- trajectory chart
    if "time_s" in analysis.series:
        st.markdown("### Hip (centre-of-mass) trajectory")
        df = pd.DataFrame(analysis.series).set_index("time_s")
        st.line_chart(df)

    with st.expander("Analysis notes & limitations"):
        for n in analysis.notes:
            st.markdown(f"- {n}")
        st.markdown("- Hip-centre is a proxy for the true centre of mass.\n"
                    "- 2-D analysis assumes a side-on, static camera; angles are projections of 3-D motion.\n"
                    "- Reference ranges are heuristics (see `REF` in `biomechanics.py`) - adapt them to the athlete and discuss with a coach.")
else:
    st.info("Pick an event, upload a clip and press **Analyze**.")
