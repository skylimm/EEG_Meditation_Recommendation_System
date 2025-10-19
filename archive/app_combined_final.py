
import sys
import streamlit as st
import os
import time
import subprocess
from muselsl import record
from pylsl import resolve_stream
from user_input import get_user_folder

import numpy as np
import json

from preprocessings import summarize_one_file, BANDS

# =========================
# Header & Global Styling
# =========================

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("./assets/logo.png", width=2000)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Dancing+Script:wght@600&family=Quicksand:wght@400;500&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Karla:wght@500;700&display=swap');

.header-wrap { text-align:center; }
.header-title { font-family:'Karla', sans-serif; font-size:32px; font-weight:700; color:#1F2937; letter-spacing:.3px; margin:2px 0; }
.header-sub   { font-family:'Karla', sans-serif; font-size:17px; color:#475569; line-height:1.3; }

.hw-title { font-family:'Karla', sans-serif; font-weight:700; color:#0F172A; font-size:22px; margin:14px 0 8px; text-align:center; }

.hw-card {
  border: 1px solid #E2E8F0;
  background: #FFFFFF;
  border-radius: 14px;
  padding: 14px 14px 16px;
  height: 100%;
  box-shadow: 0 1px 6px rgba(2,6,23,0.06);
}
.hw-icon { font-size:26px; line-height:1; text-align:center;}
.hw-step { font-family:'Karla', sans-serif; font-weight:700; color:#0F172A; font-size:16px; margin:6px 0 2px; text-align:center; }
.hw-text { font-family:'Karla', sans-serif; color:#475569; font-size:14.5px; line-height:1.55; text-align:center; }

.tip {
  background:#F0FDF4; border:1px solid #DCFCE7; color:#166534;
  padding:10px 14px; border-radius:12px; margin-top:12px; text-align:center;
  font-family:'Karla', sans-serif; font-size:14.5px;
}

/* subtle fade-in */
@keyframes fadeUp { from {opacity:0; transform:translateY(6px);} to {opacity:1; transform:translateY(0);} }
.header-wrap, .hw-title, .hw-card, .tip { animation: fadeUp .5s ease forwards; }
</style>

<div class="header-wrap">
  <div class="header-sub">
    <b> Real-Time EEG Meditation Recommender</b> — analyze your brainwaves and discover which meditation technique your brain needs <b>right now.</b>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="hw-title">How it works</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">🎧</div>
      <div class="hw-step">1. Pre Collection</div>
      <div class="hw-text">Choose <b>Live</b> (record from Muse) or <b>Upload</b> a CSV to establish your baseline state.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">📊</div>
      <div class="hw-step">2. Analyze & Meditate</div>
      <div class="hw-text">We analyze Alpha/Beta/Theta, recommend a meditation, and guide a 15-minute session.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">✨</div>
      <div class="hw-step">3. Post Collection & Results</div>
      <div class="hw-text">Collect <b>Post</b> EEG (Live or Upload), then view your before/after changes & interpretation.</div>
    </div>
    """, unsafe_allow_html=True)

# =========================
# Session State Defaults
# =========================
ss = st.session_state

defaults = dict(
    stream_found=False,
    recording_in_progress=False,
    recording_finished=False,
    search_completed=False,
    meditation_selected=False,
    post_meditation_recording_completed=False,
    post_stream_found=False,
    post_recording_in_progress=False,
    post_recording_finished=False,
    after_eeg_file=None,
    before_eeg_file=None,
    # Navigation & locks
    progress_step=1,   # 1=Pre, 2=Analysis, 3=Post, 4=Results
    lock_upload=False,
    # Meditation timer
    meditation_running=False,
    meditation_paused=False,
    meditation_completed=False,
    meditation_start_ts=None,
    meditation_elapsed=0,
    meditation_total=1 * 60,  # NOTE: set to 15*60 for full; kept 1*60 for quick testing
    audio_played=False
)
for k, v in defaults.items():
    ss.setdefault(k, v)

# =========================
# Utility helpers
# =========================
def show_fixation_overlay():
    """Display overlay with centered pulsing dot during recording."""
    placeholder = st.empty()
    placeholder.markdown("""
        <style>
            .overlay {
                position:fixed; top:0; left:0; width:100vw; height:100vh;
                background:white; display:flex; justify-content:center; align-items:center;
                z-index:9999; flex-direction:column;
            }
            .dot {
                width:30px; height:30px; background:black; border-radius:50%;
                animation:pulse 15s infinite;
            }
            @keyframes pulse {
                0%{transform:scale(1);opacity:1;}
                50%{transform:scale(1.3);opacity:0.6;}
                100%{transform:scale(1);opacity:1;}
            }
        </style>
        <div class="overlay">
            <div class="dot"></div>
            <p><b>Recording EEG... Please look at the dot..</b></p>
        </div>
        """, unsafe_allow_html=True)
    return placeholder

@st.cache_data
def load_cohort_stats(path="outputs/cohort_baseline.json"):
    with open(path, "r") as f:
        return json.load(f)

def classify_vs_cohort(val, mean, sd, z_thresh=1.0):
    if sd is None or sd == 0:
        return "normal", 0.0
    z = (val - mean) / sd
    if z > z_thresh:
        return "high", z
    elif z < -z_thresh:
        return "low", z
    else:
        return "normal", z

# =========================
# User folder & file names
# =========================
st.markdown("")
st.markdown("")
folder_path = get_user_folder()
pre_filename = "before_eeg.csv"
post_filename = "after_eeg_recording.csv"  # keep naming from live flow
record_duration = 30  # seconds (set to 300 for 5 minutes)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Pre Collection",
    "2️⃣ Brainwave Analysis & Meditation",
    "3️⃣ Post Collection",
    "4️⃣ Results & Interpretation"
])

# ----------------------------------
# Tab 1: PRE COLLECTION (Live | Upload)
# ----------------------------------
with tab1:
    # Lock rule: Only accessible content if progress_step >=1; if >=2, inputs are locked (no re-recording)
    st.header("1️⃣ Pre-Meditation EEG Collection")

    if ss.progress_step < 1:
        st.warning("🔒 Locked.")
        st.stop()

    if not folder_path:
        st.info("Please enter your name above to create a personal data folder.")
    else:
        pre_tab_live, pre_tab_upload = st.tabs(["🎧 Live Recording", "📂 Upload File"])

        # ---------- Live Recording ----------
        with pre_tab_live:
            if ss.progress_step >= 2:
                st.warning("🔒 Pre Collection is locked after moving to Brainwave Analysis.")
                st.stop()

            st.caption("Please ensure your Bluetooth and Muse 2 Headband is on before searching for Muse Device")

            search_ph = st.empty()

            def render_search_button():
                with search_ph.container():
                    if st.button("🔍 Search for Muse Device & Start Stream", key="btn_search"):
                        ss.lock_upload = True  # disable Upload tab during live search/record
                        with st.spinner("Starting Muse LSL stream..."):
                            process = subprocess.Popen([sys.executable, "start_stream.py"])
                            time.sleep(8)
                            found = any(resolve_stream('type', 'EEG') for _ in range(20) if not time.sleep(1))
                            if found:
                                ss['stream_process'] = process
                                ss.stream_found = True
                                render_connected_disabled()
                                st.success("Muse device found. You can now record EEG.")
                            else:
                                process.terminate()
                                st.error("Please ensure Muse Headband & Bluetooth is on and rerun the web.")

            def render_connected_disabled():
                with search_ph.container():
                    st.button("✅ Muse Device Connected", disabled=True, key="btn_connected")

            if ss.get("stream_found", False):
                render_connected_disabled()
            else:
                render_search_button()

            with st.expander("Review instructions below if this is your first time using the Muse 2 headband", expanded=False):
                col1_, col2_ = st.columns([1, 2])
                with col1_:
                    st.image(
                        "https://pyxis.nymag.com/v1/imgs/e09/a32/9356a8db59e0b3880c8b8d08bb7c165461-09-muse.2x.rhorizontal.w700.jpg",
                        caption="Correct EEG headband placement",
                        width=200,
                    )
                with col2_:
                    st.markdown("""
                    ### 🧠 Wearing the Muse 2 Headband
                    - Hair away from forehead/ears; band snug.
                    - Quiet room, upright posture, relaxed jaw.
                    - Keep still; eyes open; gaze at a small point.
                    """)
                st.divider()
                st.markdown("""
                ### 🎧 During EEG Recording
                - Sit quietly with eyes open; fix gaze on the dot.
                - Natural breathing; avoid excessive blinking or movement.
                - **5-Minute EEG Recording starts when dot appears.**
                """)

            record_ph = st.empty()

            def render_start_record():
                with record_ph.container():
                    if st.button("🎙 Start Pre-Meditation EEG Recording", key="btn_record"):
                        ss.recording_in_progress = True
                        render_recording_disabled()

                        with st.spinner(f"5 minutes EEG recording will start upon seeing dot.."):
                            time.sleep(5)

                        overlay = show_fixation_overlay()

                        with st.spinner(f"Recording EEG for {record_duration} seconds..."):
                            try:
                                # full_path = os.path.join(folder_path, pre_filename)
                                full_path = os.path.abspath(os.path.join(folder_path, pre_filename))

                                record(record_duration, filename=full_path)
                                if os.path.exists(full_path):
                                    st.toast(f"Recording complete! File saved: `{full_path}`")
                                    ss['before_eeg_file'] = full_path
                                    ss.recording_finished = True
                                    render_recording_ended()
                                else:
                                    st.error("Recording failed. Please refresh the page and try again.")
                            except Exception as e:
                                st.error(f"Recording error: {e}")
                            finally:
                                overlay.empty()
                                if 'stream_process' in ss and ss['stream_process'].poll() is None:
                                    ss['stream_process'].terminate()
                                    st.toast("Muse stream terminated.")
                                ss.pop('stream_process', None)
                                ss.stream_found = False
                                ss.recording_in_progress = False

            def render_recording_disabled():
                with record_ph.container():
                    st.button("🎙 Recording in Progress…", disabled=True, key="btn_recording")

            def render_recording_ended():
                with record_ph.container():
                    st.button("✅ Recording Completed", disabled=True, key="btn_recordingended")

            if ss.get("stream_found") and 'stream_process' in ss:
                if ss.get("recording_finished"):
                    render_connected_disabled()
                    render_recording_ended()
                    st.toast("🎧 Pre-meditation EEG recording completed. You can proceed to the next step.")
                elif ss.get("recording_in_progress"):
                    render_recording_disabled()
                else:
                    render_start_record()
            else:
                if not ss.get("recording_finished"):
                    pass

            # Proceed button after pre is ready (Live path)
            if ss.get('before_eeg_file') and ss.get('recording_finished'):
                st.success("✅ Pre EEG ready.")
                if st.button("➡️ Proceed to Brainwave Analysis", key="proceed_live"):
                    # Verify the file actually exists
                    if os.path.exists(ss['before_eeg_file']):
                        ss.progress_step = 2
                        st.rerun()
                    else:
                        st.error(f"File not found: {ss['before_eeg_file']}. Please record again.")

        # ---------- Upload File ----------
        with pre_tab_upload:
            # When Live mode started, disable Upload tab to prevent mixing
            if ss.lock_upload or ss.progress_step >= 2:
                st.warning("🔒 Upload is disabled (Live session active or you've moved on to the next step).")
                st.stop()

            uploaded_pre = st.file_uploader(
                "📂 Please upload your pre-meditation EEG file recorded earlier using the local recording script.",
                type=["csv"],
                key="before_upload"
            )
            if uploaded_pre:
                pre_path = os.path.join(folder_path, pre_filename)
                with open(pre_path, "wb") as f:
                    f.write(uploaded_pre.getbuffer())
                st.success("✅ Pre-Meditation EEG file uploaded successfully.")
                ss['before_eeg_file'] = pre_path
                ss.recording_finished = True
                if st.button("➡️ Proceed to Brainwave Analysis (Upload)"):
                    ss.progress_step = 2
                    st.rerun()

# ----------------------------------
# Tab 2: ANALYSIS & MEDITATION (Shared)
# ----------------------------------
with tab2:
    st.header("2️⃣ Brainwave Analysis & Meditation Recommendation")
    # Debug information
    with st.expander("🔧 Debug Info (Remove after fixing)"):
        st.write("Session State:", {
            'progress_step': ss.progress_step,
            'before_eeg_file': ss.get('before_eeg_file'),
            'recording_finished': ss.get('recording_finished'),
            'file_exists': os.path.exists(ss.get('before_eeg_file', '')) if ss.get('before_eeg_file') else False
        })


    if ss.progress_step < 2:
        st.warning("🔒 This section is locked. Please complete Pre Collection first.")
        st.stop()

    # Prevent going back to record again in Tab 1 by showing lock message there (handled).
    # Continue with analysis if pre file exists:
    if ss.get('before_eeg_file') and ss.get('recording_finished'):
        # Analyze pre file
        result = summarize_one_file(ss['before_eeg_file'])
        rel = result.rel_overall_pct

        cohort = load_cohort_stats("../outputs/cohort_baseline.json")

        focus_bands = ["alpha", "beta", "theta"]
        rows = []
        for band in focus_bands:
            band_stats = cohort.get(band, {})
            mean = band_stats.get("mean", np.nan)
            sd = band_stats.get("sd", np.nan)
            val = rel.get(band, np.nan)
            state, z = classify_vs_cohort(val, mean, sd, z_thresh=1.0)
            rows.append({
                "Band": band.capitalize(),
                "Your %": None if np.isnan(val) else round(val, 2),
                "Cohort mean %": None if np.isnan(mean) else round(mean, 2),
                "Cohort SD": None if np.isnan(sd) else round(sd, 2),
                "Z-score": None if np.isnan(val) or np.isnan(sd) else round(z, 2),
                "State": state
            })
        ss['band_states'] = {r["Band"].lower(): r["State"] for r in rows}

        # Quick Brainwave Summary
        if 'band_states' in ss:
            st.divider()
            st.markdown("<p style='text-align:center; font-size: 27px; font-weight: bold'>🧩 Quick Brainwave Summary 🧩</p>", unsafe_allow_html=True)
            st.write(
                "<p style='font-size: 12px;  color: dimgray; text-align: justify;'>⚠️ Do note that the EEG readings and interpretations are indicative and may vary between individuals. Differences can arise due to factors such as signal quality, electrode contact, individual brainwave variability, and environmental conditions during recording.</p>",
                unsafe_allow_html=True)
            states = ss['band_states']
            icons = {"high": "🔺", "low": "🔻", "normal": "⚪"}
            colors = {"high": "#ffadad", "low": "#a5d8ff", "normal": "#e9ecef"}

            cols = st.columns(len(states))
            for i, (band, state) in enumerate(states.items()):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div style='text-align:center; border-radius:10px; padding:10px; background:{colors.get(state, "#f1f3f5")};
                                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);'>
                            <b>{band.capitalize()}</b><br>{icons.get(state, "❔")} {state.capitalize()}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # Conversational lines
            summary_lines = []
            for band, state in states.items():
                if band == "alpha":
                    if state == "high":
                        summary_lines.append("🔺Your **Alpha** levels are **high**, associated with relaxed, internally focused state.")
                    elif state == "low":
                        summary_lines.append("🔻Your **Alpha** levels are **low**, suggesting reduced relaxation or heightened external focus.")
                elif band == "beta":
                    if state == "high":
                        summary_lines.append("🔺Your **Beta** levels are **high**, reflecting strong cognitive engagement but possibly stress.")
                    elif state == "low":
                        summary_lines.append("🔻Your **Beta** levels are **low**, often seen in deep relaxation or low mental effort.")
                elif band == "theta":
                    if state == "high":
                        summary_lines.append("🔺Your **Theta** levels are **high**, linked to mind-wandering or introspective awareness.")
                    elif state == "low":
                        summary_lines.append("🔻Your **Theta** levels are **low**, suggesting externally oriented attention.")
            for line in summary_lines:
                st.markdown(line)

        st.markdown("")
        st.markdown("")

        with st.expander("Want to know more about your brain wave? Click here", expanded=False):
            st.subheader("Your baseline (relative bandpower %) vs cohort")
            st.dataframe(rows, use_container_width=True)

            st.markdown("""
            <style>
            .tooltip { position: relative; display: inline-block; cursor: help; color: inherit; }
            .tooltip .tooltiptext {
              visibility: hidden; width: 450px; background-color: #555; color: #fff; text-align: left;
              border-radius: 6px; padding: 8px; position: absolute; z-index: 1; bottom: 125%; left: 150%;
              margin-left: -130px; opacity: 0; transition: opacity 0.3s; font-size: 0.85rem; line-height: 1.3;
            }
            .tooltip:hover .tooltiptext { visibility: visible; opacity: 1; }
            </style>
            """, unsafe_allow_html=True)

            st.markdown("""
            <div class="tooltip">
            ⚡ <b>Absolute Bandpower (µV²)</b>
            <span class="tooltiptext">
            The total amount of electrical activity in each band (Alpha/Beta/Delta/Theta/Gamma).
            </span>
            </div>
            """, unsafe_allow_html=True)
            abs_table = {band: [f"{result.abs_overall[band]:.3f}"] for band in BANDS}
            st.dataframe(abs_table)

            st.markdown("""
            <div class="tooltip">
            📊 <b>Relative Bandpower (%)</b>
            <span class="tooltiptext">
            How dominant each band is relative to total 1–40 Hz activity.
            </span>
            </div>
            """, unsafe_allow_html=True)
            rel_table = {band: [f"{result.rel_overall_pct[band]:.2f}%"] for band in BANDS}
            st.dataframe(rel_table)

        # -------------------------------
        # Meditation Recommendation UI
        # -------------------------------
        st.markdown("")
        st.markdown("")
        st.header("2️⃣ Meditation Type Recommendation 🧘")

        states = ss['band_states']
        rec_options = []

        # Beta
        if states.get("beta") == "high":
            rec_options.append({
                "label": "🧘 Calm your mind and reduce mental overactivity",
                "goal": "Elevated Beta indicates cognitive tension or overthinking.",
                "type": "Focused Attention Meditation (FAM)",
                "desc": "FAM promotes lower cognitive load, stress and sensory vigilance, hence regulating overactive Beta rhythms."
            })
        elif states.get("beta") == "low":
            rec_options.append({
                "label": "⚡ Boost alertness and sharpen focus",
                "goal": "Low Beta suggests lowered arousal or mental fatigue.",
                "type": "Mantra Meditation (MM)",
                "desc": "MM activates attention networks, helping to gently re-engage Beta activity for focus and mental engagement."
            })

        # Alpha
        if states.get("alpha") == "high":
            rec_options.append({
                "label": "🌿 Regulate your relaxed state and sustain awareness",
                "goal": "High Alpha reflects deep calm or internal absorption.",
                "type": "Body Scan Meditation (BSM)",
                "desc": "BSM balances relaxation with awareness — preventing over-detachment while maintaining calm."
            })
        elif states.get("alpha") == "low":
            rec_options.append({
                "label": "🌸 Cultivate calmness, mood regulation and inner ease",
                "goal": "Low Alpha may indicate restlessness or tension.",
                "type": "Loving-Kindness Meditation (LKM)",
                "desc": "Loving-Kindness meditation fosters gentle relaxation and emotional warmth, encouraging Alpha modulation."
            })

        # Theta
        if states.get("theta") == "high":
            rec_options.append({
                "label": "🎯 Refocus drifting attention and stabilise awareness",
                "goal": "High Theta can reflect mind-wandering or drowsiness.",
                "type": "Body Scan Meditation (BSM)",
                "desc": "BSM strengthens mental steadiness/alertness and reduces Theta-related wandering."
            })
        elif states.get("theta") == "low":
            rec_options.append({
                "label": "🕊 Deepen introspection and creative calm",
                "goal": "Low Theta may reflect reduced internal focus or introspection.",
                "type": "Mantra Meditation (MM)",
                "desc": "Mantra meditation gently enhances internal awareness through rhythmic repetition."
            })

        # Deduplicate + optional OM
        seen = set()
        dedup = []
        for rec in rec_options:
            if rec["type"] not in seen:
                dedup.append(rec)
                seen.add(rec["type"])
        rec_options = dedup
        if len(rec_options) == 1:
            rec_options.append({
                "label": "🪷 Explore calming, grounding meditation for general balance",
                "goal": "You may also explore practices that bring overall mental balance and grounding.",
                "type": "Open-Monitoring Meditation (OM)",
                "desc": "Maintain Balanced Awareness and Mental Clarity, helping maintain stability across Alpha, Beta, and Theta rhythms."
            })

        # Scoped button styling
        st.markdown("""
        <style>
        .big-font { font-size:20px !important; font-weight: bold; color: black; text-align: center; margin-bottom: 0px; }
        #rec-buttons { margin-top: 0.5rem !important; }
        #rec-buttons [data-testid="column"] > div { display: flex; align-items: stretch; justify-content: center; }
        #rec-buttons button {
            width: 100%; height: 80px; background: linear-gradient(145deg, #f8f9fa, #e9ecef);
            border: 2px solid #dee2e6; border-radius: 12px; font-size: 16px; font-weight: 600; color: #212529;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08); transition: all 0.2s ease-in-out; white-space: normal; word-wrap: break-word; line-height: 1.2;
        }
        #rec-buttons button:hover {
            transform: translateY(-2px) scale(1.02); border-color: #91c9f7;
            background: linear-gradient(145deg, #e6f0ff, #f8f9fa); box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        #rec-buttons div[data-testid="stHorizontalBlock"] { justify-content: center !important; align-items: stretch !important; gap: 2rem !important; }
        </style>
        """, unsafe_allow_html=True)

        st.markdown('<p class="big-font">Based on your brainwave activity, please choose what you’d like to prioritise:</p>', unsafe_allow_html=True)
        st.caption('<p style="text-align: center;">Multiple aspects of your mental state may be reflected in the EEG results. Choose the focus that <br>best matches your current goal — this will guide the recommended meditation practice.</p>', unsafe_allow_html=True)

        st.markdown('<div id="rec-buttons">', unsafe_allow_html=True)
        cols = st.columns(len(rec_options) if len(rec_options) > 0 else 1)
        for i, (col, rec) in enumerate(zip(cols, rec_options[:3])):
            with col:
                clicked = st.button(rec["label"], key=f"rec_btn_{i}", use_container_width=True, help=rec["goal"])
                if clicked:
                    ss['selected_meditation'] = rec
                    ss.meditation_selected = True
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

        # Show revealed meditation & Timer
        if 'selected_meditation' in ss:
            chosen = ss['selected_meditation']
            st.markdown(f"<h4 style='text-align: center;'><span style='color: #000000;'>Recommended Meditation:</span><br><span style='color: #0277bd;'>{chosen['type']}</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-style:italic;'> <strong>Why this fits you: </strong> {chosen['desc']}</p>", unsafe_allow_html=True)

            st.markdown("")
            st.header("3️⃣ Meditation Session (15 Minutes)")
            st.write("Read the instructions, then start the timer. A soft alert will play at the end.")

            steps = {
                "Focused Attention Meditation (FAM)": [
                    "Sit upright, breathe naturally, and count 1–4 on inhale, 5–8 on exhale. Return to 1 after 8. If you wander, gently return."
                ],
                "Open-Monitoring Meditation (OM)": [
                    "Eyes closed, open awareness to thoughts, sounds, sensations. Label gently (thinking/hearing/sensation) and return to present."
                ],
                "Body Scan Meditation (BSM)": [
                    "Move attention slowly from toes to head, noticing sensations. If wandering, return to the current body part."
                ],
                "Loving-Kindness Meditation (LKM)": [
                    "Silently repeat: ‘May you be happy/healthy/safe/live with ease,’ extending to self, friends, difficult people, all beings."
                ],
                "Mantra Meditation (MM)": [
                    "Chant ‘OM’ aloud ~10 min, then silently ~5 min. If attention drifts, return to the mantra."
                ]
            }
            for step in steps.get(chosen["type"], []):
                st.markdown(f"""
                <div style="border:none;border-radius:12px;padding:15px 18px;margin:10px 0;background:#eafafe;box-shadow:0 3px 10px rgba(0,0,0,0.1);border:1px solid #cce4ff;">
                    <strong>Meditation Instruction:<br></strong> {step}
                </div>
                """, unsafe_allow_html=True)

            # Scoped CSS for circular buttons
            st.markdown("""
            <style>
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"]{
                display: flex !important; justify-content: center !important; align-items: center !important;
                gap: 3rem !important; margin: 1.2rem 0 1.5rem 0 !important;
            }
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"] div.stButton > button {
                width: 95px; height: 95px; border-radius: 999px; font-size: 16px; font-weight: 700;
                border: 2px solid #cfe8ff; background: linear-gradient(145deg, #f7fbff, #e9f3ff);
                color: #143e62; box-shadow: 0 4px 10px rgba(0,0,0,0.08); transition: transform .15s ease, box-shadow .15s ease;
            }
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"] div.stButton:nth-of-type(1) > button {
                border-color: #8de79b; background: linear-gradient(145deg, #f6fff8, #e7ffe9); color: #0b5d1d;
            }
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"] div.stButton:nth-of-type(2) > button {
                border-color: #ffc38a; background: linear-gradient(145deg, #fff8f0, #ffefdd); color: #7a3e00;
            }
            </style>
            """, unsafe_allow_html=True)

            if not ss.meditation_completed:
                st.caption("Turn on your audio to hear the soft alert when time ends.")
                st.markdown('<div id="timer-controls-anchor"></div>', unsafe_allow_html=True)
                cols2 = st.columns([1.5, 1.5, 1, 1.5, 1.5])
                with cols2[1]:
                    start_clicked = st.button("▶️ Start Timer", key="start_meditation")
                with cols2[3]:
                    pause_clicked = st.button("⏸️ Pause Timer", key="pause_meditation")

                if start_clicked:
                    if not ss.meditation_running:
                        if ss.meditation_paused:
                            ss.meditation_start_ts = time.time() - ss.meditation_elapsed
                        else:
                            ss.meditation_start_ts = time.time()
                            ss.meditation_elapsed = 0
                        ss.meditation_running = True
                        ss.meditation_paused = False
                        st.rerun()

                if pause_clicked and ss.meditation_running:
                    ss.meditation_running = False
                    ss.meditation_paused = True
                    ss.meditation_elapsed = time.time() - ss.meditation_start_ts
                    st.rerun()

                if ss.meditation_running:
                    placeholder = st.empty()
                    prog = st.progress(0)
                    total = ss.meditation_total
                    start_ts = ss.meditation_start_ts
                    elapsed = 0
                    while elapsed < total and ss.meditation_running:
                        elapsed = int(time.time() - start_ts)
                        if elapsed > total:
                            elapsed = total
                        remaining = total - elapsed
                        mm = remaining // 60
                        ss_rem = remaining % 60
                        placeholder.markdown(f"### ⏳ Time remaining: **{mm:02d}:{ss_rem:02d}**")
                        prog.progress(int(100 * elapsed / total))
                        time.sleep(1)
                        ss.meditation_elapsed = elapsed
                    if elapsed >= total and ss.meditation_running:
                        ss.meditation_running = False
                        ss.meditation_paused = False
                        ss.meditation_completed = True
                        ss.meditation_elapsed = total
                        ss.audio_played = False
                        st.rerun()

                elif ss.meditation_paused:
                    remaining = ss.meditation_total - int(ss.meditation_elapsed)
                    mm = remaining // 60
                    ss_rem = remaining % 60
                    st.markdown(f"### ⏸️ Paused — Time remaining: **{mm:02d}:{ss_rem:02d}**")
                else:
                    mm = ss.meditation_total // 60
                    ss_rem = ss.meditation_total % 60
                    st.markdown(f"### ⏰ Ready — Total time: **{mm:02d}:{ss_rem:02d}**")

            else:
                st.success("✨ Meditation complete! Proceed to the Post-Meditation EEG Collection.")
                st.markdown("""
                <audio autoplay style="display: none;">
                    <source src="https://actions.google.com/sounds/v1/water/fountain_water_bubbling.ogg" type="audio/ogg">
                </audio>
                """, unsafe_allow_html=True)

        # Proceed to Post Collection
        if ss.meditation_completed:
            if st.button("➡️ Proceed to Post Collection"):
                ss.progress_step = 3
                st.rerun()

    else:
        st.info("Please complete Pre Collection (Live or Upload) first.")

# ----------------------------------
# Tab 3: POST COLLECTION (Live | Upload)
# ----------------------------------
with tab3:
    st.header("3️⃣ Post-Meditation EEG Collection")

    if ss.progress_step < 3:
        st.warning("🔒 This section is locked. Please complete the Meditation step first.")
        st.stop()

    if not folder_path:
        st.info("Please enter your name above to create a personal data folder.")
    else:
        post_tab_live, post_tab_upload = st.tabs(["🎧 Live Recording", "📂 Upload File"])

        # ---------- Live Recording ----------
        with post_tab_live:
            if ss.progress_step >= 4:
                st.warning("🔒 Post Collection is locked after moving to Results.")
                st.stop()

            st.caption("Please ensure your Muse 2 Headband is on and Bluetooth is connected.")

            post_full_path = os.path.join(folder_path, post_filename)
            post_duration = record_duration

            post_search_ph = st.empty()

            def render_post_search_button():
                with post_search_ph.container():
                    if st.button("🔍 Search for Muse Device & Start Stream", key="post_btn_search"):
                        with st.spinner("Searching for Muse Device..."):
                            process = subprocess.Popen([sys.executable, "start_stream.py"])
                            time.sleep(8)
                            found = any(resolve_stream('type', 'EEG') for _ in range(20) if not time.sleep(1))
                            if found:
                                ss['post_stream_process'] = process
                                ss.post_stream_found = True
                                render_post_connected_disabled()
                                st.success("Muse device found. You can now record Post-Meditation EEG.")
                                st.rerun()
                            else:
                                process.terminate()
                                st.error("Please ensure Muse Headband & Bluetooth is on and try again.")

            def render_post_connected_disabled():
                with post_search_ph.container():
                    st.button("✅ Muse Device Connected", disabled=True, key="post_btn_connected")

            if ss.get("post_stream_found", False):
                render_post_connected_disabled()
            else:
                render_post_search_button()

            with st.expander("Post-Meditation EEG Recording Instructions", expanded=False):
                st.markdown("""
                Please follow the same procedure as the pre-meditation recording:
                - Sit upright, keep still and relaxed
                - Focus on a fixed point on the screen
                - Maintain natural breathing
                - Avoid excessive blinking or movement
                """)

            post_record_ph = st.empty()

            def render_post_start_record():
                with post_record_ph.container():
                    if st.button("🎙 Start Post-Meditation EEG Recording", key="post_btn_record"):
                        ss.post_recording_in_progress = True
                        render_post_recording_disabled()
                        st.rerun()

            def render_post_recording_disabled():
                with post_record_ph.container():
                    st.button("🎙 Recording in Progress…", disabled=True, key="post_btn_recording")

            def render_post_recording_ended():
                with post_record_ph.container():
                    st.button("✅ Recording Completed", disabled=True, key="post_btn_recordingended")

            if ss.get("post_stream_found") and 'post_stream_process' in ss:
                if ss.get("post_recording_finished"):
                    render_post_connected_disabled()
                    render_post_recording_ended()
                    st.toast("🎧 Post-meditation EEG recording completed!")
                elif ss.get("post_recording_in_progress"):
                    render_post_recording_disabled()
                    with st.spinner("Post-meditation EEG recording will start upon seeing dot..."):
                        time.sleep(5)
                    overlay = show_fixation_overlay()
                    try:
                        record(post_duration, filename=post_full_path)
                        if os.path.exists(post_full_path):
                            st.toast(f"Post-meditation recording complete! File saved: `{post_full_path}`")
                            ss['after_eeg_file'] = post_full_path
                            ss.post_recording_finished = True
                            ss.post_meditation_recording_completed = True
                            ss.post_recording_in_progress = False
                            overlay.empty()
                            if ss['post_stream_process'].poll() is None:
                                ss['post_stream_process'].terminate()
                                st.toast("Muse stream terminated.")
                            del ss['post_stream_process']
                            ss.post_stream_found = False
                            st.rerun()
                        else:
                            st.error("Post-meditation recording failed. Please try again.")
                            overlay.empty()
                    except Exception as e:
                        st.error(f"Post-meditation recording error: {e}")
                        overlay.empty()
                else:
                    render_post_start_record()

        # ---------- Upload File ----------
        with post_tab_upload:
            uploaded_post = st.file_uploader(
                "📂 Please upload your post-meditation EEG file recorded earlier using the local recording script.",
                type=["csv"],
                key="after_upload"
            )
            if uploaded_post:
                post_path = os.path.join(folder_path, "after_eeg.csv")
                with open(post_path, "wb") as f:
                    f.write(uploaded_post.getbuffer())
                st.success("✅ Post-Meditation EEG file uploaded successfully.")
                ss['after_eeg_file'] = post_path
                ss.post_meditation_recording_completed = True

        # Proceed to Results
        if ss.get('after_eeg_file') and os.path.exists(ss['after_eeg_file']):
            if st.button("➡️ View Results & Interpretation"):
                ss.progress_step = 4
                st.rerun()

# ----------------------------------
# Tab 4: RESULTS & INTERPRETATION (Shared)
# ----------------------------------
with tab4:
    st.header("4️⃣ Results & Interpretation")

    if ss.progress_step < 4:
        st.warning("🔒 This section is locked. Please complete Post Collection first.")
        st.stop()

    if ss.get('after_eeg_file') and os.path.exists(ss['after_eeg_file']) \
       and ss.get('before_eeg_file') and os.path.exists(ss['before_eeg_file']):

        st.markdown('<p style="text-align: left; color: #6c757d; font-size: 0.8rem;">⚠️ EEG-based interpretations vary by individual, mood, and context — use results as guidance, not diagnosis.</p>', unsafe_allow_html=True)

        try:
            post_result = summarize_one_file(ss['after_eeg_file'])
            post_rel = post_result.rel_overall_pct

            pre_result = summarize_one_file(ss['before_eeg_file'])
            pre_rel = pre_result.rel_overall_pct

            comparison_rows = []
            focus_bands = ["alpha", "beta", "theta"]
            for band in focus_bands:
                pre_val = pre_rel.get(band, np.nan)
                post_val = post_rel.get(band, np.nan)
                change = post_val - pre_val if not np.isnan(pre_val) and not np.isnan(post_val) else np.nan
                comparison_rows.append({
                    "Band": band.capitalize(),
                    "Before (%)": round(pre_val, 2) if not np.isnan(pre_val) else "N/A",
                    "After (%)": round(post_val, 2) if not np.isnan(post_val) else "N/A",
                    "Change": f"{change:+.2f}%" if not np.isnan(change) else "N/A"
                })

            change_directions = {}
            for row in comparison_rows:
                band = row["Band"].lower()
                change_str = row["Change"]
                if change_str != "N/A":
                    change_val = float(change_str.replace('%', '').replace('+', ''))
                    if change_val > 1:
                        change_directions[band] = "up"
                    elif change_val < -1:
                        change_directions[band] = "down"
                    else:
                        change_directions[band] = "steady"

                    if band == "alpha":
                        if change_val > 1:
                            st.success(f"**{row['Band']} increased** — You’ve achieved greater relaxation and inner calm 🌿")
                        elif change_val < -1:
                            st.info(f"**{row['Band']} decreased** — You’ve achieved a more alert and engaged state ⚡")
                    elif band == "beta":
                        if change_val > 1:
                            st.info(f"**{row['Band']} increased** — You’ve achieved heightened alertness and concentration 🎯")
                        elif change_val < -1:
                            st.success(f"**{row['Band']} decreased** — You’ve achieved a calmer, less stressed mental state 🕊️")
                    elif band == "theta":
                        if change_val > 1:
                            st.success(f"**{row['Band']} increased** — You’ve achieved deeper meditative and introspective focus 💭")
                        elif change_val < -1:
                            st.info(f"**{row['Band']} decreased** — You’ve achieved better external attentiveness 🌞")

            alpha, beta, theta = (
                change_directions.get("alpha", "steady"),
                change_directions.get("beta", "steady"),
                change_directions.get("theta", "steady"),
            )

            pattern_messages = {
                ("down", "down", "up"): "🌙 Congratulations! You’ve entered a deeply introspective and restful state — your mind released tension and turned inward.",
                ("up", "down", "up"): "💫  Congratulations! You’ve reached a calm but aware state — peaceful relaxation with gentle focus.",
                ("down", "up", "down"): "⚡  Congratulations! You’ve sharpened focus and alertness — your mind feels active and engaged.",
                ("up", "down", "down"): "🌿  Congratulations! You’ve achieved calm focus — relaxed yet attentive, a balanced meditative state.",
                ("down", "down", "down"): "😌  Congratulations! Your brain settled into a low-activity rhythm — possibly deep rest or mild fatigue.",
                ("up", "up", "down"): "☀️  Congratulations! You’ve reached mindful engagement — calm yet cognitively alert.",
                ("down", "up", "up"): "💭  Congratulations! You’ve entered reflective thought — alertness and introspection intertwined."
            }

            combo_msg = pattern_messages.get((alpha, beta, theta))
            if combo_msg:
                st.markdown(f"<p style='font-size:1.3rem;'>{combo_msg}</p>", unsafe_allow_html=True)

            st.balloons()

            with st.expander("Want to know more in-depth about the result? Click here", expanded=False):
                st.subheader("📊 Before vs After Meditation Comparison")
                st.dataframe(comparison_rows, use_container_width=True)

        except Exception as e:
            st.error(f"Error analyzing post-meditation data: {e}")
            st.info("Please ensure the pre- and post-meditation recording files were created successfully.")
    else:
        st.info("Please complete both Pre and Post collections to view results.")
