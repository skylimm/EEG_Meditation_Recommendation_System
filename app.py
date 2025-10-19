import sys
from math import floor

import streamlit as st
import os
import time
import subprocess
from muselsl import record
from pylsl import resolve_stream
from user_input import get_user_folder

import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from preprocessings import summarize_one_file, BANDS
import json

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

# ---- How it works section ---
st.markdown('<div class="hw-title">How it works</div>', unsafe_allow_html=True)

c1, c2, c3 = st.columns(3, gap="large")
with c1:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">🎧</div>
      <div class="hw-step">1. Record Pre-EEG </div>
      <div class="hw-text">Wear the Muse headband and we capture a short resting EEG to understand your current brain state.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">📊</div>
      <div class="hw-step">2. Recommendations</div>
      <div class="hw-text">Receive the meditation technique that best balances your current brain state — right now, in real time.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">✨</div>
      <div class="hw-step">3. Post-EEG & Results</div>
      <div class="hw-text">After meditation, record another short EEG to see how your brainwave patterns changed.</div>
    </div>
    """, unsafe_allow_html=True)
# initializations
st.session_state.setdefault("stream_found", False)
st.session_state.setdefault("recording_in_progress", False)
st.session_state.setdefault("recording_finished", False)
st.session_state.setdefault("search_completed", False)
if "meditation_selected" not in st.session_state:
    st.session_state.meditation_selected = False
# Add these with your other session state initializations
if "post_meditation_recording_completed" not in st.session_state:
    st.session_state.post_meditation_recording_completed = False
if "post_stream_found" not in st.session_state:
    st.session_state.post_stream_found = False
if "post_recording_in_progress" not in st.session_state:
    st.session_state.post_recording_in_progress = False
if "post_recording_finished" not in st.session_state:
    st.session_state.post_recording_finished = False
if "after_eeg_file" not in st.session_state:
    st.session_state.after_eeg_file = None
st.session_state.setdefault("baseline_mode", "Research")  # "Research" or "Comfort"


# Step 1: User input and folder creation
st.markdown("")
st.markdown("")
folder_path = get_user_folder()
filename = "before_eeg.csv"
duration = 300 if st.session_state["baseline_mode"] == "Research" else 180
# duration = 60 if st.session_state["baseline_mode"] == "Research" else 10
import base64

def play_hidden_audio(file_path):
    """Play audio invisibly via HTML <audio> tag."""
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
    audio_b64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f"""
        <audio autoplay style="display:none;">
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_tag, unsafe_allow_html=True)


def show_fixation_overlay():

    placeholder = st.empty()

    if st.session_state.get("baseline_mode") == "Comfort":
        # Use Streamlit's audio component instead of HTML audio
        audio_file = open('./assets/brown_noise_3min.mp3', 'rb')
        audio_bytes = audio_file.read()

        st.audio(audio_bytes, format='audio/mp3', autoplay=True)


    placeholder.markdown("""
        <style>
            .overlay {
                position:fixed; top:0; left:0; width:100vw; height:100vh;
                background:white; display:flex; justify-content:center; align-items:center;
                z-index:9999;
                flex-direction:column;
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


if folder_path:
    full_path = os.path.join(folder_path, filename)
    st.subheader("1️⃣ Pre-Meditation EEG Collection")
    mode_locked = st.session_state.get("recording_in_progress") or st.session_state.get("recording_finished")

    mode = st.radio(
        "Choose baseline recording mode:",
        ["Research Mode (5 min, silent)", "Comfort Mode (3 min with brown noise)"],
        index=0 if st.session_state.get("baseline_mode") in [None, "Research"] else 1,
        disabled=mode_locked,
    )
    # Save chosen mode
    if not mode_locked:
        st.session_state["baseline_mode"] = "Research" if "Research" in mode else "Comfort"

    # Display contextual note
    if st.session_state["baseline_mode"] == "Comfort":
        st.caption("Comfort Mode shortens baseline to 3 minutes for better user experience.")
    else:
        st.caption("Research Mode keeps a 5-minute silent baseline for experimental consistency.")


    # Only show search functionality if no meditation has been selected yet
    if not st.session_state.get('meditation_selected', False):
        st.caption("Please ensure your Bluetooth and Muse 2 Headband is on before searching for Muse Device")

        # --- Step 2: Search for Muse Device ---
        search_ph = st.empty()


        def render_search_button():
            with search_ph.container():
                if st.button("🔍 Search for Muse Device & Start Stream", key="btn_search"):
                    with st.spinner("Starting Muse LSL stream..."):
                        process = subprocess.Popen([sys.executable, "start_stream.py"])
                        time.sleep(8)
                        found = any(resolve_stream('type', 'EEG') for _ in range(20) if not time.sleep(1))
                        if found:
                            st.session_state['stream_process'] = process
                            st.session_state.stream_found = True
                            render_connected_disabled()
                            st.success("Muse device found. You can now record EEG.")
                        else:
                            process.terminate()
                            st.error("Please ensure Muse Headband & Bluetooth is on and rerun the web.")


        def render_connected_disabled():
            with search_ph.container():
                st.button("✅ Muse Device Connected", disabled=True, key="btn_connected")


        # initial render
        if st.session_state.get("stream_found", False):
            render_connected_disabled()
        else:
            render_search_button()

        # Instruction
        with st.expander("Review instructions below if this is your first time using the Muse 2 headband",
                         expanded=False):
            # --- First row: Image + Wearing instructions ---
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(
                    "https://pyxis.nymag.com/v1/imgs/e09/a32/9356a8db59e0b3880c8b8d08bb7c165461-09-muse.2x.rhorizontal.w700.jpg",
                    caption="Correct EEG headband placement",
                    width=200,
                )

            with col2:
                st.markdown(
                    """
                    ### 🧠 Wearing the Muse 2 Headband
                    You will wear a **Muse 2 EEG headband** across your forehead and behind your ears (as shown in the image).  

                    Please ensure the following before you start the Pre-Meditation EEG Recording:
                    - Hair is tucked away from your **forehead** and **ears**.  
                    - The band feels **snug but comfortable**.  
                    - Sit in a **quiet, comfortable room** with steady lighting.  
                    - Sit upright in a chair with **back support**, feet flat on the ground, and hands resting on your lap.  
                    - Keep your **head, face, and jaw relaxed**.  
                    - **Avoid lying down** — this may cause the headband to shift or make you drowsy.  
                    """
                )

            # --- Second row: centered "During EEG Recording" section ---
            st.divider()
            left, center, right = st.columns([0.5, 4, 0.5])
            with center:
                st.markdown(
                    """
                    ### 🎧 During EEG Recording
                    - Sit **quietly with eyes open**.  
                    - Fix your gaze on a **small point** (e.g., dot or cross) on the screen.  
                    - Keep your **body as still as possible**:
                      - **Avoid blinking excessively**, clenching your jaw, or moving your head.  
                    - Maintain **natural, gentle breathing** throughout the session.
                    - **5-Minute EEG Recording starts when dot screen is displayed.**
                    """
                )

            st.info(
                "When you're ready, press **'Start Pre-Meditation EEG Recording'** button below to begin."
            )

        if st.session_state["baseline_mode"] == "Comfort":
            st.caption("Turn on volume to hear the brown noise")
        # --- Step 3: Record EEG (only if stream started) ---
        record_ph = st.empty()


        def render_start_record():
            with record_ph.container():
                if st.button("🎙 Start Pre-Meditation EEG Recording", key="btn_record"):
                    st.session_state.recording_in_progress = True
                    render_recording_disabled()

                    with st.spinner(f"{int(duration/60)} minutes EEG recording will start upon seeing dot.."):
                        time.sleep(5)  # show spinner first for 2 seconds (adjust as needed)

                    overlay = show_fixation_overlay()

                    with st.spinner(f"Recording EEG for {duration} seconds..."):
                        try:
                            record(duration, filename=full_path)
                            if os.path.exists(full_path):
                                st.toast(f"Recording complete! File saved: `{full_path}`")
                                st.session_state['before_eeg_file'] = full_path
                                st.session_state.recording_finished = True
                                render_recording_ended()
                            else:
                                st.error("Recording failed. Please refresh the page and try again.")
                        except Exception as e:
                            st.error(f"Recording error: {e}")
                        finally:
                            overlay.empty()  # remove dot

                            if st.session_state['stream_process'].poll() is None:
                                st.session_state['stream_process'].terminate()
                                st.toast("Muse stream terminated.")
                            del st.session_state['stream_process']
                            st.session_state.stream_found = False
                            st.session_state.recording_in_progress = False


        def render_recording_disabled():
            with record_ph.container():
                st.button("🎙 Recording in Progress…", disabled=True, key="btn_recording")


        def render_recording_ended():
            with record_ph.container():
                st.button("✅ Recording Completed", disabled=True, key="btn_recordingended")


        # initial render logic
        if st.session_state.get("stream_found") and 'stream_process' in st.session_state:
            if st.session_state.get("recording_finished"):
                render_connected_disabled()
                render_recording_ended()
                st.toast("🎧 Pre-meditation EEG recording completed. You can proceed to the next step.")
            elif st.session_state.get("recording_in_progress"):
                render_recording_disabled()
            else:
                render_start_record()

    else:
        st.success("✅ EEG Data Collected - Ready for Meditation")

    if st.session_state.get('before_eeg_file') and st.session_state.get('recording_finished'):
        full_path = st.session_state['before_eeg_file']
        # After recording a new user's PRE file:

        result = summarize_one_file(full_path)  # your recorded pre file
        rel = result.rel_overall_pct  # {'delta': %, 'theta': %, ...}


        @st.cache_data
        def load_cohort_stats(path="outputs/cohort_baseline.json"):
            with open(path, "r") as f:
                return json.load(f)  # {band: {mean, sd, p25, p75, ...}}


        cohort = load_cohort_stats("outputs/cohort_baseline.json")


        def classify_vs_cohort(val, mean, sd, z_thresh=1.0):
            # handle degenerate sd
            if sd is None or sd == 0:
                return "normal", 0.0
            z = (val - mean) / sd
            if z > z_thresh:
                return "high", z
            elif z < -z_thresh:
                return "low", z
            else:
                return "normal", z


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
        # Save band states (high / low / normal) to session for reuse outside expander
        st.session_state['band_states'] = {r["Band"].lower(): r["State"] for r in rows}

        # --- Quick Brainwave Feedback Summary (above expander) ---
        if 'band_states' in st.session_state:
            st.divider()
            st.markdown(
                "<p style='text-align:center; font-size: 27px; font-weight: bold'>🧩 Quick Brainwave Summary 🧩</p>",
                unsafe_allow_html=True)
            st.write(
                "<p style='font-size: 12px;  color: dimgray; text-align: justify;'>⚠️ Do note that the EEG readings and interpretations are indicative and may vary between individuals. Differences can arise due to factors such as signal quality, electrode contact, individual brainwave variability, and environmental conditions during recording.</p>",
                unsafe_allow_html=True)
            st.markdown("")
            states = st.session_state['band_states']
            icons = {"high": "🔺", "low": "🔻", "normal": "⚪"}
            colors = {"high": "#ffadad", "low": "#a5d8ff", "normal": "#e9ecef"}

            # Display color-coded chips in one row
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

            st.markdown("")
            st.markdown("")
            # Conversational summary lines
            # --- Conversational summary (aligned with report literature) ---
            summary_lines = []

            for band, state in states.items():
                if band == "alpha":
                    if state == "high":
                        summary_lines.append(
                            "🔺Your **Alpha** levels are **high**, which is typically associated with a relaxed, internally focused state. "
                            "This suggests you were calm and possibly in a meditative or restful state during the recording."
                        )
                    elif state == "low":
                        summary_lines.append(
                            "🔻Your **Alpha** levels are **low**, which may indicate reduced relaxation or heightened external focus. "
                            "This pattern is often observed when attention is directed outward or when experiencing mild stress or cognitive tension."
                        )

                elif band == "beta":
                    if state == "high":
                        summary_lines.append(
                            "🔺Your **Beta** levels are **high**, which reflect strong cognitive engagement and top-down attention control. "
                            "However, elevated Beta is also linked to mental overactivity and stress, suggesting your mind may have been alert or restless during recording."
                        )
                    elif state == "low":
                        summary_lines.append(
                            "🔻Your **Beta** levels are **low**, which may reflect reduced cognitive drive or arousal. "
                            "This often occurs in states of deep relaxation, meditation, or low mental effort."
                        )

                elif band == "theta":
                    if state == "high":
                        summary_lines.append(
                            "🔺Your **Theta** levels are **high**, commonly linked to internal focus, daydreaming, or mind-wandering. "
                            "In meditation contexts, elevated Theta can indicate deep introspective or relaxed awareness."
                        )
                    elif state == "low":
                        summary_lines.append(
                            "🔻Your **Theta** levels are **low**, which suggests reduced internal processing or diminished relaxation. "
                            "This is often seen when attention is externally oriented or the mind is actively engaged in control or planning."
                        )

            if summary_lines:
                for line in summary_lines:
                    st.markdown(f" {line}")

        # collapsible data

        st.markdown("")
        st.markdown("")

        with st.expander("Want to know more about your brain wave? Click here", expanded=False):
            st.subheader("Your baseline (relative bandpower %) vs cohort")
            st.dataframe(rows, use_container_width=True)

            # Friendly messages
            # for r in rows:
            #     if r["State"] == "high":
            #         st.warning(f"{r['Band']} looks **high** vs cohort (z={r['Z-score']}).")
            #     elif r["State"] == "low":
            #         st.info(f"{r['Band']} looks **low** vs cohort (z={r['Z-score']}).")

            # --- Tooltip style (only needs to be declared once in your app) ---
            st.markdown("""
            <style>
            .tooltip {
              position: relative;
              display: inline-block;
              cursor: help;
              color: inherit;
            }
            .tooltip .tooltiptext {
              visibility: hidden;
              width: 450px;
              background-color: #555;
              color: #fff;
              text-align: left;
              border-radius: 6px;
              padding: 8px;
              position: absolute;
              z-index: 1;
              bottom: 125%;
              left: 150%;
              margin-left: -130px;
              opacity: 0;
              transition: opacity 0.3s;
              font-size: 0.85rem;
              line-height: 1.3;
            }
            .tooltip:hover .tooltiptext {
              visibility: visible;
              opacity: 1;
            }
            </style>
            """, unsafe_allow_html=True)

            # --- Display absolute bandpower ---
            # st.markdown("### ⚡ Absolute Bandpower (µV²)")
            # --- Absolute Bandpower tooltip ---
            st.markdown("""
            <div class="tooltip">
            ⚡ <b>Absolute Bandpower (µV²)</b>
            <span class="tooltiptext">
            The <b>total amount of electricity activity</b> your brain produced in each brainwave type (Alpha/Beta/Delta/Theta/Gamma).<br>
            <b>Example: </b> <br>
            🧘‍♀️ A calm, focused mind usually shows stronger Alpha waves <br>
            💭 An active, problem-solving mind shows stronger Beta waves <br>
            </span>
            </div>
            """, unsafe_allow_html=True)

            abs_table = {band: [f"{result.abs_overall[band]:.3f}"] for band in BANDS}
            st.dataframe(abs_table)

            # --- Display relative bandpower (% of total 1–40 Hz) ---
            st.markdown("""
            <div class="tooltip">
            📊 <b>Relative Bandpower (%)</b>
            <span class="tooltiptext">
            It shows how <b>dominant</b> each frequency band is relative to overall brain activity. <br>
            <b>Example:</b> <br>
            If Beta (%) is high it’s likely you’re in an active-thinking or stressed state 🤯 
            </span>
            </div>
            """, unsafe_allow_html=True)

            # st.markdown("### 📊 Relative Bandpower (%)")
            rel_table = {band: [f"{result.rel_overall_pct[band]:.2f}%"] for band in BANDS}
            st.dataframe(rel_table)

        st.markdown("")
        st.markdown("")
        st.header("2️⃣ Meditation Type Recommendation 🧘")

        if 'band_states' in st.session_state:
            states = st.session_state['band_states']

            rec_options = []

            # Beta patterns
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

            # Alpha patterns
            if states.get("alpha") == "high":
                rec_options.append({
                    "label": "🌿 Regulate your relaxed state and sustain awareness",
                    "goal": "High Alpha reflects deep calm or internal absorption.",
                    "type": "Body Scan Meditation (BSM)",
                    "desc": "BSM  balances relaxation with awareness — preventing over-detachment while maintaining calm."
                })
            elif states.get("alpha") == "low":
                rec_options.append({
                    "label": "🌸 Cultivate calmness, mood regulation and inner ease",
                    "goal": "Low Alpha may indicate restlessness or tension.",
                    "type": "Loving-Kindness Meditation (LKM)",
                    "desc": "Loving-Kindness meditation fosters gentle relaxation and emotional warmth, encouraging Alpha modulation."
                })

            # Theta patterns
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

            # -------------------------------
            # 2️⃣ Deduplicate meditation types if multiple bands recommend same
            # -------------------------------
            seen = set()
            dedup = []
            for rec in rec_options:
                if rec["type"] not in seen:
                    dedup.append(rec)
                    seen.add(rec["type"])
            rec_options = dedup

            # -------------------------------
            # 3️⃣ Add optional “general balance” suggestion if only one band triggered
            # -------------------------------
            if len(rec_options) == 1:
                optional = {
                    "label": "🪷 Explore calming, grounding meditation for general balance",
                    "goal": "You may also explore practices that bring overall mental balance and grounding.",
                    "type": "Open-Monitoring Meditation (OM)",
                    "desc": "Maintain Balanced Awareness and Mental Clarity, helping maintain stability across Alpha, Beta, and Theta rhythms."
                }
                rec_options.append(optional)

            if rec_options:
                st.markdown("""
                <style>
                .big-font {
                    font-size:20px !important;
                    font-weight: bold;
                    color: black;
                    text-align: center;
                    margin-bottom: 0px; /* removes excess space under heading */

                }

                /* Reduce space around the info box */
                div[data-testid="stVerticalBlock"] > div:has(> div[data-testid="stVerticalBlock"] > div:has(> .stAlert)) {
                    margin-top: -1rem !important;
                    margin-bottom: 0.5rem !important;
                }

                /* Scoped button styling */
                #rec-buttons {
                    margin-top: 0.0rem !important;
                }
                #rec-buttons .option-card {
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;
                    align-items: stretch;
                    height: 100%;
                    text-align: center;
                    padding: 10px 8px;
                    min-height: 160px; /* consistent button area height */
                }
                #rec-buttons .option-caption {
                    color: #6c757d;
                    font-size: 0.9rem;
                    min-height: 35px;  /* equal caption zone */
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                #rec-buttons button {
                    height: 100px !important;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    padding: 8px 4px;
                }

                #rec-buttons [data-testid="column"] > div {
                    height: 120px;
                }

                #rec-buttons button:hover {
                    transform: translateY(-2px) scale(1.02);
                    border-color: #91c9f7;
                    background: linear-gradient(145deg, #e6f0ff, #f8f9fa);
                    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                }
                #rec-buttons button:active {
                    transform: scale(0.98);
                    box-shadow: 0 1px 3px rgba(0,0,0,0.2);
                }
                #rec-buttons div[data-testid="stHorizontalBlock"] {
                    justify-content: center !important;
                    align-items: stretch !important;
                    gap: 2rem !important;
                }
                </style>
                """, unsafe_allow_html=True)

                # Heading & compact info
                st.markdown(
                    '<p class="big-font">Based on your brainwave activity, please choose what you’d like to prioritise:</p>',
                    unsafe_allow_html=True)
                st.markdown("")
                with st.container():
                    # st.markdown('<div class="compact-info">', unsafe_allow_html=True)
                    st.caption(
                        '<p style="text-align: center;">Multiple aspects of your mental state may be reflected in the EEG results. Choose the focus that <br>best  matches your current goal — this will guide the recommended meditation practice.</p>',
                        unsafe_allow_html=True
                    )
                    # st.markdown('</div>', unsafe_allow_html=True)

                    # --- Scoped buttons with working hover tooltips ---
                    st.markdown("""
                    <style>
                    #rec-buttons {
                        margin-top: 0.5rem !important;
                    }

                    /* Equal height columns */
                    #rec-buttons [data-testid="column"] > div {
                        display: flex;
                        align-items: stretch;
                        justify-content: center;
                    }

                    /* Wrapper div for button + tooltip */
                    #rec-buttons .tooltip-wrapper {
                        position: relative;
                        width: 100%;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }

                    /* Button styling */
                    #rec-buttons button {
                        width: 100%;
                        height: 80px;
                        background: linear-gradient(145deg, #f8f9fa, #e9ecef);
                        border: 2px solid #dee2e6;
                        border-radius: 12px;
                        font-size: 16px;
                        font-weight: 600;
                        color: #212529;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.08);
                        transition: all 0.2s ease-in-out;
                        white-space: normal;
                        word-wrap: break-word;
                        line-height: 1.2;
                    }

                    #rec-buttons button:hover {
                        transform: translateY(-2px) scale(1.02);
                        border-color: #91c9f7;
                        background: linear-gradient(145deg, #e6f0ff, #f8f9fa);
                        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
                    }

                    /* Tooltip box */
                    #rec-buttons .tooltiptext {
                        visibility: hidden;
                        width: 250px;
                        background-color: #212529;
                        color: #fff;
                        text-align: left;
                        border-radius: 8px;
                        padding: 8px 10px;
                        position: absolute;
                        z-index: 100;
                        bottom: 110%;
                        opacity: 0;
                        transition: opacity 0.3s ease;
                        font-size: 0.85rem;
                        line-height: 1.3;
                    }

                    /* Show tooltip on hover */
                    #rec-buttons .tooltip-wrapper:hover .tooltiptext {
                        visibility: visible;
                        opacity: 1;
                    }

                    /* Align all horizontally */
                    #rec-buttons div[data-testid="stHorizontalBlock"] {
                        justify-content: center !important;
                        align-items: stretch !important;
                        gap: 2rem !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)

                    st.markdown('<div id="rec-buttons">', unsafe_allow_html=True)
                    cols = st.columns(len(rec_options))

                    for i, (col, rec) in enumerate(zip(cols, rec_options[:3])):
                        with col:
                            # Use st.button with help parameter for native tooltip
                            clicked = st.button(
                                rec["label"],
                                key=f"rec_btn_{i}",
                                use_container_width=True,
                                help=rec["goal"]  # This creates the native tooltip
                            )
                            if clicked:
                                st.session_state['selected_meditation'] = rec
                                st.session_state.meditation_selected = True  # ← ADD THIS LINE
                                st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)




            else:
                st.info("Your brainwave balance appears stable — you may explore any meditation style you prefer.")
                st.markdown("""
                <p style="text-align: center; font-size: 18px;">
                Since your Alpha, Beta, and Theta levels are balanced, you can select <b>any meditation type</b> to deepen your self-awareness or relaxation experience.
                </p>
                """, unsafe_allow_html=True)

                # --- Available meditation options ---
                explore_options = [
                    {
                        "label": "🎯 Enhance focus and mental sharpness",
                        "type": "Focused Attention Meditation (FAM)",
                        "desc": "Trains the mind to focus on a single object or breath — ideal for improving attentional control and mental stability."
                    },
                    {
                        "label": "🌊 Cultivate open, non-judgmental awareness",
                        "type": "Open Monitoring Meditation (OM)",
                        "desc": "Encourages observing thoughts and sensations without attachment — promotes balance across all brainwave rhythms."
                    },
                    {
                        "label": "💗 Foster emotional warmth and positivity",
                        "type": "Loving-Kindness Meditation (LKM)",
                        "desc": "Builds empathy and emotional regulation, gently enhancing relaxation and Alpha rhythm."
                    },
                    {
                        "label": "🕯️ Deepen relaxation and body awareness",
                        "type": "Body Scan Meditation (BSM)",
                        "desc": "Involves gradual awareness of body sensations, easing stress and increasing calmness."
                    },
                    {
                        "label": "🔔 Improve rhythmic focus through sound",
                        "type": "Mantra Meditation (MM)",
                        "desc": "Uses sound repetition (e.g., ‘OM’) to stabilize attention and enhance Beta–Theta balance."
                    }
                ]

                # --- Display meditation choice buttons ---
                st.markdown('<div id="rec-buttons">', unsafe_allow_html=True)
                cols = st.columns(3)

                for i, (col, rec) in enumerate(zip(cols * 2, explore_options)):  # repeat columns if >3
                    with col:
                        clicked = st.button(
                            rec["label"],
                            key=f"explore_btn_{i}",
                            use_container_width=True,
                            help=rec["desc"]
                        )
                        if clicked:
                            st.session_state['selected_meditation'] = rec
                            st.session_state.meditation_selected = True
                            st.rerun()

                st.markdown('</div>', unsafe_allow_html=True)
        # -------------------------------
        # 5️⃣ Show revealed meditation type & guidance
        # -------------------------------
        if 'selected_meditation' in st.session_state:
            chosen = st.session_state['selected_meditation']

            st.markdown(
                f"<h4 style='text-align: center;'><span style='color: #000000;'>Recommended Meditation:</span><br><span style='color: #0277bd;'>{chosen['type']}</span></h4>",
                unsafe_allow_html=True)
            st.markdown(f"<p style='font-style:italic;'> <strong>Why this fits you: </strong> {chosen['desc']}</p>",
                        unsafe_allow_html=True)

            st.markdown("")
            st.markdown("")
            st.header("3️⃣ Meditation Session (15 Minutes)")
            st.write(
                "Before beginning your session, please take a moment to read the meditation instructions carefully. \n Once ready, press **Start Meditation Timer** to begin your 15-minute meditation practice. \n When the timer ends, a soft alert will play — you can then gently open your eyes and proceed to the **Post-Meditation EEG Recording** step.")

            steps = {
                "Focused Attention Meditation (FAM)": [
                    "Sit upright and gently close your eyes.<br>Take one deep breath in, and slowly breathe out. <br> Now let your breathing settle into a steady, natural rhythm. <br> As you inhale, begin counting in your mind — one, two, three, four. <br>As you exhale, continue the count — five, six, seven, eight. "
                    "<br>When you reach eight, return to one and start again. "
                    "<br>Keep your focus on two things: the flow of your breath, and the steady count. <br>"
                    "If your mind wanders, simply notice it, let it go, and bring your attention back to the breathing and counting. "
                    "<br>Stay with this practice for the full 15 minutes."

                ],
                "Open Monitoring Meditation (OM)": [
                    "Sit comfortably with eyes closed. <br> Instead of focusing on one thing, allow whatever arises in the present moment — thoughts, feelings, body sensations, or sounds — to enter your awareness. <br> Don’t try to push anything away or hold onto it. <br> If you notice a thought, silently label it ‘thinking.’ If you notice a sound, silently label it ‘hearing.’ If you feel a sensation, silently label it ‘sensation.’ Simply observe, moment by moment, with openness and curiosity.<br>If you get lost in thought, gently return to awareness of whatever is present right now. <br>Continue observing without judgment for the full 15 minutes."
                ],
                "Body Scan Meditation (BSM)": [
                    "Lie down or sit upright comfortably. Close your eyes and let your breath flow naturally. <br> Bring attention to your toes. Notice any sensations — tingling, warmth, or nothing at all. <br> Slowly move your attention through your feet, ankles, calves, and knees. <br> Continue up through the thighs, hips, and lower back.<br>  Progress to your stomach and chest, noticing the rise and fall with each breath. <br> Then to your shoulders, arms, elbows, wrists, palms, and fingers. <br> Finally, move attention to your neck, face, and head. <br> Notice your whole body resting in awareness. <br> If your mind wanders, gently return to the body part you were focusing on. <br> Take your time moving through the body until 15 minutes have passed."
                ],
                "Loving-Kindness Meditation (LKM)": [
                    "Sit upright and close your eyes. <br> Bring to mind someone you care about deeply. Imagine them in front of you. <br>Silently repeat: ‘May you be happy. May you be healthy. May you be safe. May you live with ease.’ <br> After a few minutes, extend these wishes to yourself: ‘May I be happy. May I be healthy. May I be safe. May I live with ease.’ <br>Then expand outward — to friends, to people you don’t know, even to people you find difficult. <br> Finally, include all living beings everywhere. <br> Keep repeating the phrases with a sincere and gentle heart for 15 minutes.<br>If the mind wanders, return to the feeling of kindness and continue."
                ],
                "Mantra Meditation (MM)": [
                    "Sit comfortably with eyes closed. <br>Begin chanting the sound ‘OM’ aloud, slowly and steadily. Let the vibration resonate in your chest and head. <br>Continue chanting for about 10 minutes, keeping a natural rhythm. <br>After 10 minutes, switch to repeating the mantra silently in your mind for the remaining 5 minutes. <br>If your attention drifts, gently return to the mantra, aloud or silent. <br>Allow yourself to be fully absorbed in the sound and vibration until 15 minutes have passed."
                ]
            }

            for step in steps.get(chosen["type"], []):
                # st.markdown(step)
                st.markdown(f"""
                <div style="
                    border: none;
                    border-radius: 12px;
                    padding: 15px 18px;
                    margin: 10px 0;
                    background: #eafafe;
                    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
                    border: 1px solid #cce4ff;
                "> 
                    <strong>Meditation Instruction: <br> </strong> {step}
                    </div>
                    """, unsafe_allow_html=True)

            if "meditation_running" not in st.session_state:
                st.session_state.meditation_running = False
            if "meditation_paused" not in st.session_state:
                st.session_state.meditation_paused = False
            if "meditation_completed" not in st.session_state:
                st.session_state.meditation_completed = False
            if "meditation_start_ts" not in st.session_state:
                st.session_state.meditation_start_ts = None
            if "meditation_elapsed" not in st.session_state:
                st.session_state.meditation_elapsed = 0
            if "meditation_total" not in st.session_state:
                st.session_state.meditation_total = 1 * 60  # 15 minutes
            if "audio_played" not in st.session_state:  # NEW: Track audio play state
                st.session_state.audio_played = False

            # --- Scoped CSS (only for this section) ---
            st.markdown("""
            <style>
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"]{
                display: flex !important;
                justify-content: center !important;
                align-items: center !important;
                gap: 3rem !important;
                margin: 1.2rem 0 1.5rem 0 !important;
            }
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"] div.stButton > button {
                width: 95px;
                height: 95px;
                border-radius: 999px;
                font-size: 16px;
                font-weight: 700;
                border: 2px solid #cfe8ff;
                background: linear-gradient(145deg, #f7fbff, #e9f3ff);
                color: #143e62;
                box-shadow: 0 4px 10px rgba(0,0,0,0.08);
                transition: transform .15s ease, box-shadow .15s ease;
            }
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"] div.stButton > button:hover {
                transform: translateY(-2px) scale(1.03);
                border-color: #8cc6ff;
                box-shadow: 0 8px 18px rgba(0,0,0,0.10);
            }
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"] div.stButton:nth-of-type(1) > button {
                border-color: #8de79b;
                background: linear-gradient(145deg, #f6fff8, #e7ffe9);
                color: #0b5d1d;
            }
            #timer-controls-anchor + div[data-testid="stHorizontalBlock"] div.stButton:nth-of-type(2) > button {
                border-color: #ffc38a;
                background: linear-gradient(145deg, #fff8f0, #ffefdd);
                color: #7a3e00;
            }
            </style>
            """, unsafe_allow_html=True)

            # --- Timer UI ---
            if not st.session_state.meditation_completed:
                st.caption("Remember to turn on volume to hear the soft alert after 15 minutes end.")
                st.markdown('<div id="timer-controls-anchor"></div>', unsafe_allow_html=True)
                cols = st.columns([1.5, 1.5, 1, 1.5, 1.5])

                with cols[1]:
                    start_clicked = st.button("▶️ Start Timer", key="start_meditation")
                with cols[3]:
                    pause_clicked = st.button("⏸️ Pause Timer", key="pause_meditation")

                # --- Start logic ---
                if start_clicked:
                    if not st.session_state.meditation_running:
                        if st.session_state.meditation_paused:
                            # Resume
                            st.session_state.meditation_start_ts = time.time() - st.session_state.meditation_elapsed
                        else:
                            # Start new
                            st.session_state.meditation_start_ts = time.time()
                            st.session_state.meditation_elapsed = 0
                        st.session_state.meditation_running = True
                        st.session_state.meditation_paused = False
                        st.rerun()

                # --- Pause logic ---
                if pause_clicked and st.session_state.meditation_running:
                    st.session_state.meditation_running = False
                    st.session_state.meditation_paused = True
                    st.session_state.meditation_elapsed = time.time() - st.session_state.meditation_start_ts
                    st.rerun()

                # --- Timer Display ---
                if st.session_state.meditation_running:
                    placeholder = st.empty()
                    prog = st.progress(0)
                    total = st.session_state.meditation_total
                    start_ts = st.session_state.meditation_start_ts

                    elapsed = 0
                    while elapsed < total and st.session_state.meditation_running:
                        elapsed = int(time.time() - start_ts)
                        if elapsed > total:
                            elapsed = total
                        remaining = total - elapsed
                        mm = remaining // 60
                        ss = remaining % 60
                        placeholder.markdown(f"### ⏳ Time remaining: **{mm:02d}:{ss:02d}**")
                        prog.progress(int(100 * elapsed / total))
                        time.sleep(1)
                        st.session_state.meditation_elapsed = elapsed

                    # --- Timer completion check ---
                    if elapsed >= total and st.session_state.meditation_running:
                        st.session_state.meditation_running = False
                        st.session_state.meditation_paused = False
                        st.session_state.meditation_completed = True
                        st.session_state.meditation_elapsed = total
                        st.session_state.audio_played = False  # Reset audio state
                        st.rerun()

                elif st.session_state.meditation_paused:
                    remaining = st.session_state.meditation_total - int(st.session_state.meditation_elapsed)
                    mm = remaining // 60
                    ss = remaining % 60
                    st.markdown(f"### ⏸️ Paused — Time remaining: **{mm:02d}:{ss:02d}**")
                else:
                    # Initial state - show total time
                    mm = st.session_state.meditation_total // 60
                    ss = st.session_state.meditation_total % 60
                    st.markdown(f"### ⏰ Ready — Total time: **{mm:02d}:{ss:02d}**")

            # --- Meditation completed: show result ---
            else:
                st.success(
                    "✨ Meditation complete! You may now gently open your eyes and proceed to the Post-Meditation EEG Collection.")
                # Audio with manual play button (browsers block autoplay)

                st.markdown("""
                <audio autoplay style="display: none;">
                    <source src="https://actions.google.com/sounds/v1/water/fountain_water_bubbling.ogg" type="audio/ogg">
                </audio>
                """, unsafe_allow_html=True)

                st.markdown("")
                st.markdown("")

                st.header("4️⃣ Post-Meditation EEG Collection")

                # Post-meditation filename
                post_filename = "after_eeg_recording.csv"
                post_full_path = os.path.join(folder_path, post_filename)
                post_duration = 300 if st.session_state["baseline_mode"] == "Research" else 180
                # post_duration = 60 if st.session_state["baseline_mode"] == "Research" else 30


                # Only show post-meditation recording if not completed yet
                if not st.session_state.get('post_meditation_recording_completed', False):
                    st.caption("Please put your Muse 2 Headband back on and ensure Bluetooth is connected")

                    # --- Post-Meditation: Search for Muse Device ---
                    post_search_ph = st.empty()


                    def render_post_search_button():
                        with post_search_ph.container():
                            if st.button("🔍 Search for Muse Device & Start Stream", key="post_btn_search"):
                                with st.spinner("Searching for Muse Device..."):
                                    process = subprocess.Popen([sys.executable, "start_stream.py"])
                                    time.sleep(8)
                                    found = any(resolve_stream('type', 'EEG') for _ in range(20) if not time.sleep(1))
                                    if found:
                                        st.session_state['post_stream_process'] = process
                                        st.session_state.post_stream_found = True
                                        render_post_connected_disabled()
                                        st.success("Muse device found. You can now record Post-Meditation EEG.")
                                        st.rerun()
                                    else:
                                        process.terminate()
                                        st.error("Please ensure Muse Headband & Bluetooth is on and try again.")


                    def render_post_connected_disabled():
                        with post_search_ph.container():
                            st.button("✅ Muse Device Connected", disabled=True, key="post_btn_connected")


                    # initial render for post-meditation search
                    if st.session_state.get("post_stream_found", False):
                        render_post_connected_disabled()
                    else:
                        render_post_search_button()

                    # Post-meditation instructions
                    with st.expander("Post-Meditation EEG Recording Instructions", expanded=False):
                        st.markdown("""
                        ### 🧠 Post-Meditation Recording
                        Please follow the same procedure as the pre-meditation recording:
                        - Sit upright with back support
                        - Keep your body still and relaxed
                        - Focus on a fixed point on the screen
                        - Maintain natural breathing
                        - Avoid excessive blinking or movement
                        """)

                    # --- Post-Meditation: Record EEG ---
                    post_record_ph = st.empty()


                    def render_post_start_record():
                        with post_record_ph.container():
                            if st.button("🎙 Start Post-Meditation EEG Recording", key="post_btn_record"):
                                st.session_state.post_recording_in_progress = True
                                render_post_recording_disabled()
                                st.rerun()


                    def render_post_recording_disabled():
                        with post_record_ph.container():
                            st.button("🎙 Recording in Progress…", disabled=True, key="post_btn_recording")


                    def render_post_recording_ended():
                        with post_record_ph.container():
                            st.button("✅ Recording Completed", disabled=True, key="post_btn_recordingended")


                    # Post-meditation recording logic
                    if st.session_state.get("post_stream_found") and 'post_stream_process' in st.session_state:
                        if st.session_state.get("post_recording_finished"):
                            render_post_connected_disabled()
                            render_post_recording_ended()
                            st.toast("🎧 Post-meditation EEG recording completed!")
                        elif st.session_state.get("post_recording_in_progress"):
                            render_post_recording_disabled()

                            # Start the actual recording process
                            with st.spinner(f"{int(post_duration/60)} minutes EEG recording will start upon seeing dot.."):
                                time.sleep(5)

                            overlay = show_fixation_overlay()

                            try:
                                record(post_duration, filename=post_full_path)
                                if os.path.exists(post_full_path):
                                    st.toast(f"Post-meditation recording complete! File saved: `{post_full_path}`")
                                    st.session_state['after_eeg_file'] = post_full_path
                                    st.session_state.post_recording_finished = True
                                    st.session_state.post_meditation_recording_completed = True
                                    st.session_state.post_recording_in_progress = False
                                    overlay.empty()

                                    if st.session_state['post_stream_process'].poll() is None:
                                        st.session_state['post_stream_process'].terminate()
                                        st.toast("Muse stream terminated.")
                                    del st.session_state['post_stream_process']
                                    st.session_state.post_stream_found = False
                                    st.rerun()
                                else:
                                    st.error("Post-meditation recording failed. Please try again.")
                                    overlay.empty()
                            except Exception as e:
                                st.error(f"Post-meditation recording error: {e}")
                                overlay.empty()
                        else:
                            render_post_start_record()

                else:
                    st.success("✅ Post-Meditation EEG Recording Completed!")

                    # Show post-meditation analysis results
                    if st.session_state.get('after_eeg_file') and os.path.exists(st.session_state['after_eeg_file']):
                        st.markdown("")
                        st.header("5️⃣ Results")
                        # disclaimer
                        st.markdown(
                            '<p style="text-align: left; color: #6c757d; font-size: 0.8rem;">'
                            '⚠️ EEG-based interpretations vary by individual, mood, and context — use results as guidance, not diagnosis.'
                            '</p>',
                            unsafe_allow_html=True
                        )

                        try:
                            # Analyze post-meditation EEG data
                            post_result = summarize_one_file(st.session_state['after_eeg_file'])
                            post_rel = post_result.rel_overall_pct

                            # Get pre-meditation data for comparison
                            pre_result = summarize_one_file(st.session_state['before_eeg_file'])
                            pre_rel = pre_result.rel_overall_pct

                            # Create comparison table
                            comparison_rows = []
                            focus_bands = ["alpha", "beta", "theta"]
                            for band in focus_bands:
                                pre_val = pre_rel.get(band, np.nan)
                                post_val = post_rel.get(band, np.nan)
                                change = post_val - pre_val if not np.isnan(pre_val) and not np.isnan(
                                    post_val) else np.nan

                                comparison_rows.append({
                                    "Band": band.capitalize(),
                                    "Before (%)": round(pre_val, 2) if not np.isnan(pre_val) else "N/A",
                                    "After (%)": round(post_val, 2) if not np.isnan(post_val) else "N/A",
                                    "Change": f"{change:+.2f}%" if not np.isnan(change) else "N/A"
                                })

                            # st.subheader("📈 Interpretation")

                            change_directions = {}
                            for row in comparison_rows:
                                band = row["Band"].lower()
                                change_str = row["Change"]

                                if change_str != "N/A":
                                    change_val = float(change_str.replace('%', '').replace('+', ''))

                                    # determine up / down / steady
                                    if change_val > 1:
                                        change_directions[band] = "up"
                                    elif change_val < -1:
                                        change_directions[band] = "down"
                                    else:
                                        change_directions[band] = "steady"

                                    # Per-band message
                                    if band == "alpha":
                                        if change_val > 1:
                                            st.success(
                                                f"**{row['Band']} increased** — You’ve achieved greater relaxation and inner calm 🌿")
                                        elif change_val < -1:
                                            st.info(
                                                f"**{row['Band']} decreased** — You’ve achieved a more alert and engaged state ⚡")
                                    elif band == "beta":
                                        if change_val > 1:
                                            st.info(
                                                f"**{row['Band']} increased** — You’ve achieved heightened alertness and concentration 🎯")
                                        elif change_val < -1:
                                            st.success(
                                                f"**{row['Band']} decreased** — You’ve achieved a calmer, less stressed mental state 🕊️")
                                    elif band == "theta":
                                        if change_val > 1:
                                            st.success(
                                                f"**{row['Band']} increased** — You’ve achieved deeper meditative and introspective focus 💭")
                                        elif change_val < -1:
                                            st.info(
                                                f"**{row['Band']} decreased** — You’ve achieved better external attentiveness 🌞")

                            # -----------------------------
                            # 🌿 Combined Pattern Interpretation
                            # -----------------------------
                            alpha, beta, theta = (
                                change_directions.get("alpha", "steady"),
                                change_directions.get("beta", "steady"),
                                change_directions.get("theta", "steady"),
                            )

                            pattern_messages = {
                                ("down", "down",
                                 "up"): "🌙 Congratulations! You’ve entered a deeply introspective and restful state — your mind released tension and turned inward.",
                                ("up", "down",
                                 "up"): "💫  Congratulations! You’ve reached a calm but aware state — peaceful relaxation with gentle focus.",
                                ("down", "up",
                                 "down"): "⚡  Congratulations! You’ve sharpened focus and alertness — your mind feels active and engaged.",
                                ("up", "down",
                                 "down"): "🌿  Congratulations! You’ve achieved calm focus — relaxed yet attentive, a balanced meditative state.",
                                ("down", "down",
                                 "down"): "😌  Congratulations! Your brain settled into a low-activity rhythm — possibly deep rest or mild fatigue.",
                                ("up", "up",
                                 "down"): "☀️  Congratulations! You’ve reached mindful engagement — calm yet cognitively alert.",
                                ("down", "up",
                                 "up"): "💭  Congratulations! You’ve entered reflective thought — alertness and introspection intertwined."
                            }

                            combo_msg = pattern_messages.get((alpha, beta, theta))
                            if combo_msg:
                                # st.subheader("🧘 Overall Brainwave Pattern")
                                st.markdown(f"<p style='font-size:1.3rem;'>{combo_msg}</p>", unsafe_allow_html=True)

                            # -----------------------------
                            # 🎯 Overall Effectiveness
                            # -----------------------------
                            # st.subheader("🎯 Meditation Effectiveness")
                            st.markdown("")
                            positive_changes = sum(
                                1 for row in comparison_rows if row["Change"] != "N/A" and "+" in row["Change"])
                            total_comparable = sum(1 for row in comparison_rows if row["Change"] != "N/A")

                            if total_comparable > 0:
                                effectiveness = (positive_changes / total_comparable) * 100
                                if effectiveness >= 70:
                                    st.success(
                                        "**Great session!** Your brainwave patterns show strong positive changes 🌟")
                                elif effectiveness >= 40:
                                    st.info("**Good session!** Some positive effects detected — keep up the practice ✨")
                                else:
                                    st.info("**Session completed!** Every session trains your mind toward balance 💪")

                            st.balloons()

                            with st.expander(
                                    "Want to know more in-depth about the result? Click here",
                                    expanded=False):

                                st.subheader("📊 Before vs After Meditation Comparison")

                                st.dataframe(comparison_rows, use_container_width=True)


                        except Exception as e:
                            st.error(f"Error analyzing post-meditation data: {e}")
                            st.info("Please ensure the post-meditation recording file was created successfully.")
                    else:
                        st.warning("Post-meditation recording file not found. Please complete the recording first.")