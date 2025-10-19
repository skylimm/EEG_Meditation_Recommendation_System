import sys
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

col1, col2, col3 = st.columns([1,2,1])
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
.hw-step { font-family:'Karla', sans-serif; font-weight:700; color:#0F172A; font-size:16px; margin:6px 0 2px; text-align:center;}
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
      <div class="hw-step">1. Record & <br> Upload Pre-EEG </div>
      <div class="hw-text">Wear the Muse headband & upload a recorded EEG — we capture your resting brain activity to see your current state.</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">📊</div>
      <div class="hw-step">2. Recommendations</div>
      <div class="hw-text">Receive the meditation technique that best balances your current brain state — and choose what you’d like to focus on and prioritize, in real time.</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="hw-card">
      <div class="hw-icon">✨</div>
      <div class="hw-step">3. Post-EEG & Results</div>
      <div class="hw-text">After meditation, upload your post-session EEG — see how your brainwaves changed and how your mind responded.</div>
    </div>
    """, unsafe_allow_html=True)

# initializations
if "meditation_selected" not in st.session_state:
    st.session_state.meditation_selected = False
# Add these with your other session state initializations
if "after_eeg_file" not in st.session_state:
    st.session_state.after_eeg_file = None



# Step 1: User input and folder creation
st.markdown("")
st.markdown("")
folder_path = get_user_folder()
filename = "before_eeg.csv"
duration = 30

# -------------------------------
# 0️⃣ EEG Setup & Local Recording Guide
# -------------------------------
st.markdown("")
with st.expander("0️⃣ Setup for Local EEG Collection (You can skip if you're already familiar and done it before)", expanded=False):
    st.markdown("### Setup for Local EEG Collection (Optional)")
    st.markdown("To collect your **EEG data** for later upload and meditation recommendations, follow these steps carefully:")

    st.markdown("""    
    1. **Install Requirements**
       - [Download Python 3.8](https://www.python.org/downloads/release/python-3810/)
       - Open a terminal and install dependencies:
         ```bash
         pip install muselsl pylsl numpy pandas
         ```
    """)
    st.markdown("""
    2. **Pair Your Muse Headband**
       - Make sure your **Muse 2 HeadBand** is powered on and Bluetooth is on.""")

    st.markdown("""
    3. **Download and Unzip EEG Collection Files**
       - Download the full set of required scripts by clicking the button below.
       - Unzip them into a folder (e.g., `Downloads/EEG_Collector`).

       - After unzipping, open a terminal and navigate there:
       ```bash
       cd Downloads/EEG_Collector
       ```""")

    import zipfile
    from io import BytesIO

    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        zipf.write("eeg_data_collection.py")
        zipf.write("record_eeg.py")
        zipf.write("start_stream.py")
    buffer.seek(0)

    st.download_button(
        label="⬇️ Download EEG Data Collection Kit (ZIP)",
        data=buffer,
        file_name="EEG_Collector_Package.zip",
        mime="application/zip"
    )
    st.markdown("")

    st.markdown("""    
    4. **Start Recording**
       - Read the following instructions on proper Muse 2 headband placement and important notes for the 5-minute EEG recording session.
       - Ensure you understand the flow of EEG data collection before running the command below.
       - Run the following command to collect EEG data for 5 minutes:
         ```bash
         python eeg_data_collection.py
         ```
       - This will generate a file like `EEG_recording_YYYYMMDD_HHMMSS.csv` in your Downloads folder.
    """)

    st.markdown("""    
    5. **Return to this app**
       - Upload your recorded file under **“Pre-Meditation EEG Upload”** or **“Post-Meditation EEG Upload”**.
    """)



    # --- Wearing and recording guidance ---
    st.divider()
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(
            "https://pyxis.nymag.com/v1/imgs/e09/a32/9356a8db59e0b3880c8b8d08bb7c165461-09-muse.2x.rhorizontal.w700.jpg",
            caption="Correct EEG headband placement",
            width=200,
        )
    with col2:
        st.markdown("""
        - Position the headband across your **forehead and behind the ears**.
        - Tuck away any hair blocking electrode contact.
        - Sit upright, relaxed, and in a quiet room with stable lighting.
        - Avoid lying down — the headband may shift.
        """)

    st.markdown(                    """
                    ### 🎧 During EEG Recording
                    - Sit **quietly with eyes open**.  
                    - Fix your gaze on a **small point** (e.g., dot or cross) on the screen.  
                    - Keep your **body as still as possible**:
                      - **Avoid blinking excessively**, clenching your jaw, or moving your head.  
                    - Maintain **natural, gentle breathing** throughout the session.
                    - **5-Minute EEG Recording starts upon script run, so ensure you got the dots ready to prevent...**
                    """)

    st.info("✅ When ready, run the local scripts to collect EEG data before uploading it below.")


st.markdown("")


if folder_path:
    st.subheader("1️⃣ Pre-Meditation EEG Upload")

    # st.markdown("Please upload your pre-meditation EEG file recorded earlier using the local recording script.")

    uploaded_pre = st.file_uploader("📂 Please upload your pre-meditation EEG file recorded earlier using the local recording script.", type=["csv"], key="before_upload")

    if uploaded_pre:
        pre_path = os.path.join(folder_path, filename)
        with open(pre_path, "wb") as f:
            f.write(uploaded_pre.getbuffer())
        st.success("✅ Pre-Meditation EEG file uploaded successfully.")
        st.session_state['before_eeg_file'] = pre_path
        st.session_state.recording_finished = True


    if st.session_state.get('before_eeg_file') and st.session_state.get('recording_finished'):
        full_path = st.session_state['before_eeg_file']
        # After recording a new user's PRE file:

        result = summarize_one_file(full_path)  # your recorded pre file
        rel = result.rel_overall_pct  # {'delta': %, 'theta': %, ...}


        @st.cache_data
        def load_cohort_stats(path="outputs/cohort_baseline.json"):
            with open(path, "r") as f:
                return json.load(f)  # {band: {mean, sd, p25, p75, ...}}


        cohort = load_cohort_stats("../outputs/cohort_baseline.json")


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
            st.markdown("<p style='text-align:center; font-size: 27px; font-weight: bold'>🧩 Quick Brainwave Summary 🧩</p>", unsafe_allow_html=True)
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
                st.markdown('<p class="big-font">Based on your brainwave activity, please choose what you’d like to prioritise:</p>', unsafe_allow_html=True)
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

            st.markdown(f"<h4 style='text-align: center;'><span style='color: #000000;'>Recommended Meditation:</span><br><span style='color: #0277bd;'>{chosen['type']}</span></h4>", unsafe_allow_html=True)
            st.markdown(f"<p style='font-style:italic;'> <strong>Why this fits you: </strong> {chosen['desc']}</p>", unsafe_allow_html=True)

            st.markdown("")
            st.header("3️⃣ Meditation Session (15 Minutes)")
            st.write("Before beginning your session, please take a moment to read the meditation instructions carefully. \n Once ready, press **Start Meditation Timer** to begin your 15-minute meditation practice. \n When the timer ends, a soft alert will play — you can then gently open your eyes and proceed to the **Post-Meditation EEG Recording** step.")


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
                st.success("✨ Meditation complete! You may now gently open your eyes and proceed to the Post-Meditation EEG Collection.")
                # Audio with manual play button (browsers block autoplay)

                st.markdown("""
                <audio autoplay style="display: none;">
                    <source src="https://actions.google.com/sounds/v1/water/fountain_water_bubbling.ogg" type="audio/ogg">
                </audio>
                """, unsafe_allow_html=True)

                st.markdown("")
                st.markdown("")

                st.header("4️⃣ Post-Meditation EEG Upload")


                uploaded_post = st.file_uploader("📂 Please upload your pre-meditation EEG file recorded earlier using the local recording script.", type=["csv"],
                                                 key="after_upload")

                if uploaded_post:
                    post_path = os.path.join(folder_path, "after_eeg.csv")
                    with open(post_path, "wb") as f:
                        f.write(uploaded_post.getbuffer())
                    st.success("✅ Post-Meditation EEG file uploaded successfully.")
                    st.session_state['after_eeg_file'] = post_path
                    st.session_state.post_meditation_recording_completed = True


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

                            # -----------------------------
                            # 🌿 Combined Pattern Interpretation
                            # -----------------------------
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
                                # st.subheader("🧘 Overall Brainwave Pattern")
                                st.markdown(f"<p style='font-size:1.3rem;'>{combo_msg}</p>", unsafe_allow_html=True)



                            # -----------------------------
                            # 🎯 Overall Effectiveness
                            # -----------------------------
                            # st.subheader("🎯 Meditation Effectiveness")
                            st.markdown("")
                            positive_changes = sum(1 for row in comparison_rows if row["Change"] != "N/A" and "+" in row["Change"])
                            total_comparable = sum(1 for row in comparison_rows if row["Change"] != "N/A")

                            if total_comparable > 0:
                                effectiveness = (positive_changes / total_comparable) * 100
                                if effectiveness >= 70:
                                    st.success("**Great session!** Your brainwave patterns show strong positive changes 🌟")
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