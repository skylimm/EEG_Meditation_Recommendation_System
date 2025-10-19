
# Enhanced version with progressive tab locking and upload disable during Live Recording
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
@import url('https://fonts.googleapis.com/css2?family=Karla:wght@500;700&display=swap');
.header-wrap { text-align:center; }
.header-sub { font-family:'Karla', sans-serif; font-size:17px; color:#475569; line-height:1.3; }
</style>
<div class="header-wrap">
  <div class="header-sub"><b>EEG Meditation Recommender</b> — analyze brainwaves and discover which meditation fits you best.</div>
</div>
""", unsafe_allow_html=True)

# =========================
# Session States
# =========================
ss = st.session_state
defaults = {
    "progress_step": 1,   # 1=Pre, 2=Analysis, 3=Post, 4=Results
    "lock_upload": False,
    "before_eeg_file": None,
    "after_eeg_file": None,
    "recording_finished": False,
    "post_meditation_recording_completed": False,
    "meditation_selected": False,
    "meditation_completed": False
}
for k, v in defaults.items():
    ss.setdefault(k, v)

# =========================
# Helper
# =========================
def disable_future_tabs(current_max):
    """Render warnings for locked tabs."""
    tab_labels = ["1️⃣ Pre Collection", "2️⃣ Brainwave Analysis & Meditation", 
                  "3️⃣ Post Collection", "4️⃣ Results & Interpretation"]
    tabs = st.tabs(tab_labels)
    for i, t in enumerate(tabs, start=1):
        if i > current_max:
            with t:
                st.warning("🔒 This section is locked. Please complete the previous step first.")
                st.stop()
    return tabs

# =========================
# Folder setup
# =========================
folder_path = get_user_folder()
pre_filename = "before_eeg.csv"
post_filename = "after_eeg_recording.csv"
record_duration = 30

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = disable_future_tabs(ss.progress_step)

# ---------- TAB 1 ----------
with tab1:
    st.header("1️⃣ Pre Collection")
    if not folder_path:
        st.info("Enter your name above first.")
    else:
        pre_tab_live, pre_tab_upload = st.tabs(["🎧 Live Recording", "📂 Upload File"])

        with pre_tab_live:
            if st.button("🔍 Search for Muse Device & Start Stream", key="btn_search"):
                ss.lock_upload = True
                st.success("Simulated Muse search started... (Live mode active)")

            if st.button("🎙 Start Pre Recording", key="btn_record"):
                time.sleep(1)
                file_path = os.path.join(folder_path, pre_filename)
                with open(file_path, "w") as f: f.write("dummy EEG data")
                ss.before_eeg_file = file_path
                ss.recording_finished = True
                st.success("✅ Pre EEG recorded successfully!")

            if ss.recording_finished:
                if st.button("➡️ Proceed to Brainwave Analysis"):
                    ss.progress_step = 2
                    st.rerun()

        with pre_tab_upload:
            if ss.lock_upload:
                st.warning("🔒 Upload disabled during Live Recording session.")
                st.stop()

            uploaded = st.file_uploader("📂 Upload pre EEG CSV", type=["csv"], key="pre_upload")
            if uploaded:
                path = os.path.join(folder_path, pre_filename)
                with open(path, "wb") as f:
                    f.write(uploaded.getbuffer())
                ss.before_eeg_file = path
                ss.recording_finished = True
                st.success("✅ Uploaded successfully!")
                if st.button("➡️ Proceed to Brainwave Analysis (Upload)"):
                    ss.progress_step = 2
                    st.rerun()

# ---------- TAB 2 ----------
with tab2:
    st.header("2️⃣ Brainwave Analysis & Meditation")
    if ss.progress_step < 2:
        st.warning("🔒 Locked until Pre Collection is complete.")
        st.stop()

    if ss.before_eeg_file:
        st.success(f"Loaded: {ss.before_eeg_file}")
        st.markdown("Simulated Brainwave Analysis + Meditation Timer...")

        if st.button("✅ Finish Meditation"):
            ss.meditation_completed = True
            st.success("Meditation completed!")

        if ss.meditation_completed:
            if st.button("➡️ Proceed to Post Collection"):
                ss.progress_step = 3
                st.rerun()
    else:
        st.warning("Please complete Pre Collection first.")

# ---------- TAB 3 ----------
with tab3:
    st.header("3️⃣ Post Collection")
    if ss.progress_step < 3:
        st.warning("🔒 Locked until Meditation completed.")
        st.stop()

    post_tab_live, post_tab_upload = st.tabs(["🎧 Live Recording", "📂 Upload File"])

    with post_tab_live:
        if st.button("🎙 Start Post Recording", key="post_live_record"):
            path = os.path.join(folder_path, post_filename)
            with open(path, "w") as f: f.write("dummy POST EEG data")
            ss.after_eeg_file = path
            ss.post_meditation_recording_completed = True
            st.success("✅ Post EEG recorded successfully!")

        if ss.post_meditation_recording_completed:
            if st.button("➡️ View Results"):
                ss.progress_step = 4
                st.rerun()

    with post_tab_upload:
        uploaded_post = st.file_uploader("📂 Upload post EEG CSV", type=["csv"], key="post_upload")
        if uploaded_post:
            path = os.path.join(folder_path, "after_eeg.csv")
            with open(path, "wb") as f:
                f.write(uploaded_post.getbuffer())
            ss.after_eeg_file = path
            ss.post_meditation_recording_completed = True
            st.success("✅ Uploaded successfully!")
            if st.button("➡️ View Results (Upload)"):
                ss.progress_step = 4
                st.rerun()

# ---------- TAB 4 ----------
with tab4:
    st.header("4️⃣ Results & Interpretation")
    if ss.progress_step < 4:
        st.warning("🔒 Locked until Post Collection completed.")
        st.stop()

    if ss.after_eeg_file and ss.before_eeg_file:
        st.success("✅ Displaying results comparison...")
        st.write("Simulation: Showing before vs after EEG delta.")
    else:
        st.warning("Missing files for comparison.")
