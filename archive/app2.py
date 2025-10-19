import sys
import streamlit as st
import os
import time
import subprocess
from muselsl import record
from pylsl import resolve_stream
from user_input import get_user_folder
from archive.eeg_utils2 import (
    EEG_CHANNELS, BANDS,
    estimate_fs, preprocess,
    sliding_fft_bandpower, dwt_band_envelopes,
    per_channel_summary, overall_summary,
    plot_band_timeseries_fft, plot_band_timeseries_dwt,
    aggregate_fft_overall, aggregate_dwt_overall,
    apply_threshold_labels,
    label_relative_value, REL_RULES_DEFAULT,
    plot_overall_abs_prepost, plot_overall_rel_prepost, plot_overall_rel_change, plot_rel_change_heatmap,
    plot_band_timeseries_dwt_compare, plot_band_timeseries_fft_compare,

    plot_percent_change_lines, percent_change_timeseries_fft
)

# imports for formatting
from archive.eeg_utils2 import (
    format_overall_summary, format_single_channel_summary
)
import numpy as np
import pandas as pd

# initializations
if "analysis_ready" not in st.session_state:
    st.session_state.analysis_ready = False
if "per_ch" not in st.session_state:
    st.session_state.per_ch = {}
if "overall" not in st.session_state:
    st.session_state.overall = {}
if "method_used" not in st.session_state:
    st.session_state.method_used = "fft"  # or "dwt"


st.title("🧠 EEG Meditation System")

# Step 1: User input and folder creation
folder_path = get_user_folder()
filename = "before_eeg.csv"
duration = 5

if folder_path:
    full_path = os.path.join(folder_path, filename)
    st.header("1️⃣ Pre-Meditation EEG Collection")

    # Step 2: Search for Muse & Start stream
    if st.button("🔍 Search for Muse Device & Start Stream"):
        with st.spinner("Starting Muse LSL stream..."):
            process = subprocess.Popen([sys.executable, "start_stream.py"])
            time.sleep(8)  # Let stream stabilize

            # Check for EEG stream
            found = False
            for _ in range(10):
                streams = resolve_stream('type', 'EEG')
                if streams:
                    found = True
                    break
                time.sleep(1)

            if found:
                st.session_state['stream_process'] = process
                st.success("Muse device found. You can now record EEG.")
            else:
                process.terminate()
                st.error("Muse device not found. Please check connection and retry.")

    # Step 3: Record EEG (only if stream started)
    if 'stream_process' in st.session_state:
        if st.button("🎙 Start Pre-Meditation EEG Recording"):
            with st.spinner(f"Recording EEG for {duration} seconds..."):
                try:
                    record(duration, filename=full_path)
                    if os.path.exists(full_path):
                        st.success(f"Recording complete! File saved: `{full_path}`")
                        st.session_state['before_eeg_file'] = full_path
                    else:
                        st.error("Recording failed. File not found.")
                except Exception as e:
                    st.error(f"Recording error: {e}")
                finally:
                    # Terminate the stream process
                    if st.session_state['stream_process'].poll() is None:
                        st.session_state['stream_process'].terminate()
                        st.success("Muse stream terminated.")
                    del st.session_state['stream_process']
    else:
        st.info("Please search for Muse device first.")

    # Step 4: Band Power Analysis: per-channel + overall
    if st.session_state.get('before_eeg_file'):
        st.header("2️⃣ Band Summary")
        with st.sidebar:
            st.header("Analysis Options")
            method = st.radio("Method", ["FFT", "DWT"], index=0)

            if method == "FFT":
                win_sec = st.number_input("Window (sec)", 0.5, 10.0, 2.0, 0.5)
                step_sec = st.number_input("Step (sec)", 0.1, 5.0, 0.5, 0.1)
                window = st.selectbox("FFT Window", ["hann", "hamming", "blackman"], index=0)
            else:
                wavelet = st.text_input("Wavelet", "db4")
                levels = st.number_input("DWT Levels", 2, 10, 6, 1)
                smooth_sec = st.number_input("Envelope Smooth (sec)", 0.05, 1.0, 0.25, 0.05)

        eeg_file = st.session_state['before_eeg_file']

        if st.button("🧮 Run Analysis"):
            try:
                df = pd.read_csv(eeg_file)
                per_ch, fs = per_channel_summary(
                    df,
                    method="fft" if method == "FFT" else "dwt",
                    bands=BANDS,
                    win_sec=win_sec if method == "FFT" else 2.0,
                    step_sec=step_sec if method == "FFT" else 0.5,
                    window=window if method == "FFT" else "hann",
                    wavelet=wavelet if method == "DWT" else "db4",
                    levels=int(levels) if method == "DWT" else 6,
                    smooth_sec=smooth_sec if method == "DWT" else 0.25
                )
                overall = overall_summary(per_ch, method="fft" if method == "FFT" else "dwt", bands=BANDS)

                st.session_state["per_ch"] = per_ch
                st.session_state["method_used"] = "fft" if method == "FFT" else "dwt"

                st.success(f"Estimated fs ≈ {fs:.1f} Hz")

                method_used = "fft" if method == "FFT" else "dwt"

                st.session_state.per_ch = per_ch
                st.session_state.overall = overall
                st.session_state.method_used = "fft" if method == "FFT" else "dwt"
                st.session_state.analysis_ready = True


            except Exception as e:
                st.error(f"Analysis failed: {e}")

        if st.session_state.analysis_ready:

            st.subheader("Overall summary (across channels)")
            st.markdown(format_overall_summary(st.session_state.overall, st.session_state.method_used))

            st.subheader("Per-channel summary")
            # st.markdown(format_per_channel_summary(st.session_state.per_ch, st.session_state.method_used))
            channel_list = list(st.session_state.per_ch.keys())
            if not channel_list:
                st.info("No channel data found.")
            else:
                selected_channel = st.selectbox(
                    "Select a channel to view summary",
                    options=channel_list,
                    key="per_ch_select",  # key ensures the widget’s choice persists
                )
                with st.expander(selected_channel, expanded=True):
                    block = format_single_channel_summary(
                        st.session_state.per_ch[selected_channel],
                        st.session_state.method_used
                    )
                    st.markdown(f"```\n{block}\n```")

        else:
            st.info("Click **Run analysis** to generate band summaries.")



        with st.expander("🔎 Raw summary data"):
            st.subheader("Overall summary (across channels)")
            st.json(st.session_state.overall)

            st.subheader("Per-channel summary")
            st.json(st.session_state.per_ch)



    # Step 5: Visualization
    if st.session_state.get('before_eeg_file'):
        st.header("3️⃣ Band Time-series Visualization 📊")
        eeg_file = st.session_state['before_eeg_file']
        df = pd.read_csv(eeg_file)
        channels = [c for c in EEG_CHANNELS if c in df.columns]
        viz_options = ["Overall (across all channels)"] + channels
        choice = st.selectbox("Channel", viz_options)

        if st.button("📈 Plot"):
            try:
                # fs + preprocessing
                fs = 256.0
                if "timestamps" in df.columns:
                    fs, _ = estimate_fs(df["timestamps"].to_numpy())

                if choice == "Overall (across all channels)":
                    # ---- OVERALL plots (mean across channels) ----
                    if method == "FFT":
                        times, band_abs = aggregate_fft_overall(
                            df, channels, fs, BANDS, win_sec, step_sec, window, preprocess
                        )
                        fig = plot_band_timeseries_fft(
                            times, band_abs, BANDS, title_suffix=" – Overall (mean across channels)"
                        )
                        st.pyplot(fig)
                    else:
                        t_axis, envelopes = aggregate_dwt_overall(
                            df, channels, fs, BANDS, wavelet=wavelet, levels=int(levels),
                            smooth_sec=smooth_sec, preprocess=preprocess
                        )
                        fig = plot_band_timeseries_dwt(
                            t_axis, envelopes, BANDS, title_suffix=" – Overall (mean across channels)"
                        )
                        st.pyplot(fig)

                else:
                    # ---- SINGLE-CHANNEL plots ----
                    x = df[choice].dropna().to_numpy()
                    x = preprocess(x, fs)

                    if method == "FFT":
                        times, band_abs = sliding_fft_bandpower(x, fs, BANDS, win_sec, step_sec, window)
                        fig = plot_band_timeseries_fft(times, band_abs, BANDS, title_suffix=f" – {choice}")
                        st.pyplot(fig)
                    else:
                        t_axis, envelopes = dwt_band_envelopes(
                            x, fs, BANDS, wavelet=wavelet, levels=int(levels), smooth_sec=smooth_sec
                        )
                        fig = plot_band_timeseries_dwt(t_axis, envelopes, BANDS, title_suffix=f" – {choice}")
                        st.pyplot(fig)

            except Exception as e:
                st.error(f"Plot failed: {e}")


    # Step 6: Meditation Technique Recommendation
    # define threshold
    # which threhold do which meditation technique
    # buttons to ask
    # dominance of wbrainwaves
    # example.py

    # Step 6: Meditation Technique Recommendation
    st.header("4️⃣ Meditation Technique Recommendation")

    if st.session_state.get("per_ch"):
        per_ch = st.session_state["per_ch"]
        method_used = st.session_state.get("method_used", "fft")

        # 1) Label per-channel using your threshold engine
        labels = apply_threshold_labels(per_ch, method=method_used)

        # 2) Collapse to overall (average relative %) per band → a single label per band
        #    (If you prefer per-channel decisions, use labels[ch][band]['rel_label'] directly.)
        band_overall = {}
        for band in BANDS.keys():
            vals = []
            for ch, metrics in per_ch.items():
                key = f"{band}_rel_mean"
                if key in metrics and np.isfinite(metrics[key]):
                    vals.append(metrics[key])
            if len(vals) == 0:
                continue
            avg_rel = float(np.mean(vals))
            band_overall[band] = label_relative_value(band, avg_rel, REL_RULES_DEFAULT)  # "Low/Medium/High"

        # 3) Build options from your mapping
        #    Beta low => Concentration/Breathwork
        #    Beta high => Relaxation
        #    Theta high => Focused Attention (FAM)
        #    Theta low => Mindfulness / Non-directive
        #    Alpha low => Non-directive meditation
        #    Alpha high => Mindfulness meditation
        options = []
        if band_overall.get("Beta") == "High":
            options.append("Relaxation (reduce Beta)")
            st.session_state["chosen_meditation"] = True
        if band_overall.get("Beta") == "Low":
            options.append("Concentration / Breathwork (increase Beta)")
            st.session_state["chosen_meditation"] = True
        if band_overall.get("Theta") == "High":
            options.append("Focused Attention (reduce Theta)")
            st.session_state["chosen_meditation"] = True
        if band_overall.get("Theta") == "Low":
            options.append("Mindfulness / Non-directive (increase Theta)")
            st.session_state["chosen_meditation"] = True

        if band_overall.get("Alpha") == "High":
            options.append("Mindfulness (settle into Alpha)")
            st.session_state["chosen_meditation"] = True

        if band_overall.get("Alpha") == "Low":
            options.append("Non-directive (encourage Alpha)")
            st.session_state["chosen_meditation"] = True


        # Remove duplicates while preserving order
        dedup = []
        seen = set()
        for o in options:
            if o not in seen:
                dedup.append(o);
                seen.add(o)
        options = dedup

        st.caption(f"Overall labels (avg across channels): "
                   f"Beta={band_overall.get('Beta', '–')}, "
                   f"Theta={band_overall.get('Theta', '–')}, "
                   f"Alpha={band_overall.get('Alpha', '–')}")

        if options:
            choice = st.radio("Select what you want to prioritise:", options, key="radio_choice")
            if st.button(f"Confirm: {choice}", key="btn_confirm_choice"):
                st.session_state["selected_protocol"] = choice
                st.success(f"Selected: {choice}")
        else:
            st.info("Bands look balanced — choose any practice you prefer.")


        # (Optional) show the exact thresholds used for transparency
        with st.expander("See thresholds used"):
            st.json({"absolute_cutoffs": labels.get("ABS_CUTOFFS"),
                     "relative_rules": labels.get("REL_RULES")})

    else:
        st.info("Run the analysis first to get recommendations.")


    # Step 7: Meditation (15mins)
    # default to 15 minutes = 900 seconds
    MEDITATION_SECONDS = 9

    if "selected_protocol" in st.session_state:
        st.header("5️⃣ Start Meditation")
        st.write(f"Selected protocol: **{st.session_state['selected_protocol']}**")

        if st.button("🧘 Start meditation now", key="btn_start_meditation"):
            st.session_state["meditation_running"] = True
            st.session_state["meditation_start_ts"] = time.time()
            st.session_state["meditation_total"] = MEDITATION_SECONDS

        # Timer UI
        if st.session_state.get("meditation_running"):
            placeholder = st.empty()
            prog = st.progress(0)
            total = st.session_state["meditation_total"]
            start_ts = st.session_state["meditation_start_ts"]

            # simple second-by-second loop
            elapsed = 0
            while elapsed < total:
                elapsed = int(time.time() - start_ts)
                if elapsed > total:
                    elapsed = total
                remaining = total - elapsed
                mm = remaining // 60
                ss = remaining % 60
                placeholder.markdown(f"### ⏳ Time remaining: **{mm:02d}:{ss:02d}**")
                prog.progress(int(100 * elapsed / total))
                time.sleep(1)

            placeholder.success("Meditation completed ✅")
            prog.progress(100)
            st.session_state["meditation_running"] = False
            st.session_state["meditation_done"] = True
    else:
        st.info("Select a meditation option above to enable the timer.")

    # Step 8: Post EEG
    if st.session_state.get("meditation_done"):
        st.header("6️⃣ Post‑Meditation EEG Collection")
        after_filename = "after_eeg.csv"
        after_path = os.path.join(folder_path, after_filename)
        post_duration = duration  # reuse the same duration you used before, or let user change

        if st.button("🔍 Search for Muse Device & Start Stream (Post)"):
            with st.spinner("Starting Muse LSL stream..."):
                process = subprocess.Popen([sys.executable, "start_stream.py"])
                time.sleep(8)

                found = False
                for _ in range(10):
                    streams = resolve_stream('type', 'EEG')
                    if streams:
                        found = True
                        break
                    time.sleep(1)

                if found:
                    st.session_state['post_stream_process'] = process
                    st.success("Muse device found for post‑meditation recording.")
                else:
                    process.terminate()
                    st.error("Muse device not found. Please retry.")

        if 'post_stream_process' in st.session_state:
            if st.button(f"🎙 Start Post‑Meditation EEG Recording ({post_duration}s)"):
                with st.spinner(f"Recording EEG for {post_duration} seconds..."):
                    try:
                        record(post_duration, filename=after_path)
                        if os.path.exists(after_path):
                            st.success(f"Post‑meditation recording complete: `{after_path}`")
                            st.session_state['after_eeg_file'] = after_path
                        else:
                            st.error("Recording failed. File not found.")
                    except Exception as e:
                        st.error(f"Recording error: {e}")
                    finally:
                        if st.session_state['post_stream_process'].poll() is None:
                            st.session_state['post_stream_process'].terminate()
                            st.success("Muse stream terminated.")
                        del st.session_state['post_stream_process']


    # Step 9: Compare difference with Pre
    # Step 10: see if it actually helps (how many % increase)

    # ----------------- Step 9: Compare Pre vs Post -----------------

    if st.session_state.get('before_eeg_file') and st.session_state.get('after_eeg_file'):
        st.header("7️⃣ Compare Pre vs Post")
        method_choice = st.radio("Method for comparison", ["FFT", "DWT"], index=0, key="cmp_method")

        if st.button("🔍 Run Pre/Post Comparison"):
            try:
                df_pre = pd.read_csv(st.session_state['before_eeg_file'])
                df_post = pd.read_csv(st.session_state['after_eeg_file'])

                method_flag = "fft" if method_choice == "FFT" else "dwt"

                # Per-channel summaries
                per_pre, _ = per_channel_summary(df_pre, method=method_flag, bands=BANDS)
                per_post, _ = per_channel_summary(df_post, method=method_flag, bands=BANDS)

                # Overall (mean across channels)
                overall_pre = overall_summary(per_pre, method=method_flag, bands=BANDS)
                overall_post = overall_summary(per_post, method=method_flag, bands=BANDS)

                with st.expander("🔎 Raw summary data"):
                    st.subheader("Per-channel summary (Pre)")
                    st.json(per_pre)
                    st.subheader("Per-channel summary (Post)")
                    st.json(per_post)
                    st.subheader("Overall (mean across channels)")
                    st.json({"Pre": overall_pre, "Post": overall_post})

                # ---- Plots ----
                st.subheader("A) Overall Absolute/Amplitude (Pre vs Post)")
                fig1 = plot_overall_abs_prepost(overall_pre, overall_post, method=method_flag, bands=BANDS)
                st.pyplot(fig1)

                st.subheader("B) Overall Relative Power (Pre vs Post)")
                fig2 = plot_overall_rel_prepost(overall_pre, overall_post, bands=BANDS)
                st.pyplot(fig2)

                st.subheader("C) Change in Relative Power (percentage points)")
                fig3 = plot_overall_rel_change(overall_pre, overall_post, bands=BANDS)
                st.pyplot(fig3)

                st.subheader("D) Per-channel Heatmap of Relative Power Change")
                fig4 = plot_rel_change_heatmap(per_pre, per_post, bands=BANDS)
                st.pyplot(fig4)

                st.caption(
                    "Tip: Positive bars/cells mean the band increased after meditation; negatives mean it decreased.")

                st.subheader("E) Time-series Comparison (FFT)")
                # Determine sampling rate (fs) from timestamps, fallback to 256 Hz
                if "timestamps" in df_pre.columns and "timestamps" in df_post.columns:
                    fs_pre, _ = estimate_fs(df_pre["timestamps"].to_numpy())
                    fs_post, _ = estimate_fs(df_post["timestamps"].to_numpy())
                    fs = float(np.round((fs_pre + fs_post) / 2.0))
                else:
                    fs = 256.0


                fig_fft_cmp = plot_band_timeseries_fft_compare(df_pre, df_post, fs)
                st.pyplot(fig_fft_cmp)

                st.subheader("F) Time-series Comparison (DWT)")
                fig_dwt_cmp = plot_band_timeseries_dwt_compare(df_pre, df_post, fs)
                st.pyplot(fig_dwt_cmp)

            except Exception as e:
                st.error(f"Comparison failed: {e}")
    else:
        st.info("Need both Pre and Post EEG files to compare.")

    # ----------------- Step X: Percent-change Time-Series -----------------
    st.header("📈 Percent-change Time-Series (Before → After)")

    have_pre = st.session_state.get('before_eeg_file')
    have_post = st.session_state.get('after_eeg_file')

    if not (have_pre and have_post):
        st.info("Load both **Before** and **After** EEG files to compare.")
    else:
        df_pre = pd.read_csv(st.session_state['before_eeg_file'])
        df_post = pd.read_csv(st.session_state['after_eeg_file'])

        # Estimate fs from timestamps, fallback to 256 Hz
        if "timestamps" in df_pre.columns and "timestamps" in df_post.columns:
            fs_pre, _ = estimate_fs(df_pre["timestamps"].to_numpy())
            fs_post, _ = estimate_fs(df_post["timestamps"].to_numpy())
            fs = float(np.round((fs_pre + fs_post) / 2.0))
        else:
            fs = 256.0

        # Controls
        scope = st.selectbox("Scope", ["Overall"] + [c for c in EEG_CHANNELS if c in df_pre.columns], index=0)
        which = st.radio("Metric", ["Absolute power (µV²)", "Relative power (%)"], index=0, horizontal=True)
        bands_pick = st.multiselect("Brainwaves to display", list(BANDS.keys()), default=list(BANDS.keys()))
        win_sec = st.slider("FFT window (s)", 1.0, 4.0, 2.0, 0.5)
        step_sec = st.slider("FFT step (s)", 0.25, 2.0, 0.5, 0.25)

        # Compute
        if which.startswith("Absolute"):
            t, change_dict, ylabel, title = percent_change_timeseries_fft(
                df_pre, df_post, fs, bands=BANDS, scope=scope,
                win_sec=win_sec, step_sec=step_sec, which="absolute"
            )
        else:
            rel_mode = st.radio("Relative change mode", ["Percentage points (Post−Pre)", "Percent vs Pre"], index=0,
                                horizontal=True)
            t, change_dict, ylabel, title = percent_change_timeseries_fft(
                df_pre, df_post, fs, bands=BANDS, scope=scope,
                win_sec=win_sec, step_sec=step_sec,
                which="relative",
                rel_mode=("pp" if rel_mode.startswith("Percentage points") else "ratio")
            )

        # Plot
        fig = plot_percent_change_lines(t, change_dict, bands_select=bands_pick, ylabel=ylabel, title=title)
        st.pyplot(fig)

        st.caption("Tip: values above 0 indicate an increase after meditation; below 0 indicate a decrease.")

    # if st.button("User Feedback Form"):

