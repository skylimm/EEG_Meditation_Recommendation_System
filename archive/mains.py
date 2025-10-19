from pathlib import Path
import os
import numpy as np
import pandas as pd

from AI_datacollection import is_effective_eeg
from user_input import get_user_input
from record_eeg import record_eeg_to_csv
from archive.preprocessing import (
    apply_notch_filter, bandpass_filter, remove_artifacts_ica,
    compute_band_powers, extract_dwt_features, extract_statistical_features,
    recommend_meditation
)

def main():
    # Step 1: Collect user input
    folder_path = get_user_input()

    # Step 2: Pre-Test EEG Baseline (Before)
    print("\n=== Pre-Test EEG Baseline ===")
    print("Sit still with eyes open, avoid thinking about anything, breath normally")
    input("Press Enter to start collect Pre-meditation EEG...")
    before_eeg_file = record_eeg_to_csv(duration=25, folder_path=folder_path, filename="before_eeg.csv")
    print(before_eeg_file)

    # Step 3: Pre-Meditation Mini-Game
    print("\n=== Pre-Meditation Mini-Game ===")
    pre_game_score = 'EXAMPLE'  # Replace with your mini-game function
    print(f"Pre-Meditation Game Score: {pre_game_score}")


    # Step 4: Meditation Recommendation (AI )
    print("\n=== Meditation Recommendation ===")
    print("\n recommending meditation type")

    # preprocessing signals before throwing into model
    pre_eeg_data = pd.read_csv(before_eeg_file)
    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    # 2. Bandpass filter
    for ch in channels:
        pre_eeg_data[ch] = bandpass_filter(pre_eeg_data[ch].values)


    # 4. Extract Features (FFT, DWT, Stats)
    fft_powers = [compute_band_powers(clean_matrix[:, i]) for i in range(len(channels))]
    dwt_feats = [extract_dwt_features(clean_matrix[:, i]) for i in range(len(channels))]
    stats_feats = [extract_statistical_features(clean_matrix[:, i]) for i in range(len(channels))]

    # 5. AI Recommendation
    pre_avg_theta = np.mean([f['theta_rel'] for f in fft_powers])
    pre_avg_alpha = np.mean([f['alpha_rel'] for f in fft_powers])
    pre_avg_beta = np.mean([f['beta_rel'] for f in fft_powers])
    pre_avg_delta = np.mean([f['delta_rel'] for f in fft_powers])
    pre_avg_gamma = np.mean([f['gamma_rel'] for f in fft_powers])

    #
    # # 2. Extract Statistical Features
    # stats_features = {ch: extract_statistical_features(pre_eeg_cleaned[:, i])
    #                   for i, ch in enumerate(channels)}
    # print("Statistical Features (Pre-Meditation):", stats_features)
    #
    # # 3. DWT Features
    # dwt_features = {ch: extract_dwt_features(pre_eeg_cleaned[:, i])
    #                 for i, ch in enumerate(channels)}
    # print("DWT Features (Pre-Meditation):", dwt_features)
    #
    # # 4. Band Power Calculation (FFT)
    # delta_band = (0.5, 4)
    # theta_band = (4, 7)
    # alpha_band = (8, 12)
    # beta_band = (13, 30)
    # gamma_band = (30, 80)
    #
    # # Compute average power across all channels for each band
    # pre_delta = np.mean([calculate_band_power(pre_eeg_cleaned[:, i], band=delta_band)
    #                     for i in range(len(channels))])
    # pre_theta = np.mean([calculate_band_power(pre_eeg_cleaned[:, i], band=theta_band)
    #                     for i in range(len(channels))])
    # pre_alpha = np.mean([calculate_band_power(pre_eeg_cleaned[:, i], band=alpha_band)
    #                     for i in range(len(channels))])
    # pre_beta = np.mean([calculate_band_power(pre_eeg_cleaned[:, i], band=beta_band)
    #                     for i in range(len(channels))])
    # pre_gamma = np.mean([calculate_band_power(pre_eeg_cleaned[:, i], band=gamma_band)
    #                     for i in range(len(channels))])
    # # pre_delta = np.mean([calculate_band_power(pre_eeg_data[ch], band=delta_band) for ch in channels])
    # # pre_theta = np.mean([calculate_band_power(pre_eeg_data[ch], band=theta_band) for ch in channels])
    # # pre_alpha = np.mean([calculate_band_power(pre_eeg_data[ch], band=alpha_band) for ch in channels])
    # # pre_beta = np.mean([calculate_band_power(pre_eeg_data[ch], band=beta_band) for ch in channels])
    # # pre_gamma = np.mean([calculate_band_power(pre_eeg_data[ch], band=gamma_band) for ch in channels])
    #
    #
    print(f"\nDelta Power: {pre_avg_delta:.3f}")
    print(f"\nTheta Power: {pre_avg_theta:.3f}")
    print(f"\nAlpha Power: {pre_avg_alpha:.3f}")
    print(f"\nBeta Power : {pre_avg_beta:.3f}")
    print(f"\nGamma Power: {pre_avg_gamma:.3f}")

    recommendation = recommend_meditation(pre_avg_theta, pre_avg_alpha, pre_avg_beta)
    print(f"\nRecommended Meditation: {recommendation}")


    # Step 5: Meditation Session
    print("\n=== Meditation Session ===")
    print(f"\nRecommended Meditation: {recommendation}")
    print("\nPreparing to begin your meditation session...")
    input("\nPress Enter to START the meditation session...")
    # # Optionally record EEG during meditation
    # meditation_eeg_file = record_eeg_to_csv(duration=900, folder_path=folder_path, filename="meditation_eeg.csv")
    input("Press Enter to END the meditation session...")
    print("\nMeditation ended...")

    # Step 6: Post-Meditation EEG Measurement (After)
    print("\n=== Post-Meditation EEG Measurement ===")
    print("\nSit still with eyes open, avoid thinking about anything, breath normally")
    input("Press Enter to start collect Post-meditation EEG...")
    after_eeg_file = record_eeg_to_csv(duration=25, folder_path=folder_path, filename="after_eeg.csv")


    post_eeg_data = pd.read_csv(after_eeg_file)

    # post_delta = np.mean([calculate_band_power(post_eeg_data[ch], band=delta_band) for ch in channels])
    # post_theta = np.mean([calculate_band_power(post_eeg_data[ch], band=theta_band) for ch in channels])
    # post_alpha = np.mean([calculate_band_power(post_eeg_data[ch], band=alpha_band) for ch in channels])
    # post_beta = np.mean([calculate_band_power(post_eeg_data[ch], band=beta_band) for ch in channels])
    # post_gamma = np.mean([calculate_band_power(post_eeg_data[ch], band=gamma_band) for ch in channels])
    channels = ['TP9', 'AF7', 'AF8', 'TP10']

    # 1. Apply notch filter
    for ch in channels:
        post_eeg_data[ch] = apply_notch_filter(post_eeg_data[ch].values)

    # 2. Bandpass filter
    for ch in channels:
        post_eeg_data[ch] = bandpass_filter(post_eeg_data[ch].values)

    # 3. ICA Artifact Removal
    post_clean_matrix = remove_artifacts_ica(post_eeg_data[channels].values)

    # 4. Extract Features (FFT, DWT, Stats)
    fft_powers = [compute_band_powers(post_clean_matrix[:, i]) for i in range(len(channels))]
    dwt_feats = [extract_dwt_features(post_clean_matrix[:, i]) for i in range(len(channels))]
    stats_feats = [extract_statistical_features(post_clean_matrix[:, i]) for i in range(len(channels))]

    # 5. AI Recommendation
    post_avg_theta = np.mean([f['theta_rel'] for f in fft_powers])
    post_avg_alpha = np.mean([f['alpha_rel'] for f in fft_powers])
    post_avg_beta = np.mean([f['beta_rel'] for f in fft_powers])
    post_avg_delta = np.mean([f['delta_rel'] for f in fft_powers])
    post_avg_gamma = np.mean([f['gamma_rel'] for f in fft_powers])

    print(f"\nDelta Power: {post_avg_delta:.3f}")
    print(f"\nTheta Power: {post_avg_theta:.3f}")
    print(f"\nAlpha Power: {post_avg_alpha:.3f}")
    print(f"\nBeta Power : {post_avg_beta:.3f}")
    print(f"\nGamma Power: {post_avg_gamma:.3f}")



    # Step 7: Post-Meditation Mini-Game
    print("\n=== Post-Meditation Mini-Game ===")
    post_game_score = 'EXAMPLE SCORE'  # Replace with your mini-game function
    print(f"\nPost-Meditation Game Score: {post_game_score}")

    # Step 8: User Feedback Collection
    print("\n=== User Feedback Collection ===")
    feedback = input("Did you feel more focused after meditation? (Yes/No): ")
    print(f"\nUser Feedback: {feedback}")

    # Step 9: Collecting data for AI MODEL
    print("\n=== Collating data for AI model ===")
    effective = is_effective_eeg(pre_beta=pre_avg_beta,post_beta=post_avg_beta,pre_theta=pre_avg_theta,post_theta=post_avg_theta,
        beta_thresh=0.2, theta_thresh=0.2)

    # Create a dictionary with all the data
    session_data = {
        'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pre_delta': pre_avg_delta,
        'pre_theta': pre_avg_theta,
        'pre_alpha': pre_avg_alpha,
        'pre_beta': pre_avg_beta,
        'pre_gamma': pre_avg_gamma,
        'recommended_meditation': recommendation,
        'post_delta': post_avg_delta,
        'post_theta': post_avg_theta,
        'post_alpha': post_avg_alpha,
        'post_beta': post_avg_beta,
        'post_gamma': post_avg_gamma,
        'pre_game_score': pre_game_score,
        'post_game_score': post_game_score,
        'focus_feedback': feedback,
        'effective': effective
    }

    # Define the output file path
    results_file = os.path.join(Path(folder_path).parent, "meditation_results.csv")

    # Convert to DataFrame
    df = pd.DataFrame([session_data])

    # Append to existing file or create new one
    if os.path.exists(results_file):
        df.to_csv(results_file, mode='a', header=False, index=False)
        print(f"Data appended to existing file: {results_file}")
    else:
        df.to_csv(results_file, index=False)
        print(f"New results file created: {results_file}")

    print("\nData collection complete. Thank you for participating!")


if __name__ == "__main__":
    main()