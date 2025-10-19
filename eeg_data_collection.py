# simple_record_eeg.py
# Minimal Muse 2 EEG recording script for new users
# -------------------------------------------------
# Requirements:
#   pip install muselsl pylsl numpy pandas

import os
import time
from datetime import datetime
from record_eeg import record_eeg_to_csv  # must exist in same folder or installed package

def get_downloads_folder():
    """Cross-platform way to find user's Downloads directory."""
    home = os.path.expanduser("~")
    downloads = os.path.join(home, "Downloads")
    if not os.path.exists(downloads):
        os.makedirs(downloads, exist_ok=True)
    return downloads

def main():
    print("\n🧠 Muse EEG Recording — Simple Mode")
    print("-----------------------------------")
    print("Make sure your Muse 2 headband is turned on and Bluetooth is connected.")
    print("When ready, this program will record EEG data for 5 minutes (300 seconds).")

    # --- User confirmation ---
    input("\nPress Enter to start recording...")

    duration = 10  # 5 minutes
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = get_downloads_folder()
    filename = f"EEG_recording_{timestamp}.csv"
    save_path = os.path.join(save_dir, filename)

    print(f"\n📡 Recording started... Duration: {duration}s")
    print(f"File will be saved to: {save_path}")
    print("Do not close this window until the recording completes.\n")

    try:
        # Call your existing function to record EEG
        record_eeg_to_csv(duration=duration, folder_path=save_dir, filename=filename)
        print(f"\n✅ Recording complete! EEG data saved at:\n{save_path}")
        print("You can now upload this CSV file to the hosted web app for analysis.")
    except Exception as e:
        print(f"\n❌ Error during recording: {e}")
    finally:
        print("\nSession ended. You may close this window.")

if __name__ == "__main__":
    main()
