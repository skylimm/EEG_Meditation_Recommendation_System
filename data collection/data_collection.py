# data_collection.py  (manual PRE/POST selector; no loops)
import os
from record_eeg import record_eeg_to_csv  # uses your existing function

BASE_DIR = r"C:\Users\slex8\OneDrive - Nanyang Technological University\UNI\FYP\data_collection"

def slugify(s: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in s).strip("_")

def get_participant_folder(base_dir=BASE_DIR):
    name = input("Enter participant name: ").strip()
    if not name:
        raise ValueError("Name cannot be empty.")
    name_slug = slugify(name)
    folder = os.path.join(base_dir, name_slug)
    os.makedirs(folder, exist_ok=True)
    print(f"[OK] Using folder: {folder}")
    return folder, name_slug

def get_meditation_type():
    med = input("Enter meditation type (FAM, OM, BSM, LKM, MM): ").strip()
    if not med:
        raise ValueError("Meditation type cannot be empty.")
    return slugify(med)

def get_phase():
    phase = input("Phase? Type 'pre' or 'post': ").strip().lower()
    if phase not in ("pre", "post"):
        raise ValueError("Phase must be 'pre' or 'post'.")
    return phase

def main():
    folder_path, participant = get_participant_folder()
    med_type = get_meditation_type()
    phase = get_phase()

    filename = f"{participant}_{med_type}_{phase}.csv"
    print(f"\n=== {participant} | {med_type} | {phase.upper()} ===")
    duration = input("Duration seconds (default 300): ").strip()
    duration = int(duration) if duration else 300

    input(f"Press Enter to start {phase.upper()} recording for {duration}s...")
    saved_path = record_eeg_to_csv(duration=duration, folder_path=folder_path, filename=filename)
    print(f"[Saved] {saved_path}")

if __name__ == "__main__":
    main()
