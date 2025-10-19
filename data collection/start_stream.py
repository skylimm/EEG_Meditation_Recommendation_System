from muselsl import stream, list_muses

def start_muse_stream():
    """
    Starts the Muse LSL stream.
    """
    # List available Muse devices
    muses = list_muses()
    if not muses:
        print("No Muse devices found.")
        return

    # Start streaming from the first Muse device
    print(f"Starting Muse stream from device: {muses[0]['address']}")
    stream(muses[0]['address'], ppg_enabled=False, acc_enabled=False, gyro_enabled=False)

if __name__ == "__main__":
    start_muse_stream()