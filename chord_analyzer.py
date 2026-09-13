import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog

# --- 1. CORE LOGIC ---

def load_audio(file_path):
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    return audio_data, sample_rate

def extract_notes_by_beat(audio_data, sample_rate):
    chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    
    bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo 
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
    
    # We now return the BPM alongside the beat data so the GUI can display it!
    return beat_chroma, bpm

def match_chord_template(beat_data):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    base_major = [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0]
    base_minor = [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0]
    
    templates = {}
    for i, note in enumerate(NOTES):
        templates[f"{note} Major"] = np.roll(base_major, i)
        templates[f"{note} Minor"] = np.roll(base_minor, i)
    
    best_chord = "Unknown"
    highest_score = -1
    
    for chord_name, template in templates.items():
        score = np.dot(beat_data, template)
        if score > highest_score:
            highest_score = score
            best_chord = chord_name
            
    return best_chord

# --- 2. GUI INTERFACE ---

def process_file():
    # 1. Open a file dialog to let the user pick an audio file
    file_path = filedialog.askopenfilename(
        title="Select an Audio File",
        filetypes=[("Audio Files", "*.mp3 *.wav *.ogg *.flac")]
    )
    
    # If the user cancels or closes the window, do nothing
    if not file_path:
        return

    # 2. Update the screen to show it is loading (processing takes a few seconds)
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, f"Loading file: {file_path.split('/')[-1]}...\n")
    result_box.insert(tk.END, "Analyzing audio, please wait...\n\n")
    window.update() # Force the window to refresh the text

    try:
        # 3. Run the core logic
        audio, sr = load_audio(file_path)
        beat_chroma, bpm = extract_notes_by_beat(audio, sr)
        
        # 4. Display the BPM
        result_box.insert(tk.END, f"Detected Tempo: {bpm:.0f} BPM\n")
        result_box.insert(tk.END, "-" * 30 + "\n")
        
        # 5. Loop through beats and display the timeline
        num_beats = beat_chroma.shape[1]
        for beat in range(num_beats):
            single_beat_data = beat_chroma[:, beat]
            chord_name = match_chord_template(single_beat_data)
            result_box.insert(tk.END, f"Beat {beat + 1}: {chord_name}\n")
            
    except Exception as e:
        result_box.insert(tk.END, f"\nAn error occurred: {e}")

# --- 3. CREATE THE WINDOW ---
window = tk.Tk()
window.title("Music Chord Analyzer")
window.geometry("450x600")

# Add a title label
title_label = tk.Label(window, text="Music Chord Analyzer", font=("Helvetica", 16, "bold"))
title_label.pack(pady=15)

# Add the Browse button
browse_btn = tk.Button(window, text="Upload Audio File", command=process_file, font=("Helvetica", 12), bg="lightblue")
browse_btn.pack(pady=10)

# Add a text box to show the results
result_box = tk.Text(window, height=25, width=40, font=("Courier", 10))
result_box.pack(pady=10)

# Start the application!
window.mainloop()