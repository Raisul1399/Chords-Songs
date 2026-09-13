import librosa
import numpy as np
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox

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
    # 1. Force the file explorer to only accept MP3s
    file_path = filedialog.askopenfilename(
        title="Select an MP3 File",
        filetypes=[("MP3 Files", "*.mp3")]
    )
    
    if not file_path:
        return

    # 2. Defensive Programming: Check duration BEFORE loading the heavy audio data
    try:
        # librosa can peek at the file length instantly
        duration = librosa.get_duration(path=file_path)
        if duration > 300:
            # Pop up a formal error window!
            messagebox.showerror("File Too Long", "Please upload an MP3 file that is 5 minutes or shorter.")
            return
    except Exception as e:
        messagebox.showerror("Error", f"Could not read file: {e}")
        return

    # 3. Update the screen
    result_box.delete("1.0", tk.END)
    result_box.insert(tk.END, f"Loading file: {file_path.split('/')[-1]}...\n")
    result_box.insert(tk.END, "Analyzing audio, please wait...\n\n")
    window.update() 

    try:
        # 4. Run the core logic
        audio, sr = load_audio(file_path)
        beat_chroma, bpm = extract_notes_by_beat(audio, sr)
        
        result_box.insert(tk.END, f"Detected Tempo: {bpm:.0f} BPM\n")
        result_box.insert(tk.END, "-" * 30 + "\n")
        
        # 5. NEW LOGIC: Group identical consecutive chords!
        num_beats = beat_chroma.shape[1]
        previous_chord = None
        start_beat = 1
        
        for beat in range(num_beats):
            single_beat_data = beat_chroma[:, beat]
            current_chord = match_chord_template(single_beat_data)
            
            # Setup the very first beat
            if previous_chord is None:
                previous_chord = current_chord
                start_beat = beat + 1
                
            # If the chord CHANGES, print the previous grouped block
            elif current_chord != previous_chord:
                end_beat = beat # The beat before the change
                if start_beat == end_beat:
                    result_box.insert(tk.END, f"Beat {start_beat}: {previous_chord}\n")
                else:
                    result_box.insert(tk.END, f"Beats {start_beat}-{end_beat}: {previous_chord}\n")
                
                # Reset the tracker for the new chord
                previous_chord = current_chord
                start_beat = beat + 1

        # Print the very last chord block after the loop finishes
        if previous_chord is not None:
            end_beat = num_beats
            if start_beat == end_beat:
                result_box.insert(tk.END, f"Beat {start_beat}: {previous_chord}\n")
            else:
                result_box.insert(tk.END, f"Beats {start_beat}-{end_beat}: {previous_chord}\n")
            
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