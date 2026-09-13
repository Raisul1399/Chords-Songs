import librosa
import numpy as np

def load_audio(file_path):
    print(f"Loading {file_path}...")
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    print(f"Success! Sample rate: {sample_rate}Hz")
    return audio_data, sample_rate

def extract_notes_by_beat(audio_data, sample_rate):
    print("Detecting tempo and grouping by beats...")
    chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    
    bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo 
    print(f"Detected Tempo: {bpm:.0f} BPM")
    
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
    return beat_chroma

def match_chord_template(beat_data):
    # The 12 notes: C, C#, D, D#, E, F, F#, G, G#, A, A#, B
    # 1 means the note is in the chord, 0 means it isn't.
    templates = {
        "C Major": [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
        "C Minor": [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        "D Major": [0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0],
        "E Major": [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1],
        "E Minor": [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
        "F Major": [1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],
        "G Major": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
        "G Minor": [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],
        "A Major": [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
        "A Minor": [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    }
    
    best_chord = "Unknown"
    highest_score = -1 # Start with a negative score
    
    # Compare the real 12-note audio data against every perfect template
    for chord_name, template in templates.items():
        # np.dot multiplies overlapping notes and adds them up for a total score
        score = np.dot(beat_data, template)
        
        # If this chord scores higher than the previous best, it becomes the new winner
        if score > highest_score:
            highest_score = score
            best_chord = chord_name
            
    return best_chord

# --- EXECUTION CODE ---
my_file = "freesound_community-guitar-chords-70663.mp3" 

audio, sr = load_audio(my_file)
beat_chroma = extract_notes_by_beat(audio, sr)

print("\n--- CHORD TIMELINE ---")
num_beats = beat_chroma.shape[1]

# THE LOOP
for beat in range(num_beats):
    # Isolate the data for just this single beat
    single_beat_data = beat_chroma[:, beat]
    
    # Pass all 12 notes directly into the template matcher!
    chord_name = match_chord_template(single_beat_data)
    
    print(f"Beat {beat + 1}: {chord_name}")