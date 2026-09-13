import librosa
import numpy as np

def load_audio(file_path):
    print(f"Loading {file_path}...")
    audio_data, sample_rate = librosa.load(file_path, sr=None)
    print(f"Success! Sample rate: {sample_rate}Hz")
    print(f"Audio duration: {len(audio_data) / sample_rate:.2f} seconds")
    return audio_data, sample_rate

def extract_notes_by_beat(audio_data, sample_rate):
    print("Detecting tempo and grouping by beats...")
    chromagram = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(y=audio_data, sr=sample_rate)
    
    bpm = tempo[0] if isinstance(tempo, np.ndarray) else tempo 
    print(f"Detected Tempo: {bpm:.0f} BPM")
    
    beat_chroma = librosa.util.sync(chromagram, beat_frames, aggregate=np.median)
    print(f"Chunked the song into {beat_chroma.shape[1]} beat-sized segments.")
    return beat_chroma

def identify_strongest_notes_in_beat(beat_data):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    # No averaging needed anymore! We just sort the 12 numbers for this single beat
    top_3_indices = np.argsort(beat_data)[-3:][::-1]
    top_3_notes = [NOTES[i] for i in top_3_indices]
    return top_3_notes

def match_chord(top_notes):
    notes_set = frozenset(top_notes)
    CHORD_DICTIONARY = {
        frozenset(['C', 'E', 'G']): "C Major",
        frozenset(['C', 'D#', 'G']): "C Minor",
        frozenset(['D', 'F#', 'A']): "D Major",
        frozenset(['E', 'G#', 'B']): "E Major",
        frozenset(['E', 'G', 'B']): "E Minor",
        frozenset(['F', 'A', 'C']): "F Major",
        frozenset(['G', 'B', 'D']): "G Major",
        frozenset(['G', 'A#', 'D']): "G Minor",
        frozenset(['A', 'C#', 'E']): "A Major",
        frozenset(['A', 'C', 'E']): "A Minor",
    }
    
    if notes_set in CHORD_DICTIONARY:
        return CHORD_DICTIONARY[notes_set]
    else:
        return "Unknown"

# --- EXECUTION CODE ---
my_file = "freesound_community-guitar-chords-70663.mp3" 

audio, sr = load_audio(my_file)
beat_chroma = extract_notes_by_beat(audio, sr)

print("\n--- CHORD TIMELINE ---")
num_beats = beat_chroma.shape[1]

# THE LOOP: Walk through the song beat-by-beat
for beat in range(num_beats):
    # 1. Isolate the data for just this single beat
    single_beat_data = beat_chroma[:, beat]
    
    # 2. Find the 3 strongest notes for this beat
    strongest_notes = identify_strongest_notes_in_beat(single_beat_data)
    
    # 3. Match to dictionary
    chord_name = match_chord(strongest_notes)
    
    # 4. Print the timeline
    print(f"Beat {beat + 1}: {chord_name} (Detected Notes: {strongest_notes[0]}, {strongest_notes[1]}, {strongest_notes[2]})")