import librosa #audio analysis library
import pandas as pd 
import numpy as np 
import os
from tqdm import tqdm #progress bar during process

AUDIO_DIR="audio"
OUTPUT_PATH="features/audio_features.csv"

def extract_from_file(file_path):
    """
    -takes path of audio file, returns dict of its features
    -the dict will be a ROW for .csv file
    """
    y, sr=librosa.load(file_path, sr=None) #y=audio waveform,sr=sample rate (kHz)

    feats={
        "filename": os.path.basename(file_path),
        "dur":librosa.get_duration(y=y, sr=sr),
        "tempo":librosa.beat.tempo(y=y, sr=sr)[0], #estimated tempo (in BPM)
        "spectral_centroid":np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)) #brigthness of sound
    }

    """
    MFCCs (Mel-Frequency Cepstral Coefficients)
    -compressed representation of timbre
    """
    mfccs=librosa.feature.mfcc(y=y,sr=sr,n_mfcc=13)
    for i, co in enumerate(mfccs):
        feats[f"mfcc_{i+1}"]=np.mean(co)
    
    """
    CHROMA STFT
    -measures energy of pitches (C,D,...,B)
    """
    chrom=librosa.feature.chroma_stft(y=y,sr=sr)
    for i, chrom_bin in enumerate(chrom):
        feats[f"chroma_{i+1}"]=np.mean(chrom_bin)
    

    return feats

def main():
    all_rows=[] #list of dicts(feats{})

    for fname in tqdm(os.listdir(AUDIO_DIR)):
        if fname.endswith(".wav") or fname.endswith(".mp3"):
            fpath=os.path.join(AUDIO_DIR, fname)
            try:
                feats=extract_from_file(fpath)
                all_rows.append(feats)
            except Exception as e:
                print("Error while processing Sound Files")
            
    
    df=pd.DataFrame(all_rows)

    df.to_csv(OUTPUT_PATH, index=False)
    print("Features saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
