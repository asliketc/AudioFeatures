import seaborn as sns 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import StandardScaler

df=pd.read_csv("/Users/macbook/Desktop/dataProjects/librosa_project2/features/audio_features.csv")
chroma_cols=[]
pitch_classes=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
chroma_mapping = {f"chroma_{i+1}": pitch_classes[i] for i in range(12)}


df = df.rename(columns=chroma_mapping)

df[pitch_classes].mean().plot(kind="bar", figsize=(10, 4), color="orchid")
plt.title("Average Pitch Class Energy")
plt.xlabel("Pitch Class")
plt.ylabel("Mean Intensity")
plt.xticks(rotation=0)
plt.grid(True)
plt.tight_layout()
#plt.show()


mfcc_cols=[]
for col in df.columns:
    if col.startswith("mfcc_"):
        mfcc_cols.append(col)

#sclr=StandardScaler()
#scaled_mfcc=sclr.fit_transform(df[mfcc_cols])
mfcc_df = df[mfcc_cols].copy()
mfcc_df["filename"] = df["filename"]

df["timbre_brightness"] = df["mfcc_1"]
df["timbre_warmth"] = df[["mfcc_2", "mfcc_3", "mfcc_4"]].mean(axis=1)
df["timbre_texture"] = df[["mfcc_5", "mfcc_6", "mfcc_7", "mfcc_8"]].mean(axis=1)

mfcc_labels={
    "mfcc_1":"mfcc_1\n(Brightness)",
    "mfcc_2":"mfcc_2\n(Warmth)",
    "mfcc_3":"mfcc_3\n(Warmth)",
    "mfcc_4":"mfcc_4\n(Warmth)",
    "mfcc_5":"mfcc_5\n(Texture)",
    "mfcc_6":"mfcc_6\n(Texture)",
    "mfcc_7":"mfcc_7\n(Texture)",
    "mfcc_8":"mfcc_8\n(Texture)",
    "mfcc_9":"mfcc_9\n(Sharper)",
    "mfcc_10":"mfcc_10\n(Sharper)",
    "mfcc_11":"mfcc_11\n(Metallic)",
    "mfcc_12":"mfcc_12\n(Metallic)",
    "mfcc_13":"mfcc_13\n(Metallic)",
}
xtick_labels = [mfcc_labels[col] for col in mfcc_cols]


plt.figure(figsize=(12, 6))
sns.heatmap(
    mfcc_df[mfcc_cols], 
    cmap="magma", 
    yticklabels=df["filename"], 
    #cbar_kws={"label": "Z-score (normalized MFCC value)"}, 
     cbar_kws={"label": "Raw MFCC Value (Higher = More Intensity)"},
    annot=True,
    fmt=".1f"
)
plt.xticks(ticks=[i + 0.5 for i in range(len(xtick_labels))], labels=xtick_labels, rotation=45, ha="right")
plt.title("MFCC Heatmap with Auto-Explained Timbre Roles (color of sound)", fontsize=16, weight='bold')
plt.ylabel("Track Filename")
plt.xlabel("MFCC Coefficients (Musical Meaning in Parentheses)")

# 🔍 Footnote
plt.figtext(
    0.5, -0.1,
    "• Brightness: MFCC 1, correlates with sharpness/fullness\n"
    "• Warmth: MFCCs 2 to 4, typically mid-low resonances\n"
    "• Texture: MFCCs 5 to 8, shape & contrast\n"
    "• Others: Higher MFCCs tend to be more noise/detail focused",
    wrap=True, horizontalalignment='center', fontsize=9
)

plt.tight_layout()
plt.show()


print("Interpreted Timbre Summary Table:\n")
print(df[["filename", "timbre_brightness", "timbre_warmth", "timbre_texture"]].round(2))
