import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def create_dataframe(frame_numbers, timestamps, labels, output_csv_path):
   df = pd.DataFrame({
    "frame": frame_numbers,
    "time_sec": timestamps,
    "behavior_label": labels
    })
   df.to_csv(output_csv_path, index=False)
   print(f"Ethogram saved to: {output_csv_path}")
   return df

def plot_etoghram(df,labels):
  # Plot ethogram
   plt.figure(figsize=(12, 2))
   plt.scatter(df["frame"], df["behavior_label"], c=df["behavior_label"], cmap='tab20', marker='|', s=100)
   plt.xlabel("frames")
   plt.ylabel("Motifs")
   plt.title("Ethogram")
   plt.yticks(np.unique(labels))
   plt.tight_layout()
   plt.show()

def etoghram(input_file, output_csv_path,fps, offset):
    labels = np.load(input_file)
    frame_numbers = np.arange(len(labels)) + offset
    timestamps = (frame_numbers)/fps
    df = create_dataframe(frame_numbers, timestamps, labels, output_csv_path)
    plot_etoghram(df,labels)