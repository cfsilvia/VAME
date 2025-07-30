import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from collections import Counter
from matplotlib.colors import ListedColormap, BoundaryNorm
import os

def assign_frame_labels(raw,labels,window_size, step_size):
    n_frames = raw.shape[0]
    frame_motifs = [[] for _ in range(n_frames)]
    
    for i, label in enumerate(labels):
        start = i*step_size
        end = min(start + window_size, n_frames)
        for f in range(start, end):
            frame_motifs[f].append(label)
            
    #Since each frame has several frames we will to the label with more frequency for each frame
    frame_labels = [
        Counter(m).most_common(1)[0][0] if m else -1
        for m in frame_motifs
    ]
    
     # Fill any -1 labels (uncovered frames) with last known valid label
    last_valid = next((label for label in reversed(frame_labels) if label != -1), 0)
    frame_labels = [label if label != -1 else last_valid for label in frame_labels]
    
    data =raw.copy()
    data['motif'] = frame_labels
    
    return data

def plot_labels(data):
    y = data['BM_snout_y'].values
    labels = data['motif'].values
    x = np.arange(len(y))

    # Build line segments
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    # Create color map and normalization
    unique_labels = np.unique(labels)
    max_label = int(max(unique_labels))
    base_colors = ['white', 'red', 'green', 'blue', 'magenta', 'cyan', 'yellow', 'brown']
    cmap = ListedColormap(base_colors[:max_label + 1])
    norm = BoundaryNorm(np.arange(-0.5, max_label + 1.5), cmap.N)

    # Assign labels to each segment
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(labels[:-1])
    lc.set_linewidth(2)

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.add_collection(lc)
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(np.nanmin(y), np.nanmax(y))
    ax.set_xlabel("Frame")
    ax.set_ylabel("BM_snout_y")
    plt.colorbar(lc, ax=ax, label="Motif")
    plt.title("BM_snout_y signal colored by motif")
    plt.tight_layout()
    plt.show()

    

    

def main():
    dir =r"U:\Users\Ruthi\2025\BMR2\VAME\Your-VAME-Project-Jul21-2025\results\BMR2_with_landmarks_left_to _use\VAME\hmm-5\\"
    dir_data = r"U:\Users\Ruthi\2025\BMR2\VAME\Your-VAME-Project-Jul21-2025\videos\pose_estimation\\"
    #load the data
    raw = pd.read_csv(dir_data + "BMR2_with_landmarks_left_to _use.csv")
    labels = np.load(dir + r'5_km_label_BMR2_with_landmarks_left_to _use.npy')
    #configuration of sliding window according to config
    window_size = 30
    step_size = 1
    data_with_labels = assign_frame_labels(raw,labels,window_size, step_size)
    
    #plot and mark labels
    plot_labels(data_with_labels)
    

if __name__ == '__main__':
    main()