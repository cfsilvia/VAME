import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

dir =r"U:\Users\Ruthi\2025\BMR2\VAME\Your-VAME-Project-Jul21-2025\results\BMR2_with_landmarks_left_to _use\VAME\hmm-5\\"
dir_data = r"U:\Users\Ruthi\2025\BMR2\VAME\Your-VAME-Project-Jul21-2025\videos\pose_estimation\\"
# 1. Load your raw pose CSV
#    Adjust the path to wherever your video-1.csv lives
raw = pd.read_csv(dir_data + "BMR2_with_landmarks_left_to _use.csv")
#raw.columns = [f"{bp.lower()}_{coord.lower()}" for bp, coord in raw.columns]
# 2. Load your motif labels (k=15)
labels = np.load(dir + r'5_km_label_BMR2_with_landmarks_left_to _use.npy')

# 3. Add the labels as a new column
raw['motif'] = pd.Series(labels)

# 4. (Optional) Inspect the first few rows
print(raw[['BM_snout_x', 'BM_snout_y', 'motif']].head())

# 5. Plot nose positions, colored by motif:
plt.figure(figsize=(8,8))
scatter = plt.scatter(
    raw['BM_snout_x'], 
    raw['BM_snout_y'], 
    c=raw['motif'],      # color by motif ID
    s=1,                 # dot size
    cmap='tab20',  # a qualitative colormap
    alpha = 0.8
)
plt.axis('equal')
plt.title('Nose trajectory colored by motif')
plt.xlabel('BM_snout_x')
plt.ylabel('BM_snout_y')
# cbar = plt.colorbar(scatter, ticks=np.unique(raw['motif']))
# cbar.set_label('Motif ID')

plt.show()


# Keep only rows without NaNs in key columns
clean = raw[['BM_snout_y', 'motif']].dropna().copy()
clean['frame'] = clean.index

# Prepare coordinates
x = clean['frame'].values
y = clean['BM_snout_y'].values
motifs = clean['motif'].values

# Build segments between consecutive points
points = np.array([x, y]).T.reshape(-1, 1, 2)
segments = np.concatenate([points[:-1], points[1:]], axis=1)

# Motif per segment (use start point's motif)
segment_motifs = motifs[:-1]

# Create the line collection
lc = LineCollection(
    segments,
    cmap='tab20',
    norm=plt.Normalize(segment_motifs.min(), segment_motifs.max())
)
lc.set_array(segment_motifs)
lc.set_linewidth(1.5)

# Plot
fig, ax = plt.subplots(figsize=(14, 5))
ax.add_collection(lc)
ax.set_xlim(x.min(), x.max())
ax.set_ylim(y.min(), y.max())
ax.set_title('BM_snout_y Over Time (Filled Line Colored by Motif)')
ax.set_xlabel('Frame')
ax.set_ylabel('BM_snout_y')
cbar = plt.colorbar(lc, ax=ax, ticks=np.unique(segment_motifs))
cbar.set_label('Motif ID')
plt.tight_layout()
plt.show()