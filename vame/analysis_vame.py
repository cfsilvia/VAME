import numpy as np
import matplotlib.pyplot as plt
dir =r"U:\Users\Ruthi\2025\BMR2\VAME\Your-VAME-Project-Jul21-2025\results\BMR2_with_landmarks_left_to _use\VAME\hmm-5\\"
# Load outputs
latent = np.load(dir + 'latent_vector_BMR2_with_landmarks_left_to _use.npy')
labels = np.load(dir + '5_km_label_BMR2_with_landmarks_left_to _use.npy')
usage  = np.load(dir + 'motif_usage_BMR2_with_landmarks_left_to _use.npy')

# 1) Bar plot of motif usage
plt.figure()
plt.bar(range(len(usage)), usage)
plt.xlabel('Motif ID')
plt.ylabel('Frame count')
plt.title('Motif usage')
plt.show()

# 2) Time series of motif labels
plt.figure()
plt.plot(labels, lw=0.5)
plt.xlabel('Frame index')
plt.ylabel('Motif label')
plt.title('Motif assignments over time')
plt.show()

# 3) Scatter of first two latent dims, colored by motif
plt.figure()
plt.scatter(latent[:,0], latent[:,1], c=labels, s=1)
plt.xlabel('Latent dim 1')
plt.ylabel('Latent dim 2')
plt.title('Latent space projection')
plt.show()
