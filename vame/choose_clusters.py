from hmmlearn import hmm
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

X = np.load(r"U:\Users\Ruthi\2025\BMR10\VAME\BMR10-VAME-Project-Jul24-2025\results\BMR10_with_landmarks_left\VAME\hmm-15\\latent_vector_BMR10_with_landmarks_left.npy")
X = PCA(n_components=10).fit_transform(X)

log_likelihoods, aics, bics = [], [], []
Ks = range(2, 21)

for k in Ks:
    model = hmm.GaussianHMM(n_components=k, covariance_type='full', random_state=0)
    model.fit(X)
    logL = model.score(X)
    n_params = k * (k - 1) + k * X.shape[1] * 2  # rough param count
    log_likelihoods.append(logL)
    aics.append(2 * n_params - 2 * logL)
    bics.append(np.log(X.shape[0]) * n_params - 2 * logL)

# Plot
plt.plot(Ks, aics, label="AIC")
plt.plot(Ks, bics, label="BIC")
plt.xlabel("Number of States")
plt.ylabel("Score")
plt.title("Model Selection (HMM)")
plt.legend()
plt.grid()
plt.show()
