import mne
import pylustrator
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np
from multisim import Simulator
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import SlidingEstimator, GeneralizingEstimator, cross_val_multiscore
from mne.stats import permutation_cluster_1samp_test, bootstrap_confidence_interval
from scipy.stats import gamma as gamma_dist


def do_decoding(epochs, labels, estimator):
    scores = {' vs. '.join(np.sort(list(set(lbls)))): [] for lbls in labels}
    for epo in epochs:
        for lbls in labels:
            # Extract the data:
            data = epo.get_data()
            # Classification of category
            scores[' vs. '.join(np.sort(list(set(lbls))))].append(
                np.mean(
                    cross_val_multiscore(
                        estimator, data, lbls, cv=5, n_jobs=-1, verbose="WARNING"
                    ),
                    axis=0,
                )
            )
    return {key: np.array(val) for key, val in scores.items()}

# =================================================================================
# 1. Simulate data with various effects specifications:

# ====================================================
# 1.1. Specify fixed parameters:
n_channels = 8  # EEG system with 32 electrodes
n_subjects = 20  # Recording from 20 subjects
noise_std = 1 # Variance of the data
ch_cov = None  # Assuming that the data of each sensor are independent
tmin = -0.2
tmax = 1.0
sfreq = 50  # Simulating data at 50Hz
X = np.array([[1, 1, -1, -1] * 20, [1, -1] * 40]).T # Design matrix
cond_names = ["category", "attention"] # Name of experimental conditions
X = pd.DataFrame(X, columns=cond_names)  # Add a column for the interaction between category and attention
mapping = {"category": {1: "face", -1: "object"}, "attention": {1: "attended", -1: "unattended"}}
cate_lbl = np.array([mapping["category"][val] for val in X.to_numpy()[:, 0]]) # Labels for decoding
att_lbl = np.array([mapping["attention"][val] for val in X.to_numpy()[:, 1]]) # Labels for decoding
ch_cov = scipy.linalg.toeplitz(np.linspace(1, 0, 16)**2) # Channels by channels covariance matrix

# ====================================================
# 1.2. Specify effects:
effects_strong = [
    {"condition": 'category',
        "windows": [0.1, 0.2],
        "effect_size": 4},
    {"condition": 'attention',
        "windows": [0.4, 0.5],
        "effect_size": 4},
]
effects_medium = [
    {"condition": 'category',
        "windows": [0.1, 0.2],
        "effect_size": 0.5},
    {"condition": 'attention',
        "windows": [0.4, 0.5],
        "effect_size": 0.5},
]
effects_temp_gen = [
    {"condition": 'category',
        "windows": [[0.1, 0.2], [0.6, 0.7]],
        "effect_size": 0.5}
]

# Create kernel:
t = np.arange(0, 0.6, 1 / sfreq)  # time vector (in seconds)
kernel = gamma_dist(a=2, scale=0.05).pdf(t)

# ====================================================
# 1.3. Simulate effects:

# 1.3.1. Strong effects:
sims = Simulator(
    X,  # Design matrix
    effects_strong,    # effects to simulate
    noise_std,  # Observation noise
    n_channels,  # Number of channelss
    n_subjects,  # Number of subjects
    -0.25,
    1.0,  # Start and end of epochs
    sfreq,  # Sampling frequency of the data
    random_state=48
)
epochs_strong = sims.export_to_mne(X=X.copy(), mapping=mapping)

# 1.3.2. Effects with kernel
sims = Simulator(
    X,  # Design matrix
    effects_medium,    # effects to simulate
    noise_std,  # Observation noise
    n_channels,  # Number of channelss
    n_subjects,  # Number of subjects
    -0.25,
    1.0,  # Start and end of epochs
    sfreq,  # Sampling frequency of the data
    kern=kernel,
    random_state=48
)
epochs_kernel = sims.export_to_mne(X=X.copy(), mapping=mapping)

# 1.3.3. Effects with temporal generalization
sims = Simulator(
    X,  # Design matrix
    effects_temp_gen,    # effects to simulate
    noise_std,  # Observation noise
    n_channels,  # Number of channelss
    n_subjects,  # Number of subjects
    -0.25,
    1.0,  # Start and end of epochs
    sfreq,  # Sampling frequency of the data
    kern=kernel,
    random_state=48
)
epochs_temp_gen = sims.export_to_mne(X=X.copy(), mapping=mapping)

# =================================================================================
# 2. Perform decoding:

# Create the classifier:
clf = make_pipeline(StandardScaler(), SVC())
# Time resolved
time_decod = SlidingEstimator(clf, n_jobs=None, scoring="accuracy", verbose=True)
cross_temp_decod = GeneralizingEstimator(clf, n_jobs=None, scoring="accuracy", verbose=True)

scores_strong = do_decoding(epochs_strong, [cate_lbl, att_lbl], time_decod)
scores_kernel = do_decoding(epochs_kernel, [cate_lbl, att_lbl], time_decod)
scores_temp_gen = do_decoding(epochs_temp_gen, [cate_lbl, att_lbl], cross_temp_decod)

# =================================================================================
# 3. Plotting
pylustrator.start()
fig, ax = plt.subplots(4, 3, figsize=(8.27, 11.69))

# ====================================================
# 3.1. Upper row (data parameters):
# 3.1.1. Left: Plot montage
montage = mne.channels.make_standard_montage("biosemi16")
montage.plot(kind='topomap', show_names=False, axes=ax[0, 0], show=False)
ax[0, 0].set_title("'n_channels'", fontsize=12, fontfamily='monospace', color='purple', 
                   bbox=dict(boxstyle="round,pad=0.2", fc="whitesmoke", ec="gray", lw=0.5))

# 3.1.2. Middle: Plot the design matrix:
X = pd.DataFrame({
    "category": ['face', 'object', 'face', 'object', '...'],
    "attention": ['attended', 'attended', 'unattended', 'unattended', '...'],
})
ax[0, 1].set_axis_off()
pd.plotting.table(ax[0, 1], X, loc='upper center', cellLoc='center')
ax[0, 1].set_title("'X'", fontsize=12, fontfamily='monospace', color='purple', 
                   bbox=dict(boxstyle="round,pad=0.2", fc="whitesmoke", ec="gray", lw=0.5))

# 3.1.3. Right: Plot covariance matrix:
ax[0, 2].imshow(ch_cov, aspect='equal', cmap='RdBu_r')
ax[0, 2].set_title("'ch_cov'", fontsize=12, fontfamily='monospace', color='purple', 
                   bbox=dict(boxstyle="round,pad=0.2", fc="whitesmoke", ec="gray", lw=0.5))
ax[0, 2].set_xlabel('Channels')
ax[0, 2].set_ylabel('Channels')

# ====================================================
# 3.2. 2nd row (minimal example):

# 3.2.1. Left: Plot effects dict:
# Convert to formatted string
text = (
    '[{"condition": "category",\n'
    '  "windows": [0.1, 0.2],\n'
    '  "effect_size": 4},\n'
    ' {"condition": "attention",\n'
    '  "windows": [0.4, 0.5],\n'
    '  "effect_size": 4}]'
)
ax[1, 0].text(0, 1, text, fontsize=12, va='top', ha='left', family='monospace')
ax[1, 0].set_title("'effects'", fontsize=12, fontfamily='monospace', color='purple', 
                   bbox=dict(boxstyle="round,pad=0.2", fc="whitesmoke", ec="gray", lw=0.5))
ax[1, 0].set_axis_off()

# 3.2.2. Middle: Plot activation:
data = np.squeeze(epochs_strong[0].get_data(picks=[0]))
im = ax[1, 1].imshow(data[np.argsort(epochs_strong[0].events[:, 2]), :], aspect='auto', 
                     extent=[epochs_strong[0].times[0], epochs_strong[0].times[-1], 1, 80], 
                     origin='lower', cmap='RdBu_r')
ax[1, 1].set_ylabel('Trials')
ax[1, 1].set_xlabel('Time (s)')
ax[1, 1].set_title('Activation')
ax[1, 1].set_yticks([])

# 3.2.3. Right: Plot time resolved decoding:
# Compute the confidence intervals:
ci_low_cate, ci_up_cate = bootstrap_confidence_interval(scores_strong['face vs. object'])
ci_low_att, ci_up_att = bootstrap_confidence_interval(scores_strong['attended vs. unattended'])
# Plot
ax[1, 2].plot(epochs_strong[0].times, np.mean(scores_strong['face vs. object'], axis=0), label="category", color="b")
ax[1, 2].fill_between(epochs_strong[0].times, ci_low_cate, ci_up_cate, alpha=0.3, color="b")
ax[1, 2].plot(epochs_strong[0].times, np.mean(scores_strong['attended vs. unattended'], axis=0), label="attention", color="g")
ax[1, 2].fill_between(epochs_strong[0].times, ci_low_att, ci_up_att, alpha=0.3, color="g")
ax[1, 2].axhline(0.5, color="k", linestyle="--", label="chance")
ax[1, 2].set_xlim([epochs_strong[0].times[0], epochs_strong[0].times[-1]])
ax[1, 2].set_xlabel("Time (s)")
ax[1, 2].set_ylabel("AUC")
ax[1, 2].legend()
ax[1, 2].set_title('Decoding')

# ====================================================
# 3.3. 3rd row (kernel example):
# 3.2.1. Left: Plot effects dict:
text = (
    '[{"condition": "category",\n'
    '  "windows": [0.1, 0.2],\n'
    '  "effect_size": 0.5},\n'
    ' {"condition": "attention",\n'
    '  "windows": [0.4, 0.5],\n'
    '  "effect_size": 0.5}]'
)
ax[2, 0].text(0, 1, text, fontsize=12, va='top', ha='left', family='monospace')
ax[2, 0].set_title("'effects'", fontsize=12, fontfamily='monospace', color='purple', 
                   bbox=dict(boxstyle="round,pad=0.2", fc="whitesmoke", ec="gray", lw=0.5))
ax[2, 0].set_axis_off()


# 3.2.1. Middle: Plot kernel
ax[2, 1].plot(t, kernel)
ax[2, 1].set_xlabel("Time (s)")
ax[2, 1].set_title("'kern'", fontsize=12, fontfamily='monospace', color='purple', 
                   bbox=dict(boxstyle="round,pad=0.2", fc="whitesmoke", ec="gray", lw=0.5))

# Compute the confidence intervals:
ci_low_cate, ci_up_cate = bootstrap_confidence_interval(scores_kernel['face vs. object'])
ci_low_att, ci_up_att = bootstrap_confidence_interval(scores_kernel['attended vs. unattended'])
# Plot
ax[2, 2].plot(epochs_kernel[0].times, np.mean(scores_kernel['face vs. object'], axis=0), label="category", color="b")
ax[2, 2].fill_between(epochs_kernel[0].times, ci_low_cate, ci_up_cate, alpha=0.3, color="b")
ax[2, 2].plot(epochs_kernel[0].times, np.mean(scores_kernel['attended vs. unattended'], axis=0), label="attention", color="g")
ax[2, 2].fill_between(epochs_kernel[0].times, ci_low_att, ci_up_att, alpha=0.3, color="g")
ax[2, 2].axhline(0.5, color="k", linestyle="--", label="chance")
ax[2, 2].set_xlim([epochs_kernel[0].times[0], epochs_kernel[0].times[-1]])
ax[2, 2].set_xlabel("Time (s)")
ax[2, 2].set_ylabel("AUC")
ax[2, 2].set_title('Decoding')

# ====================================================
# 3.4. 4th row (temporal generalization):

# 3.2.1. Left: Plot effects dict:
text = (
    '[{"condition": "category",\n'
    '  "windows": [[0.1, 0.2],\n          [0.6, 0.7]],\n'
    '  "effect_size": 0.5}'
)
ax[3, 0].text(0, 1, text, fontsize=12, va='top', ha='left', family='monospace')
ax[3, 0].set_title("'effects'", fontsize=12, fontfamily='monospace', color='purple', 
                   bbox=dict(boxstyle="round,pad=0.2", fc="whitesmoke", ec="gray", lw=0.5))
ax[3, 0].set_axis_off()

# 3.2.2. Middle: Plot time resolved decoding:
time_res_decoding = np.array([np.diag(scores_temp_gen['face vs. object'][i, :, :])for i in range(scores_temp_gen['face vs. object'].shape[0])])
ci_low_cate, ci_up_cate = bootstrap_confidence_interval(time_res_decoding)
# Plot time resolves:
ax[3, 1].plot(epochs_temp_gen[0].times, np.mean(time_res_decoding, axis=0), label="category", color="b")
ax[3, 1].fill_between(epochs_temp_gen[0].times, ci_low_cate, ci_up_cate, alpha=0.3, color="b")
ax[3, 1].axhline(0.5, color="k", linestyle="--", label="chance")
ax[3, 1].set_xlim([epochs_temp_gen[0].times[0], epochs_temp_gen[0].times[-1]])
ax[3, 1].set_xlabel("Time (s)")
ax[3, 1].set_ylabel("AUC")
ax[3, 1].set_title("Decoding")
# 3.2.3. Middle: Plot temporal generalization
im = ax[3, 2].imshow(np.mean(scores_temp_gen['face vs. object'], axis=0), cmap="RdBu_r", 
               origin="lower", extent=epochs_temp_gen[0].times[[0, -1, 0, -1]])
ax[3, 2].axhline(0.0, color="k")
ax[3, 2].axvline(0.0, color="k")
ax[3, 2].xaxis.set_ticks_position("bottom")
ax[3, 2].set_xlabel(
    'Condition: Testing Time (s)',
)
ax[3, 2].set_ylabel('Condition: Training Time (s)')
ax[3, 2].set_title('Temporal generalization')
fig.colorbar(im, ax=ax[3, 2], label="Performance (ROC AUC)")

plt.tight_layout()
#% start: automatic generated code from pylustrator
plt.figure(1).ax_dict = {ax.get_label(): ax for ax in plt.figure(1).axes}
import matplotlib as mpl
getattr(plt.figure(1), '_pylustrator_init', lambda: ...)()
plt.figure(1).set_size_inches(20.960000/2.54, 19.000000/2.54, forward=True)
plt.figure(1).ax_dict["<colorbar>"].set(position=[0.9155, 0.06277, 0.02727, 0.1532], ylabel='')
plt.figure(1).ax_dict["<colorbar>"].get_yaxis().get_label().set(text='')
plt.figure(1).axes[0].set(position=[0.04508, 0.8422, 0.1829, 0.1268])
plt.figure(1).axes[0].set_position([0.049181, 0.765435, 0.173719, 0.188549])
plt.figure(1).axes[1].set(position=[0.3026, 0.7288, 0.2653, 0.2425])
plt.figure(1).axes[1].set_position([0.301515, 0.596774, 0.264390, 0.360550])
plt.figure(1).axes[2].set(position=[0.7382, 0.8422, 0.1829, 0.1291])
plt.figure(1).axes[2].set_position([0.739923, 0.765435, 0.173719, 0.191882])
plt.figure(1).axes[3].set_position([0.018076, 0.321180, 0.182232, 0.264405])
plt.figure(1).axes[3].title.set(visible=False)
plt.figure(1).axes[3].texts[0].set(position=(-0.0116, 1.304), fontsize=9.)
plt.figure(1).axes[4].set(position=[0.3015, 0.5349, 0.2767, 0.1312])
plt.figure(1).axes[4].yaxis.labelpad = -1.279052
plt.figure(1).axes[5].legend(loc=(0.6758, 0.5356), frameon=False, fontsize=6.)
plt.figure(1).axes[5].set(position=[0.6991, 0.5346, 0.2765, 0.1314])
plt.figure(1).axes[5].yaxis.labelpad = -23.423687
plt.figure(1).axes[5].get_legend().set(visible=True)
plt.figure(1).axes[5].get_yaxis().get_label().set(position=(448.9, 1.061), fontsize=12., rotation=0.)
plt.figure(1).axes[6].set_position([0.018076, -0.045751, 0.182232, 0.264405])
plt.figure(1).axes[6].title.set(visible=False)
plt.figure(1).axes[6].texts[0].set(position=(-0.0116, 1.841), fontsize=9.)
plt.figure(1).axes[7].set(position=[0.3015, 0.3098, 0.2767, 0.1312])
plt.figure(1).axes[8].set(position=[0.699, 0.3098, 0.2767, 0.1312], ylabel='')
plt.figure(1).axes[8].get_yaxis().get_label().set(text='')
plt.figure(1).axes[9].set(visible=True)
plt.figure(1).axes[9].set_position([0.019019, -0.183560, 0.182273, 0.264354])
plt.figure(1).axes[9].title.set(visible=False)
plt.figure(1).axes[9].texts[0].set(position=(-0.0167, 1.511), fontsize=9.)
plt.figure(1).axes[10].set(position=[0.3015, 0.08473, 0.2767, 0.1312])
plt.figure(1).axes[11].set(position=[0.7679, 0.063, 0.139, 0.1529], xlabel='Testing Time (s)', ylabel='Training time (s)')
plt.figure(1).axes[11].get_xaxis().get_label().set(position=(0.5, 253.2), text='Testing Time (s)')
plt.figure(1).axes[11].get_yaxis().get_label().set(position=(447., 0.5), text='Training time (s)')
plt.figure(1).text(0.0890, 0.4516, "'effects'", transform=plt.figure(1).transFigure, fontsize=12., color='#800080ff')  # id=plt.figure(1).texts[0].new
plt.figure(1).text(0.0890, 0.2302, "'effects'", transform=plt.figure(1).transFigure, fontsize=12., color='#800080ff')  # id=plt.figure(1).texts[1].new
plt.figure(1).text(0.0893, 0.7802, "'effects'", transform=plt.figure(1).transFigure, fontsize=12., color='#800080ff')  # id=plt.figure(1).texts[2].new
plt.figure(1).texts[2].set_position([0.089034, 0.673160])
plt.figure(1).text(0.9363, 0.1997, 'AUC', transform=plt.figure(1).transFigure, fontsize=12.)  # id=plt.figure(1).texts[3].new
plt.figure(1).text(0.0053, 0.9731, 'A', transform=plt.figure(1).transFigure, fontsize=16., weight='bold')  # id=plt.figure(1).texts[4].new
plt.figure(1).texts[4].set_position([0.005295, 0.960005])
plt.figure(1).text(0.0053, 0.7802, 'B', transform=plt.figure(1).transFigure, fontsize=16., weight='bold')  # id=plt.figure(1).texts[5].new
plt.figure(1).texts[5].set_position([0.005316, 0.673160])
plt.figure(1).text(0.0053, 0.4500, 'C', transform=plt.figure(1).transFigure, fontsize=16., weight='bold')  # id=plt.figure(1).texts[6].new
plt.figure(1).text(0.0053, 0.2302, 'D', transform=plt.figure(1).transFigure, fontsize=16., weight='bold')  # id=plt.figure(1).texts[7].new
plt.figure(1).text(0.2725, 0.4516, 'a.u.', transform=plt.figure(1).transFigure, )  # id=plt.figure(1).texts[8].new
plt.figure(1).text(0.6664, 0.4483, 'AUC', transform=plt.figure(1).transFigure, fontsize=12.)  # id=plt.figure(1).texts[9].new
#% end: automatic generated code from pylustrator

plt.savefig('./paper/figure1.png')
plt.close()