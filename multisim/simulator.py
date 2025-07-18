import numpy as np
import pandas as pd
import os
from typing import List, Dict, Optional, Sequence


class Simulator:
    """
    Simulate epoched MEG/EEG data with multivariate patterns.

    This generator follows the general spirit of Friston & Zeidman's
    **SPM `DEMO_CVA_RSA.m`** but adds *time-resolved* effects so that each
    experimental manipulation can switch on/off within an epoch.

    Parameters
    ----------
    X : pandas.DataFrame, shape (n_epochs, n_conditions)
        Trial-by-trial design matrix. **Must have named columns**.
    effects : list[dict]
        Each dict describes **one multivariate pattern**.  Required keys

        ================ ================================
        Key              Meaning
        ================ ================================
        ``condition``    Column name in ``X``.
        ``windows``      List of ``(start, end)`` time pairs (seconds).
        ``effect_size``  Mahalanobis distance (:math:`d'`).
        ``effect_amp``   Amplitude of β-weights directly.
        ================ ================================

        Specify either effect_size or effect_amp, not both.

        *One dict → one pattern.*  To share a pattern across several windows,
        list them in a single dict.  To have different patterns, provide
        multiple dicts with the same ``condition``.
    noise_std : float
        Standard deviation of additive Gaussian noise.
    n_channels : int
        Number of channels (sensors/components).
    n_subjects : int
        Number of subjects.
    tmin, tmax : float
        Epoch start and end (seconds).
    sfreq : float
        Sampling frequency (Hz).
    ch_cov : ndarray shape (n_channels, n_channels), optional
        Channel covariance. Defaults to ``None`` (identity matrix)
    kern : 1-D ndarray, optional
        Optional causal temporal kernel. Defaults to ``None``.
    intersub_noise_std : float, optional
        Standard deviation of between-subject variability in effect amplitude. Defaults to 0.
    random_state : int | numpy.random.Generator | None
        Seed or generator for reproducibility.

    Notes
    -----
    **Mathematical relation between ``effect_size`` and ``effect_amp``**

    For a single pattern :math:`v` (vector across channels) with amplitude :math:`a`,
    noise standard deviation :math:`\\sigma` and channel covariance :math:`\\Sigma`, the theoretical
    Mahalanobis distance between condition centroids is

    .. math::

        d' = \\frac{a}{\\sigma} \\cdot \\sqrt{v^T \\Sigma^{-1} v}

    where :math:`\\sigma` = ``noise_std``, :math:`\\Sigma` = ``ch_cov`` (channel covariance),
    and :math:`v` = channel pattern (unit vector across channels).

    We draw :math:`v` from a standard normal and **normalise it to unit Mahalanobis
    norm** (:math:`‖v‖_{Σ^{-1}} = 1`).  Therefore the distance simplifies to

    .. math::

        d' = \\frac{a}{\\sigma}

    so the amplitude we must inject is

    .. math::

        a = d'\\sigma

    If the user supplies ``effect_amp`` directly, that value is taken instead
    and the conversion above is skipped.

    References
    ----------
    .. [1] Friston, K., & Zeidman, P. "DEMO_CVA_RSA.m", Statistical Parametric Mapping (SPM).
           Available at: https://github.com/spm/spm/blob/main/toolbox/DEM/DEMO_CVA_RSA.m

    Examples
    --------
    Simulating a simple dataset with 20 subjects and a single experimental effect:

    >>> import numpy as np
    >>> from multisim import Simulator
    >>> X = pd.DataFrame(np.random.randn(100, 1), columns=["category"]) # 100 trials, 1 experimental condition
    >>> t_win = np.array([[0.2, 0.5]])  # Effect between 200-500 ms
    >>> effects = [
        {"condition": 'category',
         "windows": [0.1, 0.3],
         "effect_size": 0.5
        }
    ]
    >>> sims = Simulator(
    ...   X, noise_std=0.1, n_channels=64, n_subjects=20,
    ...   tmin=-0.2, tmax=0.8, sfreq=250,
    ...   t_win=t_win, effects=effects
    ...   )
    >>> sim.summary()  # Should return 20 subjects
    """

    def __init__(
        self,
        X: pd.DataFrame,
        effects: List[Dict],
        noise_std: float,
        n_channels: int,
        n_subjects: int,
        tmin: float,
        tmax: float,
        sfreq: float,
        ch_cov: Optional[np.ndarray] = None,
        kern: Optional[np.ndarray] = None,
        intersub_noise_std: float = 0.0,
        random_state: Optional[object] = None,
    ) -> None:
        # 1. Initialize class:
        self.rng = np.random.default_rng(random_state)
        self.X = X.copy()
        self.noise_std = float(noise_std)
        self.n_channels = int(n_channels)
        self.n_subjects = int(n_subjects)
        self.tmin, self.tmax, self.sfreq = float(tmin), float(tmax), float(sfreq)
        self.intersub_noise_std = float(intersub_noise_std)
        self.n_epochs, self.n_exp_conditions = X.shape

        if self.tmax <= self.tmin:
            raise ValueError("tmax must be greater than tmin.")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive.")

        # 2. Number of samples per epoch:
        self.n_samples = int(round((self.tmax - self.tmin) * self.sfreq)) - 1
        if self.n_samples <= 0:
            raise ValueError("Derived n_samples must be > 0. Check tmin/tmax/sfreq.")

        # 3. Specify channel covariance
        if ch_cov is None:
            ch_cov = np.eye(n_channels)
        ch_cov = np.asarray(ch_cov, float)
        if ch_cov.shape != (n_channels, n_channels):
            raise ValueError("ch_cov must have shape (n_channels, n_channels).")
        self.ch_cov = ch_cov

        # 4. Prepare kernel --------------------------------------------------------
        if kern is not None:
            kern = np.asarray(kern, float).squeeze()
            if kern.ndim != 1 or kern.size == 0:
                raise ValueError("kern must be a 1-D non-empty array.")
            # Normalize kernel
            kern = kern / kern.sum()
        self.kern = kern

        # 5. Prepare effects -------------------------------------------------
        self.effects = self._parse_effects(effects)

        # 6. pre-compute design matrices ----------------------------------
        self._tvec = np.linspace(
            self.tmin, self.tmax, self.n_samples + 1, endpoint=False
        )[1:]
        Xt = np.eye(self.n_samples)  # [T, T]
        self._X_full = np.kron(self.X.values, Xt)  # [N*T, P*T]
        self._X0 = np.ones((self.n_epochs * self.n_samples, 1))  # intercept

        # 7. simulate ------------------------------------------------------
        self.data: List[np.ndarray] = []
        for _ in range(self.n_subjects):
            self.data.append(self._simulate_one_subject())

    def _parse_effects(self, effects_in: Sequence[Dict]) -> List[Dict]:
        """Validate and canonicalise the user-supplied effects list."""
        if not isinstance(effects_in, (list, tuple)) or len(effects_in) == 0:
            raise ValueError("effects must be a non-empty list of dictionaries.")

        effect_val: List[Dict] = []
        for i, eff in enumerate(effects_in):
            if not isinstance(eff, dict):
                raise TypeError(f"Effect #{i} is not a dict.")
            if "condition" not in eff or "windows" not in eff:
                raise ValueError(f"Effect #{i}: must have 'condition' and 'windows'.")

            cond = eff["condition"]
            if cond not in self.X.columns:
                raise ValueError(f"Effect #{i}: unknown condition '{cond}'.")
            col_idx = self.X.columns.get_loc(cond)

            windows = eff["windows"]
            # normalise windows to list[tuple]
            if isinstance(windows, (list, tuple)) and len(windows) > 0:
                if isinstance(windows[0], (int, float)):
                    windows = [tuple(windows)]  # single pair
                else:
                    windows = [tuple(w) for w in windows]
            else:
                raise ValueError(f"Effect #{i}: 'windows' malformed.")

            if "effect_amp" in eff and "effect_size" in eff:
                raise ValueError(
                    f"Effect #{i}: specify *either* effect_size or effect_amp, not both."
                )
            if "effect_amp" not in eff and "effect_size" not in eff:
                raise ValueError(
                    f"Effect #{i}: must provide effect_size or effect_amp."
                )

            # now you don’t need the extra trace-based denominator:
            if "effect_size" in eff:
                amp = float(eff["effect_size"]) * self.noise_std
            else:
                amp = float(eff["effect_amp"]) / np.sqrt(self.n_channels)

            effect_val.append({"col_idx": col_idx, "windows": windows, "base_amp": amp})
        return effect_val

    def _simulate_one_subject(self) -> np.ndarray:
        """Simulate data for a single subject → (epochs, channels, samples)."""
        # 1. initialise betas
        B3d = np.zeros(
            (self.n_samples, self.n_exp_conditions, self.n_channels), dtype=float
        )

        # 2. Loop through each effect:
        for eff in self.effects:
            # subject‑specific amplitude
            amp = self.rng.normal(loc=eff["base_amp"], scale=self.intersub_noise_std)
            # generate random pattern
            v = self.rng.standard_normal(self.n_channels)
            # Normalize it to 1 Mahalanobis distance:
            v /= np.sqrt(v @ np.linalg.pinv(self.ch_cov) @ v)  # unit Mahalanobis norm

            # build time resolved activation time course
            act = np.zeros(self.n_samples, float)
            for t0, t1 in eff["windows"]:
                act[(self._tvec >= t0) & (self._tvec <= t1)] = 1.0
            # Convolve with kernel if specified
            if self.kern is not None:
                act = np.convolve(act, self.kern, mode="full")[: self.n_samples]
            # Normalize activation to 1:
            peak = act.max()
            if peak == 0:
                continue  # no active time points
            act *= amp / peak  # scale so max = amp

            # insert into β (broadcast (time,1)*(1,channel))
            B3d[:, eff["col_idx"], :] += act[:, None] * v

        # flatten to (condition*time, channels) with correct ordering
        B = np.transpose(B3d, (1, 0, 2)).reshape(
            self.n_exp_conditions * self.n_samples, self.n_channels
        )
        B = B @ self.ch_cov  # apply Σ once
        # Generate subject data by multiplying design matrix with betas:
        sub_data = self._X_full @ B
        # Add intercept:
        sub_data += self._X0 @ (self.rng.standard_normal((1, self.n_channels)) / 16.0)
        # Add spatially covaried noise:
        noise = self.rng.standard_normal(
            (self.n_epochs * self.n_samples, self.n_channels)
        )
        sub_data += (noise @ self.ch_cov) * self.noise_std

        # reshape to (epochs, channels, samples)
        return sub_data.reshape(
            self.n_epochs, self.n_samples, self.n_channels
        ).transpose(0, 2, 1)

    def summary(self) -> str:
        """Textual summary of the simulated dataset."""
        return (
            f"Subjects  : {len(self.data)}\n"
            f"Epochs    : {self.n_epochs}\n"
            f"Samples   : {self.n_samples}\n"
            f"Channels  : {self.n_channels}\n"
            f"Conditions: {self.n_exp_conditions} ({', '.join(self.X.columns)})"
        )

    def generate_events(
        self,
        X: np.ndarray = None,
        mapping: dict = None,
    ) -> tuple:
        """
        Generate MNE-compatible events and event_id dictionary from a design matrix.

        Parameters
        ----------
        X : np.ndarray, shape (n_trials, n_conditions), optional
            Design matrix to use. If None, defaults to self.X.
        cond_names : list of str, optional
            Names of the experimental conditions (columns of X).
            If None, defaults to ["cond1", "cond2", ..., "condN"].
        mapping : dict, optional
            Dictionary specifying custom label mapping per condition.
            Format: {condition_name: {original_value: label, ...}, ...}

        Returns
        -------
        events : np.ndarray, shape (n_trials, 3)
            Array of [onset_sample, 0, event_id] per trial.
        event_id : dict
            Dictionary mapping label string to event integer.
        """
        if X is None:
            X = self.X.copy()

        n_trials = X.shape[0]

        # Replace condition id based on mapping:
        if mapping is not None:
            X.replace(mapping, inplace=True)

        # build one combined label per trial:  "condA_val1/condB_val2"
        labels = X.apply(
            lambda row: "/".join(f"{c}_{v}" for c, v in row.items()), axis=1
        ).to_numpy()

        # Unique combinations and event ID mapping
        unique_labels = np.unique(labels)
        event_id = {label: idx + 1 for idx, label in enumerate(unique_labels)}
        # Event numbers for each trial
        event_nums = np.array([event_id[label] for label in labels])

        # Onset sample of each trial (first sample + number of sample in each trial + 1 for all trials but the first)
        onsets = (
            int(round(-self.tmin * self.sfreq))
            + self.data[0].shape[2] * np.arange(n_trials)
            + (np.arange(n_trials) > 0).astype(int)
        )

        # Events array [onset, 0, event_id]
        events = np.column_stack([onsets, np.zeros_like(onsets), event_nums])

        return events, event_id

    def export_to_mne(
        self,
        ch_type: str = "eeg",
        X: np.ndarray = None,
        mapping: dict = None,
        verbose: str = "ERROR",
    ) -> list:
        """
        Export the simulated data to MNE-Python format.

        Parameters
        ----------
        ch_type : str, optional
            Type of simulated channels, e.g., ``'eeg'`` or ``'meg'``.
            Default is ``'eeg'``.
        X : np.ndarray, shape (n_trials, n_conditions), optional
            Design matrix to use. If None, defaults to self.X.
        mapping : dict, optional
            Dictionary specifying custom label mapping per condition.
            Format: {condition_name: {original_value: label, ...}, ...}
        verbose : string, optional
            Verbosity level for MNE, see https://mne.tools/stable/generated/mne.verbose.html

        Returns
        -------
        list of mne.EpochsArray
            List of MNE-Python EpochsArray objects for each subject.
        """
        try:
            from mne import create_info, EpochsArray
        except ImportError:
            raise ImportError(
                "MNE-Python could not be imported. Use the following installation method "
                "appropriate for your environment:\n\n"
                "    pip install mne\n"
                "    conda install -c conda-forge mne"
            )
        # Create events:
        events, event_id = self.generate_events(X, mapping)

        # Prepare info for epochs object in the end
        info = create_info(
            [f"CH{n:03}" for n in range(self.n_channels)],
            ch_types=[ch_type] * self.n_channels,
            sfreq=self.sfreq,
            verbose=verbose,
        )
        epochs_list = [
            EpochsArray(
                data,
                info,
                tmin=self.tmin,
                events=events,
                event_id=event_id,
                verbose=verbose,
            )
            for data in self.data
        ]
        return epochs_list

    def export_to_eeglab(
        self,
        X: np.ndarray = None,
        mapping: dict = None,
        root: str = ".",
        fname_template: str = "sub-{:02d}.set",
    ) -> None:
        """
        Export the simulated data to EEGLAB format (save to file).

        Parameters
        ----------
        ch_type : str
            Type of simulated channels, e.g., 'eeg' or 'meg'. Default is 'eeg'.
        root : str
            Directory where the files will be saved. Default is current directory '.'.
        fname_template : str
            Filename template for each subject, with a placeholder for the subject index.
            Default is 'subject_{:02d}.set'.

        Returns
        -------
        None

        Notes
        -----
        Requires 'mne' and 'eeglabio' packages.
        Each subject's data will be saved as a separate EEGLAB file.
        """
        try:
            from eeglabio.epochs import export_set
        except ImportError:
            raise ImportError(
                "eeglabio could not be imported. Install it with:\n\n"
                "    pip install eeglabio\n"
                "    conda install -c conda-forge eeglabio"
            )

        # Ensure fname_template has a placeholder
        if "{" not in fname_template:
            # No placeholder: append a subject number at the end
            if fname_template.endswith(".set"):
                fname_template = fname_template[:-4] + "_{:02d}.set"
            else:
                fname_template = fname_template + "_{:02d}.set"

        if not os.path.exists(root):
            os.makedirs(root)

        # Create events:
        events, event_id = self.generate_events(X, mapping)

        for i, epo in enumerate(self.data):
            filename = fname_template.format(i)
            filepath = os.path.join(root, filename)
            export_set(
                filepath,
                epo,
                self.sfreq,
                events,
                self.tmin,
                self.tmax,
                [f"CH{n:03}" for n in range(self.n_channels)],
                event_id=event_id,
                ch_locs=None,
                annotations=None,
                ref_channels="common",
                precision="single",
            )

        return None
