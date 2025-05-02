import numpy as np
import os
from typing import Optional


class Simulator:
    """
    Simulate epoched MEG/EEG data with multivariate patterns.

    This class generates simulated EEG/MEG data with predefined experimental
    effects, allowing for controlled evaluation of analysis methods. Effects are
    introduced at specified time windows, serving as ground truth signals.

    The method used here is based on the approach implemented in **SPM's `DEMO_CVA_RSA.m`**
    function, originally developed by **Karl Friston and Peter Zeidman** [1]_.
    Our implementation extends their method by incorporating **time-resolved effects**,
    allowing for dynamic experimental manipulations over specified time windows.

    Parameters
    ----------
    X : array, shape (n_epochs, n_experimental_conditions)
        Between-trial design matrix specifying experimental manipulations.
    noise_std : float
        Standard deviation of additive Gaussian noise (applied before channel covariance).
    n_channels : int
        Number of channels (e.g., sensors or components) in the simulated data.
    n_subjects : int
        Number of subjects to simulate.
    tmin : float
        Start time (in seconds) of each epoch.
    tmax : float
        End time (in seconds) of each epoch.
    sfreq : float
        Sampling frequency in Hz.
    t_win : array, shape (n_effects, 2)
        Time windows (start, end) in seconds where each experimental effect is nonzero.
    effects : array, shape (n_effects,)
        Indices of the experimental conditions (columns of ``X``) associated with
        time-locked effects. Each entry corresponds to a row in ``t_win``.
    effects_amp : array, shape (n_effects,) | None, optional
        Amplitudes of the effects. Effects are simulated by sampling beta parameters
        from a normal distribution across channels, with ``effects_amp`` defining
        the variance of the distribution. Do not specify if you use effect size
        and reciprocally. Default is None (use effect size instead).
    effect_size : array, shape (n_effects,), optional
        Standardized multivariate effect size (Mahalanobis distance) for each effect.
        Internally converted to an amplitude :math:`a` (which scales the spread of the betas)
        by solving:

        .. math::

            d' = \\frac{a}{\\sigma} \\cdot \\sqrt{v^T \\Sigma^{-1} v}

        where :math:`\\sigma` = ``noise_std``, :math:`\\Sigma` = ``ch_cov`` (channel covariance),
        and :math:`v` = channel pattern (unit vector across channels).

        Thus the injected β-weights satisfy a Mahalanobis distance of d'
        between condition centroids, yielding theoretical decoding
        accuracy ≈ :math:`\\Phi(d'/2)`. Do not use if you specify effects_amp directly.
        Default is 0.5.
    ch_cov : array, shape (n_channels, n_channels) | None, optional
        Channel data covariance matrix for the simulated data. If None, an identity
        matrix is used (i.e., no cross-channel correlations).
    kern : array, shape (n_samples, ), optional
        Temporal kernel. Should be 1 dimensional. If none, no temporal kernel is applied.
    intersub_noise_std : float, optional
        Inter-subject standard deviation in the effect amp. If not specified, assume no
        inter-subject variability.

    Attributes
    ----------
    data : list of numpy array(s) of shape (n_epochs, n_channels, n_samples)
        Simulated data for each subject.

    Raises
    ------
    ValueError
        If the number of rows in ``t_win`` does not match the length of ``effects``.

    Notes
    -----
    - This function follows the **same methodological principles** as ``DEMO_CVA_RSA.m`` from SPM,
      but extends it by adding time-resolved experimental effects.
    - The original implementation in SPM was developed by **Karl Friston and Peter Zeidman**.
    - The generated data follows an event-related structure, suitable for 
      classification and decoding analyses.
    - Effects are injected into selected experimental conditions based on ``X``.

    References
    ----------
    .. [1] Friston, K., & Zeidman, P. "DEMO_CVA_RSA.m", Statistical Parametric Mapping (SPM).
           Available at: https://github.com/spm/spm/blob/main/toolbox/DEM/DEMO_CVA_RSA.m

    Examples
    --------
    Simulating a simple dataset with 20 subjects and a single experimental effect:

    >>> import numpy as np
    >>> from meeg_simulator import simulate_data
    >>> X = np.random.randn(100, 2)  # 100 trials, 2 experimental conditions
    >>> t_win = np.array([[0.2, 0.5]])  # Effect between 200-500 ms
    >>> effects = np.array([1])  # Effect corresponds to second column of X
    >>> epochs_list = simulate_data(X, noise_std=0.1, n_channels=64, n_subjects=20,
    ...                             tmin=-0.2, tmax=0.8, sfreq=250,
    ...                             t_win=t_win, effects=effects)
    >>> print(len(epochs_list))  # Should return 20 subjects
    """

    def __init__(
        self,
        X: np.ndarray,
        noise_std: float,
        n_channels: int,
        n_subjects: int,
        tmin: float,
        tmax: float,
        sfreq: float,
        t_win: np.ndarray,
        effects: np.ndarray,
        effects_amp: Optional[np.ndarray] = None,
        effect_size: Optional[np.ndarray] = None,
        ch_cov: Optional[np.ndarray] = None,
        kern: Optional[np.ndarray] = None,
        intersub_noise_std: Optional[float] = 0,
    ):
        # ---------------------------------------------------------------------
        # 1. Check inputs & compute the number of samples
        # ---------------------------------------------------------------------
        if len(effects) != len(t_win):
            raise ValueError(
                "The dimension of 'effects' and 't_win' do not match! "
                "There should be exactly one time window for each effect."
            )
        if ch_cov is None:
            ch_cov = np.eye(n_channels)  # Identity covariance matrix
        elif ch_cov.shape != (n_channels, n_channels):
            raise ValueError(
                f"ch_cov must be shape ({n_channels}, {n_channels}), "
                f"but got {ch_cov.shape}."
            )
        if kern is not None:
            if len(kern) == 0 or kern.ndim != 1:
                raise ValueError("kern must be a 1-D numpy array.")
            # Ensure float dtype for later maths
            kern = kern.astype(float)
            # Normalize the kernel (so that it sums up to 1):
            kern = kern / kern.sum()

        # ------------------------------------------------------------------
        #  Resolve effect amplitudes (channel-norm–corrected)
        # ------------------------------------------------------------------
        if effects_amp is not None and effect_size is not None:
            raise ValueError("Pass either 'effects_amp' or 'effect_size', not both.")

        if effects_amp is not None:
            effects_amp = np.asarray(effects_amp, float) / np.sqrt(n_channels)

        elif effect_size is not None:
            effect_size = np.asarray(effect_size, float)
            if effect_size.shape != (len(effects),):
                raise ValueError("'effect_size' must match len(effects).")
            # Normalize effect size by the channel covariance matrix:
            inv_cov = np.linalg.pinv(ch_cov)  # pseudoinverse
            denom = np.sqrt(np.trace(inv_cov))  # = √n_channels  if Σ = I
            effects_amp = effect_size * noise_std / denom  # guarantees d′ = effect_size

        else:  # default: effect size of 0.5 (mahlanobis distance)
            inv_cov = np.linalg.pinv(ch_cov)  # safe inverse
            denom = np.sqrt(np.trace(inv_cov))  # = √n_channels if Σ = I
            effects_amp = (
                np.full(len(effects), 0.5) * noise_std / denom
            )  # guarantees d′ = effect_size

        self.X = X
        self.noise_std = noise_std
        self.n_channels = n_channels
        self.n_subjects = n_subjects
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.t_win = t_win
        self.effects = effects
        self.effects_amp = effects_amp
        self.ch_cov = ch_cov
        self.n_epochs, self.n_exp_conditions = X.shape
        self.kern = kern

        # Make sure we get an integer sample count:
        n_samples = int(round((tmax - tmin) * sfreq)) - 1
        if n_samples <= 0:
            raise ValueError(
                "Derived 'n_samples' must be positive. Check tmin, tmax, and sfreq."
            )

        # ---------------------------------------------------------------------
        # 2. Create a time vector for each epoch and the FIR (identity) within-trial design
        # ---------------------------------------------------------------------
        t = np.linspace(tmin, tmax, n_samples + 1, endpoint=False)[1:]
        # Identity matrix => one column per time point
        Xt = np.eye(n_samples)  # shape => [n_samples, n_samples]

        # ---------------------------------------------------------------------
        # 3. Construct the full design matrix with the Kronecker product
        #    shape => [n_epochs*n_samples, self.n_exp_conditions*n_samples]
        # ---------------------------------------------------------------------
        X_full = np.kron(X, Xt)

        # ---------------------------------------------------------------------
        # 4. Intercept term across all trials and samples
        #    shape => [n_epochs*n_samples, 1]
        # ---------------------------------------------------------------------
        X0 = np.ones((self.n_epochs * n_samples, 1), dtype=float)

        # ---------------------------------------------------------------------
        # 5. Prepare outputs: loop over subjects and simulate data
        # ---------------------------------------------------------------------
        self.data = []

        # Each subject gets unique random draws
        for _ in range(n_subjects):
            # ---------------------------------------------------------------------
            # 6. Build a "time activation" matrix for each experimental condition
            #    cv.shape => [n_samples, self.n_exp_conditions]
            #    This marks time points in which each condition is active (1) or inactive (0)
            # ---------------------------------------------------------------------
            cv = np.zeros((n_samples, self.n_exp_conditions), dtype=float)
            # Preallocate for storing subject specific amplitudes:
            subject_effect_amp = np.zeros(effects_amp.shape)
            for idx, eff_cond in enumerate(effects):
                # Identify which samples lie in the desired time window
                start_t, end_t = t_win[idx]
                # Mask for t in [start_t, end_t]
                mask = (t >= start_t) & (t <= end_t)
                subject_effect_amp[idx] = np.random.normal(
                    loc=effects_amp[idx], scale=intersub_noise_std
                )
                cv[mask, eff_cond] = subject_effect_amp[idx]

            # --------  OPTIONAL  temporal convolution  (causal only)  --------
            if kern is not None:
                for c in range(self.n_exp_conditions):
                    # causal (forward-only) convolution, keep length = n_samples
                    cv[:, c] = np.convolve(cv[:, c], kern, mode="full")[:n_samples]
                    # Ensure that the max value after convolution matches the effect_amp specified
                    peak = np.max(np.abs(cv[:, c]))
                    if peak <= 0:
                        raise RuntimeError("Convolution dropped all signal!")
                    cv[:, c] *= subject_effect_amp[idx] / peak

            # Flatten cv to shape [n_samples*self.n_exp_conditions] so we can build a diagonal "selector"
            # and multiply it by random effects. That ensures only certain (time, condition) combos
            # end up non-zero.
            cv_diagonal = np.diag(
                cv.T.flatten()
            )  # shape => [n_samples*self.n_exp_conditions, n_samples*self.n_exp_conditions]

            # --------------------------------------------------
            # 7a. Build subject-specific effect weights
            #     shape => [n_samples*self.n_exp_conditions, n_channels]
            # --------------------------------------------------
            # Generate beta parameters randomy sampled from a standard normal distribution,
            # but using CV to set the effects to their desired effect sizes. Adding channel covariance across effects
            B = (
                cv_diagonal
                @ np.random.randn(n_samples * self.n_exp_conditions, n_channels)
                @ ch_cov
            )  # shape => [n_samples*self.n_exp_conditions, n_channels]

            # --------------------------------------------------
            # 7b. Combine with the full design
            #     shape => [n_epochs*n_samples, n_channels]
            # --------------------------------------------------
            sub_data = X_full @ B

            # --------------------------------------------------
            # 7c. Add intercept
            #     shape => [n_epochs*n_samples, n_channels]
            # --------------------------------------------------
            B0 = np.random.randn(1, n_channels) / 16.0
            intercept = X0 @ B0
            sub_data += intercept

            # --------------------------------------------------
            # 7d. Add noise (channel-correlated)
            # --------------------------------------------------
            noise = np.random.randn(self.n_epochs * n_samples, n_channels)
            # Apply channel covariance
            noise = noise @ ch_cov
            noise *= noise_std

            sub_data += noise

            # --------------------------------------------------
            # 7e. Store simulated data for this subject
            # --------------------------------------------------
            # Reshape the data:
            # Reshape to [N, T, C] (trials x time x channels)
            sub_data = np.transpose(
                sub_data.reshape([self.n_epochs, len(t), n_channels]), [0, 2, 1]
            )
            self.data.append(sub_data)

    def generate_events(
        self,
        X: np.ndarray = None,
        cond_names: list = None,
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
            X = self.X

        n_trials, n_conds = X.shape

        # Set default condition names if not provided
        if cond_names is None:
            cond_names = [f"cond{i + 1}" for i in range(n_conds)]

        if len(cond_names) != n_conds:
            raise ValueError(
                f"Number of columns in X ({n_conds}) must match number of cond_names ({len(cond_names)})."
            )

        labels = []
        for trial in X:
            parts = []
            for name, value in zip(cond_names, trial):
                if mapping and name in mapping:
                    label_value = mapping[name].get(value, str(value))
                else:
                    label_value = str(value)
                parts.append(f"{name}_{label_value}")
            labels.append("/".join(parts))

        labels = np.array(labels)

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
        events = np.zeros((n_trials, 3), dtype=int)
        events[:, 0] = onsets
        events[:, 2] = event_nums

        return events, event_id

    def export_to_mne(
        self,
        ch_type: str = "eeg",
        X: np.ndarray = None,
        cond_names: list = None,
        mapping: dict = None,
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
        cond_names : list of str, optional
            Names of the experimental conditions (columns of X).
            If None, defaults to ["cond1", "cond2", ..., "condN"].
        mapping : dict, optional
            Dictionary specifying custom label mapping per condition.
            Format: {condition_name: {original_value: label, ...}, ...}

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
                f"    pip install mne\n"
                f"    conda install -c conda-forge mne"
            )
        # Create events:
        events, event_id = self.generate_events(X, cond_names, mapping)

        # Prepare info for epochs object in the end
        info = create_info(
            [f"CH{n:03}" for n in range(self.n_channels)],
            ch_types=[ch_type] * self.n_channels,
            sfreq=self.sfreq,
        )
        epochs_list = [
            EpochsArray(data, info, tmin=self.tmin, events=events, event_id=event_id)
            for data in self.data
        ]
        return epochs_list

    def export_to_eeglab(
        self,
        X: np.ndarray = None,
        cond_names: list = None,
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
        events, event_id = self.generate_events(X, cond_names, mapping)

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
