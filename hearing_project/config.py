# Hearing project: shared configuration - Jake Abraham
"""
PLAIN WORDS
-----------
Every number that more than one file needs lives here: sample rate, frame sizes, how many
filterbank channels, the shape of the belief-network tree, training settings, the STR levels,
and which dataset folders to use. Changing a value here changes it everywhere, so experiments
stay consistent. Values marked [K] come from Keshavarzi et al. 2021, [B] from Bellur & Elhilali
2020, and [NEW] are choices made for this project.

PSEUDOCODE / NOTES FOR THE IMPLEMENTING AGENT
---------------------------------------------
- This file is real, importable Python. Keep it free of heavy imports (no jax here).
- Other modules do `from config import CFG` and read attributes; never hard-code these numbers.
- If you add a variant or dataset, add it here first.
"""

from dataclasses import dataclass, field
from pathlib import Path


###########################################################
#                     AUDIO / FRAMING                     #
###########################################################

@dataclass(frozen=True)
class AudioCfg:
    sample_rate: int = 16_000          # [K] VCTK downsampled 48k -> 16k
    frame_len: int = 80                # [K] 5 ms Hann window
    hop_len: int = 40                  # [K] 50% overlap -> 2.5 ms hop
    n_fft: int = 128                   # [NEW] zero-pad 80 -> 128 for a power-of-two FFT (65 bins)
    n_bands: int = 64                  # [K] 64 gammatone channels
    f_min_hz: float = 50.0             # [K]
    f_max_hz: float = 8_000.0          # [K]
    log_floor: float = 1e-8            # [NEW] floor before log-compression of band energies
    irm_beta: float = 0.5              # [K] IRM = (S^2 / (S^2 + V^2)) ** beta  (sqrt form, as in proposal Fig. 1)
    lstm_context_frames: int = 3       # [K] baseline RNN sees frames j-2, j-1, j


###########################################################
#               DISTRIBUTED BELIEF NETWORK                #
###########################################################

@dataclass(frozen=True)
class DBNCfg:
    n_layers: int = 4                  # [B] 4 layers -> 1 + 2 + 4 + 8 = 15 local sub-networks (LSNs)
    units_per_lsn: int = 48            # [NEW] slim; Bellur used 300. "bellur_size" preset below.
    basis_width: int = 12              # [NEW] frames per CRBM basis (12 x 2.5 ms = 30 ms); Bellur: 6 x 6 ms = 36 ms. Slowest LSN then sees ~0.45 s
    pool_left: int = 1                 # [B] eta  (left child keeps the parent's rate)
    pool_right: int = 3                # [B] tau  (right child is 3x slower)
    sparsity_target: float = 0.05      # [B] s
    sparsity_weight: float = 5.0       # [B] lambda
    cd_steps: int = 1                  # [NEW] CD-1
    lr: float = 1e-3
    epochs_per_layer: int = 10
    backend: str = "crbm"              # "crbm" (faithful) or "sparse_conv_ae" (fallback if CD is unstable)
    readout_dim_per_lsn: int = 16      # [NEW] each LSN is projected to 16 dims before fusion (15 x 16 = 240)


BELLUR_SIZE = DBNCfg(units_per_lsn=300, basis_width=6, readout_dim_per_lsn=50)  # upper-bound experiment


###########################################################
#                        ATTENTION                        #
###########################################################

@dataclass(frozen=True)
class AttnCfg:
    n_memories: tuple = ("speech", "transient")   # [NEW] Bellur stored only the attended class
    n_free_bases: int = 2              # [B-like] extra unconstrained bases that absorb "everything else"
    coherence_tau_ms: float = 200.0    # [NEW] time constant of the running correlation (online version of Eq. 9)
    beta_threshold: float = -0.1       # [B] Eq. 10 threshold
    gate_temperature: float = 0.05     # [NEW] soft (sigmoid) gate instead of a hard 0/1 step
    alpha_mix: float = 0.7             # [B] Eq. 11  F_out = alpha F + (1 - alpha) F_hat
    learn_alpha_beta: bool = True      # [NEW] allow fine-tuning of alpha / beta in stage 5
    nnls_iters: int = 5                # [NEW] projected-gradient steps for per-frame memory activation


###########################################################
#                     MASK ESTIMATOR                      #
###########################################################

@dataclass(frozen=True)
class MaskNetCfg:
    lstm_sizes: tuple = (128, 64)      # [K]
    dense_out: int = 64                # [K] one gain per gammatone band
    lr: float = 1e-3                   # [K] Adam(0.9, 0.999, 1e-8)
    batch_size: int = 32               # [NEW] utterances per batch (Keshavarzi used 1500 3-frame windows)
    epochs: int = 30                   # [NEW] Keshavarzi used 5 on a smaller fixed set; we mix dynamically
    grad_clip: float = 1.0
    seq_len_frames: int = 800          # [NEW] 2 s training crops at 2.5 ms hop
    use_stream_selection: bool = True  # [NEW] learned per-LSN weights (Bellur's "future work" stream selection)


###########################################################
#                          DATA                           #
###########################################################

@dataclass(frozen=True)
class DataCfg:
    root: Path = Path("data")
    vctk_dir: Path = Path("data/vctk")
    fsd50k_dir: Path = Path("data/fsd50k")
    esc50_dir: Path = Path("data/esc50")
    librispeech_dir: Path = Path("data/librispeech")
    musan_dir: Path = Path("data/musan")
    demand_dir: Path = Path("data/demand")
    n_train_talkers: int = 80          # [K] 80 talkers for training
    n_val_talkers: int = 10
    n_test_talkers: int = 20
    train_str_db: tuple = (-5, -10, -15)                   # [K]
    test_str_db: tuple = (-5, -10, -15, -20, -25, -30)     # [K] went to -20; we extend to stress-test
    str_peak_window_ms: float = 5.0    # [K] transient RMS measured in 5 ms around its peak
    background_snr_db: tuple = (10, 5) # [NEW] stretch condition: transients + babble/DEMAND
    heldout_transient_fraction: float = 0.3
    grandma_class: str = "Crumpling_and_crinkling"         # ALWAYS held out of training. Verify name in FSD50K vocabulary.csv
    # Candidate FSD50K classes for transients. VERIFY each against FSD50K/ground_truth/vocabulary.csv.
    transient_classes: tuple = (
        "Crumpling_and_crinkling", "Keys_jangling", "Knock", "Door", "Slam", "Glass", "Shatter",
        "Dishes_and_pots_and_pans", "Cutlery_and_silverware", "Tap", "Zipper_(clothing)", "Scissors",
        "Drawer_open_or_close", "Coin_(dropping)", "Finger_snapping", "Clapping", "Typing", "Packing_tape_and_duct_tape",
    )
    seed: int = 0


###########################################################
#                        VARIANTS                         #
###########################################################

VARIANTS = {
    # id: (short name, description)
    "A": ("NP", "no processing"),
    "B": ("MCTR", "non-neural multi-channel transient reduction"),
    "C": ("RNN-3", "Keshavarzi baseline, 3-frame windows"),
    "D": ("RNN-stream", "baseline LSTM, stateful over utterance (context control)"),
    "E": ("RNN-big", "stateful baseline with parameter count matched to H (size control)"),
    "F": ("DBN-replace", "DBN+attention -> LSTM, no raw filterbank skip (original proposal)"),
    "G": ("Hybrid-noattn", "filterbank + DBN -> LSTM, no attention"),
    "H": ("Hybrid", "PROPOSED: filterbank + DBN + speech attention -> LSTM"),
    "I": ("IRM", "oracle ideal ratio mask (upper bound)"),
}


@dataclass(frozen=True)
class Config:
    audio: AudioCfg = field(default_factory=AudioCfg)
    dbn: DBNCfg = field(default_factory=DBNCfg)
    attn: AttnCfg = field(default_factory=AttnCfg)
    mask: MaskNetCfg = field(default_factory=MaskNetCfg)
    data: DataCfg = field(default_factory=DataCfg)
    out_dir: Path = Path("runs")


CFG = Config()
