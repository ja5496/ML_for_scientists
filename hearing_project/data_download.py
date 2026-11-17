# Hearing project: dataset download and organisation - Jake Abraham
"""
PLAIN WORDS
-----------
Downloads the open datasets and sorts them into train / validation / test lists.

- Speech comes from VCTK, the same corpus Keshavarzi et al. used. Talkers are split so that no
  test talker is ever heard during training.
- Transient sounds (crinkles, keys, doors, glass, cutlery...) come from FSD50K, a large open
  library of human-labelled Freesound clips. Keshavarzi's own 24 transient recordings were never
  released, so we build an equivalent (and bigger) set ourselves.
- Some transient *categories* are held out entirely so the final test uses sound types the network
  has never heard. "Crumpling and crinkling" is always held out — that's the grandma test.
- A separate pile of general sounds (speech, everyday sounds, music) is collected for the
  unsupervised pretraining of the belief network, as Bellur & Elhilali did with TIMIT/BBC/music.
  Those are swapped for open equivalents (LibriSpeech, FSD50K, MUSAN).

Output: JSON manifests (lists of file paths + metadata) in data/manifests/.

PSEUDOCODE FOR THE IMPLEMENTING AGENT
-------------------------------------
Sources (check network access first; the sandbox may need these hosts allow-listed):
    VCTK         https://datashare.ed.ac.uk/handle/10283/3443          (CC BY 4.0)
    FSD50K       https://zenodo.org/records/4060432                    (per-clip CC licenses)
    ESC-50       https://github.com/karolpiczak/ESC-50                 (CC BY-NC, optional)
    LibriSpeech  https://www.openslr.org/12   (train-clean-100 is enough)  (CC BY 4.0)
    MUSAN        https://www.openslr.org/17   (music subset only)         (CC BY 4.0)
    DEMAND       https://zenodo.org/records/1227121                    (CC BY-SA 3.0, stretch test)

1. download_all(): fetch + checksum + extract each archive into CFG.data.<name>_dir.
   Skip datasets that already exist. Never re-download silently over existing data.
2. build_speech_manifest():
     - list VCTK speakers with metadata (gender, accent) from speaker-info.txt
     - deterministic shuffle with CFG.data.seed; stratify by gender
     - assign n_train / n_val / n_test talkers (disjoint)
     - resample 48k -> 16k on the fly or cache resampled .wav (use soxr / resampy)
     - drop the leading/trailing silence (VCTK has long silences), keep >= 1.0 s
3. build_transient_manifest():
     - read FSD50K ground_truth/dev.csv + eval.csv + vocabulary.csv
     - ASSERT each name in CFG.data.transient_classes exists in vocabulary.csv; print the missing ones
     - keep clips whose labels include a transient class; prefer single-label clips
     - trim each clip to the strongest event: find peak |x|, keep [peak - 50 ms, peak + 450 ms]
     - reject clips with no clear onset: require (peak level) - (median level) > 20 dB
     - class split: grandma_class -> test only; remaining classes split 70/30 (seeded)
       into "seen" (train/val/test) and "unseen" (test only)
     - record license per clip (FSD50K provides it) so the final report can cite them
4. build_pretrain_manifest():
     - LibriSpeech speakers NOT overlapping VCTK test talkers (they never overlap; still assert)
     - FSD50K clips whose labels contain NO test-only transient classes
     - MUSAN music
     - target roughly 3 h speech + 4 h sounds + 2 h music (Bellur's proportions)
5. build_background_manifest(): DEMAND noises + babble (sum of 4–8 held-out LibriSpeech talkers)
6. write_manifests(): JSON, one file per split, each entry = {path, duration, label(s), license, talker}
"""

import json
from pathlib import Path

from config import CFG


def download_all():
    """Fetch every dataset listed above into CFG.data.root. Idempotent."""
    raise NotImplementedError


def build_speech_manifest():
    """Return {'train': [...], 'val': [...], 'test': [...]} of VCTK utterances with disjoint talkers."""
    raise NotImplementedError


def build_transient_manifest():
    """Return {'train': [...], 'val': [...], 'test_seen': [...], 'test_unseen': [...], 'test_grandma': [...]}."""
    raise NotImplementedError


def build_pretrain_manifest():
    """Return a list of generic sound files for unsupervised DBN pretraining (no test classes / talkers)."""
    raise NotImplementedError


def build_background_manifest():
    """Return background noise files (DEMAND + synthetic babble) for the stretch condition."""
    raise NotImplementedError


def write_manifests(manifests: dict, out_dir: Path = CFG.data.root / "manifests"):
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, entries in manifests.items():
        (out_dir / f"{name}.json").write_text(json.dumps(entries, indent=1, default=str))


if __name__ == "__main__":
    download_all()
    write_manifests({
        **{f"speech_{k}": v for k, v in build_speech_manifest().items()},
        **{f"transient_{k}": v for k, v in build_transient_manifest().items()},
        "pretrain": build_pretrain_manifest(),
        "background": build_background_manifest(),
    })
