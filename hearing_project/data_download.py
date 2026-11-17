# Hearing project: dataset download and organisation - Jake Abraham
"""
PLAIN WORDS
-----------
Downloads the open datasets and turns them into compact, ready-to-use packs (see audio_store.py).

- SPEECH: VCTK, the same corpus Keshavarzi et al. used. 80 training talkers x 20 sentences (= their
  1,600), 10 validation talkers x 10, and 20 test talkers x 15 (= their 300). Talkers never overlap
  between splits. Only those ~2,000 sentences are fetched: if the server allows it, they are read
  straight out of the 11-GB zip over the network without downloading the whole thing.
- TRANSIENTS: FSD50K (open, human-labelled Freesound clips). Each clip is cut to the half-second
  around its loudest moment; clips that aren't actually "spiky" are rejected, as are clips that
  contain speech or music. Whole transient categories are held out for testing, and "crumpling and
  crinkling" is ALWAYS held out — the grandma test.
- The belief-network pretraining data (LibriSpeech, MUSAN) and background noise (DEMAND) are only
  needed from stage 2 on and are not downloaded in this first pass.

The proposal's datashare link is checked automatically: its file list is printed, which tells us
whether it is VCTK.

USAGE
-----
    python data_download.py            # everything for the first pass (needs network access, see below)
    python data_download.py --check    # only test which hosts are reachable

Hosts that must be allowed in the environment's network settings:
    datashare.ed.ac.uk   (VCTK)          zenodo.org   (FSD50K)
Later stages also need: www.openslr.org (LibriSpeech, MUSAN).

IMPLEMENTATION NOTES (status: IMPLEMENTED; network paths untested until hosts are allowed)
-----------------------------------------------------------------------------------------
- RangeFile: a seekable file object over HTTP Range requests, so zipfile can read single members.
  Falls back to a full download when the server ignores Range.
- VCTK URL: resolved via the DSpace 7 REST API (items/<uuid>/bundles -> bitstreams), trying the
  proposal's item first, then handle 10283/3443, then the legacy bitstream URL.
- FSD50K eval audio is a split zip (.z01 + .zip): downloaded fully, merged with `zip -s 0`, then only
  the selected transient clips are extracted. Peak disk use ~13 GB, archives deleted afterwards.
"""

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_store import frame_levels_db, save_pack, to_mono_16k, trim_silence
from config import CFG

D = CFG.data
DATASHARE = "https://datashare.ed.ac.uk"
PROPOSAL_ITEM_UUID = "30e7453c-9ea8-48b4-8e18-f96d0dc62928"   # link given in the research proposal
VCTK_HANDLE = "10283/3443"
VCTK_LEGACY_URL = f"{DATASHARE}/bitstream/handle/{VCTK_HANDLE}/VCTK-Corpus-0.92.zip"
ZENODO_FSD50K = "https://zenodo.org/api/records/4060432"
USER_AGENT = "hearing-project-data-fetcher/0.1"


###########################################################
#                     HTTP HELPERS                        #
###########################################################

def _request(url: str, headers: dict = None, timeout: float = 60):
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url: str) -> dict:
    with _request(url, {"Accept": "application/json"}) as r:
        return json.loads(r.read().decode())


def download(url: str, dest: Path, expected_size: int = None, chunk: int = 1 << 22):
    """Stream url -> dest with resume support and a progress line."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if expected_size and have == expected_size:
        print(f"  [skip] {dest.name} already complete")
        return dest
    headers = {"Range": f"bytes={have}-"} if have else {}
    with _request(url, headers, timeout=120) as r:
        mode = "ab" if (have and r.status == 206) else "wb"
        done = have if mode == "ab" else 0
        total = expected_size or (done + int(r.headers.get("Content-Length", 0)))
        with open(dest, mode) as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    print(f"\r  {dest.name}: {done / 1e9:6.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print()
    return dest


class RangeFile:
    """Read-only, seekable file over HTTP Range requests (block-cached). Enough for zipfile."""

    def __init__(self, url: str, block: int = 1 << 20, max_blocks: int = 64):
        self.url, self.block, self.max_blocks = url, block, max_blocks
        self.pos = 0
        self.cache = OrderedDict()
        with _request(url, {"Range": "bytes=0-0"}) as r:
            if r.status != 206 or "Content-Range" not in r.headers:
                raise IOError("server ignores Range requests")
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _block(self, i):
        if i in self.cache:
            self.cache.move_to_end(i)
            return self.cache[i]
        start, end = i * self.block, min(self.size, (i + 1) * self.block) - 1
        for attempt in range(4):
            try:
                with _request(self.url, {"Range": f"bytes={start}-{end}"}, timeout=120) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 3:
                    raise
        self.cache[i] = data
        if len(self.cache) > self.max_blocks:
            self.cache.popitem(last=False)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = max(0, min(n, self.size - self.pos))
        out = bytearray()
        while len(out) < n:
            i, off = divmod(self.pos, self.block)
            chunk = self._block(i)[off: off + n - len(out)]
            out += chunk
            self.pos += len(chunk)
        return bytes(out)

    def close(self):
        self.cache.clear()


def fetch_member(url: str, zi: zipfile.ZipInfo) -> bytes:
    """Fetch ONE zip member with a single exact Range request (thread-safe; no shared file object)."""
    import zlib
    start = zi.header_offset
    end = start + 30 + len(zi.filename.encode()) + zi.compress_size + 2048       # slack for the local extra field
    for attempt in range(8):
        try:
            with _request(url, {"Range": f"bytes={start}-{end}"}, timeout=120) as r:
                raw = r.read()
            break
        except OSError:
            if attempt == 7:
                raise
            time.sleep(min(30, 2 ** attempt))          # server sometimes drops connections under load
    n_name, n_extra = int.from_bytes(raw[26:28], "little"), int.from_bytes(raw[28:30], "little")
    data = raw[30 + n_name + n_extra: 30 + n_name + n_extra + zi.compress_size]
    if zi.compress_type == zipfile.ZIP_STORED:
        return data
    if zi.compress_type == zipfile.ZIP_DEFLATED:
        return zlib.decompressobj(-15).decompress(data)
    raise ValueError(f"unsupported compression {zi.compress_type}")


def open_remote_zip(url: str, local_fallback: Path) -> zipfile.ZipFile:
    try:
        zf = zipfile.ZipFile(RangeFile(url))
        print(f"  reading zip remotely via Range requests ({url})")
        return zf
    except (IOError, zipfile.BadZipFile) as e:
        print(f"  Range read unavailable ({e}); downloading the whole archive to {local_fallback}")
        download(url, local_fallback)
        return zipfile.ZipFile(local_fallback)


def check_hosts():
    ok = {}
    for host in ("https://datashare.ed.ac.uk", "https://zenodo.org", "https://www.openslr.org"):
        try:
            with _request(host, timeout=15) as r:
                ok[host] = r.status
        except Exception as e:  # noqa: BLE001 — report any failure
            ok[host] = f"BLOCKED/UNREACHABLE ({type(e).__name__})"
    for h, s in ok.items():
        print(f"  {h:32s} {s}")
    return all(isinstance(s, int) for s in ok.values())


###########################################################
#                         VCTK                            #
###########################################################

def dspace_bitstreams(item_uuid: str) -> list:
    """[(name, size_bytes, content_url)] for every file in a DSpace 7 item."""
    out = []
    bundles = get_json(f"{DATASHARE}/server/api/core/items/{item_uuid}/bundles")
    for b in bundles.get("_embedded", {}).get("bundles", []):
        bs = get_json(b["_links"]["bitstreams"]["href"])
        for f in bs.get("_embedded", {}).get("bitstreams", []):
            out.append((f["name"], f.get("sizeBytes"), f["_links"]["content"]["href"]))
    return out


def resolve_vctk_url() -> str:
    """Find the VCTK zip. Also reports what the proposal's datashare link contains."""
    try:
        files = dspace_bitstreams(PROPOSAL_ITEM_UUID)
        print("  proposal's datashare item contains:", [f[0] for f in files])
        for name, _, url in files:
            if re.match(r"VCTK-Corpus.*\.zip$", name):
                return url
    except Exception as e:  # noqa: BLE001
        print(f"  could not list the proposal's item ({e})")
    try:
        item = get_json(f"{DATASHARE}/server/api/pid/find?id=hdl:{VCTK_HANDLE}")
        for name, _, url in dspace_bitstreams(item["uuid"]):
            if re.match(r"VCTK-Corpus.*\.zip$", name):
                return url
    except Exception as e:  # noqa: BLE001
        print(f"  handle lookup failed ({e}); trying the legacy URL")
    return VCTK_LEGACY_URL


VCTK_AUDIO_RE = re.compile(r"(?:^|/)(p\d{3}|s\d+)_(\d{3})(?:_mic1)?\.(flac|wav)$")


def parse_speaker_info(text: str) -> dict:
    """{speaker_id: gender} from VCTK speaker-info.txt (handles '225' and 'p225' styles)."""
    info = {}
    for line in text.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 3 and parts[0][0] in "ps0123456789":
            sid = parts[0] if parts[0][0] in "ps" else "p" + parts[0]
            info[sid] = parts[2].upper()
    return info


def split_talkers(gender: dict, available: dict, seed: int = D.seed) -> dict:
    """Gender-stratified, disjoint talker split. available: {talker: [member names]}."""
    need = max(D.utts_per_talker)
    rng = np.random.default_rng(seed)
    talkers = sorted(t for t, m in available.items() if len(m) >= need + 24)
    by_g = defaultdict(list)
    for t in talkers:
        by_g[gender.get(t, "?")].append(t)
    for g in by_g:
        rng.shuffle(by_g[g])
    # interleave genders so every split gets ~equal F/M
    order = [t for pair in zip(*[by_g[g] for g in sorted(by_g)]) for t in pair]
    order += [t for g in by_g for t in by_g[g] if t not in order]
    n_tr, n_va, n_te = D.n_train_talkers, D.n_val_talkers, D.n_test_talkers
    assert len(order) >= n_tr + n_va + n_te, f"only {len(order)} usable talkers"
    return {"test": order[:n_te], "val": order[n_te:n_te + n_va], "train": order[n_te + n_va:n_te + n_va + n_tr]}


def build_speech_packs(seed: int = D.seed):
    url = resolve_vctk_url()
    zf = open_remote_zip(url, D.vctk_dir / "VCTK-Corpus.zip")
    names = zf.namelist()
    info_name = next(n for n in names if n.endswith("speaker-info.txt"))
    gender = parse_speaker_info(zf.read(info_name).decode(errors="ignore"))
    available = defaultdict(list)
    for n in names:
        m = VCTK_AUDIO_RE.search(n)
        if m and ("mic2" not in n):
            available[m.group(1)].append(n)
    split = split_talkers(gender, available, seed)
    rng = np.random.default_rng(seed + 1)
    remote = isinstance(zf.fp, RangeFile)

    def load(n):
        raw = fetch_member(url, zf.getinfo(n)) if remote else zf.read(n)
        x, fs = sf.read(io.BytesIO(raw), dtype="float32")
        return trim_silence(to_mono_16k(x, fs))

    from concurrent.futures import ThreadPoolExecutor
    for (name, talkers), k in zip(split.items(), (D.utts_per_talker[2], D.utts_per_talker[1], D.utts_per_talker[0])):
        # skip utterances 001–024 (read passages shared by every talker) to avoid text overlap;
        # fetch k + 3 candidates per talker so a few too-short ones can be dropped
        wanted = []
        for t in talkers:
            members = sorted(n for n in available[t] if int(VCTK_AUDIO_RE.search(n).group(2)) > 24)
            wanted += [(t, n) for n in rng.permutation(members)[: k + 3]]
        with ThreadPoolExecutor(6 if remote else 4) as ex:
            audio = list(ex.map(lambda tn: load(tn[1]), wanted))
        clips, meta = [], []
        for (t, n), x in zip(wanted, audio):
            if len(x) >= 16_000 and sum(m["talker"] == t for m in meta) < k:
                clips.append(x)
                meta.append(dict(talker=t, gender=gender.get(t, "?"), file=n, license="CC BY 4.0 (VCTK)"))
        print(f"  {name}: {len(meta)} utterances from {len(talkers)} talkers")
        save_pack(D.cache_dir / f"speech_{name}.npz", clips, meta)
    (D.cache_dir / "speech_split.json").write_text(json.dumps(split, indent=1))


###########################################################
#                        FSD50K                           #
###########################################################

def fetch_fsd50k(splits=D.fsd50k_splits) -> tuple:
    """Download ground truth + audio for `splits`; return (merged zip paths, ground_truth dir)."""
    rec = get_json(ZENODO_FSD50K)
    files = {f["key"]: f for f in rec["files"]}
    out = D.fsd50k_dir
    gt_zip = download(files["FSD50K.ground_truth.zip"]["links"]["self"], out / "FSD50K.ground_truth.zip",
                      files["FSD50K.ground_truth.zip"]["size"])
    with zipfile.ZipFile(gt_zip) as z:
        z.extractall(out)
    merged = {}
    for sp in splits:
        parts = sorted(k for k in files if k.startswith(f"FSD50K.{sp}_audio."))
        for k in parts:
            download(files[k]["links"]["self"], out / k, files[k]["size"])
        m = out / f"FSD50K.{sp}_audio.merged.zip"
        if not m.exists():
            subprocess.run(["zip", "-q", "-s", "0", f"FSD50K.{sp}_audio.zip", "--out", m.name], cwd=out, check=True)
        for k in parts:
            (out / k).unlink(missing_ok=True)
        merged[sp] = m
    gt_dir = next(out.glob("**/vocabulary.csv")).parent
    return merged, gt_dir


def assign_transient_class(labels: list) -> str:
    if any(l in D.transient_exclude_labels for l in labels):
        return None
    for c in D.transient_classes:
        if c in labels:
            return c
    return None


def cut_transient(x: np.ndarray, fs: int = 16_000):
    """Half-second around the loudest moment, or None if the clip is not spiky enough."""
    lv = frame_levels_db(x, fs, win_ms=5.0)
    if lv.max() - np.median(lv) < D.transient_min_peakiness_db:
        return None
    p = int(np.argmax(np.abs(x)))
    a = max(0, p - int(fs * D.transient_pre_peak_ms / 1000))
    b = min(len(x), p + int(fs * D.transient_post_peak_ms / 1000))
    y = x[a:b].copy()
    fade_in, fade_out = int(0.005 * fs), int(0.02 * fs)
    if p - a > fade_in:
        y[:fade_in] *= np.linspace(0, 1, fade_in)
    if len(y) > fade_out:
        y[-fade_out:] *= np.linspace(1, 0, fade_out)
    return y


def split_transient_classes(classes: list, seed: int = D.seed) -> dict:
    rng = np.random.default_rng(seed)
    others = sorted(c for c in classes if c != D.grandma_class)
    rng.shuffle(others)
    n_unseen = int(round(D.heldout_transient_fraction * len(others)))
    return {"grandma": [D.grandma_class], "unseen": sorted(others[:n_unseen]), "seen": sorted(others[n_unseen:])}


def build_transient_packs(seed: int = D.seed):
    merged, gt_dir = fetch_fsd50k()
    vocab = {row[1] for row in csv.reader(open(gt_dir / "vocabulary.csv"))}
    missing = [c for c in D.transient_classes if c not in vocab]
    if missing:
        print("  WARNING: not in FSD50K vocabulary (ignored):", missing)
    clips_by_class = defaultdict(list)
    for sp, zpath in merged.items():
        rows = list(csv.DictReader(open(gt_dir / f"{sp}.csv")))
        with zipfile.ZipFile(zpath) as z:
            names = {Path(n).stem: n for n in z.namelist() if n.endswith(".wav")}
            for row in rows:
                c = assign_transient_class(row["labels"].split(","))
                if c is None or row["fname"] not in names:
                    continue
                x, fs = sf.read(io.BytesIO(z.read(names[row["fname"]])), dtype="float32")
                y = cut_transient(to_mono_16k(x, fs))
                if y is not None:
                    clips_by_class[c].append((y, dict(label=c, file=f"{sp}/{row['fname']}.wav", license="see FSD50K metadata")))
        # merged archive is kept (~6 GB) so the packs can be rebuilt with different class choices;
        # delete data/fsd50k/*.merged.zip by hand when disk space is needed
    print("  clips per class:", {c: len(v) for c, v in sorted(clips_by_class.items())})
    groups = split_transient_classes([c for c in clips_by_class if len(clips_by_class[c]) >= 5], seed)
    (D.cache_dir).mkdir(parents=True, exist_ok=True)
    (D.cache_dir / "transient_split.json").write_text(json.dumps(groups, indent=1))
    rng = np.random.default_rng(seed + 2)
    packs = defaultdict(list)
    for c in groups["seen"]:
        items = [clips_by_class[c][i] for i in rng.permutation(len(clips_by_class[c]))]
        n = len(items); a, b = int(0.8 * n), int(0.9 * n)
        packs["train"] += items[:a]; packs["val"] += items[a:b]; packs["test_seen"] += items[b:]
    for c in groups["unseen"]:
        packs["test_unseen"] += clips_by_class[c]
    packs["test_grandma"] += clips_by_class.get(D.grandma_class, [])
    for name, items in packs.items():
        if items:
            save_pack(D.cache_dir / f"transient_{name}.npz", [i[0] for i in items], [i[1] for i in items])
            print(f"  transient_{name}: {len(items)} clips")


###########################################################
#                          MAIN                           #
###########################################################

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only test host reachability")
    ap.add_argument("--only", choices=["speech", "transients"], default=None)
    args = ap.parse_args()
    reachable = check_hosts()
    if args.check:
        sys.exit(0 if reachable else 1)
    if not shutil.which("zip"):
        sys.exit("`zip` is required to merge FSD50K's split archive")
    if args.only in (None, "speech"):
        build_speech_packs()
    if args.only in (None, "transients"):
        build_transient_packs()
