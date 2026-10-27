"""Download the agents' checkpoints into ``$HUGSIM_AGENT_CKPTS``, laid out as agents/README.md says.

    python agents/download_checkpoints.py                       # every agent, into $HUGSIM_AGENT_CKPTS
    python agents/download_checkpoints.py --agents drivor toad  # some of them
    python agents/download_checkpoints.py --root /path --verify # also re-hash the files already there

Only what is missing is fetched. A download goes to ``<file>.part``, resumes after an interruption, is
checked against its sha256 and only then renamed, so a file under its final name is always complete.
Files that are byte-identical between agents (DrivoR's weights for ``drivor_paper`` and ``toad``, the
GTRS vocabulary for ZTRS) are downloaded once and symlinked. Plain HTTPS with the standard library: no
Hugging Face client (whose parallel downloads can stall) and no ``gdown``.

Downloading stays an explicit step, like ``prepare_benchmark.py``: an episode never fetches anything,
and a missing checkpoint still stops the agent with the path it expected.
"""

import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.request

UNIAD = "https://github.com/OpenDriveLab/UniAD/releases/download/v1.0.1/uniad_base_e2e.pth"
VAD = "https://drive.usercontent.google.com/download?id=1FLX-4LVm4z-RskghFbxGuYlcYOQmV5bS&export=download&confirm=t"
LTF = "https://huggingface.co/autonomousvision/navsim_baselines/resolve/main/ltf/ltf_seed_0.ckpt"
DRIVOR = "https://github.com/valeoai/DrivoR/releases/download/model_weights/drivor_Nav1_25epochs.pth"
GTRS = "https://huggingface.co/Zzxxxxxxxx/gtrs/resolve/main"
ZTRS = "https://huggingface.co/Zzxxxxxxxx/ZTRS/resolve/main/ztrs_vov.ckpt"
# NVlabs/GTRS at a fixed commit (its default branch is master), so the vocabulary cannot change under a run.
VOCAB = "https://raw.githubusercontent.com/NVlabs/GTRS/92a740def80610e4096962d25cc837b21e72ef78/traj_final"
# BeyondDrive's README (wjl2244/BeyondDrive, Google Drive): LTFv7 and its LTF* baseline.
LTFV7 = "https://drive.usercontent.google.com/download?id=1ctVDgZf5yAr0oHiUKUe0CSOWWv3fRdZA&export=download&confirm=t"
LTF_STAR = "https://drive.usercontent.google.com/download?id=1t49K8OJqI-DdPedEQ5q0Z2Mkf3Vcc5du&export=download&confirm=t"
# VaVAM's v1.0.0 release (valeoai/VideoActionModel): VaVAM-B, the model the ECO paper drives, and its image tokenizer.
VAVAM = "https://github.com/valeoai/VideoActionModel/releases/download/v1.0.0"

#: agent -> [(path under the root, source, sha256)]; a source starting with "=" is another entry's path,
#: linked rather than downloaded again.
#: The sha256s are those of the files every garage run has used (checked 2026-10-22: each download
#: here is byte-identical to them; DrivoR's is the one its release documents).
CHECKPOINTS = {
    "uniad": [("uniad/uniad_base_e2e.pth", UNIAD, "0ad0c2f5dc9788a41c313305779ea49346aeb742d1f6bb5ad25c46f9beffc990")],
    "vad": [("vad/VAD_base.pth", VAD, "4086547b921fee23139b09f0b79fd53a8b66dd02dd791ea80454d8237cefe4c4")],
    "ltf": [("ltf/ltf_seed_0.ckpt", LTF, "d08e97ba6beb937412697d619171553041622859e026a06935a2e0e118e6309d")],
    "drivor": [("drivor/drivor_Nav1_25epochs.pth", DRIVOR, "e1a678f201e4f1ab93d117caad42782cd7ead293bdced2b5f80212bc92426ae3")],
    "drivor_paper": [("drivor_paper/drivor_Nav1_25epochs.pth", "=drivor/drivor_Nav1_25epochs.pth", None)],
    "toad": [("toad/drivor_Nav1_25epochs.pth", "=drivor/drivor_Nav1_25epochs.pth", None)],
    "drivor_eco": [("drivor_eco/drivor_Nav1_25epochs.pth", "=drivor/drivor_Nav1_25epochs.pth", None)],
    "ltfv7": [("ltfv7/LTFv7-PDMS_89.841.ckpt", LTFV7, "ae772b368376e137318d0027df75ed3d12349efb2dec9c1854003e43832d3da7")],
    "ltf_star": [("ltf_star/LTF_-PDMS_88.7.ckpt", LTF_STAR, "a6d159ef4edd6f91faa63240fe786eb8513d79e3f2e10aedfe35a696104694f1")],
    "gtrs": [("gtrs/gtrs_dense_vov.ckpt", f"{GTRS}/gtrs_dense_vov.ckpt", "8bab5fb86fe9071365e53a18328d5fa1c91000515f9446bcb05eadc34ae5b95d"),
             ("gtrs/16384.npy", f"{VOCAB}/16384.npy", "e8c29cfc25add59ae8b64769a4554c6518878726178c0bd889fc8518ebe1261d")],
    "ztrs": [("ztrs/ztrs_vov.ckpt", ZTRS, "26d45c0efc47f1823896ed9fe909d84e3c1f483f56ee49274c09de0838fa3415"),
             ("ztrs/16384.npy", "=gtrs/16384.npy", None)],
    "gtrs_aug": [("gtrs_aug/gtrs_aug_vov.ckpt", f"{GTRS}/gtrs_aug_vov.ckpt", "c1c433be768781f98b16d946b41fae4c4cc31d7b1715ee6b8b683dcf637ddd26"),
                 ("gtrs_aug/8192.npy", f"{VOCAB}/8192.npy", "cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35")],
    "vavam": [("vavam/VAM_width_1024_pretrained_139k.pt", f"{VAVAM}/VAM_width_1024_pretrained_139k.pt", "129e887db30477d458933369964c98b51fdbeb5041eaf8efca4594e2ca56de18"),
              ("vavam/VQ_ds16_16384_llamagen_encoder.jit", f"{VAVAM}/VQ_ds16_16384_llamagen_encoder.jit", "b5e2c47d04f8ff8a6a47433252aaeefaa8be16be83db6521521a78fefae7e228")],
    "vavam_eco": [("vavam_eco/VAM_width_1024_pretrained_139k.pt", "=vavam/VAM_width_1024_pretrained_139k.pt", None),
                  ("vavam_eco/VQ_ds16_16384_llamagen_encoder.jit", "=vavam/VQ_ds16_16384_llamagen_encoder.jit", None)],
}
SHA256 = {path: sha for files in CHECKPOINTS.values() for path, _, sha in files}
SOURCE = {path: src for files in CHECKPOINTS.values() for path, src, _ in files}


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check(path, rel):
    """Raise if ``path`` does not hash to the sha256 recorded for ``rel`` (when one is recorded)."""
    want = SHA256.get(rel)
    if want is None:
        return
    got = sha256(path)
    if got != want:
        raise RuntimeError(f"{rel}: sha256 {got} does not match the expected {want}")


def fetch(url, dest, attempts=5):
    """Download ``url`` to ``dest``.part (resuming what is there), then rename it to ``dest``."""
    part = dest + ".part"
    for attempt in range(1, attempts + 1):
        have = os.path.getsize(part) if os.path.exists(part) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "hugsim-garage", **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                if have and resp.status != 206:  # the server ignored the range: start over
                    have = 0
                total = resp.headers.get("Content-Length")
                total = int(total) + have if total else None
                if "text/html" in (resp.headers.get("Content-Type") or "") and not have:
                    raise RuntimeError(f"{url} answered an HTML page, not the file (a quota or sign-in page?)")
                done, last = have, 0.0
                with open(part, "ab" if have else "wb") as f:
                    for block in iter(lambda: resp.read(1 << 22), b""):
                        f.write(block)
                        done += len(block)
                        if time.time() - last > 5:
                            pct = f" {100 * done / total:.0f}%" if total else ""
                            print(f"    {done / 1e6:.0f} MB{pct}", flush=True)
                            last = time.time()
            if total and os.path.getsize(part) != total:
                raise RuntimeError(f"got {os.path.getsize(part)} of {total} bytes")
            os.replace(part, dest)
            return
        except (urllib.error.URLError, OSError, RuntimeError) as e:
            client_error = isinstance(e, urllib.error.HTTPError) and 400 <= e.code < 500 and e.code != 429
            if attempt == attempts or client_error or (isinstance(e, RuntimeError) and "HTML" in str(e)):
                raise
            print(f"    attempt {attempt} failed ({e}); retrying", flush=True)
            time.sleep(5 * attempt)


def install(root, rel, verify):
    """Make ``root/rel`` present and correct. Returns what was done."""
    dest = os.path.join(root, rel)
    src = SOURCE[rel]
    if os.path.lexists(dest) and not os.path.exists(dest):
        os.remove(dest)  # a dangling symlink
    if os.path.exists(dest):
        if verify:
            check(dest, rel if not src.startswith("=") else src[1:])
            return "verified"
        return "present"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if src.startswith("="):
        target = os.path.join(root, src[1:])
        if not os.path.exists(target):
            install(root, src[1:], verify)
        os.symlink(os.path.relpath(target, os.path.dirname(dest)), dest)
        return f"linked to {src[1:]}"
    print(f"[{rel}] <- {src}", flush=True)
    fetch(src, dest)
    try:
        check(dest, rel)
    except RuntimeError:
        os.remove(dest)
        raise
    return f"downloaded, {os.path.getsize(dest) / 1e6:.0f} MB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("HUGSIM_AGENT_CKPTS"), help="default: $HUGSIM_AGENT_CKPTS")
    ap.add_argument("--agents", nargs="+", choices=sorted(CHECKPOINTS), default=sorted(CHECKPOINTS))
    ap.add_argument("--verify", action="store_true", help="also check the sha256 of files already present")
    args = ap.parse_args()
    if not args.root:
        raise SystemExit("set HUGSIM_AGENT_CKPTS or pass --root")
    os.makedirs(args.root, exist_ok=True)
    free = shutil.disk_usage(args.root).free
    if free < 8e9:
        print(f"warning: {free / 1e9:.1f} GB free under {args.root}; all checkpoints take about 5.3 GB", file=sys.stderr)
    failed = []
    for agent in args.agents:
        for rel, _, _ in CHECKPOINTS[agent]:
            try:
                print(f"{rel}: {install(args.root, rel, args.verify)}", flush=True)
            except Exception as e:  # report every file, then fail
                print(f"{rel}: FAILED: {e}", file=sys.stderr, flush=True)
                failed.append(rel)
    if failed:
        raise SystemExit(f"{len(failed)} file(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
