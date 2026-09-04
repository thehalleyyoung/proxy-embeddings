"""
Does the choice of audio embedder change the answer?

Every audio-diversity number in this project has so far been a CLAP number
(plus MERT as a cross-check), and the steering result hangs on CLAP's text
tower: its text-to-audio alignment measured at -0.09, i.e. no pre-render
signal at all. Before concluding that cross-modal steering is impossible for
music, the embedder itself has to be treated as a variable.

MusicCoCa and MuLan -- the models one would reach for first -- are Google-
internal and have never been released. The closest open substitutes:

  clap_fused    laion/clap-htsat-unfused. General audio-text CLAP, trained
                heavily on captioned sound events. Our incumbent.
  clap_music    laion/larger_clap_music. Same architecture and objective,
                training re-weighted toward MUSIC-text pairs. If caption
                vocabulary was the bottleneck, this should show it.
  muq_mulan     OpenMuQ/MuQ-MuLan-large (Tencent, 2025): a MuLan-style
                music-text joint embedding over the MuQ self-supervised music
                encoder -- the nearest open analogue of MusicCoCa's role.
  mert          m-a-p/MERT-v1-95M. No text tower; pure self-supervised music
                representation. The control: what does similarity look like
                when the embedder never saw a caption?

Questions, in order of consequence:

  1. AGREEMENT. Correlation of pairwise track similarities across embedders.
     If low, "audio diversity" is embedder-relative and every number needs
     its embedder stated -- same lesson as the text-cone finding.
  2. STEERABILITY. For each embedder with a text tower: does similarity of
     the PROMPTS (through the text tower) predict similarity of the TRACKS
     (through the audio tower)? This is what pre-render steering needs, and
     what CLAP-fused failed. Reported as a Spearman over prompt pairs.
  3. VERDICT STABILITY. Do naive-vs-conditioned diversity gaps survive an
     embedder swap, and does the track that each embedder calls "most
     redundant" stay the same?

All tracks are windowed (10 s / 5 s hop) and aggregated as [mean, std, mean
|delta|] so a static piece and a developing one do not collide; MuQ handles
longer context natively but gets the same windowing for comparability.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
from audio_domain import load_audio  # noqa: E402

CACHE = HERE / "figures" / "audio_emb_cache"
CACHE.mkdir(parents=True, exist_ok=True)
WIN_S, HOP_S = 10.0, 5.0


def _agg(W: np.ndarray) -> np.ndarray:
    if len(W) == 1:
        z = np.zeros_like(W[0])
        v = np.concatenate([W[0], z, z])
    else:
        v = np.concatenate([W.mean(0), W.std(0), np.abs(np.diff(W, axis=0)).mean(0)])
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _windows(y: np.ndarray, sr: int):
    w, h = int(WIN_S * sr), int(HOP_S * sr)
    if len(y) < w:
        return [np.pad(y, (0, w - len(y)))]
    return [y[i:i + w] for i in range(0, len(y) - w + 1, h)]


def _tensor(out):
    import torch
    if isinstance(out, torch.Tensor):
        return out
    for a in ("audio_embeds", "text_embeds", "pooler_output", "last_hidden_state"):
        v = getattr(out, a, None)
        if v is not None:
            return v if v.dim() == 2 else v.mean(dim=1)
    raise TypeError(type(out))


class ClapEmbedder:
    def __init__(self, model_name: str, tag: str):
        import torch
        from transformers import ClapModel, AutoProcessor
        self.torch = torch
        self.tag = tag
        self.model = ClapModel.from_pretrained(model_name).eval()
        self.proc = AutoProcessor.from_pretrained(model_name)
        self.sr = 48000
        self.has_text = True

    def audio(self, wav: Path) -> np.ndarray:
        y = load_audio(wav, self.sr)
        wins = _windows(y, self.sr)
        vecs = []
        with self.torch.no_grad():
            for i in range(0, len(wins), 8):
                inp = self.proc(audio=wins[i:i + 8], sampling_rate=self.sr,
                                return_tensors="pt")
                f = _tensor(self.model.get_audio_features(**inp))
                f = f / f.norm(dim=-1, keepdim=True)
                vecs.append(f.numpy())
        return _agg(np.vstack(vecs))

    def text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            i = self.proc(text=[t[:900] for t in texts], return_tensors="pt",
                          padding=True, truncation=True)
            f = _tensor(self.model.get_text_features(**i))
            f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()


class MuqMulanEmbedder:
    def __init__(self):
        import torch
        from muq import MuQMuLan
        self.torch = torch
        self.tag = "muq_mulan"
        self.model = MuQMuLan.from_pretrained("OpenMuQ/MuQ-MuLan-large").eval()
        self.sr = 24000
        self.has_text = True

    def audio(self, wav: Path) -> np.ndarray:
        y = load_audio(wav, self.sr)
        wins = _windows(y, self.sr)
        vecs = []
        with self.torch.no_grad():
            for w in wins:
                t = self.torch.tensor(w[None, :], dtype=self.torch.float32)
                f = self.model(wavs=t)
                f = f / f.norm(dim=-1, keepdim=True)
                vecs.append(f.numpy())
        return _agg(np.vstack(vecs))

    def text(self, texts: list[str]) -> np.ndarray:
        with self.torch.no_grad():
            f = self.model(texts=[t[:512] for t in texts])
            f = f / f.norm(dim=-1, keepdim=True)
        return f.numpy()


class MertEmbedder:
    def __init__(self):
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        self.torch = torch
        self.tag = "mert"
        name = "m-a-p/MERT-v1-95M"
        self.model = AutoModel.from_pretrained(name, trust_remote_code=True).eval()
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
        self.sr = 24000
        self.has_text = False

    def audio(self, wav: Path) -> np.ndarray:
        y = load_audio(wav, self.sr)
        w, h = int(5.0 * self.sr), int(2.5 * self.sr)
        wins = ([y[i:i + w] for i in range(0, len(y) - w + 1, h)]
                or [np.pad(y, (0, w - len(y)))])
        vecs = []
        with self.torch.no_grad():
            for win in wins:
                inp = self.fe(win, sampling_rate=self.sr, return_tensors="pt")
                hs = self.model(**inp, output_hidden_states=True).hidden_states
                sel = self.torch.stack(hs[5:9]).mean(dim=0).mean(dim=1).squeeze(0)
                vecs.append((sel / sel.norm()).numpy())
        return _agg(np.vstack(vecs))


def track_matrix(emb, wavs: list[Path], set_tag: str) -> np.ndarray:
    key = CACHE / f"{emb.tag}__{set_tag}__{len(wavs)}.npy"
    if key.exists():
        return np.load(key)
    rows = []
    for i, w in enumerate(wavs):
        rows.append(emb.audio(w))
        if (i + 1) % 10 == 0:
            print(f"    {emb.tag}: {i + 1}/{len(wavs)}", flush=True)
    M = np.stack(rows)
    np.save(key, M)
    return M


def sim_upper(M: np.ndarray) -> np.ndarray:
    S = M @ M.T
    return S[np.triu_indices(len(M), 1)]


def vendi_centered(M: np.ndarray) -> float:
    X = M - M.mean(0, keepdims=True)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    X = X[n.ravel() > 1e-9] / n[n.ravel() > 1e-9]
    w = np.linalg.eigvalsh((X.T @ X) / len(X))
    w = np.clip(w, 1e-12, None); w = w / w.sum()
    return float(np.exp(-(w * np.log(w)).sum()))


def main():
    from scipy.stats import spearmanr
    # ---- track sets ----------------------------------------------------
    sets = {}
    for tag, d in [("steer", HERE / "real" / "lyria_steer" / "audio"),
                   ("naive", HERE / "real" / "lyria_naive" / "audio"),
                   ("ihd", HERE / "real" / "lyria_ihd" / "audio")]:
        wavs = sorted(d.glob("*.wav")) if d.exists() else []
        if len(wavs) >= 10:
            sets[tag] = wavs
            print(f"set {tag}: {len(wavs)} tracks")
    prompts = {}
    for tag, d in [("steer", HERE / "real" / "lyria_steer"),
                   ("naive", HERE / "real" / "lyria_naive"),
                   ("ihd", HERE / "real" / "lyria_ihd")]:
        f = d / "log.jsonl" if (d / "log.jsonl").exists() else d / "corpus.jsonl"
        if f.exists():
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            prompts[tag] = [r.get("prompt") or r.get("text") for r in rows]

    embedders = []
    embedders.append(ClapEmbedder("laion/clap-htsat-unfused", "clap_fused"))
    embedders.append(ClapEmbedder("laion/larger_clap_music", "clap_music"))
    try:
        embedders.append(MuqMulanEmbedder())
    except Exception as e:
        print(f"muq_mulan unavailable: {str(e)[:150]}")
    embedders.append(MertEmbedder())

    out = {"sets": {t: len(w) for t, w in sets.items()}, "embedders": {},
           "agreement": {}, "steerability": {}, "verdicts": {}}
    mats = {}
    for emb in embedders:
        out["embedders"][emb.tag] = {"has_text": emb.has_text}
        for t, wavs in sets.items():
            print(f"  embedding {t} with {emb.tag} ...", flush=True)
            mats[(emb.tag, t)] = track_matrix(emb, wavs, t)

    # ---- 1. agreement --------------------------------------------------
    for t in sets:
        tags = [e.tag for e in embedders]
        rows = {}
        for i, a in enumerate(tags):
            for b in tags[i + 1:]:
                r = float(spearmanr(sim_upper(mats[(a, t)]),
                                    sim_upper(mats[(b, t)])).statistic)
                rows[f"{a}~{b}"] = round(r, 4)
        out["agreement"][t] = rows
        print(f"agreement[{t}]: {rows}")

    # ---- 2. steerability ----------------------------------------------
    for emb in embedders:
        if not emb.has_text:
            continue
        for t, wavs in sets.items():
            pl = prompts.get(t, [])
            idx = [int(w.stem) for w in wavs]
            ptexts = [pl[i] for i in idx if i < len(pl)]
            if len(ptexts) != len(wavs) or len(ptexts) < 10:
                continue
            T = emb.text(ptexts)
            r = float(spearmanr(sim_upper(T), sim_upper(mats[(emb.tag, t)])).statistic)
            out["steerability"].setdefault(emb.tag, {})[t] = round(r, 4)
    print("steerability (prompt-sim vs track-sim, per embedder):")
    print(json.dumps(out["steerability"], indent=2))

    # ---- 3. verdict stability ------------------------------------------
    for t in sets:
        row = {}
        for emb in embedders:
            M = mats[(emb.tag, t)]
            S = M @ M.T
            np.fill_diagonal(S, -np.inf)
            nn = 1.0 - S.max(axis=1)
            i, j = np.unravel_index(np.argmax(S), S.shape)
            row[emb.tag] = {
                "vendi_centered": round(vendi_centered(M), 2),
                "median_nn_dist": round(float(np.median(nn)), 4),
                "most_redundant_pair": [int(i), int(j)],
            }
        out["verdicts"][t] = row
    for t, row in out["verdicts"].items():
        print(f"verdicts[{t}]:")
        for e, v in row.items():
            print(f"  {e:12s} vendiC={v['vendi_centered']:7.2f} "
                  f"medNN={v['median_nn_dist']:.4f} pair={v['most_redundant_pair']}")

    json.dump(out, open(HERE / "figures" / "audio_embedder_compare.json", "w"), indent=2)
    print("wrote figures/audio_embedder_compare.json")


if __name__ == "__main__":
    main()
