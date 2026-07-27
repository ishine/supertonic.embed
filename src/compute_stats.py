"""Recompute the paper's statistics from committed CSVs.

Answers: (1) is the preset -> proposed improvement statistically significant?
         (2) do the paper's reported 95% CIs reproduce as bootstrap CIs of the mean?
"""
import csv
import os

import numpy as np
from scipy import stats

RNG = np.random.default_rng(0)
B = 20000
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, one level above src/


def load(path):
    with open(f"{ROOT}/{path}", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def boot_ci_mean(x, b=B, alpha=0.05):
    x = np.asarray(x, float)
    idx = RNG.integers(0, len(x), size=(b, len(x)))
    means = x[idx].mean(axis=1)
    return np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])


def boot_ci_diff(a, b_, b=B, alpha=0.05):
    """Paired bootstrap CI of mean(a - b_)."""
    d = np.asarray(a, float) - np.asarray(b_, float)
    idx = RNG.integers(0, len(d), size=(b, len(d)))
    return np.percentile(d[idx].mean(axis=1), [100 * alpha / 2, 100 * (1 - alpha / 2)])


def paired_report(name, opt, pre):
    opt = np.asarray(opt, float)
    pre = np.asarray(pre, float)
    d = opt - pre
    w = stats.wilcoxon(opt, pre)
    t = stats.ttest_rel(opt, pre)
    # paired Cohen's d
    dz = d.mean() / d.std(ddof=1)
    lo, hi = boot_ci_mean(opt)
    dlo, dhi = boot_ci_diff(opt, pre)
    print(f"\n### {name}  (n={len(d)})")
    print(f"  preset   mean = {pre.mean():.4f}  (sd {pre.std(ddof=1):.4f})")
    print(f"  proposed mean = {opt.mean():.4f}  (sd {opt.std(ddof=1):.4f})")
    print(f"  proposed 95% bootstrap CI of mean = [{lo:.3f}, {hi:.3f}]")
    print(f"  improvement  = {d.mean():+.4f}   95% CI [{dlo:+.4f}, {dhi:+.4f}]")
    print(f"  Wilcoxon signed-rank: W={w.statistic:.1f}, p={w.pvalue:.3e}")
    print(f"  paired t-test:        t={t.statistic:.2f},  p={t.pvalue:.3e}")
    print(f"  Cohen's d_z = {dz:.2f}   |  improved in {int((d > 0).sum())}/{len(d)} speakers")
    return dict(mean=opt.mean(), ci=(lo, hi), p=w.pvalue, dz=dz)


print("=" * 68)
print("PAIRED SIGNIFICANCE TESTS  (preset baseline -> proposed), SupertonicTTS")
print("=" * 68)

wav = load("results/_layer3_seedtts/sim_wavlm_all.csv")
spk = load("results/_layer3_seedtts/sim_all_speakers.csv")
wer = load("results/_layer3_seedtts/wer_all.csv")

assert len(wav) == len(spk) == len(wer) == 44, (len(wav), len(spk), len(wer))
# align by id
wav_by = {r["id"]: r for r in wav}
spk_by = {r["id"]: r for r in spk}
ids = sorted(wav_by)

opt_w = [float(wav_by[i]["opt_wavlm"]) for i in ids]
pre_w = [float(wav_by[i]["pre_wavlm"]) for i in ids]
opt_e = [float(spk_by[i]["opt_ecapa"]) for i in ids]
pre_e = [float(spk_by[i]["pre_ecapa"]) for i in ids]
opt_r = [float(spk_by[i]["opt_resnet"]) for i in ids]
pre_r = [float(spk_by[i]["pre_resnet"]) for i in ids]

r_w = paired_report("SIM_W  (WavLM-base-plus-sv)", opt_w, pre_w)
r_e = paired_report("SIM_E  (ECAPA-TDNN)", opt_e, pre_e)
r_r = paired_report("SIM_R  (ResNet)", opt_r, pre_r)

print("\n" + "=" * 68)
print("PAPER'S REPORTED CIs vs RECOMPUTED BOOTSTRAP CIs")
print("=" * 68)
for label, got, claimed in [
    ("SIM_W", r_w["ci"], "[.84, .89]"),
    ("SIM_E", r_e["ci"], "[.43, .48]"),
    ("SIM_R", r_r["ci"], "[.42, .48]"),
]:
    print(f"  {label}: recomputed [{got[0]:.3f}, {got[1]:.3f}]   paper {claimed}")

print("\n" + "=" * 68)
print("CROSS-METRIC AGREEMENT (architecture independence)")
print("=" * 68)
rho, p = stats.spearmanr(opt_e, opt_r)
print(f"  Spearman rho(ECAPA, ResNet) on proposed = {rho:.4f}  (p={p:.2e})")
rho2, p2 = stats.spearmanr(opt_w, opt_e)
print(f"  Spearman rho(WavLM-SV, ECAPA)           = {rho2:.4f}  (p={p2:.2e})")

print("\n" + "=" * 68)
print("WER  (proposed only; no preset WER stored)")
print("=" * 68)
w_arr = np.array([float(r["wer"]) for r in wer])
lo, hi = boot_ci_mean(w_arr)
print(f"  mean WER = {w_arr.mean()*100:.2f}%   95% CI [{lo*100:.2f}%, {hi*100:.2f}%]")
print(f"  median   = {np.median(w_arr)*100:.2f}%   max = {w_arr.max()*100:.2f}%")
print(f"  speakers with WER=0: {int((w_arr == 0).sum())}/44")

print("\n" + "=" * 68)
print("SUBGROUP CHECK (original 20 vs extended 24) - protocol consistency")
print("=" * 68)
for g in ("original", "extended"):
    sel = [i for i in ids if wav_by[i]["group"] == g]
    ow = np.array([float(wav_by[i]["opt_wavlm"]) for i in sel])
    oe = np.array([float(spk_by[i]["opt_ecapa"]) for i in sel])
    print(f"  {g:9s} n={len(sel):2d}  SIM_W {ow.mean():.4f}   SIM_E {oe.mean():.4f}")
g1 = [float(spk_by[i]["opt_ecapa"]) for i in ids if wav_by[i]["group"] == "original"]
g2 = [float(spk_by[i]["opt_ecapa"]) for i in ids if wav_by[i]["group"] == "extended"]
u = stats.mannwhitneyu(g1, g2)
print(f"  Mann-Whitney U (SIM_E, original vs extended): p={u.pvalue:.3f}")
print("  -> a LARGE p means the two batches are consistent (good)")

print("\n" + "=" * 68)
print("PER-SPEAKER RANGE")
print("=" * 68)
print(f"  SIM_W range: {min(opt_w):.4f} - {max(opt_w):.4f}")
print(f"  SIM_E range: {min(opt_e):.4f} - {max(opt_e):.4f}")
print(f"  speakers below preset baseline (SIM_E): "
      f"{sum(1 for a, b_ in zip(opt_e, pre_e) if a < b_)}/44")
