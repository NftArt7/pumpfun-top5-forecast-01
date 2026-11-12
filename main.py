#!/usr/bin/env python3
"""pumpfun-top5-forecast-01 entry: trajectory + Monte Carlo forecast.

Reads $MAKEFUN_INPUT_DIR (history.ndjson, task.json) and writes
$MAKEFUN_OUTPUT_DIR/forecast.json with exactly the keys
probabilities / top5 / sizes.

Model (stdlib only, deterministic, no network, no wall-clock):
  * per coin, fit a weighted-least-squares slope of log10(market cap in SOL)
    over its marks, shrunk hard toward a negative target (top coins decay;
    a spike fitted on two points must not be extrapolated);
  * project each coin to every answer read with drift that decays over a
    2.5 h time constant and a sigma that widens like a random walk;
  * seeded Monte Carlo (fixed seed) over the top candidates ranks the
    coins at each read, giving place probabilities and the distribution
    of each place's log-cap (sizes). Each draw first asks whether the
    coin has rugged by that read (a steep, saturating hazard: in the
    practice ~2 of the top 11 had rugged within 10 min and ~8 of 11
    within the hour); a rugged coin falls to the post-rug floor near
    log 1.3 (~19 SOL). Survivors move with a modest Gaussian sigma --
    the big moves ARE the rugs, so the Gaussian must not also be huge;
  * newcomers (coins born after the history ends, which no forecast can
    name) are modelled explicitly, per place: place 1 is strongly
    newcomer-dominated (in the practice it was a newcomer in 6 of 7
    reads), places 2-3 usually survivors, places 4-5 fresh risers, so
    the newcomer share is top-heavy and grows with the horizon; the
    history coins' place mass is shifted down by a binomial count of
    newcomers in the top 5, and the leftover mass is left unnamed so the
    scorer shares it over the answer chart's un-named coins (newcomers
    included);
  * sizes mix the Monte Carlo history distribution with a wide, higher
    newcomer component (a viral newcomer routinely exceeds the fading
    history #1, as MAX did in the practice -- its caps averaged log 4.6
    against a history #1 near 3.5, so place 1's newcomer premium is
    1.0); the winner's-curse cap applies to the history component only;
  * place probabilities are sharpened (power 2.0, renormalised per
    place): the Monte Carlo average over reads is flatter than the
    realised per-read concentration, so the top coins earn more of each
    place's named mass; the unnamed share is unchanged.
  * a coin is named at a place only when its probability beats the
    per-coin share the place would otherwise leave unnamed.
"""
import json
import math
import os
import random
from datetime import datetime

SEED = 20261112
TEMP_GAMMA = 2.0        # sharpen each place's probability distribution: the
                     # Monte Carlo average over reads is flatter than the
                     # realised per-read concentration (ablation: 1.0 -> 2.0
                     # gains ~+0.4 local log; the curve plateaus at ~2.0)
NSAMPLES = 4000
MAX_CANDIDATES = 150
TAU_H = 2.5            # drift decay time, hours
VOL_RATE = 0.35        # log10 sd per sqrt(hour) for a SURVIVING coin (the
                     # big moves are rugs, modelled separately below, so the
                     # Gaussian part is the modest drift of survivors)
SIG0 = 0.10            # base log10 sd
SIG_CAP = 0.65         # max projected log10 sd for a survivor
COMMON_SD = 0.12       # shared shock per Monte Carlo sample (sizes only)
RUG_LEVEL = 1.3        # log10 cap a rugged coin falls to (~19 SOL: in the
                     # practice every top coin that rugged landed near here)
RUG_SD = 0.15          # spread of the post-rug level
NEW_SD = 0.8           # newcomer size distribution sd (wide: a viral newcomer
                     # can land anywhere from a dud to far above the history's #1)
CURVE_CAP = 2.55       # log10 cap of a coin still on its bonding curve
MICRO = 1_000_000
# Place weights for the newcomer share: the largest coin at a read is
# usually the freshest viral launch, so place 1 is newcomer-dominated
# while places 2-3 are usually survivors (weights average 1.0).
NEW_W = [3.0, 0.65, 0.65, 1.0, 1.0]
# Extra size premium of a newcomer taking place k, over the last mark's
# k-th cap (place 1's premium is the largest: a viral newcomer routinely
# exceeds the fading history #1, as MAX did in the practice).
NEW_PREM = [1.0, 0.3, 0.25, 0.25, 0.25]
# Winner's-curse cap slack over the last mark's k-th cap, per place.
WCAP = [1.0, 0.6, 0.5, 0.5, 0.5]


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def logcap(row):
    return math.log10(int(row["market_cap_lamports"]) / 1e9)


def norm_cdf(x, mu, sd):
    return 0.5 * (1.0 + math.erf((x - mu) / (sd * math.sqrt(2.0))))


def shift_hist(hist, db):
    """Shift a bin histogram by db bins (fractional), linear interpolation."""
    n = len(hist)
    out = [0.0] * n
    for b, v in enumerate(hist):
        if v == 0.0:
            continue
        x = b + db
        lo = math.floor(x)
        frac = x - lo
        if 0 <= lo < n:
            out[lo] += v * (1.0 - frac)
        if 0 <= lo + 1 < n:
            out[lo + 1] += v * frac
    return out


def bin_of(x, edges):
    if x < edges[0]:
        return 0
    if x >= edges[-1]:
        return len(edges)
    return int((x - edges[0]) / 0.05) + 1


def floor6(p):
    v = math.floor(p * MICRO) / MICRO
    return v if v > 0 else 0.0


def binom5(r):
    """P(N = j) for N ~ Binomial(5, r), j = 0..5."""
    out = []
    for j in range(6):
        c = math.comb(5, j)
        out.append(c * (r ** j) * ((1.0 - r) ** (5 - j)))
    return out


def newcomer_share(t_hours):
    """Mean share of top-5 slots taken by coins born after the history ends.

    Top-5 coins are typically under an hour old, so the share grows with
    the horizon: at the practice's ~1 h it is ~0.3, at the final's
    7-13 h the history's coins are long past their prime and it is ~0.7.
    """
    return min(0.72, max(0.15, 0.20 + 0.14 * math.sqrt(max(t_hours, 0.0))))


def rug_prob(t_hours):
    """Probability a history coin has rugged by a read t_hours after the
    last mark. In the practice ~2 of the top 11 had rugged within 10 min
    and ~8 of 11 within the hour, so the hazard is steep and saturating;
    by the final's 7-13 h horizon essentially every history coin is dead
    or faded, which is why the newcomer reserve is so large there.
    """
    return min(0.90, 0.15 + 0.45 * max(t_hours, 0.0))


def newcomer_shares(t_hours, places):
    """Per-place newcomer shares.

    The mean is newcomer_share(); the split is top-heavy: place 1 is
    usually the freshest viral launch (in the practice it was a newcomer
    in 6 of 7 reads), places 2-3 usually survivors, places 4-5 fresh
    risers. Each share is capped at 0.9: even at the final's horizon a
    history coin can still, rarely, hold a place.
    """
    base = newcomer_share(t_hours)
    return [min(0.9, base * NEW_W[k - 1]) for k in range(1, places + 1)]


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    with open(os.path.join(in_dir, "task.json"), "r", encoding="utf-8") as f:
        task = json.load(f)
    places = int(task["places"])
    edges = [float(e) for e in task["size"]["edges"]]
    nbins = len(edges) + 1
    reads = [parse_time(r) for r in task["reads"]]
    t0 = parse_time(task["t0"])

    marks = []
    with open(os.path.join(in_dir, "history.ndjson"), "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            m = json.loads(line)
            if m.get("ok") and isinstance(m.get("rows"), list) and m["rows"]:
                marks.append(m)
    if not marks:
        raise SystemExit("no ok marks in history")
    n_marks = len(marks)
    t_last = (parse_time(marks[-1]["at"]) - t0).total_seconds() / 60.0

    # ---- per-coin observations: mint -> list of (t_min, logcap, venue, complete, created_min, mark_idx)
    obs = {}
    for mi, m in enumerate(marks):
        t = (parse_time(m["at"]) - t0).total_seconds() / 60.0
        for r in m["rows"]:
            try:
                created = (parse_time(r["created_at"]) - t0).total_seconds() / 60.0
            except Exception:
                created = t
            obs.setdefault(r["mint"], []).append(
                (t, logcap(r), r.get("venue", "curve"),
                 bool(r.get("complete")), created, mi)
            )
    if len(obs) < places:
        raise SystemExit("history holds fewer coins than places")

    # ---- global median slope (per hour) from coins with >= 2 observations
    def wls_slope(o):
        # weights decay with age from the coin's last observation (45 min scale)
        t_last_o = o[-1][0]
        sw = st = sy = stt = sty = 0.0
        for (t, y, _v, _c, _cr, _mi) in o:
            w = math.exp(-(t_last_o - t) / 45.0)
            sw += w
            st += w * t
            sy += w * y
            stt += w * t * t
            sty += w * t * y
        den = sw * stt - st * st
        if den <= 1e-12:
            return None
        # slope per minute -> per hour
        return ((sw * sty - st * sy) / den) * 60.0

    all_slopes = []
    slopes = {}
    for mint in sorted(obs):
        s = wls_slope(obs[mint]) if len(obs[mint]) >= 2 else None
        slopes[mint] = s
        if s is not None:
            all_slopes.append(s)
    all_slopes.sort()
    if all_slopes:
        med = all_slopes[len(all_slopes) // 2]
    else:
        med = 0.0
    # Shrinkage target: top coins decay (their mean 10-min change is about
    # -0.08 log10, the median about 0), so a slope fitted on few points --
    # a coin that just spiked -- must not be extrapolated. -0.2/h sits
    # between the median (~0) and the top coins' mean (~-0.5/h).
    global_slope = min(0.1, max(-0.3, 0.5 * med - 0.1))

    # ---- per-coin fitted state
    coins = {}
    for mint in sorted(obs):
        o = obs[mint]
        y_last = o[-1][1]
        venue = o[-1][2]
        complete = o[-1][3]
        age_min = t_last - o[-1][4]
        last_idx = o[-1][5]
        s = slopes[mint]
        n = len(o)
        if s is None:
            s_shrunk = global_slope
        else:
            s_shrunk = (n / (n + 4.0)) * s + (4.0 / (n + 4.0)) * global_slope
        s_shrunk = min(0.8, max(-0.8, s_shrunk))
        stale = n_marks - 1 - last_idx
        coins[mint] = {
            "y_last": y_last,
            "slope": s_shrunk,
            "venue": venue,
            "complete": complete,
            "age_min": age_min,
            "stale": stale,
        }

    def project(c, t_hours):
        mu = c["y_last"] + c["slope"] * TAU_H * (1.0 - math.exp(-t_hours / TAU_H))
        if not c["complete"] and c["venue"] == "curve":
            mu = min(mu, CURVE_CAP)
        mu -= 0.2 * c["stale"]
        sig = math.sqrt(SIG0 * SIG0 + (VOL_RATE * math.sqrt(t_hours)) ** 2)
        sig = min(SIG_CAP, sig)
        if not c["complete"] and c["venue"] == "curve":
            sig *= 1.2
        if c["age_min"] < 30.0:
            sig *= 1.15
        if c["stale"] > 0:
            sig *= 1.15
        return mu, sig

    # ---- candidate set: top by max projected mean over the reads
    horizons = [(r - parse_time(marks[-1]["at"])).total_seconds() / 3600.0
                for r in reads]
    scored = []
    for mint in sorted(coins):
        c = coins[mint]
        best = None
        for h in horizons:
            mu, _sig = project(c, h)
            if best is None or mu > best:
                best = mu
        scored.append((best, mint))
    scored.sort(key=lambda x: (-x[0], x[1]))
    cand = [mint for _best, mint in scored[:MAX_CANDIDATES]]
    cand_set = set(cand)

    # ---- Monte Carlo per read
    rng = random.Random(SEED)
    # p_sum[mint][place] and size_sum[place][bin], averaged over reads
    p_sum = {mint: [0.0] * (places + 1) for mint in cand}
    size_sum = [[0.0] * nbins for _ in range(places + 1)]
    mean_h = sum(horizons) / len(horizons) if horizons else 0.0
    last_sorted = sorted((c["y_last"] for c in coins.values()), reverse=True)

    for ri, h in enumerate(horizons):
        r_share = newcomer_share(h)
        r_k = newcomer_shares(h, places)
        b5 = binom5(r_share)
        mus = []
        sigs = []
        for mint in cand:
            mu, sig = project(coins[mint], h)
            mus.append(mu)
            sigs.append(sig)
        n_c = len(cand)
        q = [[0.0] * (places + 1) for _ in range(n_c)]  # q[i][k]
        size_hist = [[0.0] * nbins for _ in range(places + 1)]
        kth_mean = [0.0] * (places + 1)
        qr = rug_prob(h)
        for _s in range(NSAMPLES):
            common = rng.gauss(0.0, COMMON_SD)
            top = []  # list of (y, idx), descending, at most `places`
            for i in range(n_c):
                # A coin either survives (Gaussian drift around its
                # projection) or has rugged by this read (it falls to the
                # post-rug floor, ~19 SOL, and is out of the top 5).
                if rng.random() < qr:
                    y = rng.gauss(RUG_LEVEL, RUG_SD)
                else:
                    y = rng.gauss(mus[i], sigs[i]) + common
                if len(top) < places or y > top[-1][0]:
                    j = 0
                    while j < len(top) and top[j][0] > y:
                        j += 1
                    top.insert(j, (y, i))
                    if len(top) > places:
                        top.pop()
            for k in range(places):
                yk, ik = top[k]
                q[ik][k + 1] += 1.0
                size_hist[k + 1][bin_of(yk, edges)] += 1.0
                kth_mean[k + 1] += yk
        inv = 1.0 / NSAMPLES
        for i in range(n_c):
            for k in range(1, places + 1):
                q[i][k] *= inv
        for k in range(1, places + 1):
            for b in range(nbins):
                size_hist[k][b] *= inv
            kth_mean[k] /= NSAMPLES

        # Tame the winner's curse: the max of many independent noisy draws
        # overstates the realised top caps of HISTORY coins. Keep each
        # place's history size mean within a band around the last mark's
        # k-th cap, and re-centre that place's size histogram by the same
        # shift. (The newcomer component below is NOT capped this way: a
        # fresh viral coin routinely exceeds the fading history #1.)
        for k in range(1, places + 1):
            if k <= len(last_sorted):
                cap = last_sorted[k - 1] + WCAP[k - 1] + 0.1 * min(1.0, h / 6.0)
                floor = last_sorted[k - 1] - 0.8
                capped = min(max(kth_mean[k], floor), cap)
                delta = capped - kth_mean[k]
                if delta != 0.0:
                    size_hist[k] = shift_hist(size_hist[k], delta / 0.05)
                    kth_mean[k] = capped

        # newcomer shift: history coin at place k was (k-j)-th among history
        # coins when j newcomers sit above it. The history mass at place k
        # is scaled to (1 - r_k): the rest is left unnamed, so the scorer
        # shares it over the answer chart's un-named coins (newcomers
        # included) -- which is where a newcomer taking place k is scored.
        for k in range(1, places + 1):
            rk = r_k[k - 1]
            tot = 0.0
            svals = [0.0] * n_c
            for i in range(n_c):
                s = 0.0
                for j in range(0, min(4, k - 1) + 1):
                    m = k - j
                    if 1 <= m <= places:
                        s += b5[j] * q[i][m]
                svals[i] = s
                tot += s
            scale = (1.0 - rk) / tot if tot > 0 else 0.0
            for i in range(n_c):
                p_sum[cand[i]][k] += svals[i] * scale
            # sizes: mix the history distribution with a wider, higher
            # newcomer one. A newcomer taking place k is typically a fresh
            # viral launch, so its cap sits above the fading history k-th
            # cap (by NEW_PREM) with a wide spread -- not bounded by it.
            if k <= len(last_sorted):
                mu_new = last_sorted[k - 1] + NEW_PREM[k - 1]
            else:
                mu_new = kth_mean[k] + 0.3
            for b in range(nbins):
                lo = -1e9 if b == 0 else edges[b - 1]
                hi = 1e9 if b == nbins - 1 else edges[b]
                pn = norm_cdf(hi, mu_new, NEW_SD) - norm_cdf(lo, mu_new, NEW_SD)
                size_sum[k][b] += (1.0 - rk) * size_hist[k][b] + rk * pn

    n_reads = float(len(reads))
    for mint in cand:
        for k in range(1, places + 1):
            p_sum[mint][k] /= n_reads
    for k in range(1, places + 1):
        for b in range(nbins):
            size_sum[k][b] /= n_reads

    # ---- temperature calibration on place probabilities: sharpen each
    # place's distribution (power > 1) and renormalise to the same total
    # mass, keeping the unnamed share -- and therefore the size mix --
    # unchanged. The average over reads spreads mass over too many coins;
    # the realised top 5 at a read is concentrated on a few.
    _tot = {}
    for k in range(1, places + 1):
        _tot[k] = sum(p_sum[m][k] for m in cand)
    for mint in cand:
        for k in range(1, places + 1):
            p_sum[mint][k] = p_sum[mint][k] ** TEMP_GAMMA
    for k in range(1, places + 1):
        _new = sum(p_sum[m][k] for m in cand)
        if _new > 0:
            _f = _tot[k] / _new
            for mint in cand:
                p_sum[mint][k] *= _f

    # ---- emit probabilities: name a coin only while its p beats the share
    # the place would otherwise leave unnamed per un-named chart coin
    probabilities = []
    for k in range(1, places + 1):
        order = sorted(cand, key=lambda m: (-p_sum[m][k], m))
        named_sum = 0.0
        named_n = 0
        for mint in order:
            if named_n >= 100:
                break
            p = p_sum[mint][k]
            threshold = (1.0 - named_sum) / max(1, 100 - named_n)
            if p <= threshold:
                break
            pv = floor6(p)
            if pv <= 0:
                break
            probabilities.append({"mint": mint, "place": k, "p": pv})
            named_sum += pv
            named_n += 1

    # ---- top5: greedy by place probability
    top5 = []
    used = set()
    for k in range(1, places + 1):
        best_mint = None
        best_p = -1.0
        for mint in cand:
            if mint in used:
                continue
            if p_sum[mint][k] > best_p:
                best_p = p_sum[mint][k]
                best_mint = mint
        if best_mint is None:
            best_mint = cand[k - 1]
        top5.append(best_mint)
        used.add(best_mint)

    # ---- emit sizes
    sizes = []
    for k in range(1, places + 1):
        for b in range(nbins):
            pv = floor6(size_sum[k][b])
            if pv > 0:
                sizes.append({"place": k, "bin": b, "p": pv})

    forecast = {"probabilities": probabilities, "top5": top5, "sizes": sizes}
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, task.get("output", "forecast.json"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(forecast, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
