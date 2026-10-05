#!/usr/bin/env python
"""
Synthetic league simulator.

Simulates 8 teams x 30 athletes day by day (training, matches, sleep, nutrition,
wellness, physical qualities, injuries) with a *known* causal structure defined in
`causal.py`, then writes the raw source-system extracts as parquet files.

    python data_gen/generate.py --out data/raw

Raw tables mimic what real vendor feeds look like, including defects the dbt
staging layer has to deal with: duplicate sessions, two GPS vendors with
different speed units, sensor glitches, and incomplete self-reporting.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import causal as C  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)

START = pd.Timestamp("2024-07-01")
END = pd.Timestamp("2026-09-30")
BURN = 35  # days simulated before START so rolling windows are warm

POSITIONS = ["GK", "CB", "FB", "DM", "CM", "AM", "W", "ST"]
PER_TEAM_COMPOSITION = {"GK": 3, "CB": 5, "FB": 4, "DM": 3, "CM": 4, "AM": 2, "W": 4, "ST": 5}
POS_YOYO = {"GK": -550, "CB": -150, "FB": 150, "DM": 80, "CM": 150, "AM": 20, "W": 120, "ST": -100}
POS_SPRINT = {"GK": 0.16, "CB": 0.06, "FB": -0.07, "DM": 0.02, "CM": 0.0, "AM": -0.02, "W": -0.09, "ST": -0.05}
POS_CMJ = {"GK": 1.5, "CB": 1.5, "FB": 0.0, "DM": 0.0, "CM": -0.5, "AM": -0.5, "W": 0.5, "ST": 1.0}
POS_HEIGHT = {"GK": 7, "CB": 4, "FB": -2, "DM": 1, "CM": -1, "AM": -2, "W": -3, "ST": 2}
POS_WEIGHT = {"GK": 6, "CB": 4, "FB": -2, "DM": 2, "CM": -1, "AM": -3, "W": -3, "ST": 2}
POS_DIST = {"GK": 0.45, "CB": 0.90, "FB": 1.05, "DM": 1.00, "CM": 1.05, "AM": 1.02, "W": 1.04, "ST": 0.95}
POS_HSR = {"GK": 0.10, "CB": 0.70, "FB": 1.10, "DM": 0.85, "CM": 0.95, "AM": 1.00, "W": 1.25, "ST": 1.15}

BODY_PARTS = ["hamstring", "ankle", "knee", "groin", "calf", "quadriceps", "foot", "lower_back", "hip", "shoulder", "other"]
BP_BASE = np.array([20, 14, 12, 11, 9, 6, 6, 6, 5, 3, 8], dtype=float)
BP_DAYS = np.array([14, 18, 35, 14, 12, 14, 20, 10, 14, 18, 8], dtype=float)
BP_TYPE = {"hamstring": "muscle strain", "ankle": "ligament sprain", "knee": "ligament sprain",
           "groin": "muscle strain", "calf": "muscle strain", "quadriceps": "muscle strain",
           "foot": "bone stress / contusion", "lower_back": "muscle strain", "hip": "tendinopathy",
           "shoulder": "contusion", "other": "contusion"}

# session type codes
NONE, MATCH, TRAIN, GYM, RECOV, REHAB, INDIV = 0, 1, 2, 3, 4, 5, 6
TYPE_NAME = {MATCH: "match", TRAIN: "training", GYM: "gym", RECOV: "recovery", REHAB: "rehab", INDIV: "individual"}
W_BY_CODE = {MATCH: 5.0, TRAIN: 1.0, GYM: 0.4, RECOV: 0.2, REHAB: 0.15, INDIV: 0.4}

# day-type codes (team level, microcycle position)
DT_OFF, DT_RECOV, DT_LIGHT, DT_MOD, DT_HARD, DT_MATCH, DT_OFFSEASON, DT_DEFAULT = range(8)
DT_PARAMS = {DT_RECOV: (40, 2.5), DT_LIGHT: (50, 3.5), DT_MOD: (70, 5.0), DT_HARD: (85, 6.8), DT_DEFAULT: (70, 5.5)}

REF_LOAD = 350.0  # typical daily sRPE load (AU) used as a scale for adaptation


def phase_of(d: pd.Timestamp) -> int:
    """0 = off-season, 1 = pre-season, 2 = in-season"""
    if d.month == 6 or (d.month == 7 and d.day <= 5):
        return 0
    if d.month == 7 or (d.month == 8 and d.day <= 9):
        return 1
    return 2


def tail_mean(arr, t, n, incl=False):
    lo = max(0, t - n + 1) if incl else max(0, t - n)
    hi = t + 1 if incl else t
    if hi <= lo:
        return np.full(arr.shape[0], np.nan)
    return np.nanmean(arr[:, lo:hi], axis=1)


def simulate(seed: int, n_teams: int, per_team: int):
    rng = np.random.default_rng(seed)
    sim_start = START - pd.Timedelta(days=BURN)
    dates = pd.date_range(sim_start, END)
    T = len(dates)
    N = n_teams * per_team
    K = n_teams

    # ------------------------------------------------------------------ athletes
    team_idx = np.repeat(np.arange(K), per_team)
    pos_list = []
    for _ in range(K):
        for p, n in PER_TEAM_COMPOSITION.items():
            pos_list += [p] * int(n * per_team / 30)
    pos = np.array(pos_list[:N])
    pos_code = np.array([POSITIONS.index(p) for p in pos])
    members = [np.where(team_idx == k)[0] for k in range(K)]

    consc = rng.normal(0, 1, N)                       # lifestyle conscientiousness (confounder)
    team_q = rng.normal(0, 0.25, K)
    talent = rng.normal(0, 0.45, N) + team_q[team_idx]
    age0 = np.clip(rng.normal(26.5, 4.3, N), 19, 37)
    pa = lambda d: np.array([d[p] for p in pos])  # noqa: E731
    yoyo_b = 2100 + pa(POS_YOYO) + 120 * consc + rng.normal(0, 250, N)
    sprint_b = 4.20 + pa(POS_SPRINT) - 0.03 * consc + rng.normal(0, 0.10, N)
    cmj_b = 38 + pa(POS_CMJ) + 1.0 * consc + rng.normal(0, 3.5, N)
    squat_b = 1.70 + 0.12 * consc + rng.normal(0, 0.22, N)
    nordic_b = 340 + 12 * consc + rng.normal(0, 48, N)
    gym_freq = np.clip(1.3 + 0.5 * consc + rng.normal(0, 0.5, N), 0.0, 3.0)
    sleep_b = np.clip(7.3 + 0.35 * consc + rng.normal(0, 0.55, N), 5.3, 9.3)
    prot_b = np.clip(1.55 + 0.16 * consc + rng.normal(0, 0.28, N), 0.8, 2.6)
    carb_b = np.clip(5.2 + 0.4 * consc + rng.normal(0, 0.9, N), 2.5, 8.5)
    hyd_b = np.clip(2.9 + 0.2 * consc + rng.normal(0, 0.45, N), 1.6, 4.5)
    p_well = np.clip(0.86 + 0.06 * consc + rng.normal(0, 0.05, N), 0.5, 0.99)
    p_nut = np.clip(0.68 + 0.10 * consc + rng.normal(0, 0.10, N), 0.25, 0.95)
    p_sleep = np.clip(0.92 + rng.normal(0, 0.03, N), 0.7, 0.99)
    frailty_log = rng.normal(0, 0.35, N)
    hrv_base = np.exp(rng.normal(np.log(70), 0.25, N))
    rhr_base = rng.normal(56, 5, N)
    height = 181 + pa(POS_HEIGHT) + rng.normal(0, 5, N)
    weight = 77 + pa(POS_WEIGHT) + rng.normal(0, 5, N)
    bodyfat_b = np.clip(rng.normal(11.0, 2.3, N) - 0.5 * consc, 6, 20)
    foot = rng.choice(["right", "left", "both"], N, p=[0.70, 0.23, 0.07])
    intensity_mult = rng.normal(1.0, 0.07, N)

    join_t = np.zeros(N, dtype=int)
    new_signing = rng.random(N) < 0.15
    join_t[new_signing] = rng.integers(BURN, T - 120, new_signing.sum())
    signed_date = np.array([(dates[j] if j >= BURN else dates[0] - pd.Timedelta(days=int(rng.integers(30, 1200))))
                            for j in join_t])
    birth_date = np.array([sim_start - pd.Timedelta(days=int(a * 365.25)) for a in age0])
    vendor = np.where(team_idx < K // 2, "VENDOR_A", "VENDOR_B")

    # ------------------------------------------------------------------ fixtures
    phases = np.array([phase_of(d) for d in dates])
    dows = np.array([d.dayofweek for d in dates])
    is_match = np.zeros((K, T), dtype=bool)
    fixtures = []
    for k in range(K):
        match_dow = 6 if k % 2 else 5
        for t in range(T):
            if phases[t] == 0:
                continue
            ph = phases[t]
            p_sat = 0.93 if ph == 2 else 0.85
            p_mid = 0.20 if ph == 2 else 0.30
            if dows[t] == match_dow and rng.random() < p_sat:
                is_match[k, t] = True
            elif dows[t] == 2 and rng.random() < p_mid and not is_match[k, max(0, t - 2):t].any():
                is_match[k, t] = True
    # remove midweek matches that sit right before a weekend match (no 2-day congestion)
    for k in range(K):
        for t in range(T - 2):
            if is_match[k, t] and is_match[k, t + 1]:
                is_match[k, t] = False
    days_to_next = np.full((K, T), 99)
    days_since = np.full((K, T), 99)
    for k in range(K):
        nxt = 99
        for t in range(T - 1, -1, -1):
            if is_match[k, t]:
                nxt = 0
            elif nxt < 99:
                nxt += 1
            days_to_next[k, t] = nxt
        prv = 99
        for t in range(T):
            if is_match[k, t]:
                prv = 0
            elif prv < 99:
                prv += 1
            days_since[k, t] = prv
    fixture_id = {}
    fid = 0
    for k in range(K):
        for t in np.where(is_match[k])[0]:
            fid += 1
            fixture_id[(k, t)] = fid
            ph = phases[t]
            comp = "Friendly" if ph == 1 else ("Cup" if dows[t] == 2 else "League")
            if dates[t] >= START:
                fixtures.append(dict(fixture_id=fid, team_id=f"T{k+1:02d}", fixture_date=dates[t].date(),
                                     competition=comp, home_away=rng.choice(["H", "A"]),
                                     opponent_strength=int(rng.integers(1, 6))))
    fixtures_df = pd.DataFrame(fixtures)
    opp = {r.fixture_id: r.opponent_strength for r in fixtures_df.itertuples()} if len(fixtures_df) else {}
    home = {r.fixture_id: r.home_away for r in fixtures_df.itertuples()} if len(fixtures_df) else {}

    # daytype matrix
    daytype = np.full((K, T), DT_DEFAULT, dtype=int)
    for k in range(K):
        for t in range(T):
            dn, ds = days_to_next[k, t], days_since[k, t]
            if is_match[k, t]:
                daytype[k, t] = DT_MATCH
            elif phases[t] == 0:
                daytype[k, t] = DT_OFFSEASON
            elif dn == 1:
                daytype[k, t] = DT_LIGHT
            elif dn == 2:
                daytype[k, t] = DT_MOD
            elif dn == 3:
                daytype[k, t] = DT_HARD
            elif dn == 4:
                daytype[k, t] = DT_MOD
            elif ds == 1:
                daytype[k, t] = DT_RECOV
            elif ds == 2:
                daytype[k, t] = DT_OFF
            else:
                daytype[k, t] = DT_DEFAULT if rng.random() < 0.6 else DT_OFF

    # ------------------------------------------------------------------ state
    load = np.zeros((N, T), dtype=np.float32)
    sleep_h = np.full((N, T), np.nan, dtype=np.float32)
    sleep_eff = np.full((N, T), np.nan, dtype=np.float32)
    deep_pct = np.full((N, T), np.nan, dtype=np.float32)
    prot = np.full((N, T), np.nan, dtype=np.float32)
    carb = np.full((N, T), np.nan, dtype=np.float32)
    fat = np.full((N, T), np.nan, dtype=np.float32)
    hyd = np.full((N, T), np.nan, dtype=np.float32)
    hrv = np.full((N, T), np.nan, dtype=np.float32)
    rhr = np.full((N, T), np.nan, dtype=np.float32)
    sore = np.full((N, T), np.nan, dtype=np.float32)
    fatig = np.full((N, T), np.nan, dtype=np.float32)
    stress = np.full((N, T), np.nan, dtype=np.float32)
    mood = np.full((N, T), np.nan, dtype=np.float32)
    played = np.zeros((N, T), dtype=np.float32)
    sprint_hist = np.zeros((N, T), dtype=np.float32)
    aer_hist = np.zeros((N, T), dtype=np.float32)
    S_type = np.zeros((N, T, 2), dtype=np.int8)
    S_dur = np.zeros((N, T, 2), dtype=np.float32)
    S_rpe = np.zeros((N, T, 2), dtype=np.float32)

    yoyo, sprint, cmj, squat, nordic = yoyo_b.copy(), sprint_b.copy(), cmj_b.copy(), squat_b.copy(), nordic_b.copy()
    bodyfat = bodyfat_b.copy()
    inj_until = np.full(N, -1)
    inj_start = np.full(N, -1)
    inj_days = np.zeros(N)
    prior_inj = np.zeros(N)
    sleep_dev = np.zeros(N)
    hrv_dev = np.zeros(N)
    stress_state = rng.normal(4, 1, N)

    injuries, matches, tests = [], [], []
    test_anchor = START + pd.Timedelta(days=(7 - START.dayofweek) % 7)  # first Monday on/after START

    for t in range(T):
        d = dates[t]
        ph = phases[t]
        active = t >= join_t
        unavail_now = inj_until >= t
        available = active & ~unavail_now
        dsj = np.maximum(t - join_t, 0)
        prev_played = played[:, t - 1] if t > 0 else np.zeros(N)
        team_match_today = is_match[team_idx, t]
        dt_team = daytype[team_idx, t]

        # --- 1. sleep (night ending morning t)
        sleep_dev = 0.35 * sleep_dev + rng.normal(0, 0.7, N)
        s = (sleep_b + sleep_dev - 0.55 * (prev_played > 0) - 0.25 * team_match_today
             + (0.4 if ph == 0 else 0.0))
        sleep_h[:, t] = np.clip(s, 3.5, 10.8)
        sleep_eff[:, t] = np.clip(90 - 2.2 * np.maximum(0, 7.5 - s) + rng.normal(0, 2.5, N), 60, 98)
        deep_pct[:, t] = np.clip(18 + rng.normal(0, 3, N) + 0.5 * (s - 7.5), 8, 30)

        # --- 2. wellness (depends on load of previous days + sleep)
        l1 = load[:, t - 1] if t > 0 else np.zeros(N)
        l2 = load[:, t - 2] if t > 1 else np.zeros(N)
        z_sleep = (sleep_h[:, t] - sleep_b) / 0.8
        z_l1, z_l2 = (l1 - REF_LOAD) / 300, (l2 - REF_LOAD) / 300
        nut_prev = np.nan_to_num((carb[:, t - 1] - carb_b) / 1.0, nan=0.0) if t > 0 else np.zeros(N)
        hrv_dev = 0.45 * hrv_dev + 0.06 * z_sleep - 0.05 * z_l1 - 0.03 * z_l2 + 0.02 * nut_prev + rng.normal(0, 0.07, N)
        hrv[:, t] = hrv_base * np.exp(hrv_dev)
        rhr[:, t] = rhr_base - 5 * hrv_dev + rng.normal(0, 1.5, N)
        stress_state = 0.9 * stress_state + 0.1 * 4 + rng.normal(0, 0.5, N)
        sore[:, t] = np.clip(2.0 + 0.0045 * l1 + 1.2 * (prev_played > 0) - 0.4 * z_sleep + rng.normal(0, 0.8, N), 1, 10)
        fatig[:, t] = np.clip(3.0 + 0.004 * l1 - 0.7 * z_sleep + 0.2 * stress_state + rng.normal(0, 0.9, N), 1, 10)
        stress[:, t] = np.clip(stress_state + rng.normal(0, 0.5, N), 1, 10)
        mood[:, t] = np.clip(7.0 + 0.35 * z_sleep - 0.25 * (stress_state - 4) + rng.normal(0, 0.8, N), 1, 10)

        # --- 3. plan sessions
        slot_type = np.zeros((N, 2), dtype=np.int8)
        slot_dur = np.zeros((N, 2))
        slot_rpe = np.zeros((N, 2))

        # lineups on match days
        mins = np.zeros(N)
        fitz = C.fitness_composite(yoyo, sprint, cmj, squat)
        for k in range(K):
            if not is_match[k, t]:
                continue
            mem = members[k]
            av = available[mem]
            sc = np.where(av, talent[mem] + 0.25 * fitz[mem] + rng.normal(0, 0.45, len(mem)), -99.0)
            gk = pos_code[mem] == 0
            gk_ord = np.argsort(-np.where(gk, sc, -999))
            out_ord = np.argsort(-np.where(~gk, sc, -999))
            starters = [gk_ord[0]] + list(out_ord[:10])
            subs = list(out_ord[10:15])
            for j, i in enumerate(starters):
                if sc[i] < -90:
                    continue
                m = 90 if (j == 0 or rng.random() < 0.5) else int(rng.integers(55, 90))
                mins[mem[i]] = m
            used = rng.choice(len(subs), size=min(3, len(subs)), replace=False)
            for u in used:
                i = subs[u]
                if sc[i] > -90:
                    mins[mem[i]] = int(rng.integers(8, 35))

        phase_mult = 1.15 if ph == 1 else 1.0
        indiv_noise = intensity_mult * rng.normal(1.0, 0.10, N)
        for i in range(N):
            if not active[i]:
                continue
            if unavail_now[i]:
                if inj_days[i] >= 5 and t >= inj_start[i] + 0.5 * inj_days[i]:
                    slot_type[i, 0], slot_dur[i, 0], slot_rpe[i, 0] = REHAB, 40, 3.0
                continue
            dt = dt_team[i]
            if dt == DT_MATCH:
                if mins[i] > 0:
                    rp = float(np.clip(rng.normal(7.4, 0.7), 5.5, 9.5))
                    slot_type[i, 0], slot_dur[i, 0], slot_rpe[i, 0] = MATCH, mins[i], rp
                else:
                    slot_type[i, 0], slot_dur[i, 0], slot_rpe[i, 0] = TRAIN, 55 * rng.normal(1, 0.1), 5.5
            elif dt == DT_OFFSEASON:
                if rng.random() < 0.25:
                    slot_type[i, 0], slot_dur[i, 0], slot_rpe[i, 0] = INDIV, 30, 4.0
            elif dt == DT_OFF:
                pass
            elif dt == DT_RECOV:
                if prev_played[i] >= 45:
                    slot_type[i, 0], slot_dur[i, 0], slot_rpe[i, 0] = RECOV, 40, 2.5
                else:
                    slot_type[i, 0], slot_dur[i, 0], slot_rpe[i, 0] = TRAIN, 70, 6.0
            else:
                du, rp = DT_PARAMS[dt]
                slot_type[i, 0], slot_dur[i, 0], slot_rpe[i, 0] = TRAIN, du, rp
            if slot_type[i, 0] in (TRAIN, RECOV):
                slot_dur[i, 0] *= phase_mult * indiv_noise[i] * rng.normal(1, 0.06)
                slot_rpe[i, 0] = float(np.clip(slot_rpe[i, 0] * phase_mult * indiv_noise[i], 1.5, 9.5))
                if rng.random() < 0.02:   # unplanned extra work (trial match, extra conditioning)
                    slot_dur[i, 0] *= 1 + rng.uniform(0.3, 0.9)
                slot_dur[i, 0] = min(slot_dur[i, 0], 140.0)   # no 3-hour training sessions
            # second slot: gym or pre-season double
            if dt in (DT_MOD, DT_HARD, DT_DEFAULT) and slot_type[i, 0] == TRAIN:
                if ph == 1 and rng.random() < 0.55:
                    slot_type[i, 1], slot_dur[i, 1], slot_rpe[i, 1] = TRAIN, 50, 5.0
                elif rng.random() < gym_freq[i] / 3.0:
                    slot_type[i, 1], slot_dur[i, 1], slot_rpe[i, 1] = GYM, 55 * rng.normal(1, 0.1), 6.0

        planned_load = (slot_dur * slot_rpe).sum(1)

        # --- 4. nutrition (intake for day t, driven by planned load)
        lf = np.clip(planned_load / 600.0, 0, 1.3)
        carb[:, t] = np.clip(carb_b * (0.82 + 0.28 * lf) + rng.normal(0, 0.8, N), 1.5, 12.0)
        prot[:, t] = np.clip(prot_b + 0.1 * (prev_played > 0) + rng.normal(0, 0.25, N), 0.6, 3.2)
        fat[:, t] = np.clip(1.0 + rng.normal(0, 0.15, N), 0.5, 2.0)
        hyd[:, t] = np.clip(hyd_b * (1 + 0.08 * lf) + rng.normal(0, 0.3, N), 1.2, 5.0)

        # --- 5. injury hazard from state at the start of day t
        if t > 0:
            cs = np.cumsum(load[:, :t], axis=1)
            def csum(n):  # sum of last n days before t
                lo = max(0, t - n)
                tot = cs[:, -1] - (cs[:, lo - 1] if lo > 0 else 0)
                return tot
            acute = csum(7) / np.minimum(7, np.maximum(np.minimum(t, dsj), 1))
            chronic = csum(28) / np.minimum(28, np.maximum(np.minimum(t, dsj), 1))
        else:
            acute = chronic = np.full(N, REF_LOAD)
        acwr = np.where(chronic > 5, acute / np.maximum(chronic, 1e-6), 1.0)
        acwr = np.clip(acwr, 0, 5)
        hrv_hist = hrv[:, max(0, t - 28):t]
        if hrv_hist.shape[1] >= 5:
            hrv_mu, hrv_sd = np.nanmean(hrv_hist, axis=1), np.nanstd(hrv_hist, axis=1, ddof=1)
            hrv_z = (hrv[:, t] - hrv_mu) / np.maximum(hrv_sd, 1e-3)
        else:
            hrv_z = np.zeros(N)
        dsr = np.where(inj_until >= 0, t - inj_until, np.nan)
        s3, s7 = tail_mean(sleep_h, t, 3, incl=True), tail_mean(sleep_h, t, 7, incl=True)
        c3, p7, h3 = tail_mean(carb, t, 3), tail_mean(prot, t, 7), tail_mean(hyd, t, 3)
        fill = lambda x, v: np.where(np.isnan(x), v, x)  # noqa: E731
        feat = dict(
            yoyo_ir1_m=yoyo, sprint_30m_s=sprint, cmj_cm=cmj, squat_1rm_rel=squat, nordic_n=nordic,
            sleep_hours_3d=fill(s3, sleep_b), sleep_hours_7d=fill(s7, sleep_b),
            carbs_gpkg_3d=fill(c3, carb_b), protein_gpkg_7d=fill(p7, prot_b), hydration_l_3d=fill(h3, hyd_b),
            acwr=acwr, hrv_z=np.nan_to_num(hrv_z), days_since_return=dsr, prior_injuries=prior_inj,
            age_years=age0 + (t - BURN) / 365.25,
        )
        exposure = np.zeros(N)
        for sl in range(2):
            for code, w in W_BY_CODE.items():
                m = slot_type[:, sl] == code
                exposure += np.where(m, w * slot_dur[:, sl] / 60.0, 0.0)
        lam = np.exp(C.injury_log_rate(feat, frailty_log)) * exposure
        p_inj = 1 - np.exp(-lam)
        got_hurt = (rng.random(N) < p_inj) & available & (exposure > 0)
        for i in np.where(got_hurt)[0]:
            w = BP_BASE.copy()
            weak = max(0.0, (340 - nordic[i]) / 55.0)
            w[0] *= 1 + 0.8 * weak
            bp = int(rng.choice(len(BODY_PARTS), p=w / w.sum()))
            dout = int(np.clip(rng.lognormal(np.log(BP_DAYS[bp]), 0.7), 1, 200))
            main_type = slot_type[i, 0]
            where = TYPE_NAME.get(int(main_type), "training")
            if where not in ("match", "training", "gym"):
                where = "training"
            contact = rng.random() < (0.35 if where == "match" else 0.08)
            injuries.append(dict(athlete_idx=i, t=t, body_part=BODY_PARTS[bp], injury_type=BP_TYPE[BODY_PARTS[bp]],
                                 mechanism="contact" if contact else "non-contact", days_out=dout,
                                 occurred_during=where))
            inj_until[i], inj_start[i], inj_days[i] = t + dout, t, dout
            prior_inj[i] += 1
            f_cut = rng.uniform(0.3, 0.7)
            slot_dur[i, :] *= f_cut
            if mins[i] > 0:
                mins[i] = max(1, int(mins[i] * rng.uniform(0.3, 0.8)))
                slot_dur[i, 0] = mins[i]

        # record sessions / load
        S_type[:, t, :], S_dur[:, t, :], S_rpe[:, t, :] = slot_type, slot_dur, slot_rpe
        load[:, t] = (slot_dur * slot_rpe).sum(1)
        played[:, t] = mins
        sprint_hist[:, t], aer_hist[:, t] = sprint, yoyo

        # --- 6. match performance
        if is_match[:, t].any() and t >= BURN:
            comps = C.rating_components(feat)
            sig = sum(comps.values())
            for i in np.where((mins >= 15) & active)[0]:
                k = team_idx[i]
                fx = fixture_id[(k, t)]
                rating = (6.95 + talent[i] + sig[i] - 0.08 * (opp.get(fx, 3) - 3)
                          + (0.10 if home.get(fx) == "H" else 0.0) + rng.normal(0, 0.45))
                rating = float(np.clip(rating, 3.0, 10.0))
                pc = POSITIONS[pos_code[i]]
                att = pc in ("W", "ST", "AM")
                xg = max(0.0, rng.gamma(2.0, 0.12 if att else 0.03) * (mins[i] / 90) * (1 + 0.15 * (rating - 6.5)))
                xa = max(0.0, rng.gamma(2.0, 0.07 if pc in ("W", "AM", "CM", "FB") else 0.02) * (mins[i] / 90))
                matches.append(dict(
                    fixture_id=fx, athlete_idx=i, t=t, minutes_played=int(mins[i]),
                    started=bool(mins[i] >= 55 and mins[i] != 0 and rng.random() < 0.95),
                    performance_rating=round(rating, 2),
                    goals=int(rng.poisson(xg * 1.0)), assists=int(rng.poisson(xa)),
                    xg=round(xg, 3), xa=round(xa, 3),
                    pass_accuracy_pct=round(float(np.clip(78 + 3.0 * (rating - 6.5) + (4 if pc in ("CB", "DM", "CM", "GK") else 0)
                                                           + rng.normal(0, 3.5), 45, 98)), 1),
                    duels_won_pct=round(float(np.clip(50 + 4.0 * (rating - 6.5) + rng.normal(0, 6), 15, 85)), 1)))

        # --- 7. physical tests every 6 weeks (Monday), staggered per team by attendance randomness
        if d >= test_anchor and (d - test_anchor).days % 42 == 0:
            for i in np.where(available)[0]:
                if rng.random() < 0.9:
                    tests.append(dict(
                        athlete_idx=i, t=t,
                        cmj_cm=round(float(cmj[i] + rng.normal(0, 1.0)), 1),
                        sprint_30m_s=round(float(sprint[i] + rng.normal(0, 0.03)), 3),
                        yoyo_ir1_m=int(round(float(yoyo[i] + rng.normal(0, 80)) / 40) * 40),
                        squat_1rm_rel=round(float(squat[i] + rng.normal(0, 0.05)), 2),
                        nordic_n=round(float(nordic[i] + rng.normal(0, 15)), 0),
                        body_fat_pct=round(float(bodyfat[i] + rng.normal(0, 0.4)), 1)))

        # --- 8. adapt physical qualities (slow first-order dynamics toward load-dependent targets)
        ratio = np.clip(chronic / REF_LOAD, 0, 1.6)
        det = np.where(inj_until >= t, 1.0, 0.0)
        k_adapt = 0.012
        yoyo += k_adapt * (yoyo_b + 280 * np.clip(ratio - 0.9, -0.6, 0.4) - 120 * det - yoyo) + rng.normal(0, 2, N)
        sprint += k_adapt * (sprint_b - 0.04 * np.clip(ratio - 0.9, -0.6, 0.4) - 0.015 * (gym_freq - 1.3) + 0.01 * det - sprint) + rng.normal(0, 0.0006, N)
        cmj += k_adapt * (cmj_b + 1.6 * (gym_freq - 1.3) - 1.0 * det - cmj) + rng.normal(0, 0.04, N)
        squat += k_adapt * (squat_b + 0.10 * (gym_freq - 1.3) - 0.05 * det - squat) + rng.normal(0, 0.003, N)
        nordic += k_adapt * (nordic_b + 18 * (gym_freq - 1.3) - 8 * det - nordic) + rng.normal(0, 0.6, N)

    return dict(
        dates=dates, N=N, K=K, T=T, team_idx=team_idx, pos=pos, pos_code=pos_code, age0=age0,
        height=height, weight=weight, foot=foot, signed_date=signed_date, birth_date=birth_date,
        vendor=vendor, join_t=join_t, load=load, sleep_h=sleep_h, sleep_eff=sleep_eff, deep_pct=deep_pct,
        prot=prot, carb=carb, fat=fat, hyd=hyd, hrv=hrv, rhr=rhr, sore=sore, fatig=fatig, stress=stress, mood=mood,
        p_well=p_well, p_nut=p_nut, p_sleep=p_sleep, S_type=S_type, S_dur=S_dur, S_rpe=S_rpe,
        sprint_hist=sprint_hist, aer_hist=aer_hist, injuries=injuries, matches=matches, tests=tests,
        fixtures_df=fixtures_df, rng=rng,
    )


# ---------------------------------------------------------------------- tables
def aid(i):
    return f"A{i+1:04d}"


def build_tables(sim: dict) -> dict[str, pd.DataFrame]:
    rng = sim["rng"]
    dates, N, T = sim["dates"], sim["N"], sim["T"]
    team_idx, pos = sim["team_idx"], sim["pos"]
    out = {}
    batch = lambda dts: pd.to_datetime(dts) + pd.Timedelta(hours=26)  # noqa: E731  (nightly ingest)

    out["athletes"] = pd.DataFrame(dict(
        athlete_id=[aid(i) for i in range(N)], team_id=[f"T{k+1:02d}" for k in team_idx], position=pos,
        birth_date=pd.to_datetime(sim["birth_date"]).date, height_cm=np.round(sim["height"], 1),
        weight_kg=np.round(sim["weight"], 1), dominant_foot=sim["foot"],
        signed_date=pd.to_datetime(sim["signed_date"]).date,
        _loaded_at=pd.Timestamp(END) + pd.Timedelta(days=1)))

    mask_t = np.arange(T) >= BURN
    ts_idx = np.where(mask_t)[0]

    # sessions ---------------------------------------------------------------
    rows = []
    for slot in (0, 1):
        ii, tt = np.where((sim["S_dur"][:, :, slot] > 0) & mask_t[None, :])
        typ = sim["S_type"][ii, tt, slot]
        dur = sim["S_dur"][ii, tt, slot].astype(float)
        rpe = sim["S_rpe"][ii, tt, slot].astype(float)
        rows.append(pd.DataFrame(dict(i=ii, t=tt, typ=typ, dur=dur, rpe=rpe, slot=slot)))
    s = pd.concat(rows, ignore_index=True)
    s = s[s.dur >= 5].reset_index(drop=True)
    n = len(s)
    pcs = np.array([POS_DIST[p] for p in pos])[s.i.values]
    phs = np.array([POS_HSR[p] for p in pos])[s.i.values]
    gps_session = np.isin(s.typ.values, [MATCH, TRAIN, RECOV, INDIV])
    is_match_s = s.typ.values == MATCH
    aer_z = (sim["aer_hist"][s.i.values, s.t.values] - 2100) / 350
    dist = s.dur.values * (35 + 10.5 * s.rpe.values) * pcs * (1 + 0.04 * aer_z) * rng.normal(1, 0.07, n)
    hsr_frac = (0.01 + 0.0095 * s.rpe.values) * np.where(is_match_s, 1.0, 0.75) * phs
    hsr = dist * np.clip(hsr_frac * rng.normal(1, 0.12, n), 0.005, 0.2)
    spr = rng.poisson(np.where(is_match_s, 0.20, 0.08 * s.rpe.values / 6) * s.dur.values * phs)
    vmax = 32.0 + (4.2 - sim["sprint_hist"][s.i.values, s.t.values]) * 9 + rng.normal(0, 0.7, n)
    vmax = vmax * np.clip(0.78 + 0.03 * s.rpe.values, 0.8, 1.0)
    vmax = np.where(is_match_s, np.maximum(vmax, 31.0 - 0.0 * vmax), vmax)
    pl = dist * 0.11 * (1 + 0.04 * s.rpe.values)
    hr = 112 + 6.0 * s.rpe.values + rng.normal(0, 4, n)
    veh = sim["vendor"][s.i.values]
    speed_raw = np.where(veh == "VENDOR_B", vmax / 3.6, vmax)
    sessions = pd.DataFrame(dict(
        session_id=[f"S{x:08d}" for x in range(1, n + 1)],
        athlete_id=[aid(x) for x in s.i.values],
        session_date=dates[s.t.values].date,
        session_type=[TYPE_NAME[int(c)] for c in s.typ.values],
        duration_min=np.round(s.dur.values, 1), rpe=np.round(s.rpe.values, 1),
        total_distance_m=np.where(gps_session, np.round(dist, 0), np.nan),
        hsr_distance_m=np.where(gps_session, np.round(hsr, 0), np.nan),
        sprint_count=np.where(gps_session, spr, np.nan),
        max_speed_raw=np.where(gps_session, np.round(speed_raw, 2), np.nan),
        speed_unit=np.where(veh == "VENDOR_B", "ms", "kmh"),
        player_load=np.where(gps_session, np.round(pl, 1), np.nan),
        avg_hr_bpm=np.where(gps_session, np.round(hr, 0), np.nan),
        gps_vendor=veh,
        _loaded_at=batch(dates[s.t.values])))
    # defects: device failures, outliers, duplicate deliveries
    gps_cols = ["total_distance_m", "hsr_distance_m", "sprint_count", "max_speed_raw", "player_load", "avg_hr_bpm"]
    dead = rng.random(n) < 0.012
    sessions.loc[dead, gps_cols] = np.nan
    glitch = (rng.random(n) < 0.0015) & sessions.max_speed_raw.notna()
    sessions.loc[glitch, "max_speed_raw"] = 99.9
    dup = sessions.sample(frac=0.004, random_state=7).copy()
    dup["_loaded_at"] = dup["_loaded_at"] + pd.Timedelta(hours=6)
    out["training_sessions"] = pd.concat([sessions, dup], ignore_index=True)

    # wellness / sleep / nutrition -----------------------------------------------
    def daily(frame_fn, p_log, extra_mask=None):
        ii, tt = np.meshgrid(np.arange(N), ts_idx, indexing="ij")
        keep = (rng.random((N, len(ts_idx))) < p_log[:, None]) & (tt >= sim["join_t"][:, None])
        ii, tt = ii[keep], tt[keep]
        return ii, tt

    ii, tt = daily(None, sim["p_well"])
    well = pd.DataFrame(dict(
        athlete_id=[aid(x) for x in ii], wellness_date=dates[tt].date,
        hrv_rmssd_ms=np.round(sim["hrv"][ii, tt], 1), resting_hr_bpm=np.round(sim["rhr"][ii, tt], 0),
        soreness_1_10=np.round(sim["sore"][ii, tt], 0), fatigue_1_10=np.round(sim["fatig"][ii, tt], 0),
        stress_1_10=np.round(sim["stress"][ii, tt], 0), mood_1_10=np.round(sim["mood"][ii, tt], 0),
        _loaded_at=batch(dates[tt])))
    bad = rng.random(len(well)) < 0.003
    well.loc[bad, "hrv_rmssd_ms"] = rng.choice([0.0, 412.0, 999.0], bad.sum())
    out["wellness_daily"] = well

    ii, tt = daily(None, sim["p_sleep"])
    out["sleep_daily"] = pd.DataFrame(dict(
        athlete_id=[aid(x) for x in ii], sleep_date=dates[tt].date,
        sleep_hours=np.round(sim["sleep_h"][ii, tt], 2), sleep_efficiency_pct=np.round(sim["sleep_eff"][ii, tt], 1),
        deep_sleep_pct=np.round(sim["deep_pct"][ii, tt], 1), source_device="wearable",
        _loaded_at=batch(dates[tt])))

    ii, tt = daily(None, sim["p_nut"])
    wkg = sim["weight"][ii]
    noise = lambda: rng.lognormal(0, 0.10, len(ii))  # noqa: E731  self-report error
    pg, cg, fg = sim["prot"][ii, tt] * wkg * noise(), sim["carb"][ii, tt] * wkg * noise(), sim["fat"][ii, tt] * wkg * noise()
    kcal = 4 * pg + 4 * cg + 9 * fg
    nut = pd.DataFrame(dict(
        athlete_id=[aid(x) for x in ii], nutrition_date=dates[tt].date, kcal=np.round(kcal, 0),
        protein_g=np.round(pg, 0), carbs_g=np.round(cg, 0), fat_g=np.round(fg, 0),
        hydration_l=np.round(sim["hyd"][ii, tt] * noise(), 2), _loaded_at=batch(dates[tt])))
    nut.loc[rng.random(len(nut)) < 0.02, "kcal"] = np.nan
    out["nutrition_daily"] = nut

    # injuries -----------------------------------------------------------------
    inj = pd.DataFrame(sim["injuries"])
    inj = inj[inj.t >= BURN].reset_index(drop=True)
    out["injuries"] = pd.DataFrame(dict(
        injury_id=[f"I{x:06d}" for x in range(1, len(inj) + 1)], athlete_id=[aid(x) for x in inj.athlete_idx],
        injury_date=dates[inj.t.values].date, body_part=inj.body_part, injury_type=inj.injury_type,
        mechanism=inj.mechanism, days_out=inj.days_out, occurred_during=inj.occurred_during,
        _loaded_at=batch(dates[inj.t.values])))

    # fixtures / matches ----------------------------------------------------------
    out["fixtures"] = sim["fixtures_df"].assign(_loaded_at=pd.Timestamp(END) + pd.Timedelta(days=1))
    m = pd.DataFrame(sim["matches"])
    out["match_player_stats"] = pd.DataFrame(dict(
        match_stat_id=[f"M{x:07d}" for x in range(1, len(m) + 1)], fixture_id=m.fixture_id,
        athlete_id=[aid(x) for x in m.athlete_idx], match_date=dates[m.t.values].date,
        minutes_played=m.minutes_played, started=m.started, performance_rating=m.performance_rating,
        goals=m.goals, assists=m.assists, xg=m.xg, xa=m.xa, pass_accuracy_pct=m.pass_accuracy_pct,
        duels_won_pct=m.duels_won_pct, _loaded_at=batch(dates[m.t.values])))

    # physical tests -----------------------------------------------------------------
    te = pd.DataFrame(sim["tests"])
    out["physical_tests"] = pd.DataFrame(dict(
        test_id=[f"P{x:06d}" for x in range(1, len(te) + 1)], athlete_id=[aid(x) for x in te.athlete_idx],
        test_date=dates[te.t.values].date, cmj_cm=te.cmj_cm, sprint_30m_s=te.sprint_30m_s,
        yoyo_ir1_m=te.yoyo_ir1_m, squat_1rm_rel=te.squat_1rm_rel, nordic_n=te.nordic_n,
        body_fat_pct=te.body_fat_pct, _loaded_at=batch(dates[te.t.values])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--teams", type=int, default=8)
    ap.add_argument("--per-team", type=int, default=30)
    a = ap.parse_args()
    t0 = time.time()
    sim = simulate(a.seed, a.teams, a.per_team)
    tables = build_tables(sim)
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, df in tables.items():
        df.to_parquet(out / f"{name}.parquet", index=False)
        manifest[name] = len(df)
    manifest["_total_rows"] = int(sum(manifest.values()))
    manifest["_window"] = [str(START.date()), str(END.date())]
    (out / "_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
