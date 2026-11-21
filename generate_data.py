"""Simulate a two-week checkout-redesign A/B test (synthetic data).

Scenario: a marketplace tests a redesigned one-page checkout (treatment) against the
existing multi-step checkout (control). The unit is a user who reached checkout.

Everything is simulated with a fixed seed. No real company or customer data.
The true effects built in are:
  - conversion: +1.9 pp on mobile, +0.2 pp on desktop (so the overall lift is ~+1.25 pp)
  - order value: unchanged
  - refund rate among buyers: 4.0% control vs 4.4% treatment (small, hard to detect)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 7
N_USERS = 32_000
START = pd.Timestamp("2026-09-01")
DAYS = 14
OUT = Path(__file__).parent / "data" / "experiment.csv"

P_CONTROL = {"mobile": 0.095, "desktop": 0.135}      # control conversion by device
LIFT = {"mobile": 0.019, "desktop": 0.002}           # true absolute lift of treatment
MEDIAN_ORDER = {"mobile": 22.0, "desktop": 25.0}     # EUR, lognormal order values
REFUND = {"control": 0.040, "treatment": 0.044}      # refund rate among buyers


def main():
    rng = np.random.default_rng(SEED)
    df = pd.DataFrame({"user_id": np.arange(1, N_USERS + 1)})
    df["arm"] = np.where(rng.random(N_USERS) < 0.5, "control", "treatment")   # random 50/50
    df["device"] = np.where(rng.random(N_USERS) < 0.62, "mobile", "desktop")
    days = rng.integers(0, DAYS, N_USERS)
    df["exposure_date"] = (START + pd.to_timedelta(days, unit="D")).strftime("%Y-%m-%d")

    p = df.device.map(P_CONTROL) + np.where(df.arm == "treatment", df.device.map(LIFT), 0.0)
    df["converted"] = (rng.random(N_USERS) < p).astype(int)

    value = rng.lognormal(np.log(df.device.map(MEDIAN_ORDER).values), 0.6)
    df["order_value"] = np.where(df.converted == 1, np.round(value, 2), 0.0)

    refund = rng.random(N_USERS) < df.arm.map(REFUND).values
    df["refunded"] = (refund & (df.converted == 1)).astype(int)

    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT.relative_to(OUT.parent.parent)}: {len(df):,} users")
    print(df.groupby("arm").agg(users=("user_id", "size"), conversion=("converted", "mean")))


if __name__ == "__main__":
    main()
