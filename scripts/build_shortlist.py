"""
Build a shortlist of ~30-40 representative car models from the CarDekho
'Car details v3.csv' dataset.

Why this exists: the raw CSV is used-car *listings* (many rows per model,
varying by trim/year/mileage/ownership). Your project needs one row per
*model* with representative specs -- so this script groups listings by a
normalized (make, base_model) key, takes the median of numeric specs
across all listings of that model, and ranks models by how often they
appear (a cheap proxy for how common/relevant that model is).

It also attaches known Bharat NCAP / Global NCAP safety star ratings
where I have confirmed data (small seed dict below) and leaves the rest
as NaN with a TODO flag -- most models in a used-car dataset (2011-2020
era) predate the 2023+ Bharat NCAP program, so expect to fill this in
manually for your final shortlist. That's normal, not a bug: only ~33
cars total have been Bharat NCAP tested as of late 2026.
"""
from __future__ import annotations
import re
import pandas as pd

RAW_PATH = "data/raw/Car details v3.csv"
OUT_PATH = "data/processed/cars_shortlist.csv"
SHORTLIST_SIZE = 35

# Known multi-word makes (checked first so "Maruti Suzuki" doesn't split
# into make="Maruti", model="Suzuki ...")
KNOWN_MAKES = [
    "Maruti Suzuki", "Maruti", "Hyundai", "Honda", "Tata", "Mahindra",
    "Kia", "Toyota", "Renault", "Nissan", "Ford", "Volkswagen", "Skoda",
    "MG", "Jeep", "Chevrolet", "Datsun", "Fiat", "Mitsubishi", "Volvo",
    "BMW", "Audi", "Mercedes-Benz", "Land Rover",
]

# Trim/variant tokens that signal "stop here, rest is trim not model name".
# Deliberately conservative: only true ALL-CAPS trim codes (case-sensitive
# check, so real model words like "Swift" or "Creta" don't false-match),
# plus numeric displacement tokens and known keyword variants.
_TRIM_KEYWORDS = {
    "i-vtec", "vtec", "kappa", "kappa2", "bsiv", "bsvi",
    "turbo", "cvt", "amt", "dct", "ivt", "diesel", "petrol",
}


def _is_trim_token(tok: str) -> bool:
    if tok.isupper() and 2 <= len(tok) <= 6:
        return True  # e.g. VDI, VXI, ZXI, SX, EX
    if re.match(r"^[\d.]+$", tok):
        return True  # e.g. 1.2, 1.6
    if re.match(r"^\d+cc$", tok, re.IGNORECASE):
        return True  # e.g. 1198cc
    if tok.lower() in _TRIM_KEYWORDS:
        return True
    return False

# Confirmed Bharat NCAP / Global NCAP ratings (adult occupant protection
# stars). Extend this as you source more -- do NOT invent ratings for
# models not on this list; leave them NaN instead.
SAFETY_SEED = {
    ("tata", "nexon"): 5,
    ("tata", "punch"): 5,
    ("mahindra", "thar roxx"): 5,
    ("mahindra", "xuv400"): 5,
    ("hyundai", "tucson"): 5,
    ("kia", "syros"): 5,
    ("toyota", "innova hycross"): 5,
    ("maruti", "dzire"): 5,       # new-gen only; old-gen scored lower
    ("maruti suzuki", "dzire"): 5,
}


def split_make_model(full_name: str) -> tuple[str, str]:
    for make in KNOWN_MAKES:
        if full_name.lower().startswith(make.lower() + " "):
            return make, full_name[len(make):].strip()
    parts = full_name.split(" ", 1)
    return (parts[0], parts[1] if len(parts) > 1 else "")


def extract_base_model(model_part: str) -> str:
    tokens = model_part.split()
    base_tokens = []
    for tok in tokens:
        if _is_trim_token(tok):
            break
        base_tokens.append(tok)
    return " ".join(base_tokens) if base_tokens else model_part


def parse_mileage(val) -> float | None:
    if pd.isna(val):
        return None
    m = re.search(r"[\d.]+", str(val))
    return float(m.group()) if m else None


def parse_engine_cc(val) -> float | None:
    if pd.isna(val):
        return None
    m = re.search(r"[\d.]+", str(val))
    return float(m.group()) if m else None


def parse_power_bhp(val) -> float | None:
    if pd.isna(val):
        return None
    m = re.search(r"[\d.]+", str(val))
    return float(m.group()) if m else None


def build_shortlist(raw_path: str = RAW_PATH, size: int = SHORTLIST_SIZE) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    df["make"], df["model_part"] = zip(*df["name"].map(split_make_model))
    df["base_model"] = df["model_part"].map(extract_base_model)
    df["mileage_kmpl"] = df["mileage"].map(parse_mileage)
    df["engine_cc"] = df["engine"].map(parse_engine_cc)
    df["max_power_bhp"] = df["max_power"].map(parse_power_bhp)

    grouped = (
        df.groupby(["make", "base_model"])
        .agg(
            listing_count=("name", "count"),
            median_selling_price=("selling_price", "median"),
            median_km_driven=("km_driven", "median"),
            median_mileage_kmpl=("mileage_kmpl", "median"),
            median_engine_cc=("engine_cc", "median"),
            median_max_power_bhp=("max_power_bhp", "median"),
            seats=("seats", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            most_common_fuel=("fuel", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
            most_common_transmission=("transmission", lambda s: s.mode().iloc[0] if not s.mode().empty else None),
        )
        .reset_index()
        .sort_values("listing_count", ascending=False)
    )

    shortlist = grouped.head(size).copy()
    shortlist["safety_ncap_stars"] = shortlist.apply(
        lambda r: SAFETY_SEED.get((r["make"].lower(), r["base_model"].lower())),
        axis=1,
    )
    shortlist["safety_needs_manual_lookup"] = shortlist["safety_ncap_stars"].isna()

    print(f"Grouped {len(df)} listings into {len(grouped)} distinct models.")
    print(f"Took top {len(shortlist)} by listing frequency.")
    print(
        f"Safety ratings pre-filled: {(~shortlist['safety_needs_manual_lookup']).sum()} "
        f"/ {len(shortlist)} -- expect to manually fill most of the rest."
    )
    return shortlist


if __name__ == "__main__":
    import os

    os.makedirs("data/processed", exist_ok=True)
    result = build_shortlist()
    result.to_csv(OUT_PATH, index=False)
    print(f"\nWrote shortlist to {OUT_PATH}")
    print(result.to_string())
