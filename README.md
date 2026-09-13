# Auto-Recommend — Build Log

## Week 1 progress
- [x] Project scaffold
- [x] TFN (Triangular Fuzzy Number) library — core/fuzzy_number.py
  - construction (crisp, from mean/std, clipped)
  - arithmetic (+, -, *, /)
  - vertex distance (for TOPSIS)
  - centroid defuzzification
  - width (uncertainty indicator)
  - weighted_sum, fuzzy_positive_ideal, fuzzy_negative_ideal helpers
- [x] 17 unit tests, all passing (tests/test_fuzzy_number.py)
- [x] pyfdm integration — core/mcdm_adapter.py
  - converts our TFN matrices into pyfdm's array format
  - rank_with_topsis(), rank_with_vikor() wrappers
  - elicit_weights_fahp() for fuzzy-AHP weight elicitation
  - rank_correlation() for the TOPSIS<->VIKOR agreement gate (rho >= 0.85)
- [x] End-to-end smoke test (examples/rank_three_cars.py) — 3 toy cars,
      ranked with both fTOPSIS and fVIKOR, agreement checked. Passes.
- [ ] Car dataset (~30-40 cars: specs + reviews) -- THIS IS THE NEXT BLOCKER

## IMPORTANT: pyfdm install gotcha
`pip install pyfdm` pulls v1.2.0 from PyPI, whose sdist is missing
CHANGELOG.md and fails to build. Install straight from GitHub instead:
```
pip install "git+https://github.com/jwieckowski/pyfdm.git"
```
This gets you the full v1.2.0 including fAHP (subjective weighting),
which isn't in the last working PyPI release (1.1.13).

## Run tests
```
pip install -r requirements.txt
pip install "git+https://github.com/jwieckowski/pyfdm.git"
python -m pytest tests/ -v
PYTHONPATH=. python examples/rank_three_cars.py
```

## Next up (Week 2)
- Source car spec dataset (CarDekho Kaggle dataset) for ~30-40 models
- Collect review text (Edmunds Kaggle dataset) per car
- Build aspect-sentiment scoring (PyABSA, pretrained) -> feed into
  TFN.from_mean_std_clipped -> build real decision matrices for
  rank_with_topsis / rank_with_vikor
