"""
Smoke test proving the full chain works: build a small fuzzy decision
matrix using our TFN objects, rank it with fuzzy TOPSIS and fuzzy VIKOR
via pyfdm, and check TOPSIS/VIKOR agreement -- exactly the pattern the
real MCDM engine (Module 3) will use, just with 3 toy cars instead of 30-40.
"""
from core.fuzzy_number import TFN
from core.mcdm_adapter import rank_with_topsis, rank_with_vikor, rank_correlation

# 3 cars x 3 criteria: [safety, comfort, price]
# safety, comfort are benefit criteria (higher = better)
# price is a cost criterion (lower = better) -- so a "cheap" car should
# still win if safety/comfort are strong enough, once weighted.
cars = ["Hyundai Creta", "Kia Seltos", "Tata Nexon"]

decision_matrix = [
    # safety            comfort            price (in lakh INR, cost)
    [TFN(0.85, 0.91, 0.95), TFN(0.78, 0.84, 0.90), TFN(11.5, 12.3, 13.0)],  # Creta
    [TFN(0.80, 0.86, 0.90), TFN(0.75, 0.80, 0.85), TFN(11.0, 11.8, 12.5)],  # Seltos
    [TFN(0.88, 0.93, 0.97), TFN(0.65, 0.70, 0.75), TFN(9.0, 9.8, 10.5)],   # Nexon
]

weights = [0.40, 0.25, 0.35]  # safety weighted highest, matches AHP-style priorities
types = [1, 1, -1]  # safety=benefit, comfort=benefit, price=cost

topsis_result = rank_with_topsis(decision_matrix, weights, types)
vikor_result = rank_with_vikor(decision_matrix, weights, types)

print("=== Fuzzy TOPSIS ===")
for idx in topsis_result["ranking_indices"]:
    print(f"  {cars[idx]}: CC = {topsis_result['closeness_coefficients'][idx]:.4f}")

print("\n=== Fuzzy VIKOR ===")
for idx in vikor_result["ranking_indices"]:
    print(f"  {cars[idx]}: Q = {vikor_result['q_scores'][idx]:.4f}")

agreement = rank_correlation(
    topsis_result["ranking_indices"], vikor_result["ranking_indices"]
)
print(f"\nTOPSIS <-> VIKOR rank correlation: {agreement:.3f}")
print("(Design doc's evaluation gate: rho >= 0.85 means the ranking is trustworthy)")
