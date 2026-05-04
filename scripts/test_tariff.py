import sys; sys.path.insert(0, 'E:/AX_Contest')
from src.tariff import estimate_bill, list_supported_contracts

print("지원 계약종별:")
for c in list_supported_contracts():
    print(f"  {c}")

print("\n요금 비교:")
tests = [
    ("주택용", 300, 1),
    ("주택용(고압)", 300, 1),
    ("일반용(갑)I", 1000, 1),
    ("일반용(갑)II", 1000, 7),
    ("일반용(을)저압", 1000, 7),
    ("일반용(을)고압A", 1000, 7),
    ("일반용(을)고압B", 1000, 7),
    ("일반용", 1000, 7),
]
for ct, kwh, m in tests:
    r = estimate_bill(kwh, ct, m)
    print(f"  {ct:20s} {kwh:>5}kWh  m={m}  → {r['total_won']:>10,}원  ({r['tariff_type']})")
