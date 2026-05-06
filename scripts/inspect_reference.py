import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd

REF_DIR = ROOT / "자료실"
files = [
    ('시도별 주택용', str(REF_DIR / '③각 시도별 주택용 시간대별 전력사용량 상세정보.xlsx')),
    ('주거용도', str(REF_DIR / '⑦ 주거용도 시간대별 전력사용량 상세정보.xlsx')),
]

with open(str(REF_DIR / '_schema.txt'), 'w', encoding='utf-8') as out:
    for name, path in files:
        out.write(f'\n{"="*60}\n')
        out.write(f'{name}\n')
        out.write(f'{"="*60}\n')
        df = pd.read_excel(path, header=None, nrows=12)
        for i, row in df.iterrows():
            vals = [str(v) for v in row if str(v) != 'nan']
            if vals:
                line = ' | '.join(vals)
                out.write(f'  row {i}: {line}\n')

print('saved to _schema.txt')
