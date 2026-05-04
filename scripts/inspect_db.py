"""안심구역 DB 접속 탐색 — DB로 데이터가 제공될 경우 사용.

DB 종류별 접속 시도 + 테이블 목록 + 샘플 조회 + source_dsz.yaml 생성.

사용법:
  python -m scripts.inspect_db --type oracle --host localhost --port 1521 --sid KEPCO --user analyst --password ****
  python -m scripts.inspect_db --type postgresql --host localhost --port 5432 --dbname kepco --user analyst --password ****
"""
from __future__ import annotations

import argparse
import io as _stdio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.stdout = _stdio.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd


def connect_oracle(host, port, sid, user, password):
    import cx_Oracle
    dsn = cx_Oracle.makedsn(host, port, sid=sid)
    conn = cx_Oracle.connect(user, password, dsn)
    return conn, "oracle"


def connect_postgresql(host, port, dbname, user, password):
    import psycopg2
    conn = psycopg2.connect(host=host, port=port, dbname=dbname, user=user, password=password)
    return conn, "postgresql"


def list_tables(conn, db_type):
    if db_type == "oracle":
        q = "SELECT table_name FROM user_tables ORDER BY table_name"
    else:
        q = "SELECT table_name FROM information_schema.tables WHERE table_schema='public' ORDER BY table_name"
    return pd.read_sql(q, conn)


def sample_table(conn, table_name, nrows=5):
    q = f"SELECT * FROM {table_name} WHERE ROWNUM <= {nrows}" if "oracle" else f"SELECT * FROM {table_name} LIMIT {nrows}"
    return pd.read_sql(q, conn)


def main() -> int:
    parser = argparse.ArgumentParser(description="안심구역 DB 탐색")
    parser.add_argument("--type", choices=["oracle", "postgresql"], required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--sid", default=None, help="Oracle SID")
    parser.add_argument("--dbname", default=None, help="PostgreSQL DB name")
    parser.add_argument("--user", required=True)
    parser.add_argument("--password", required=True)
    args = parser.parse_args()

    if args.port is None:
        args.port = 1521 if args.type == "oracle" else 5432

    print(f"[connect] {args.type}://{args.host}:{args.port}")
    try:
        if args.type == "oracle":
            conn, db_type = connect_oracle(args.host, args.port, args.sid, args.user, args.password)
        else:
            conn, db_type = connect_postgresql(args.host, args.port, args.dbname, args.user, args.password)
        print("  연결 성공!")
    except Exception as e:
        print(f"  연결 실패: {e}")
        print("\n  필요한 패키지:")
        print("    Oracle: pip install cx_Oracle")
        print("    PostgreSQL: pip install psycopg2-binary")
        return 1

    # 테이블 목록
    print(f"\n{'='*60}")
    print("테이블 목록:")
    print(f"{'='*60}")
    tables = list_tables(conn, db_type)
    for _, row in tables.iterrows():
        print(f"  {row.iloc[0]}")

    # LP 후보 테이블 탐색
    lp_keywords = ["lp", "load", "profile", "meter", "ami", "usage", "전력", "검침", "사용"]
    candidates = []
    for _, row in tables.iterrows():
        tname = str(row.iloc[0]).lower()
        if any(kw in tname for kw in lp_keywords):
            candidates.append(row.iloc[0])

    if candidates:
        print(f"\n{'='*60}")
        print(f"LP 데이터 후보 테이블: {candidates}")
        print(f"{'='*60}")
        for tname in candidates[:3]:
            print(f"\n  --- {tname} ---")
            try:
                sample = sample_table(conn, tname)
                print(f"  컬럼: {sample.columns.tolist()}")
                print(sample.head().to_string())
            except Exception as e:
                print(f"  조회 실패: {e}")

    conn.close()
    print("\n[done] 연결 종료")
    print("\n다음 단계:")
    print("  1. LP 테이블 확인 후 아래와 같이 데이터 추출:")
    print("     df = pd.read_sql('SELECT * FROM LP_TABLE', conn)")
    print("  2. df.to_csv('lp_data.csv', index=False)")
    print("  3. python -m scripts.inspect_lp --path lp_data.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
