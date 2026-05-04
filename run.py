"""안심구역 대화형 실행기 — 메뉴에서 선택만 하면 됨.

사용법:
  python run.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ask(prompt, options=None, default=None):
    """사용자 입력 받기."""
    if options:
        print(f"\n  {prompt}")
        for i, opt in enumerate(options, 1):
            marker = " (기본)" if opt == default else ""
            print(f"    {i}. {opt}{marker}")
        while True:
            raw = input(f"  선택 [1-{len(options)}]: ").strip()
            if not raw and default:
                return default
            try:
                idx = int(raw) - 1
                if 0 <= idx < len(options):
                    return options[idx]
            except ValueError:
                pass
            print(f"    1~{len(options)} 사이 숫자를 입력하세요")
    else:
        raw = input(f"  {prompt} [{default}]: ").strip()
        return raw if raw else default


def run_cmd(cmd):
    """명령어 실행."""
    print(f"\n{'='*60}")
    print(f"실행: {cmd}")
    print(f"{'='*60}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode


def main():
    print("""
╔══════════════════════════════════════════════════════╗
║   전기요금 과다발생 사전 예측모델 — 안심구역 실행기   ║
║   2026 AX 경진대회 · 한국전력공사 지정과제            ║
╚══════════════════════════════════════════════════════╝
    """)

    # 소스 설정
    configs = list(Path("configs").glob("source_*.yaml"))
    config_names = [str(c) for c in configs]
    if not config_names:
        config_names = ["configs/source_dsz.yaml"]

    source = ask("데이터 소스 선택:", config_names, config_names[0])
    train_end = ask("학습 종료 기간 (예: 2023-12):", default="2023-12")

    print(f"\n  소스: {source}")
    print(f"  학습 종료: {train_end}")

    # 메뉴
    while True:
        print(f"""
┌──────────────────────────────────────────────────┐
│  실행할 작업을 선택하세요                         │
├──────────────────────────────────────────────────┤
│  0. LP 데이터 구조 탐색 (최초 1회)               │
│  1. 전체 파이프라인 (원스톱)                      │
│  2. BTM 전략 비교 (A/B/C)                        │
│  3. 기상 버전 비교                                │
│  4. 피처 선택                                     │
│  5. Ablation Study (최적 조합 탐색)                │
│  6. 하이퍼파라미터 튜닝                           │
│  7. 배치 추론 (학습된 모델로 예측)                │
│  8. 프로파일러 단독                               │
│  9. 베이스라인 평가                               │
│  10. 대시보드 실행 (Streamlit)                    │
│                                                  │
│  s. 소스/기간 변경                                │
│  q. 종료                                          │
└──────────────────────────────────────────────────┘""")

        choice = input("\n  번호 입력: ").strip().lower()

        if choice == "q":
            print("\n  종료합니다.")
            break

        elif choice == "s":
            source = ask("데이터 소스:", config_names, source)
            train_end = ask("학습 종료 기간:", default=train_end)
            print(f"\n  변경됨: {source} / {train_end}")
            continue

        elif choice == "0":
            lp_path = ask("LP 데이터 경로 (파일 또는 디렉터리):", default="/path/to/lp")
            run_cmd(f'python -m scripts.inspect_lp --path "{lp_path}"')

        elif choice == "1":
            resume = ask("이전 체크포인트에서 이어서?", ["새로 실행", "이어서 실행"], "새로 실행")
            skip_profile = ask("프로파일러 스킵?", ["아니오", "예"], "아니오")
            skip_shap = ask("SHAP 분석 스킵?", ["아니오", "예"], "아니오")
            cmd = f'python -m scripts.run_full_analysis --source {source} --train-end {train_end}'
            if resume == "이어서 실행":
                cmd += " --resume"
            if skip_profile == "예":
                cmd += " --skip-profile"
            if skip_shap == "예":
                cmd += " --skip-shap"
            run_cmd(cmd)

        elif choice == "2":
            run_cmd(f'python -m scripts.compare_btm_strategies --source {source} --train-end {train_end}')

        elif choice == "3":
            run_cmd(f'python -m scripts.compare_weather_versions --source {source} --train-end {train_end}')

        elif choice == "4":
            corr_th = ask("상관 임계값:", default="0.85")
            vif_th = ask("VIF 임계값:", default="10.0")
            run_cmd(f'python -m scripts.select_features --source {source} --train-end {train_end} --corr-threshold {corr_th} --vif-threshold {vif_th}')

        elif choice == "5":
            run_cmd(f'python -m scripts.ablation_study --source {source} --train-end {train_end}')

        elif choice == "6":
            timeout = ask("튜닝 제한 시간 (초):", default="600")
            n_trials = ask("최대 시도 횟수:", default="100")
            run_cmd(f'python -m scripts.tune_lgbm --source {source} --train-end {train_end} --timeout {timeout} --n-trials {n_trials}')

        elif choice == "7":
            weights = ask("모델 가중치 경로:", default="weights/dsz_lgbm/")
            run_cmd(f'python -m scripts.predict_batch --source {source} --weights {weights}')

        elif choice == "8":
            run_cmd(f'python -m scripts.profile --source {source}')

        elif choice == "9":
            run_cmd(f'python -m scripts.run_baselines --source {source}')

        elif choice == "10":
            port = ask("포트:", default="8765")
            run_cmd(f'python -m streamlit run ui/app.py --server.port {port}')

        else:
            print("  올바른 번호를 입력하세요")

        input("\n  [Enter] 메뉴로 돌아가기...")


if __name__ == "__main__":
    main()
