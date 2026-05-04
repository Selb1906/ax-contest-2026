"""안심구역 대화형 실행기 — 1~4명 팀 병렬 작업 지원.

사용법:
  python run.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def ask(prompt, options=None, default=None):
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
    print(f"\n{'='*60}")
    print(f"실행: {cmd}")
    print(f"{'='*60}\n")
    return subprocess.run(cmd, shell=True).returncode


# ── 팀 작업 분배 ──

TEAM_PLANS = {
    1: {
        "description": "1명 — 전체 파이프라인 순차 실행",
        "roles": {
            "PC 1": [
                "0. LP 구조 탐색 (inspect_lp)",
                "1. 전체 파이프라인 (Step 1~12 자동)",
            ],
        },
    },
    2: {
        "description": "2명 — 메인 파이프라인 + Ablation 병렬",
        "roles": {
            "PC 1 (메인)": [
                "0. LP 구조 탐색 → source_dsz.yaml 생성 (공유)",
                "1. 전체 파이프라인 Step 1~9 (--skip-shap으로 빠르게)",
                "→ Step 9 완료 후 PC 2에 weights/ 전달",
                "6. SHAP 분석 (Step 6 단독)",
                "   결과를 explain_results/에 저장",
            ],
            "PC 2 (Ablation)": [
                "※ PC 1이 Step 5 완료 후 시작 (weights/ 필요)",
                "5. Ablation Study",
                "6. 하이퍼파라미터 튜닝 (남은 시간)",
                "7. Sliding 상세 평가 (최적 모델로)",
            ],
        },
    },
    3: {
        "description": "3명 — 메인 + Ablation·튜닝 + SHAP·기상",
        "roles": {
            "PC 1 (메인)": [
                "0. LP 구조 탐색 → yaml + weights 공유",
                "1. Step 1~5 (학습까지) + Step 5.5 Sliding 평가",
                "→ Step 5 완료 즉시 weights/ 를 PC 2,3에 공유",
                "   Step 5 이후 대기 → PC 2 Ablation 완료 대기",
                "   → 최종 모델로 Sliding 재평가 + 스케일 아웃",
            ],
            "PC 2 (Ablation + 튜닝)": [
                "※ PC 1의 weights/ 수령 후 시작",
                "Step 10: Ablation Study (BTM/기상/기준온도/피처/그룹)",
                "Step 11: 최적 조합 재학습 + 피처 자동 선정",
                "Step 12: Optuna 튜닝 → Step 12.5 Sliding 재평가",
            ],
            "PC 3 (SHAP + 기상)": [
                "※ PC 1의 weights/ 수령 후 시작",
                "Step 6: SHAP 분석",
                "Step 7~8: 기상 최적화 + 민감도 분석",
                "→ explain_results/, weather_opt/ 저장",
            ],
        },
    },
    4: {
        "description": "4명 — 최대 병렬",
        "roles": {
            "PC 1 (메인)": [
                "0. LP 구조 탐색 → yaml + weights 공유",
                "1. Step 1~5 + Step 5.5 Sliding 평가",
                "→ weights/ 공유 후 대기 → 최종 Sliding + 스케일 아웃",
            ],
            "PC 2 (Ablation + 튜닝)": [
                "※ PC 1의 weights/ 수령 후 시작",
                "Step 10: Ablation Study",
                "Step 11: 최적 조합 재학습",
                "Step 12: Optuna 튜닝 + Step 12.5 Sliding",
            ],
            "PC 3 (SHAP + 기상)": [
                "※ PC 1의 weights/ 수령 후 시작",
                "Step 6: SHAP 분석",
                "Step 7~8: 기상 최적화 + 민감도",
            ],
            "PC 4 (BTM solar + 프로파일러)": [
                "BTM 태양광 발전량 예측 모델 작업 (btm_solar.py)",
                "프로파일러 결과 상세 검토 + 보고서 초안",
            ],
        },
    },
}


def show_team_plan(n_members, source, train_end):
    plan = TEAM_PLANS[n_members]
    print(f"""
╔══════════════════════════════════════════════════════╗
║  팀 작업 분배 — {plan['description']:40s}║
╚══════════════════════════════════════════════════════╝

  소스: {source}
  학습 종료: {train_end}
""")
    for role, tasks in plan["roles"].items():
        print(f"  ┌─ {role}")
        for task in tasks:
            print(f"  │  {task}")
        print(f"  └{'─'*50}")

    print(f"""
  공유 방법:
    - source_dsz.yaml: PC 1이 생성 → USB or 동일 경로에 복사
    - weights/: PC 1이 Step 5 완료 후 다른 PC에 복사
    - 결과물: 각 PC에서 해당 디렉터리에 저장 → 마지막에 합치기

  주의:
    - 모든 PC에 동일한 코드(git clone)가 있어야 함
    - 데이터 경로(source_dsz.yaml)가 동일해야 함
    - weights/를 공유받기 전에 Ablation/튜닝 실행 불가
""")


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

    # 팀 모드
    n_members = int(ask("작업 인원 (1~4):", default="1"))
    n_members = max(1, min(4, n_members))
    show_team_plan(n_members, source, train_end)

    if n_members > 1:
        my_role = ask("내 역할:", list(TEAM_PLANS[n_members]["roles"].keys()))
        print(f"\n  → {my_role} 작업을 시작합니다.")

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
│  t. 팀 작업 분배 다시 보기                        │
│  s. 소스/기간 변경                                │
│  q. 종료                                          │
└──────────────────────────────────────────────────┘""")

        choice = input("\n  번호 입력: ").strip().lower()

        if choice == "q":
            print("\n  종료합니다.")
            break
        elif choice == "t":
            show_team_plan(n_members, source, train_end)
            continue
        elif choice == "s":
            source = ask("데이터 소스:", config_names, source)
            train_end = ask("학습 종료 기간:", default=train_end)
            continue

        elif choice == "0":
            lp_path = ask("LP 데이터 경로:", default="/path/to/lp")
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
