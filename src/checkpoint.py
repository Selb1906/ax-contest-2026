"""체크포인트 + 진행 상황 로깅 — 중단 안전 + 실시간 모니터링.

모든 분석 단계에서:
  1. 시작/종료/에러를 로그에 기록
  2. 중간 결과를 즉시 저장 (중단 시 이어서 가능)
  3. 진행률을 실시간 출력
"""
from __future__ import annotations

import json


class SkipStep(Exception):
    """resume 시 이미 완료된 Step을 건너뛰기 위한 예외."""
    pass
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class Progress:
    """진행 상황 추적."""
    total: int = 0
    current: int = 0
    step_name: str = ""
    start_time: float = 0
    sub_label: str = ""
    _skipped: bool = False

    def start(self, step_name: str, total: int = 0):
        self.step_name = step_name
        self.total = total
        self.current = 0
        self.start_time = time.time()
        print(f"\n{'='*60}")
        print(f"[START] {step_name}")
        if total > 0:
            print(f"  대상: {total:,}건")
        print(f"{'='*60}")

    def update(self, n: int = 1, label: str = ""):
        self.current += n
        self.sub_label = label
        elapsed = time.time() - self.start_time
        if self.total > 0:
            pct = self.current / self.total * 100
            rate = self.current / elapsed if elapsed > 0 else 0
            eta = (self.total - self.current) / rate if rate > 0 else 0
            bar_len = 30
            filled = int(bar_len * self.current / self.total)
            bar = "█" * filled + "░" * (bar_len - filled)
            status = f"  [{bar}] {pct:5.1f}% ({self.current:,}/{self.total:,})"
            status += f" | {elapsed:.0f}s elapsed | ETA {eta:.0f}s"
            if label:
                status += f" | {label}"
            print(f"\r{status}", end="", flush=True)
        else:
            if label:
                print(f"  [{elapsed:.1f}s] {label}")

    def finish(self, summary: str = ""):
        elapsed = time.time() - self.start_time
        if self.total > 0:
            print()
        print(f"[DONE] {self.step_name} — {elapsed:.1f}s")
        if summary:
            print(f"  {summary}")


class CheckpointManager:
    """체크포인트 관리자 — 중간 결과 저장 + 복원."""

    def __init__(self, base_dir: str | Path = "checkpoints"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.base_dir / "run_log.jsonl"
        self.progress = Progress()

    def _log(self, event: str, data: dict | None = None):
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "event": event,
            "step": self.progress.step_name,
        }
        if data:
            entry.update(data)
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def has_checkpoint(self, step_name: str) -> bool:
        return (self.base_dir / f"{step_name}.done").exists()

    def mark_done(self, step_name: str, summary: dict | None = None):
        marker = self.base_dir / f"{step_name}.done"
        marker.write_text(json.dumps(summary or {}, ensure_ascii=False, default=str))

    def save_intermediate(self, name: str, data: Any):
        """중간 결과 저장 — DataFrame, dict, 또는 문자열."""
        p = self.base_dir / name
        p.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, pd.DataFrame):
            # CSV + Parquet 이중 저장
            data.to_csv(str(p) + ".csv", index=False, encoding="utf-8-sig")
            try:
                data.to_parquet(str(p) + ".parquet", index=False)
            except Exception:
                pass
            self._log("save", {"file": name, "rows": len(data), "format": "csv+parquet"})

        elif isinstance(data, dict):
            with open(str(p) + ".json", "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            self._log("save", {"file": name, "format": "json"})

        elif isinstance(data, str):
            with open(str(p) + ".txt", "w", encoding="utf-8") as f:
                f.write(data)

    def load_intermediate(self, name: str, fmt: str = "parquet") -> Any:
        """중간 결과 로드."""
        if fmt == "parquet":
            p = self.base_dir / f"{name}.parquet"
            if p.exists():
                return pd.read_parquet(p)
        elif fmt == "csv":
            p = self.base_dir / f"{name}.csv"
            if p.exists():
                return pd.read_csv(p, encoding="utf-8-sig")
        elif fmt == "json":
            p = self.base_dir / f"{name}.json"
            if p.exists():
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
        return None

    @contextmanager
    def step(self, step_name: str, total: int = 0, skip_if_done: bool = True):
        """단계 실행 컨텍스트 — 시작/완료/에러 자동 로깅.

        Usage:
            with ckpt.step("profiler", total=19) as prog:
                for i, stat in enumerate(stats):
                    run_stat(stat)
                    prog.update(1, label=stat.name)
        """
        if skip_if_done and self.has_checkpoint(step_name):
            print(f"\n[SKIP] {step_name} — 이전 체크포인트 존재", flush=True)
            self._log("skip", {"step": step_name})
            self.progress._skipped = True
            yield self.progress
            return
        self.progress._skipped = False

        self.progress.start(step_name, total)
        self._log("start")

        try:
            yield self.progress
            self.progress.finish()
            self.mark_done(step_name)
            self._log("done", {"elapsed": time.time() - self.progress.start_time})
        except KeyboardInterrupt:
            elapsed = time.time() - self.progress.start_time
            print(f"\n\n[중단] {step_name} — {elapsed:.1f}s, "
                  f"{self.progress.current}/{self.progress.total}건 완료")
            self._log("interrupted", {
                "elapsed": elapsed,
                "completed": self.progress.current,
                "total": self.progress.total,
            })
            raise
        except Exception as e:
            elapsed = time.time() - self.progress.start_time
            print(f"\n[에러] {step_name} — {e}")
            self._log("error", {"error": str(e), "elapsed": elapsed})
            raise
