#!/usr/bin/env python3
"""
download_full_release.py — Safe, resumable, atomic, parallel downloader for
the EEG 2025 Challenge full releases from the public S3 bucket.

S3 bucket pattern  : s3://nmdatasets/NeurIPS25/{release}_L100_bdf
Local destination  : ./data/EEG2025r{n}/...   (matches eegdash convention)
Temp area          : ./data/EEG2025r{n}/.tmp/...

Safety model
------------
* Every file is first written to a sibling `.tmp` path, then `os.replace()`d
  atomically into its final location only after size verification.
* Interrupts (Ctrl+C / SIGTERM): pending parallel downloads are cancelled,
  running ones are allowed to finish or time out, and any partial `.tmp`
  files are unlinked. Nothing partial ever appears in the final tree.
  A second Ctrl+C force-quits immediately; any in-flight `.tmp` files
  are then orphaned, but the next run's startup cleanup removes them.
* Stale `.tmp` files from previous interrupted runs are cleaned on startup.

New features (vs. the original version)
---------------------------------------
1. Retry with exponential backoff: up to MAX_RETRIES attempts per file,
   sleeping 1s, 2s, 4s, 8s, 16s between them. Applied to both network
   errors and size mismatches.
2. Disk-space pre-flight: refuses to start if free space on the cache
   filesystem is less than missing_bytes * 1.1 (10% safety margin).
3. Task filter: by default only downloads Challenge-1-relevant tasks
   (contrastChangeDetection, surroundSupp, RestingState) plus global
   metadata (files without any `task-<name>` token). Override with
   --tasks / --all-tasks.
4. Parallel downloads: ThreadPoolExecutor with configurable workers
   (default 6). Each worker uses its own thread-local s3fs instance
   to avoid async/thread issues in s3fs. Atomicity is preserved.
5. Per-file timeout: max(60, ceil(size_MB * 5)) seconds wall-clock per
   attempt, enforced by a daemon thread + join(timeout=...). Plus
   botocore-level connect/read timeouts as a secondary line of defence.

Usage
-----
    # Default: R5, Challenge-1 tasks only, 6 workers
    python3 download_full_release.py

    # Multiple releases, more workers
    python3 download_full_release.py R5 R6 --workers 8

    # No task filter (download everything)
    python3 download_full_release.py R5 --all-tasks

    # Custom task list
    python3 download_full_release.py R5 --tasks contrastChangeDetection,RestingState

    # Dry run — list what would be downloaded
    python3 download_full_release.py --dry-run R5

    # Custom cache root and retries
    python3 download_full_release.py --cache-dir /data R5 --max-retries 3

Interrupt safely with Ctrl+C at any time.
"""

from __future__ import annotations

import argparse
import math
import os
import re
import shutil
import signal
import sys
import threading
import time
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import s3fs
except ImportError:
    sys.exit(
        "ERROR: s3fs is not installed. Install it with:\n"
        "    pip install s3fs\n"
        "or use the same Python interpreter that runs train.py."
    )


# ─── Configuration ──────────────────────────────────────────────────────────
S3_BUCKET_ROOT = "s3://nmdatasets/NeurIPS25"
S3_REGION = "us-east-2"

# All valid releases. Prefer eegdash's authoritative mapping so this script
# automatically picks up new releases; fall back to the known set as of R11.
try:
    from eegdash.const import RELEASE_TO_OPENNEURO_DATASET_MAP as _REL_MAP

    VALID_RELEASES = set(_REL_MAP.keys())
except Exception:
    VALID_RELEASES = {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"}

# Challenge-1 relevant tasks (target + useful passive tasks for pretraining).
# Names are the bare token used inside `task-<name>` in BIDS filenames.
CHALLENGE1_TASKS = ("contrastChangeDetection", "surroundSupp", "RestingState")

# Retry / backoff
DEFAULT_MAX_RETRIES = 5
RETRY_BACKOFF_BASE_S = 1.0  # -> 1, 2, 4, 8, 16 s

# Disk safety factor
DISK_SAFETY_FACTOR = 1.10

# Parallelism
DEFAULT_WORKERS = 6

# Per-file timeout: max(MIN, size_MB * SCALE) seconds
TIMEOUT_MIN_S = 60
TIMEOUT_PER_MB_S = 5.0

# botocore-level timeouts (seconds)
BOTO_CONNECT_TIMEOUT_S = 15
BOTO_READ_TIMEOUT_S = 60

# BIDS-style task token extractor: matches the first `task-<Name>` in a path.
_TASK_RE = re.compile(r"task-([A-Za-z0-9]+)")


# ─── Graceful shutdown ──────────────────────────────────────────────────────
_interrupted = False
_print_lock = threading.Lock()


def _log(msg: str, *, err: bool = False) -> None:
    with _print_lock:
        print(msg, file=sys.stderr if err else sys.stdout, flush=True)


def _install_signal_handlers() -> None:
    def _handler(signum, frame):  # noqa: ARG001
        global _interrupted
        _interrupted = True
        _log(
            "\n[!] Interrupt received — cancelling pending downloads, letting "
            "running ones finish cleanly. Press Ctrl+C again to force-quit.",
            err=True,
        )
        # Restore default so a second Ctrl+C exits immediately.
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        try:
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
        except Exception:
            pass

    signal.signal(signal.SIGINT, _handler)
    try:
        signal.signal(signal.SIGTERM, _handler)
    except Exception:
        pass


# ─── Helpers ────────────────────────────────────────────────────────────────
def _human(n_bytes: int | float) -> str:
    size = float(n_bytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            return f"{size:6.2f} {unit}"
        size /= 1024.0
    return f"{size:6.2f} PiB"


def _make_fs() -> s3fs.S3FileSystem:
    return s3fs.S3FileSystem(
        anon=True,
        client_kwargs={"region_name": S3_REGION},
        config_kwargs={
            "connect_timeout": BOTO_CONNECT_TIMEOUT_S,
            "read_timeout": BOTO_READ_TIMEOUT_S,
            "retries": {"max_attempts": 3, "mode": "standard"},
        },
    )


# Thread-local s3fs: s3fs mixes asyncio + threads in subtle ways; creating a
# fresh instance per worker thread is the safest cross-version pattern.
_thread_local = threading.local()


def _fs_for_thread() -> s3fs.S3FileSystem:
    fs = getattr(_thread_local, "fs", None)
    if fs is None:
        fs = _make_fs()
        _thread_local.fs = fs
    return fs


def _cleanup_stale_tmp(tmp_root: Path) -> int:
    """Remove any leftover .tmp files from a previous interrupted run."""
    if not tmp_root.exists():
        return 0
    removed = 0
    for p in tmp_root.rglob("*"):
        if p.is_file():
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def _file_passes_task_filter(
    rel_path: str, allowed_tasks: set[str] | None
) -> bool:
    """
    True if the file should be downloaded under the given task filter.

    Rules:
      - allowed_tasks is None  -> no filter, keep everything.
      - rel_path has no `task-<X>` token -> global metadata, always keep.
      - rel_path has `task-<X>` -> keep iff X in allowed_tasks.
    """
    if allowed_tasks is None:
        return True
    m = _TASK_RE.search(rel_path)
    if m is None:
        return True  # e.g. participants.tsv, dataset_description.json, README
    return m.group(1) in allowed_tasks


def _list_remote_files(
    fs: s3fs.S3FileSystem, bucket_uri: str
) -> list[tuple[str, str, int]]:
    """Recursively list every file in the S3 bucket. Returns (uri, rel, size)."""
    bucket_noscheme = bucket_uri.replace("s3://", "", 1).rstrip("/")
    _log(f"[*] Listing {bucket_uri} (this can take a minute)…")
    try:
        detail = fs.find(bucket_noscheme, detail=True)
    except FileNotFoundError:
        _log(f"[!] Bucket does not exist: {bucket_uri}", err=True)
        return []
    except Exception as e:
        _log(f"[!] Failed to list {bucket_uri}: {e}", err=True)
        return []

    files: list[tuple[str, str, int]] = []
    for key, meta in detail.items():
        if meta.get("type") != "file":
            continue
        rel = key[len(bucket_noscheme):].lstrip("/")
        if not rel:
            continue
        size_raw = meta.get("size", meta.get("Size"))
        try:
            size_int = int(size_raw) if size_raw is not None else 0
        except Exception:
            size_int = 0
        files.append((f"s3://{key}", rel, size_int))
    return files


def _check_disk_space(cache_dir: Path, needed_bytes: int) -> None:
    """Raise SystemExit if free space < needed_bytes * DISK_SAFETY_FACTOR."""
    if needed_bytes <= 0:
        return
    required = int(needed_bytes * DISK_SAFETY_FACTOR)
    # Use the nearest existing ancestor (cache_dir may have just been created
    # on a filesystem we want to measure).
    probe = cache_dir
    while not probe.exists():
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    if usage.free < required:
        raise SystemExit(
            f"[!] Not enough free disk space at {probe}:\n"
            f"    need   {_human(required)}  "
            f"(missing {_human(needed_bytes)} + 10% safety)\n"
            f"    have   {_human(usage.free)} free  "
            f"/ {_human(usage.total)} total.\n"
            f"    Free up space, use --cache-dir on a larger volume, or "
            f"apply a stricter --tasks filter."
        )
    _log(
        f"[*] Disk OK: need {_human(required)} (incl. 10% safety), "
        f"have {_human(usage.free)} free."
    )


# ─── Per-file download with timeout + retry ────────────────────────────────
def _sliced_sleep(total_s: float) -> None:
    """Sleep up to total_s seconds in 0.5s slices; abort early on interrupt."""
    if total_s <= 0:
        return
    slice_s = 0.5
    slept = 0.0
    while slept < total_s and not _interrupted:
        time.sleep(min(slice_s, total_s - slept))
        slept += slice_s


def _timeout_for_size(size_bytes: int) -> float:
    size_mb = max(size_bytes, 0) / (1024.0 * 1024.0)
    return float(max(TIMEOUT_MIN_S, math.ceil(size_mb * TIMEOUT_PER_MB_S)))


def _fs_get_with_timeout(
    fs: s3fs.S3FileSystem, s3_uri: str, tmp_path: Path, timeout_s: float
) -> None:
    """
    Run fs.get(s3_uri, tmp_path) with a hard wall-clock timeout.

    If the timeout fires, raises TimeoutError. The worker thread is left as a
    daemon; botocore's own read/connect timeouts will eventually tear it
    down, and its tmp file will be cleaned up on next run (or retry).
    """
    holder: dict[str, BaseException | None] = {"exc": None}
    done_event = threading.Event()

    def _worker() -> None:
        try:
            fs.get(s3_uri, str(tmp_path))
        except BaseException as e:  # noqa: BLE001 - we re-raise on main thread
            holder["exc"] = e
        finally:
            done_event.set()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    finished = done_event.wait(timeout=timeout_s)
    if not finished:
        raise TimeoutError(
            f"download exceeded {timeout_s:.0f}s wall-clock timeout"
        )
    if holder["exc"] is not None:
        raise holder["exc"]


def _download_one(
    s3_uri: str,
    final_path: Path,
    tmp_path: Path,
    expected_size: int,
    max_retries: int,
) -> tuple[str, str | None]:
    """
    Download a single file: S3 → tmp → atomic rename to final_path, with
    retry/backoff and per-attempt timeout.

    Returns (status, err_msg) where status is:
        "skipped"      — already complete
        "downloaded"   — successfully downloaded this run
        "failed"       — all retries exhausted
        "interrupted"  — gave up because the global interrupt flag is set
    """
    # Skip if final already exists with correct size (race-safety net; the
    # main pre-scan should have filtered these out already).
    if final_path.exists():
        try:
            if expected_size and final_path.stat().st_size == expected_size:
                return ("skipped", None)
        except OSError:
            pass
        # Corrupt/mismatched → remove and re-download
        try:
            final_path.unlink()
        except OSError:
            pass

    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    last_err: str | None = None
    timeout_s = _timeout_for_size(expected_size)

    for attempt in range(1, max_retries + 1):
        # Use a unique tmp path per attempt: a zombie worker thread from a
        # previously timed-out attempt could still be trickling bytes into
        # the old path, and we must not race with it on the same file.
        attempt_tmp = tmp_path.with_suffix(tmp_path.suffix + f".att{attempt}")

        if _interrupted:
            # Drop any in-progress tmp and bail.
            if attempt_tmp.exists():
                try:
                    attempt_tmp.unlink()
                except OSError:
                    pass
            return ("interrupted", last_err)

        # Defensive cleanup: attempt_tmp is unique per attempt, so normally
        # won't exist — only trips on leftovers from a prior aborted run.
        if attempt_tmp.exists():
            try:
                attempt_tmp.unlink()
            except OSError:
                pass

        fs = _fs_for_thread()
        try:
            _fs_get_with_timeout(fs, s3_uri, attempt_tmp, timeout_s)
        except KeyboardInterrupt:
            if attempt_tmp.exists():
                try:
                    attempt_tmp.unlink()
                except OSError:
                    pass
            raise
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            if attempt_tmp.exists():
                try:
                    attempt_tmp.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                sleep_s = RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
                _log(
                    f"    [attempt {attempt + 1}/{max_retries}] {s3_uri}: "
                    f"{last_err} — sleeping {sleep_s:.0f}s",
                    err=True,
                )
                _sliced_sleep(sleep_s)
                continue
            # Final attempt failed
            return ("failed", last_err)

        # Verify size
        try:
            actual = attempt_tmp.stat().st_size
        except OSError as e:
            last_err = f"stat failed: {e}"
            if attempt < max_retries:
                _sliced_sleep(RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1)))
                continue
            return ("failed", last_err)

        if expected_size and actual != expected_size:
            last_err = (
                f"size mismatch: expected {expected_size}, got {actual}"
            )
            try:
                attempt_tmp.unlink()
            except OSError:
                pass
            if attempt < max_retries:
                sleep_s = RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
                _log(
                    f"    [attempt {attempt + 1}/{max_retries}] {s3_uri}: "
                    f"{last_err} — sleeping {sleep_s:.0f}s",
                    err=True,
                )
                _sliced_sleep(sleep_s)
                continue
            return ("failed", last_err)

        # Atomic move into the final tree
        try:
            os.replace(attempt_tmp, final_path)
        except OSError as e:
            last_err = f"atomic move failed: {e}"
            if attempt_tmp.exists():
                try:
                    attempt_tmp.unlink()
                except OSError:
                    pass
            if attempt < max_retries:
                _sliced_sleep(RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1)))
                continue
            return ("failed", last_err)

        return ("downloaded", None)

    return ("failed", last_err)


# ─── Core release driver ────────────────────────────────────────────────────
def download_release(
    release: str,
    cache_dir: Path,
    *,
    dry_run: bool,
    allowed_tasks: set[str] | None,
    workers: int,
    max_retries: int,
) -> tuple[int, int, int, int]:
    """
    Download one full release.

    Returns (n_skipped, n_downloaded, n_failed, bytes_downloaded).
    """
    if release not in VALID_RELEASES:
        _log(
            f"[!] Unknown release {release}; valid: {sorted(VALID_RELEASES)}",
            err=True,
        )
        return (0, 0, 0, 0)

    bucket_uri = f"{S3_BUCKET_ROOT}/{release}_L100_bdf"
    # Match eegdash's local folder convention: EEG2025r{n}
    dataset_folder = f"EEG2025r{release[1:]}"
    final_root = cache_dir / dataset_folder
    tmp_root = final_root / ".tmp"

    _log(f"\n{'=' * 70}")
    _log(f"Release : {release}  ({bucket_uri})")
    _log(f"Final   : {final_root}")
    _log(f"Tmp     : {tmp_root}")
    _log(f"Filter  : {sorted(allowed_tasks) if allowed_tasks else '(none)'}")
    _log(f"Workers : {workers}    Retries/file: {max_retries}")
    _log(f"{'=' * 70}")

    final_root.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)

    stale = _cleanup_stale_tmp(tmp_root)
    if stale:
        _log(f"[*] Removed {stale} stale tmp file(s) from previous run.")

    fs = _make_fs()
    all_files = _list_remote_files(fs, bucket_uri)
    if not all_files:
        _log(f"[!] No files found for release {release}.", err=True)
        return (0, 0, 0, 0)

    # Apply task filter
    if allowed_tasks is not None:
        before = len(all_files)
        files = [
            (uri, rel, sz)
            for (uri, rel, sz) in all_files
            if _file_passes_task_filter(rel, allowed_tasks)
        ]
        _log(
            f"[*] Task filter: {len(files)}/{before} files match "
            f"(tasks={sorted(allowed_tasks)} + global metadata)."
        )
    else:
        files = all_files

    total_remote_bytes = sum(sz for _, _, sz in files)
    _log(f"[*] After filter: {len(files)} files, total {_human(total_remote_bytes)}")

    # Pre-scan: how many already complete locally?
    already_ok = 0
    missing_bytes = 0
    work: list[tuple[str, str, int]] = []
    for s3_uri, rel, size in files:
        final_path = final_root / rel
        try:
            local_size = final_path.stat().st_size if final_path.exists() else -1
        except OSError:
            local_size = -1
        if size > 0 and local_size == size:
            already_ok += 1
        else:
            missing_bytes += size
            work.append((s3_uri, rel, size))

    _log(
        f"[*] Already complete: {already_ok}/{len(files)} "
        f"({_human(total_remote_bytes - missing_bytes)})"
    )
    _log(
        f"[*] To download:     {len(work)}/{len(files)} "
        f"({_human(missing_bytes)})"
    )

    if dry_run:
        _log("[*] Dry-run mode — nothing will be downloaded.")
        return (already_ok, 0, 0, 0)

    if not work:
        _log("[✓] Release already fully downloaded. Nothing to do.")
        return (already_ok, 0, 0, 0)

    # Disk-space pre-flight
    _check_disk_space(cache_dir, missing_bytes)

    # ── Parallel download loop ────────────────────────────────────────
    n_skipped = already_ok
    n_downloaded = 0
    n_failed = 0
    bytes_done = 0
    t0 = time.time()
    total_work = len(work)
    completed = 0

    # Progress reporting: protected by lock since workers and main share it.
    counters_lock = threading.Lock()

    def _maybe_progress() -> None:
        # Caller must already hold counters_lock.
        if completed % 25 == 0 or completed == total_work:
            elapsed = max(time.time() - t0, 1e-6)
            rate = bytes_done / elapsed
            remaining_bytes = max(missing_bytes - bytes_done, 0)
            eta_s = remaining_bytes / rate if rate > 0 else float("inf")
            _log(
                f"    progress: {completed}/{total_work} files  "
                f"{_human(bytes_done)}/{_human(missing_bytes)} "
                f"@ {_human(int(rate))}/s  eta {eta_s / 60:5.1f} min"
            )

    try:
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="dl"
        ) as ex:
            futures = {}
            for s3_uri, rel, size in work:
                final_path = final_root / rel
                tmp_path = tmp_root / rel
                fut = ex.submit(
                    _download_one,
                    s3_uri,
                    final_path,
                    tmp_path,
                    size,
                    max_retries,
                )
                futures[fut] = (s3_uri, rel, size)

            for fut in as_completed(futures):
                # Cancel any not-yet-started work if user interrupted.
                if _interrupted:
                    for f in futures:
                        if not f.done() and not f.running():
                            f.cancel()

                s3_uri, rel, size = futures[fut]
                try:
                    status, err = fut.result()
                except CancelledError:
                    # Cancelled above because of interrupt; not a failure.
                    status, err = "interrupted", None
                except Exception as e:  # noqa: BLE001
                    status, err = "failed", f"{type(e).__name__}: {e}"

                with counters_lock:
                    completed += 1
                    short = rel if len(rel) < 68 else "…" + rel[-67:]
                    tag = f"[{completed:5d}/{total_work}]"
                    if status == "downloaded":
                        n_downloaded += 1
                        bytes_done += size
                        _log(f"{tag} ✓ {_human(size):>10s}  {short}")
                    elif status == "skipped":
                        n_skipped += 1
                        _log(f"{tag} = {_human(size):>10s}  {short}  (already complete)")
                    elif status == "interrupted":
                        # Counted separately from failed; don't spam.
                        pass
                    else:  # "failed"
                        n_failed += 1
                        _log(
                            f"{tag} ✗ {_human(size):>10s}  {short}  "
                            f"FAILED after {max_retries} attempts: {err}",
                            err=True,
                        )
                    _maybe_progress()
    except KeyboardInterrupt:
        _log("[!] Interrupted — cancelling remaining work.", err=True)
        # The ThreadPoolExecutor context manager will wait for running
        # workers to finish; tmp files they created will be removed on
        # next-run startup via _cleanup_stale_tmp.

    # Final cleanup of tmp tree — only needed if something went wrong.
    # On a clean successful run every attempt file has already been moved
    # atomically into the final tree, so tmp_root is empty.
    if n_failed > 0 or _interrupted:
        _cleanup_stale_tmp(tmp_root)

    elapsed = time.time() - t0
    _log(
        f"\n[release {release} summary] "
        f"skipped={n_skipped}  downloaded={n_downloaded}  failed={n_failed}  "
        f"bytes={_human(bytes_done)}  elapsed={elapsed / 60:.1f} min"
    )
    return (n_skipped, n_downloaded, n_failed, bytes_done)


# ─── Entry point ────────────────────────────────────────────────────────────
def _parse_tasks_arg(
    tasks_csv: str | None, all_tasks: bool
) -> set[str] | None:
    if all_tasks:
        return None
    if tasks_csv is None:
        return set(CHALLENGE1_TASKS)
    parts = [p.strip() for p in tasks_csv.split(",") if p.strip()]
    if not parts:
        return set(CHALLENGE1_TASKS)
    return set(parts)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Safe, resumable, atomic, parallel downloader for EEG 2025 full "
            "releases. Includes retry/backoff, disk pre-flight, task filter, "
            "and per-file timeouts."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "releases",
        nargs="*",
        default=["R5"],
        help="Releases to download (default: R5). Valid: R1..R11",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data"),
        help="Local cache root (default: ./data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded; don't download.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated task names to keep (e.g. "
            "'contrastChangeDetection,RestingState'). Files without any "
            "task-<name> token (global metadata) are always kept. "
            f"Default: {','.join(CHALLENGE1_TASKS)} (Challenge 1)."
        ),
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Disable the task filter; download every file in the bucket.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel download workers (default: {DEFAULT_WORKERS}).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help=(
            f"Max attempts per file (default: {DEFAULT_MAX_RETRIES}, "
            "backoff 1s/2s/4s/8s/16s)."
        ),
    )
    args = parser.parse_args()

    # Validation
    unknown = [r for r in args.releases if r not in VALID_RELEASES]
    if unknown:
        _log(
            f"[!] Unknown release(s): {unknown}. "
            f"Valid: {sorted(VALID_RELEASES)}",
            err=True,
        )
        return 2
    if args.workers < 1:
        _log("[!] --workers must be >= 1", err=True)
        return 2
    if args.max_retries < 1:
        _log("[!] --max-retries must be >= 1", err=True)
        return 2

    allowed_tasks = _parse_tasks_arg(args.tasks, args.all_tasks)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    _install_signal_handlers()

    totals = [0, 0, 0, 0]  # skipped, downloaded, failed, bytes
    for release in args.releases:
        if _interrupted:
            break
        try:
            result = download_release(
                release,
                args.cache_dir,
                dry_run=args.dry_run,
                allowed_tasks=allowed_tasks,
                workers=args.workers,
                max_retries=args.max_retries,
            )
        except SystemExit:
            raise
        except KeyboardInterrupt:
            _log("[!] Interrupted.", err=True)
            break
        for i, v in enumerate(result):
            totals[i] += v

    _log(f"\n{'=' * 70}")
    _log(
        f"GRAND TOTAL  skipped={totals[0]}  downloaded={totals[1]}  "
        f"failed={totals[2]}  bytes={_human(totals[3])}"
    )
    _log(f"{'=' * 70}")

    return 1 if totals[2] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
