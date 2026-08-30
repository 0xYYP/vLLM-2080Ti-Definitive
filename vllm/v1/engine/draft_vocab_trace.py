# -*- coding: utf-8 -*-
"""draft-vocab 真实负载词频统计补丁（本仓库维护扩展）。

用途
----
用户正常使用一周，服务端自动收集**最终采样输出 token id** 的频率分布，
一周后喂给 prepare/build_draft_vocab.py 生成与真实工作负载匹配的
MTP draft 受限词表（替换当前"113 条中文 prompt 模拟采样"的 16384 表）。

启用
----
- 起服务前设 ``DRAFT_VOCAB_TRACE=/path/to/draft_vocab_freq.json``；
  未设置则不启用（零开销、完全向后兼容）。
- 每条请求的 target 输出 token（经 MTP verify 后的最终采样，含思考链与正文）
  都会累加到进程级 Counter；定期 flush（默认每 300s / 2000 请求，
  可用 ``DRAFT_VOCAB_TRACE_FLUSH_SECS`` / ``DRAFT_VOCAB_TRACE_FLUSH_REQS`` 调整）。
- 输出 JSON：``{"token_id": count}``（升序 key），进程退出时兜底 flush。
  重启服务会在相同路径上**续写累积**（load 旧文件后继续累加）。

口径说明
--------
- 采集点：``vllm/v1/engine/output_processor.py`` 中
  ``req_state.detokenizer.update(new_token_ids, ...)`` 的入参 ——
  即每个 decode 步真正输出的 token，与 draft-vocab 表服务目标一致；
  被拒绝的 MTP 草稿 token 不计数（正确）。
- target 路径不受当前 draft 表影响（拒绝采样保证精确），
  因此使用期间可继续启用旧 16384 表服务，统计仍是纯真实分布。
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time


class DraftVocabTrace:
    """进程级 token id 频率采集器（模块单例）。"""

    def __init__(self, path: str):
        self.path = path
        self._counter: collections.Counter = collections.Counter()
        self._lock = threading.Lock()
        self._last_flush_ts = time.monotonic()
        self._since_flush = 0
        try:
            self._flush_secs = max(10, int(os.environ.get("DRAFT_VOCAB_TRACE_FLUSH_SECS", "300")))
        except ValueError:
            self._flush_secs = 300
        try:
            self._flush_reqs = max(1, int(os.environ.get("DRAFT_VOCAB_TRACE_FLUSH_REQS", "2000")))
        except ValueError:
            self._flush_reqs = 2000
        self._load_existing()
        self._start_flush_thread()

    def _start_flush_thread(self) -> None:
        """后台 daemon 线程定期 flush，不依赖 record 触发。

        record 仅在 decode 步被调用；请求结束到下一个请求到来期间没有
        record，若 flush 只挂在 record 上，counter 会滞留内存不落盘。
        线程以 flush_secs/2 为间隔（至少 5s）定期写盘，进程退出由
        atexit flush 兜底。
        """
        def _loop():
            interval = max(5.0, self._flush_secs / 2)
            while True:
                time.sleep(interval)
                try:
                    self._flush()
                except Exception:
                    pass

        t = threading.Thread(target=_loop, name="draft-vocab-trace-flusher",
                             daemon=True)
        t.start()

    def _load_existing(self) -> None:
        """重启续写：把已有 JSON 并入计数。"""
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                for k, v in data.items():
                    try:
                        self._counter[int(k)] = int(v)
                    except (ValueError, TypeError):
                        continue
                logger_debug("draft-vocab trace: merged %d existing entries", len(data))
        except FileNotFoundError:
            pass
        except (OSError, json.JSONDecodeError) as exc:
            logger_debug("draft-vocab trace: ignore unreadable %s (%s)", self.path, exc)

    def record(self, token_ids: list[int]) -> None:
        if not token_ids:
            return
        # 锁内只做计数与 flush 判定；磁盘写出由 _flush（锁外）完成，
        # 避免非重入锁自锁（错误版本曾在锁内调用 _flush_locked）。
        need_flush = False
        with self._lock:
            for tid in token_ids:
                self._counter[tid] += 1
            self._since_flush += 1
            if (self._since_flush >= self._flush_reqs
                    or time.monotonic() - self._last_flush_ts >= self._flush_secs):
                self._since_flush = 0
                self._last_flush_ts = time.monotonic()
                need_flush = True
        if need_flush:
            self._flush()

    def _snapshot(self) -> dict:
        """锁内取计数快照（供 _flush 在锁外写出）。"""
        with self._lock:
            return {str(k): v for k, v in self._counter.items()}

    def _flush(self) -> None:
        if not self._counter:
            return
        path = self.path
        tmp = path + ".tmp"
        payload = self._snapshot()
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, path)
        except OSError as exc:
            logger_debug("draft-vocab trace: flush failed (%s)", exc)

    def flush(self) -> None:
        self._flush()


_TRACE: DraftVocabTrace | None = None
_TRACE_ATOMIC = threading.Lock()


def _logger_debug(msg, *args):
    try:
        from vllm.logger import init_logger
        init_logger(__name__).debug(msg, *args)
    except Exception:
        pass


logger_debug = _logger_debug


def init_draft_vocab_trace() -> DraftVocabTrace | None:
    """按 env 初始化模块单例（幂等）。在服务启动早期调用一次。"""
    global _TRACE
    if _TRACE is not None:
        return _TRACE
    path = os.environ.get("DRAFT_VOCAB_TRACE")
    if not path:
        return None
    with _TRACE_ATOMIC:
        if _TRACE is None:
            _TRACE = DraftVocabTrace(path)
    return _TRACE


def record_draft_vocab_trace(token_ids) -> None:
    """记一条输出 token 序列（线程安全，开销 O(n) 哈希累加）。"""
    trace = _TRACE
    if trace is None:
        return
    trace.record(token_ids)


def flush_draft_vocab_trace() -> None:
    trace = _TRACE
    if trace is not None:
        trace.flush()