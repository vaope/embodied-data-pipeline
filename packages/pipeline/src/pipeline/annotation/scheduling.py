"""Bounded preparation/task pools and a run-wide, priority HTTP admission gate.

Only the network call holds an API slot. Image decoding, budget waits, cache hits
and retry backoff do not. Threads are used to preserve pinned synchronous callers.
"""
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
import heapq
import json
from pathlib import Path
import sys
import threading
import time

from .core import write

_CURRENT = None
_RUN_LOCK = threading.Lock()


def current():
    return _CURRENT


class Runtime:
    def __init__(self, out, config, stop):
        c = config.get('execution', {})
        self.limit = int(c.get('api_concurrency', 48))
        self.task_workers = int(c.get('request_workers', self.limit * 2))
        self.decode_workers = int(c.get('decode_workers', 4))
        if min(self.limit, self.task_workers, self.decode_workers) < 1:
            raise ValueError('SCHEDULER_WORKERS_INVALID')
        if self.task_workers < self.limit:
            raise ValueError('REQUEST_WORKERS_BELOW_API_CONCURRENCY')
        self.out, self.stop = Path(out), stop
        self.out.mkdir(parents=True, exist_ok=True)
        self.cv = threading.Condition()
        self.pending, self.sequence, self.active, self.peak = [], 0, 0, 0
        self.started = self.last = time.monotonic()
        self.busy_slot_s = self.full_s = 0.
        self.completed = self.network_errors = 0
        self.successes = 0
        self.decode_active = self.decode_peak = 0
        self.decode_wait_s = 0.
        self.decode_slots = threading.BoundedSemaphore(self.decode_workers)
        # Bounds both queued tasks and prepared image payloads; never queue an entire batch's images.
        self.task_slots = threading.BoundedSemaphore(self.task_workers)
        self.tasks = ThreadPoolExecutor(max_workers=self.task_workers, thread_name_prefix='av-request')
        self.log_lock = threading.Lock()
        self.invocation = time.time_ns()
        self.last_snapshot = 0.

    def event(self, kind, **fields):
        row = dict(invocation=self.invocation, t=time.time(), kind=kind, **fields)
        with self.log_lock:
            with (self.out / 'scheduler_events.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(row) + '\n')
            if time.monotonic() - self.last_snapshot >= 1.:
                write(self.out / 'scheduler_summary.json', self.summary())
                self.last_snapshot = time.monotonic()

    def _integrate(self):
        now = time.monotonic()
        dt = now - self.last
        self.busy_slot_s += self.active * dt
        self.full_s += dt if self.active == self.limit else 0.
        self.last = now

    def _check_stop(self):
        if self.stop.is_set():
            from .backends import BudgetExceeded
            raise BudgetExceeded('RUN_STOPPED: scheduler stopping')

    @contextmanager
    def network(self, directory):
        stage = Path(directory).name
        priority = {'task': 0, 's2': 0, 'caption-a': 1}.get(stage, 1)
        queued = time.monotonic()
        with self.cv:
            ticket = (priority, self.sequence)
            self.sequence += 1
            heapq.heappush(self.pending, ticket)
            try:
                while self.active >= self.limit or self.pending[0] != ticket:
                    self._check_stop()
                    self.cv.wait(.1)
                self._check_stop()
            except BaseException:
                self.pending.remove(ticket)
                heapq.heapify(self.pending)
                self.cv.notify_all()
                raise
            heapq.heappop(self.pending)
            self._integrate()
            self.active += 1
            self.peak = max(self.peak, self.active)
            active, waiting = self.active, len(self.pending)
            self.cv.notify_all()
        start = time.monotonic()
        sent_at = time.time()
        error = None
        info = {}
        try:
            yield info
        except BaseException as exc:
            error = type(exc).__name__
            raise
        finally:
            end = time.monotonic()
            with self.cv:
                self._integrate()
                self.active -= 1
                self.completed += 1
                self.network_errors += bool(error or (info.get('status_code') or 0) >= 400)
                self.successes += info.get('status_code') == 200
                self.cv.notify_all()
            self.event('request', directory=str(directory), stage=stage,
                       queue_s=start - queued, network_s=end - start,
                       sent_at=sent_at,
                       status_code=info.get('status_code'),
                       active_at_start=active, queued_at_start=waiting, error=error)

    @contextmanager
    def decoding(self):
        start = time.monotonic()
        while not self.decode_slots.acquire(timeout=.1):
            self._check_stop()
        entered = time.monotonic()
        try:
            self._check_stop()
            with self.cv:
                self.decode_wait_s += time.monotonic() - start
                self.decode_active += 1
                self.decode_peak = max(self.decode_peak, self.decode_active)
            try:
                yield
            finally:
                with self.cv:
                    self.decode_active -= 1
        finally:
            self.decode_slots.release()
            self.event('decode', wait_s=entered - start, work_s=time.monotonic() - entered)

    def map(self, function, items):
        """Submit bounded independent work, collect in input order, drain on failure.

        The caller must be an episode coordinator, never a request-pool worker.
        """
        futures = []
        stream = getattr(getattr(sys.stdout, 'local', None), 'stream', None)

        def call(item):
            self._check_stop()
            if stream is None:
                return function(item)
            from .runner import thread_stdout
            with thread_stdout(stream):
                return function(item)

        try:
            for item in items:
                while not self.task_slots.acquire(timeout=.1):
                    self._check_stop()
                try:
                    self._check_stop()
                    future = self.tasks.submit(call, item)
                except BaseException:
                    self.task_slots.release()
                    raise
                future.add_done_callback(lambda _: self.task_slots.release())
                futures.append(future)
            return [f.result() for f in futures]
        finally:
            # No audit context/log may close while its children are still running.
            wait(futures)

    def summary(self):
        with self.cv:
            self._integrate()
            wall = max(time.monotonic() - self.started, 1e-9)
            return dict(api_limit=self.limit, request_workers=self.task_workers,
                        decode_workers=self.decode_workers, wall_s=wall,
                        active=self.active, queued=len(self.pending), peak_inflight=self.peak,
                        network_attempts=self.completed, network_errors=self.network_errors,
                        successful_responses=self.successes, successful_requests_per_min=self.successes * 60 / wall,
                        average_inflight=self.busy_slot_s / wall,
                        slot_utilization=self.busy_slot_s / (wall * self.limit),
                        time_at_full_fraction=self.full_s / wall,
                        decode_peak=self.decode_peak, decode_wait_s=self.decode_wait_s,
                        scope='this invocation; includes startup/tail; cache hits excluded')

    def close(self):
        self.tasks.shutdown(wait=True)
        write(self.out / 'scheduler_summary.json', self.summary())


@contextmanager
def use_runtime(out, config, stop):
    global _CURRENT
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError('ANNOTATION_RUN_ALREADY_ACTIVE')
    runtime = None
    try:
        runtime = Runtime(out, config, stop)
        _CURRENT = runtime
        yield runtime
    finally:
        try:
            if runtime:
                runtime.close()
        finally:
            _CURRENT = None
            _RUN_LOCK.release()
