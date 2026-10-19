"""Bounded cleanup of a subprocess launched in its own Linux session.

Create the guard immediately after ``Popen(..., start_new_session=True)`` and
call ``guard.stop()`` in ``finally``, even when ``process.wait()`` succeeds.
The shell wrapper may exit while Isaac or PX4 remains in its process group.
Only this new group is signalled; names, environment matches and unrelated
process IDs are never used as kill targets. Detached descendants which create
their own session/group are deliberately outside this helper's authority.
"""
from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import time


@dataclass(frozen=True)
class GroupMember:
    pid: int
    state: str
    parent: int
    group: int
    session: int
    start_ticks: int


def _read_stat(path):
    """Parse proc stat without treating spaces/parentheses in comm as fields."""
    try:
        raw = path.read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    end = raw.rfind(")")
    fields = raw[end + 2:].split()
    if end < 0 or len(fields) < 20:
        raise RuntimeError(f"Invalid process identity record: {path}")
    return GroupMember(int(raw.split("(", 1)[0]), fields[0], int(fields[1]),
                       int(fields[2]), int(fields[3]), int(fields[19]))


class OwnedProcessGroup:
    """Track the session/group identity and wait for all live members to exit.

    Linux keeps a process-group ID allocated while that group has members.
    Checking the leader's start time also rejects observable PID reuse. Once
    an empty group is observed the guard is permanently closed and never
    signals that numeric ID again. Zombies are already dead and do not prevent
    completion; the direct Popen child is still waited/reaped separately.

    ``proc_root`` exists for filesystem-only tests. Production uses /proc in
    the same PID namespace as the Popen child.
    """

    def __init__(self, process, *, proc_root=Path("/proc")):
        self.process = process
        self.pid = int(process.pid)
        self.proc_root = Path(proc_root)
        leader = _read_stat(self.proc_root / str(self.pid) / "stat")
        if leader is None:
            raise RuntimeError("Capture the owned process group immediately after Popen")
        if (self.pid <= 1 or leader.parent != os.getpid()
                or leader.group != self.pid or leader.session != self.pid
                or self.pid == os.getpgrp()):
            raise RuntimeError("Child must be launched by this process with start_new_session=True")
        self.start_ticks = leader.start_ticks
        self._closed = False
        self.signals_sent = []

    def members(self):
        """Return live owned-group members, with conservative identity checks."""
        if self._closed:
            return []
        leader = _read_stat(self.proc_root / str(self.pid) / "stat")
        if leader is not None and (leader.start_ticks != self.start_ticks
                                   or leader.group != self.pid or leader.session != self.pid):
            raise RuntimeError(f"Refusing reused or changed process-group identity {self.pid}")
        members = []
        for directory in self.proc_root.iterdir():
            if not directory.name.isdecimal():
                continue
            member = _read_stat(directory / "stat")
            if member is None or member.group != self.pid:
                continue
            if member.session != self.pid or member.start_ticks < self.start_ticks:
                raise RuntimeError(f"Refusing unowned member in process group {self.pid}")
            if member.state not in ("Z", "X", "x"):
                members.append(member)
        if not members:
            self._closed = True
        return sorted(members, key=lambda member: member.pid)

    def _wait_empty(self, timeout, poll_interval):
        deadline = time.monotonic() + timeout
        while True:
            # Reap the wrapper promptly, but never use its exit as the group's
            # completion condition: descendants can still be running.
            self.process.poll()
            if not self.members():
                return True
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            time.sleep(min(poll_interval, remaining))

    def stop(self, *, sigint_timeout=25.0, sigterm_timeout=10.0,
             sigkill_timeout=5.0, poll_interval=.1):
        """Interrupt, terminate, then kill remaining group members; reap wrapper.

        Returns the wrapper's exit code. Raises when ownership cannot be
        established or live members survive the final deadline, rather than
        silently leaking a simulator. All waits have finite upper bounds.
        """
        import math
        durations = (sigint_timeout, sigterm_timeout, sigkill_timeout, poll_interval)
        if any(not math.isfinite(t) or t < 0 for t in durations) or poll_interval == 0:
            raise ValueError("Cleanup timeouts must be finite/nonnegative; poll interval positive")
        remaining = self.members()
        for sig, timeout in ((signal.SIGINT, sigint_timeout),
                             (signal.SIGTERM, sigterm_timeout),
                             (signal.SIGKILL, sigkill_timeout)):
            if not remaining:
                break
            # Refresh ownership immediately before signalling the group. If it
            # disappeared between this check and killpg, ESRCH is harmless.
            remaining = self.members()
            if not remaining:
                break
            try:
                os.killpg(self.pid, sig)
            except ProcessLookupError:
                pass
            else:
                self.signals_sent.append(int(sig))
            if self._wait_empty(timeout, poll_interval):
                remaining = []
                break
            remaining = self.members()
        # wait() reaps only the direct child we launched. Grandchild zombies
        # belong to their reaper and are neither live nor killable by us.
        try:
            code = self.process.wait(timeout=.2)
        except subprocess.TimeoutExpired:
            code = None
        if remaining:
            details = ", ".join(f"{member.pid}:{member.state}" for member in remaining)
            raise RuntimeError(f"Owned group {self.pid} still has live members after SIGKILL: {details}")
        if code is None:
            raise RuntimeError(f"Owned wrapper {self.pid} could not be reaped")
        return code
