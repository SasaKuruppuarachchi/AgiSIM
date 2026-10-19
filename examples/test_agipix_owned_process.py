"""Process hygiene tests with fake /proc records; no subprocess is launched."""
import os
import signal
import subprocess

import pytest

import agipix_owned_process as owned


PGID = 900001


def stat_record(root, pid, *, group=PGID, session=PGID, state="S", start=100,
                parent=None, name="python ) (worker"):
    fields = [state, str(os.getpid() if parent is None else parent), str(group), str(session)]
    fields += ["0"] * 15 + [str(start)]
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    (directory / "stat").write_text(f"{pid} ({name}) " + " ".join(fields) + "\n")


class Process:
    pid = PGID

    def __init__(self, proc):
        self.proc = proc
        self.reaped = False

    def poll(self):
        leader = owned._read_stat(self.proc / str(self.pid) / "stat")
        return -15 if leader is None or leader.state == "Z" else None

    def wait(self, timeout):
        code = self.poll()
        if code is None:
            raise subprocess.TimeoutExpired("fake owned wrapper", timeout)
        self.reaped = True
        return code


@pytest.fixture
def rig(tmp_path, monkeypatch):
    stat_record(tmp_path, PGID)
    process = Process(tmp_path)
    clock = [0.0]
    monkeypatch.setattr(owned.time, "monotonic", lambda: clock[0])
    def sleep(duration):
        clock[0] += duration
    monkeypatch.setattr(owned.time, "sleep", sleep)
    return tmp_path, process, clock


def test_stat_parser_handles_spaces_and_parentheses(tmp_path):
    stat_record(tmp_path, PGID, state="R", start=98765)
    info = owned._read_stat(tmp_path / str(PGID) / "stat")
    assert (info.pid, info.group, info.session, info.start_ticks, info.state) == (
        PGID, PGID, PGID, 98765, "R")


def test_wrapper_exit_does_not_hide_live_descendant(rig, monkeypatch):
    proc, process, clock = rig
    stat_record(proc, PGID + 1, start=101, parent=PGID)
    # An unrelated active process must never influence cleanup or its target.
    stat_record(proc, PGID + 2, group=12, session=12, start=20)
    guard = owned.OwnedProcessGroup(process, proc_root=proc)
    seen = []
    def send(group, sig):
        assert group == PGID
        seen.append(sig)
        if sig == signal.SIGTERM:
            stat_record(proc, PGID, state="Z")  # shell exits, Isaac remains
        elif sig == signal.SIGKILL:
            stat_record(proc, PGID + 1, state="Z", start=101, parent=PGID)
    monkeypatch.setattr(owned.os, "killpg", send)
    assert guard.stop() == -15
    assert seen == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert clock[0] == pytest.approx(35.)
    assert process.reaped
    assert guard.members() == []
    assert owned._read_stat(proc / str(PGID + 2) / "stat").state == "S"


def test_wrapper_already_exited_before_stop_still_cleans_child(rig, monkeypatch):
    proc, process, _ = rig
    guard = owned.OwnedProcessGroup(process, proc_root=proc)
    stat_record(proc, PGID, state="Z")
    stat_record(proc, PGID + 1, start=101, parent=PGID)
    signals = []
    def send(group, sig):
        signals.append((group, sig))
        stat_record(proc, PGID + 1, state="Z", start=101, parent=PGID)
    monkeypatch.setattr(owned.os, "killpg", send)
    assert guard.stop() == -15
    assert signals == [(PGID, signal.SIGINT)]
    assert process.reaped


def test_zombies_need_no_signals_and_closed_guard_never_kills_reused_id(rig, monkeypatch):
    proc, process, _ = rig
    guard = owned.OwnedProcessGroup(process, proc_root=proc)
    stat_record(proc, PGID, state="Z")
    stat_record(proc, PGID + 1, state="Z", start=101)
    signals = []
    monkeypatch.setattr(owned.os, "killpg", lambda *args: signals.append(args))
    assert guard.stop() == -15
    assert process.reaped and not signals
    stat_record(proc, PGID, start=9999)
    assert guard.members() == []  # closed permanently, no recycled group touched


def test_identity_reuse_is_refused_without_signalling(rig, monkeypatch):
    proc, process, _ = rig
    guard = owned.OwnedProcessGroup(process, proc_root=proc)
    stat_record(proc, PGID, start=9999)
    signals = []
    monkeypatch.setattr(owned.os, "killpg", lambda *args: signals.append(args))
    with pytest.raises(RuntimeError, match="reused or changed"):
        guard.stop()
    assert not signals


def test_survivors_after_kill_are_reported_and_wrapper_reaped(rig, monkeypatch):
    proc, process, clock = rig
    guard = owned.OwnedProcessGroup(process, proc_root=proc)
    stat_record(proc, PGID, state="Z")
    stat_record(proc, PGID + 1, state="D", start=101)
    seen = []
    monkeypatch.setattr(owned.os, "killpg", lambda group, sig: seen.append(sig))
    with pytest.raises(RuntimeError, match=f"{PGID + 1}:D"):
        guard.stop()
    assert seen == [signal.SIGINT, signal.SIGTERM, signal.SIGKILL]
    assert clock[0] == pytest.approx(40.)
    assert process.reaped


def test_rejects_process_without_new_session(rig):
    proc, process, _ = rig
    stat_record(proc, PGID, group=os.getpgrp(), session=12)
    with pytest.raises(RuntimeError, match="start_new_session"):
        owned.OwnedProcessGroup(process, proc_root=proc)


def test_esrch_race_is_harmless_and_wrapper_reaped(rig, monkeypatch):
    proc, process, _ = rig
    guard = owned.OwnedProcessGroup(process, proc_root=proc)
    def send(group, sig):
        stat_record(proc, PGID, state="Z")
        raise ProcessLookupError()
    monkeypatch.setattr(owned.os, "killpg", send)
    assert guard.stop() == -15
    assert process.reaped
