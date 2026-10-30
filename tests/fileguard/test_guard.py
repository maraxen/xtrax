"""xtrax.fileguard: quota parsing, Lustre refusal, group quota, slack, persistence, refusals.

Parsing and refusal cases are ported from alynxr ``tests/test_storage_quota.py`` (the code this
package lifts); the group-quota, concurrency-slack and emitter cases are new (spec 261030 §3.4).
"""

import json
import os
import stat
from pathlib import Path

import pytest

from xtrax import fileguard
from xtrax.fileguard import (
    LIMIT_UNKNOWN_EVENT,
    READING_EVENT,
    REFUSAL_EVENT,
    FileCapGuard,
    QuotaRefused,
    QuotaUnreadable,
    check_concurrency_slack,
    describe_location,
    fs_type,
    parse_df_inodes,
    parse_lfs_quota,
    read_quota,
    read_quotas,
)

LFS_ONE_LINE = """\
Disk quotas for usr marielle (uid 12345):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
/orcd/scratch/orcd/011  461234567       0 1073741824       -  461023  1000000 1000000       -
"""

LFS_TWO_LINE = """\
Disk quotas for usr marielle (uid 12345):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
/orcd/scratch/orcd/011
                461234567       0 1073741824       -  461023  1000000 1000000       -
"""

LFS_OVER = LFS_ONE_LINE.replace("461023  1000000 1000000", "1000001*  1000000 1000000")
LFS_UNLIMITED = LFS_ONE_LINE.replace("461023  1000000 1000000", "461023        0       0")
LFS_SOFT_ONLY = LFS_ONE_LINE.replace("461023  1000000 1000000", "461023  500000       0")
LFS_NEAR = LFS_ONE_LINE.replace("461023  1000000 1000000", "799990  1000000 1000000")
LFS_GROUP_FULL = """\
Disk quotas for grp so3 (gid 5000):
     Filesystem  kbytes   quota   limit   grace   files   quota   limit   grace
/orcd/scratch/orcd/011  1  0  0  -  4999990  5000000 5000000  -
"""

DF_OUT_CLEAN = """\
Filesystem       Inodes  IUsed   IFree IUse% Mounted on
/dev/sdb1       6553600 123456 6430144    2% /data
"""

DF_UNLIMITED = """\
Filesystem      Inodes IUsed IFree IUse% Mounted on
tmpfs                -     0     0     -  /dev/shm
"""


def _runner(
    fstype, *, lfs=(0, LFS_ONE_LINE), lfs_group=None, df=(0, DF_OUT_CLEAN), stat_rc=0, calls=None
):
    """A fake runner: ``stat -f`` answers ``fstype``; lfs (user/group) and df answer as given.

    ``lfs`` may be an exception class to raise (e.g. FileNotFoundError, as a missing binary does).
    ``lfs_group`` answers ``lfs quota -g``; it defaults to ``lfs``.
    """

    def run(argv):
        if calls is not None:
            calls.append(argv)
        if argv[0] == "stat":
            return stat_rc, f"{fstype}\n"
        if argv[0] == "lfs":
            answer = lfs_group if (argv[2] == "-g" and lfs_group is not None) else lfs
            if isinstance(answer, type) and issubclass(answer, BaseException):
                raise answer("lfs")
            return answer
        if argv[0] == "df":
            return df
        raise AssertionError(f"unexpected argv {argv!r}")

    return run


@pytest.fixture
def events():
    emitted: list[tuple[str, dict]] = []
    return emitted


def _emit_into(emitted):
    return lambda name, fields: emitted.append((name, fields))


def _named(emitted, name):
    return [kw for n, kw in emitted if n == name]


def _plain_dir(tmp_path: Path) -> Path:
    d = tmp_path / "out"
    d.mkdir()
    os.chmod(d, 0o755)  # no setgid
    return d


def _guard(tmp_path, runner, emitted=None, **kw):
    return FileCapGuard(
        _plain_dir(tmp_path),
        runner=runner,
        user="marielle",
        emit=_emit_into(emitted if emitted is not None else []),
        **kw,
    )


# --- parsing -----------------------------------------------------------------


def test_parse_lfs_single_line_layout():
    r = parse_lfs_quota(LFS_ONE_LINE, "/s")
    assert (r.files_used, r.files_limit, r.source) == (461023, 1000000, "lfs")


def test_parse_lfs_two_line_layout_matches_single_line():
    a, b = parse_lfs_quota(LFS_ONE_LINE, "/s"), parse_lfs_quota(LFS_TWO_LINE, "/s")
    assert (a.files_used, a.files_limit) == (b.files_used, b.files_limit)


def test_parse_lfs_over_quota_star_is_stripped():
    assert parse_lfs_quota(LFS_OVER, "/s").files_used == 1000001


def test_parse_lfs_unlimited_is_none():
    assert parse_lfs_quota(LFS_UNLIMITED, "/s").files_limit is None


def test_parse_lfs_soft_quota_used_when_hard_limit_zero():
    assert parse_lfs_quota(LFS_SOFT_ONLY, "/s").files_limit == 500000


@pytest.mark.parametrize(
    "text",
    ["", "garbage", "Filesystem kbytes\n/x 1 2 3", "junk /x 1 2 3 - 4 5 6 -", "/x a b c - d e f -"],
)
def test_parse_lfs_unparseable_raises(text):
    with pytest.raises(ValueError):
        parse_lfs_quota(text, "/s")


def test_parse_df_inodes():
    r = parse_df_inodes(DF_OUT_CLEAN, "/data")
    assert (r.files_used, r.files_limit, r.source, r.scope) == (123456, 6553600, "df", "filesystem")


def test_parse_df_inodes_dash_limit_is_unlimited():
    assert parse_df_inodes(DF_UNLIMITED, "/dev/shm").files_limit is None


def test_parse_df_unparseable_raises():
    with pytest.raises(ValueError):
        parse_df_inodes("nonsense", "/x")


# --- filesystem type and quota reading ---------------------------------------


def test_fs_type_reads_stat_output(tmp_path):
    assert fs_type(tmp_path, _runner("lustre")) == "lustre"


def test_fs_type_failure_refuses_rather_than_guessing(tmp_path, events):
    with pytest.raises(QuotaUnreadable):
        fs_type(tmp_path, _runner("lustre", stat_rc=1), emit=_emit_into(events))
    assert _named(events, REFUSAL_EVENT)


def test_fs_type_of_missing_path_uses_nearest_existing_ancestor(tmp_path):
    calls: list = []
    fs_type(tmp_path / "not" / "yet", _runner("ext2/ext3", calls=calls))
    assert calls[0][-1] == str(tmp_path)


def test_lustre_lfs_ok_gives_the_per_user_reading(tmp_path):
    calls: list = []
    r = read_quota(tmp_path, _runner("lustre", calls=calls), user="marielle")
    assert (r.source, r.files_used, r.scope, r.principal) == ("lfs", 461023, "user", "marielle")
    assert r.command.startswith("lfs quota -u marielle")
    assert not any(c[0] == "df" for c in calls)


def test_negative_control_lustre_lfs_missing_is_refused_not_df(tmp_path, events):
    calls: list = []
    run = _runner("lustre", lfs=FileNotFoundError, calls=calls)
    with pytest.raises(QuotaUnreadable, match="refusing rather than using a filesystem-wide df"):
        read_quota(tmp_path, run, user="u", emit=_emit_into(events))
    assert not any(c[0] == "df" for c in calls)
    assert _named(events, REFUSAL_EVENT)[0]["fs_type"] == "lustre"


def test_negative_control_lustre_lfs_nonzero_is_refused_not_df(tmp_path):
    with pytest.raises(QuotaUnreadable):
        read_quota(tmp_path, _runner("lustre", lfs=(1, "")), user="u")


def test_lustre_lfs_unparseable_is_refused_not_df(tmp_path):
    with pytest.raises(QuotaUnreadable, match="did not parse"):
        read_quota(tmp_path, _runner("lustre", lfs=(0, "nope")), user="u")


def test_ext4_uses_df_inodes_and_never_lfs(tmp_path):
    calls: list = []
    r = read_quota(tmp_path, _runner("ext2/ext3", calls=calls), user="u")
    assert r.source == "df"
    assert not any(c[0] == "lfs" for c in calls)


def test_df_failure_raises(tmp_path):
    with pytest.raises(RuntimeError):
        read_quota(tmp_path, _runner("ext2/ext3", df=(1, "")), user="u")


def test_read_quota_default_user_is_getpass(tmp_path, monkeypatch):
    monkeypatch.setattr(fileguard._node.getpass, "getuser", lambda: "someone")
    calls: list = []
    read_quota(tmp_path, _runner("lustre", calls=calls))
    assert ["lfs", "quota", "-u", "someone"] == calls[1][:4]


def test_read_quotas_reads_the_group_on_lustre_when_asked(tmp_path):
    rs = read_quotas(
        tmp_path, _runner("lustre", lfs_group=(0, LFS_GROUP_FULL)), user="u", group="so3"
    )
    assert [(r.scope, r.principal) for r in rs] == [("user", "u"), ("group", "so3")]
    assert rs[1].files_limit == 5000000


def test_read_quotas_off_lustre_returns_only_the_filesystem_reading(tmp_path):
    rs = read_quotas(tmp_path, _runner("ext2/ext3"), user="u", group="so3")
    assert [r.scope for r in rs] == ["filesystem"]


# --- location ----------------------------------------------------------------


def test_describe_location_reports_setgid(tmp_path):
    d = tmp_path / "g"
    d.mkdir()
    os.chmod(d, 0o2775)
    if not os.stat(d).st_mode & stat.S_ISGID:
        pytest.skip("filesystem does not keep the setgid bit")
    info = describe_location(d / "child" / "file")
    assert info.setgid is True
    assert info.path == str(d)
    assert info.gid == os.stat(d).st_gid


def test_describe_location_plain_dir_is_not_setgid(tmp_path):
    assert describe_location(_plain_dir(tmp_path)).setgid is False


# --- guard -------------------------------------------------------------------


def test_negative_control_outside_writer_near_cap_refuses(tmp_path, events):
    g = _guard(tmp_path, _runner("lustre", lfs=(0, LFS_NEAR)), events)
    with pytest.raises(QuotaRefused, match="R-X1"):
        g.check(100)
    refusal = _named(events, REFUSAL_EVENT)[0]
    assert (refusal["projected"], refusal["scope"]) == (800090, "user")


def test_positive_control_low_reading_passes(tmp_path):
    assert _guard(tmp_path, _runner("lustre")).check(100) == 461123


def test_refusal_does_not_increment_counter(tmp_path):
    g = _guard(tmp_path, _runner("lustre", lfs=(0, LFS_NEAR)))
    with pytest.raises(QuotaRefused):
        g.check(100)
    assert g.files_written_since_read == 0


def test_reread_after_reread_every_refuses_on_fresh_reading(tmp_path):
    answers = iter([(0, LFS_ONE_LINE), (0, LFS_NEAR)])
    run = _runner("lustre")

    def flip(argv):
        return next(answers) if argv[0] == "lfs" else run(argv)

    g = _guard(tmp_path, flip, reread_every=10)
    g.check(10)  # triggers the second (near-cap) real read
    assert g.n_reads == 2
    with pytest.raises(QuotaRefused):
        g.check(100)


def test_unlimited_never_refuses(tmp_path, events):
    g = _guard(tmp_path, _runner("lustre", lfs=(0, LFS_UNLIMITED)), events)
    g.check(10**9)
    assert not _named(events, REFUSAL_EVENT)


def test_limit_none_emits_limit_unknown_event_and_no_refusal(tmp_path, events):
    _guard(tmp_path, _runner("lustre", lfs=(0, LFS_UNLIMITED)), events)
    assert _named(events, LIMIT_UNKNOWN_EVENT)[0]["refusal_id"] == "R-X1-limit-unknown"


def test_every_real_reading_is_emitted(tmp_path, events):
    g = _guard(tmp_path, _runner("lustre"), events, reread_every=5)
    g.check(5)
    assert len(_named(events, READING_EVENT)) == 2


def test_negative_n_new_files_rejected(tmp_path):
    with pytest.raises(ValueError):
        _guard(tmp_path, _runner("lustre")).check(-1)


@pytest.mark.parametrize("kw", [{"cap_fraction": 0.0}, {"cap_fraction": 1.5}, {"reread_every": 0}])
def test_bad_guard_arguments_rejected(tmp_path, kw):
    with pytest.raises(ValueError):
        _guard(tmp_path, _runner("lustre"), **kw)


def test_record_reports_last_reading_location_and_counters(tmp_path):
    g = _guard(tmp_path, _runner("lustre"))
    g.check(3)
    rec = g.record()
    assert rec["last_reading"]["files_used"] == 461023
    assert rec["files_written_since_read"] == 3
    assert rec["location"]["setgid"] is False


def _setgid_dir(tmp_path):
    d = tmp_path / "shared"
    d.mkdir()
    os.chmod(d, 0o2775)
    if not os.stat(d).st_mode & stat.S_ISGID:
        pytest.skip("filesystem does not keep the setgid bit")
    return d


def test_negative_control_setgid_dir_on_lustre_refuses_on_the_full_group_quota(tmp_path, events):
    d = _setgid_dir(tmp_path)
    run = _runner("lustre", lfs_group=(0, LFS_GROUP_FULL))
    g = FileCapGuard(d, runner=run, user="marielle", emit=_emit_into(events))
    assert [r.scope for r in g.last_readings] == ["user", "group"]
    with pytest.raises(QuotaRefused, match="group files_limit"):
        g.check(1)
    assert _named(events, REFUSAL_EVENT)[0]["scope"] == "group"


def test_setgid_group_check_can_be_disabled(tmp_path):
    d = _setgid_dir(tmp_path)
    run = _runner("lustre", lfs_group=(0, LFS_GROUP_FULL))
    g = FileCapGuard(d, runner=run, user="marielle", check_setgid_group=False)
    assert g.check(1) == 461024


def test_record_path_persists_the_readings_atomically(tmp_path):
    target = tmp_path / "state" / "quota_last.json"
    g = _guard(tmp_path, _runner("lustre"), record_path=target, reread_every=2)
    doc = json.loads(target.read_text())
    assert doc["readings"][0]["files_used"] == 461023
    assert doc["location"]["setgid"] is False
    g.check(2)
    assert len(list(target.parent.iterdir())) == 1  # no leftover temp files


def test_no_record_path_writes_nothing(tmp_path):
    _guard(tmp_path, _runner("lustre"))
    assert [p.name for p in tmp_path.iterdir()] == ["out"]


def test_unreadable_quota_at_construction_writes_no_record(tmp_path):
    target = tmp_path / "q.json"
    with pytest.raises(QuotaUnreadable):
        _guard(tmp_path, _runner("lustre", lfs=(1, "")), record_path=target)
    assert not target.exists()


# --- concurrency slack -------------------------------------------------------


def test_slack_admits_when_overshoot_fits():
    check_concurrency_slack(1_000_000, n_jobs=100, reread_every=1000, cap_fraction=0.8)


def test_negative_control_slack_refuses_when_jobs_can_jointly_overshoot():
    with pytest.raises(QuotaRefused, match="concurrent guards"):
        check_concurrency_slack(1_000_000, n_jobs=200, reread_every=1000, cap_fraction=0.8)


def test_slack_unlimited_never_refuses():
    check_concurrency_slack(None, n_jobs=10**6, reread_every=10**6, cap_fraction=0.99)


def test_slack_rejects_nonpositive_arguments():
    with pytest.raises(ValueError):
        check_concurrency_slack(10, n_jobs=0, reread_every=1, cap_fraction=0.5)


# --- emitter -----------------------------------------------------------------


def test_default_emitter_falls_back_to_logging_without_cisternal(monkeypatch, caplog):
    import builtins

    real_import = builtins.__import__

    def no_cisternal(name, *a, **kw):
        if name == "cisternal":
            raise ImportError(name)
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_cisternal)
    emit = fileguard.default_emitter()
    assert emit is fileguard.log_emitter
    with caplog.at_level("INFO", logger="xtrax.fileguard"):
        emit(READING_EVENT, {"files_used": 1})
    assert READING_EVENT in caplog.text
