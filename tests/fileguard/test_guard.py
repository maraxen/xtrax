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
    parse_orcd_report,
    parse_quota_rows,
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
    fstype,
    *,
    lfs=(0, LFS_ONE_LINE),
    lfs_group=None,
    df=(0, DF_OUT_CLEAN),
    quota=FileNotFoundError,
    quota_group=None,
    stat_rc=0,
    calls=None,
):
    """A fake runner: ``stat -f`` answers ``fstype``; lfs (user/group) and df answer as given.

    ``lfs`` may be an exception class to raise (e.g. FileNotFoundError, as a missing binary does).
    ``lfs_group`` answers ``lfs quota -g``; it defaults to ``lfs``. ``quota`` answers
    ``quota -w -u`` (default: binary missing) and ``quota_group`` answers ``quota -w -g``.
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
        if argv[0] == "quota":
            answer = quota_group if (argv[2] == "-g" and quota_group is not None) else quota
            if isinstance(answer, type) and issubclass(answer, BaseException):
                raise answer("quota")
            return answer
        raise AssertionError(f"unexpected argv {argv!r}")

    return run


@pytest.fixture(autouse=True)
def _no_host_orcd(monkeypatch):
    """Tests never read the host's ~/orcd report unless they ask for it."""
    monkeypatch.setattr(fileguard._node, "default_orcd_dir", lambda: None)


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
    kw.setdefault("cap_fraction", 0.80)  # the ported alynxr cases were written against 0.80
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


# --- NFS: per-user `quota -w`, matched to the df device (A9, Engaging 2026-10-30) --------------

# Captured on Engaging (login node, 2026-10-30): home is quota-limited per user, scratch is not
# reported by `quota` at all, and scratch's `df -i` is the whole fstor018 server.
ENGAGING_QUOTA_U = """\
Disk quotas for user maarxaru (uid 226996): 
     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace
nfs001.lb:/home 40162568  204472320 209715200          217187  1000000 1000000        
nfs001.lb:/home 40162568  204472320 209715200          217187  1000000 1000000        
"""
ENGAGING_QUOTA_G_NONE = "Disk quotas for group maarxaru (gid 226996): none\n"
DF_I_HOME = """\
Filesystem         Inodes    IUsed      IFree IUse% Mounted on
nfs001.lb:/home  2000000000  900000000  1100000000   45% /orcd/home/002
"""
DF_I_SCRATCH = """\
Filesystem         Inodes    IUsed      IFree IUse% Mounted on
fstor018.ib:/  6301144320 29575803 6271568517    1% /orcd/scratch/orcd/011
"""
QUOTA_OVER = """\
Disk quotas for user u (uid 1):
     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace
nfs001.lb:/home 10  100 200   1000001*  1000000 1000000  6days
"""
QUOTA_BLOCKS_OVER = """\
Disk quotas for user u (uid 1):
     Filesystem  blocks   quota   limit   grace   files   quota   limit   grace
nfs001.lb:/home 300*  100 200  7days  5  1000 2000
"""


def test_parse_quota_rows_engaging_home():
    assert parse_quota_rows(ENGAGING_QUOTA_U) == {"nfs001.lb:/home": (217187, 1000000)}


def test_parse_quota_rows_files_over_quota_with_grace():
    assert parse_quota_rows(QUOTA_OVER) == {"nfs001.lb:/home": (1000001, 1000000)}


def test_parse_quota_rows_blocks_grace_shifts_the_file_cells():
    assert parse_quota_rows(QUOTA_BLOCKS_OVER) == {"nfs001.lb:/home": (5, 2000)}


def test_parse_quota_rows_none_has_no_rows():
    assert parse_quota_rows(ENGAGING_QUOTA_G_NONE) == {}


def test_parse_quota_rows_short_row_raises():
    with pytest.raises(ValueError):
        parse_quota_rows("     Filesystem  blocks\n/dev/x 1 2\n")


def test_nfs_home_reads_the_per_user_quota_before_the_filesystem(tmp_path):
    run = _runner("nfs", df=(0, DF_I_HOME), quota=(0, ENGAGING_QUOTA_U))
    rs = read_quotas(tmp_path, run, user="maarxaru")
    assert [(r.scope, r.source) for r in rs] == [("user", "quota"), ("filesystem", "df")]
    assert (rs[0].files_used, rs[0].files_limit) == (217187, 1000000)
    assert read_quota(tmp_path, run, user="maarxaru").scope == "user"


def test_negative_control_nfs_guard_refuses_on_the_user_cap_the_df_figure_would_pass(tmp_path):
    near = ENGAGING_QUOTA_U.replace("217187", "799990")
    run = _runner("nfs", df=(0, DF_I_HOME), quota=(0, near))
    g = _guard(tmp_path, run)
    with pytest.raises(QuotaRefused, match="user files_limit 1000000"):
        g.check(100)
    # The filesystem-wide df reading alone (2e9 inodes, 45% used) would have admitted it.
    (tmp_path / "x").mkdir()
    fs_only = _guard(tmp_path / "x", _runner("nfs", df=(0, DF_I_HOME)))
    assert fs_only.check(100) == 900000100


def test_nfs_scratch_without_a_user_row_is_flagged_not_silently_trusted(tmp_path, events):
    run = _runner("nfs", df=(0, DF_I_SCRATCH), quota=(0, ENGAGING_QUOTA_U))
    rs = read_quotas(tmp_path, run, user="maarxaru", emit=_emit_into(events))
    assert [r.scope for r in rs] == ["filesystem"]
    unknown = _named(events, LIMIT_UNKNOWN_EVENT)
    assert unknown[0]["scope"] == "user"
    assert unknown[0]["device"] == "fstor018.ib:/"
    assert "no user row" in unknown[0]["detail"]


def test_missing_quota_binary_is_flagged(tmp_path, events):
    rs = read_quotas(tmp_path, _runner("nfs", df=(0, DF_I_HOME)), user="u", emit=_emit_into(events))
    assert [r.scope for r in rs] == ["filesystem"]
    assert "not installed" in _named(events, LIMIT_UNKNOWN_EVENT)[0]["detail"]


def test_unparseable_quota_output_is_flagged(tmp_path, events):
    bad = "     Filesystem  blocks\nnfs001.lb:/home 1 2\n"
    run = _runner("nfs", df=(0, DF_I_HOME), quota=(0, bad))
    read_quotas(tmp_path, run, user="u", emit=_emit_into(events))
    assert "did not parse" in _named(events, LIMIT_UNKNOWN_EVENT)[0]["detail"]


def test_nfs_group_quota_none_is_flagged_and_user_still_enforced(tmp_path, events):
    run = _runner(
        "nfs",
        df=(0, DF_I_HOME),
        quota=(0, ENGAGING_QUOTA_U),
        quota_group=(0, ENGAGING_QUOTA_G_NONE),
    )
    rs = read_quotas(tmp_path, run, user="maarxaru", group="so3", emit=_emit_into(events))
    assert [r.scope for r in rs] == ["user", "filesystem"]
    assert _named(events, LIMIT_UNKNOWN_EVENT)[0]["scope"] == "group"


# --- MIT ORCD (Engaging) per-user report: ~/orcd/.quota ---------------------------------------

# Exact text of ~/orcd/.quota as captured on Engaging 2026-10-30 (fixture, not a finding).
ORCD_REPORT = """\
                               QUOTA REPORT
 Space   | Usage (GB) | Limit (GB) | % Used |  Files | Limit | % Used
---------+------------+------------+--------+--------+-------+--------
 HOME    |       38.3 |      200.0 |  19.15 | 217.2K |  1.0M |  21.72
 SCRATCH |      207.5 |     1024.0 |  20.26 | 815.3K |  1.0M |  81.53
 POOL    |      125.6 |     1024.0 |  12.27 |   1.1M |  1.1M |  99.36
Last updated: Fri Oct 30 15:01:06 EDT 2026
"""


def _orcd_tree(tmp_path, report=ORCD_REPORT):
    """A home with an orcd/ dir: the report plus scratch/pool symlinks to separate roots."""
    home = tmp_path / "home"
    orcd = home / "orcd"
    orcd.mkdir(parents=True)
    roots = {name: tmp_path / f"{name}_root" for name in ("scratch", "pool")}
    for name, root in roots.items():
        root.mkdir()
        (orcd / name).symlink_to(root)
    (orcd / ".quota").write_text(report)
    (tmp_path / "compute").mkdir()
    return home, orcd, roots


def _orcd_read(tmp_path, target, events=None, **kw):
    home, orcd, roots = _orcd_tree(tmp_path)
    return read_quotas(
        target(home, roots),
        _runner("nfs", df=(0, DF_I_SCRATCH)),
        user="u",
        emit=_emit_into(events if events is not None else []),
        orcd_dir=orcd,
        home=home,
        **kw,
    )


def test_parse_orcd_report_is_conservative_and_fraction_exact():
    rows = parse_orcd_report(ORCD_REPORT)
    assert set(rows) == {"HOME", "SCRATCH", "POOL"}
    # limit = rounded limit minus half a rounding unit; used = the precise % of that limit
    assert rows["SCRATCH"] == (774535, 950000)
    assert rows["POOL"] == (1043280, 1050000)
    assert rows["HOME"] == (206340, 950000)


@pytest.mark.parametrize("text", ["", "no table here", ORCD_REPORT.replace("81.53", "n/a")])
def test_parse_orcd_report_unparseable_raises(text):
    with pytest.raises(ValueError):
        parse_orcd_report(text)


def test_orcd_space_roots_follow_the_symlinks(tmp_path):
    home, orcd, roots = _orcd_tree(tmp_path)
    got = fileguard.orcd_space_roots(orcd, home)
    assert got["SCRATCH"] == roots["scratch"].resolve()
    assert got["POOL"] == roots["pool"].resolve()
    assert got["HOME"] == home.resolve()


def test_scratch_path_gets_the_orcd_scratch_row_first(tmp_path):
    rs = _orcd_read(tmp_path, lambda home, roots: roots["scratch"] / "runs" / "new")
    assert (rs[0].source, rs[0].principal) == ("orcd_report", "u:SCRATCH")
    assert (rs[0].files_used, rs[0].files_limit) == (774535, 950000)
    assert rs[-1].scope == "filesystem"


def test_path_through_the_home_symlink_resolves_to_scratch_not_home(tmp_path):
    rs = _orcd_read(tmp_path, lambda home, roots: home / "orcd" / "scratch" / "x")
    assert rs[0].principal == "u:SCRATCH"


def test_negative_control_pool_at_99_percent_refuses_where_df_alone_would_admit(tmp_path):
    home, orcd, roots = _orcd_tree(tmp_path)
    target = roots["pool"] / "out"
    target.mkdir()
    run = _runner("nfs", df=(0, DF_I_SCRATCH))
    g = FileCapGuard(target, runner=run, user="u", orcd_dir=orcd, check_setgid_group=False)
    with pytest.raises(QuotaRefused, match="R-X1"):
        g.check(1)
    blind = FileCapGuard(target, runner=run, user="u", orcd_dir=None, check_setgid_group=False)
    blind.check(1)  # the server-wide df reading alone admits it


def test_path_in_no_orcd_space_is_flagged(tmp_path, events):
    rs = _orcd_read(tmp_path, lambda home, roots: home.parent / "compute", events)
    assert [r.scope for r in rs] == ["filesystem"]
    flagged = [e for e in _named(events, LIMIT_UNKNOWN_EVENT) if "ORCD" in e["detail"]]
    assert flagged


def test_negative_control_stale_orcd_report_is_refused(tmp_path):
    home, orcd, roots = _orcd_tree(tmp_path)
    old = 1_000_000_000
    os.utime(orcd / ".quota", (old, old))
    with pytest.raises(QuotaUnreadable, match="stale"):
        read_quotas(
            roots["scratch"],
            _runner("nfs", df=(0, DF_I_SCRATCH)),
            user="u",
            orcd_dir=orcd,
            home=home,
        )


def test_unparseable_orcd_report_is_refused(tmp_path):
    home, orcd, roots = _orcd_tree(tmp_path, report="garbage\n")
    with pytest.raises(QuotaUnreadable, match="ORCD quota report"):
        read_quotas(
            roots["scratch"],
            _runner("nfs", df=(0, DF_I_SCRATCH)),
            user="u",
            orcd_dir=orcd,
            home=home,
        )


def test_auto_detects_orcd_dir_under_home(tmp_path, monkeypatch):
    monkeypatch.undo()  # drop the autouse "no host orcd" patch for this test
    home, orcd, roots = _orcd_tree(tmp_path)
    monkeypatch.setattr(fileguard._node.Path, "home", classmethod(lambda cls: home))
    assert fileguard.default_orcd_dir() == orcd
    rs = read_quotas(roots["pool"], _runner("nfs", df=(0, DF_I_SCRATCH)), user="u", home=home)
    assert rs[0].principal == "u:POOL"


def test_orcd_disabled_with_none(tmp_path):
    home, orcd, roots = _orcd_tree(tmp_path / "b")
    rs = read_quotas(
        roots["scratch"], _runner("nfs", df=(0, DF_I_SCRATCH)), user="u", orcd_dir=None
    )
    assert all(r.source != "orcd_report" for r in rs)


def test_default_cap_is_095_and_admits_scratch_at_81_percent(tmp_path):
    assert fileguard.DEFAULT_CAP_FRACTION == 0.95
    home, orcd, roots = _orcd_tree(tmp_path)
    g = FileCapGuard(
        roots["scratch"],
        runner=_runner("nfs", df=(0, DF_I_SCRATCH)),
        user="u",
        orcd_dir=orcd,
        check_setgid_group=False,
    )
    assert g.cap_fraction == 0.95
    g.check(1)  # SCRATCH at 81.53% is under the default cap
    with pytest.raises(QuotaRefused):
        g.check(200_000)  # ...and a burst that would cross 95% is refused
