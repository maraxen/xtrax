"""Tests for configurable bundle_js in xtrax.eda (#3126)."""

from __future__ import annotations

import pytest

from xtrax.eda import render


@pytest.fixture
def sample_plan():
    from xtrax.tiling.plan import BatchPlan

    return BatchPlan(decisions=())


def test_render_html_bundle_js_true_contains_no_cdn(sample_plan) -> None:
    pytest.importorskip("vl_convert")
    html_out = render(sample_plan, fmt="html", backend="vega", bundle_js=True)
    assert isinstance(html_out, str)
    assert len(html_out) > 100_000
    assert "cdn.jsdelivr.net" not in html_out
    assert "vegaEmbed" in html_out or "vega" in html_out.lower()


def test_render_html_bundle_js_false_contains_cdn(sample_plan) -> None:
    html_out = render(sample_plan, fmt="html", backend="vega", bundle_js=False)
    assert isinstance(html_out, str)
    assert "cdn.jsdelivr.net" in html_out


def test_matplotlib_rejects_bundle_js(sample_plan) -> None:
    with pytest.raises((NotImplementedError, ValueError), match="bundle_js"):
        render(sample_plan, fmt="png", backend="matplotlib", bundle_js=True)


def test_eda_report_helper(sample_plan) -> None:
    pytest.importorskip("vl_convert")
    from xtrax.eda.report import render_report

    # By default, bundle_js=True
    report_html = render_report(sample_plan)
    assert isinstance(report_html, str)
    assert "cdn.jsdelivr.net" not in report_html

    # When overridden to False, includes CDN
    cdn_report = render_report(sample_plan, bundle_js=False)
    assert isinstance(cdn_report, str)
    assert "cdn.jsdelivr.net" in cdn_report
