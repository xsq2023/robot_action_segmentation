from robot_tas.sampler import build_ffmpeg_select_expression


def test_build_ffmpeg_select_expression_preserves_requested_ids() -> None:
    expr = build_ffmpeg_select_expression([0, 15, 30])
    assert expr == "select=eq(n\\,0)+eq(n\\,15)+eq(n\\,30)"
