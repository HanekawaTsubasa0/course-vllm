import pytest

from course_vllm.stages import all_stage_overviews, normalize_stage, stage_overview


def test_normalize_stage_accepts_week_name_and_number():
    assert normalize_stage("week04") == "week04"
    assert normalize_stage("4") == "week04"
    assert normalize_stage(4) == "week04"


def test_stage_overview_exposes_required_outputs():
    overview = stage_overview("week02")
    assert overview["week"] == 2
    assert overview["code_status"] == "implemented"
    assert overview["required_outputs"]


def test_all_stage_overviews_cover_full_course():
    stages = all_stage_overviews()
    assert stages[0]["key"] == "week01"
    assert stages[-1]["key"] == "week16"
    assert len(stages) == 16


def test_non_ascend_stages_are_not_placeholders():
    for overview in all_stage_overviews():
        if overview["key"] == "week14":
            continue
        assert overview["code_status"] == "implemented"


def test_normalize_stage_rejects_unknown_stage():
    with pytest.raises(ValueError):
        normalize_stage("week99")
