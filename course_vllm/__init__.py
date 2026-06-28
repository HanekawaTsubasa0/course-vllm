from course_vllm.stages import all_stage_overviews, stage_overview

__all__ = ["Engine", "SamplingParams", "all_stage_overviews", "stage_overview"]


def __getattr__(name: str):
    if name == "Engine":
        from course_vllm.engine.engine import Engine

        return Engine
    if name == "SamplingParams":
        from course_vllm.engine.sampler import SamplingParams

        return SamplingParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
