import shutil
from pathlib import Path
from uuid import uuid4

from memory import EditOp, MemoryBank, RevisionPlanStep


def _make_local_test_dir() -> Path:
    path = Path("log") / "test_memory" / uuid4().hex
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_memory_bank_retrieves_successful_dsv_cf_examples():
    test_dir = _make_local_test_dir()
    memory_bank = MemoryBank(storage_path=test_dir / "memory")

    plan_steps = [
        RevisionPlanStep(
            step_id="step_1",
            target_span="intro",
            edit_type="Evidence",
            target_metrics=["overall.DSV-CF", "ssv.WLV"],
            risk_constraints=["preserve attribution fidelity"],
            rationale="Need more evidence and clearer attribution.",
            suggested_operations=["Add a concrete statistic", "Move the key claim earlier"],
        )
    ]
    edit_ops = [
        EditOp(
            edit_type="Evidence",
            target_span="intro",
            op_pattern="add-credible-stat",
        )
    ]

    old_metrics = {
        "ssv": {"WLV": 1.0, "DPA": 1.0, "CP": 6.0, "SI": 6.0},
        "isi": {"AA": 8.0, "FA": 7.0, "KC": 6.0, "AD": 6.0},
        "overall": {"DSV-CF": 4.25},
    }
    new_metrics = {
        "ssv": {"WLV": 2.5, "DPA": 2.0, "CP": 6.4, "SI": 6.4},
        "isi": {"AA": 8.2, "FA": 7.2, "KC": 6.5, "AD": 6.2},
        "overall": {"DSV-CF": 5.20},
    }

    step_record = memory_bank.add_step_from_edit(
        article_id="article_001",
        engine_id="gpt-5.2",
        query="How to improve GEO visibility?",
        round_id=0,
        from_version=0,
        to_version=1,
        planner_plans=plan_steps,
        applied_ops=edit_ops,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
    )

    assert step_record.delta_metrics["overall.DSV-CF"] > 0

    examples = memory_bank.retrieve_for_planner(
        article_id="article_002",
        current_metrics=old_metrics,
        engine_id="gpt-5.2",
        top_k=5,
    )

    assert len(examples) == 1
    assert examples[0].engine_id == "gpt-5.2"
    assert "dsv_cf_delta" in examples[0].summary
    shutil.rmtree(test_dir, ignore_errors=True)


def test_creator_record_extracts_best_patterns():
    test_dir = _make_local_test_dir()
    memory_bank = MemoryBank(storage_path=test_dir / "memory")

    step_record = memory_bank.add_step_from_edit(
        article_id="article_001",
        engine_id="gemini-3-pro",
        query="medical advice bullets",
        round_id=0,
        from_version=0,
        to_version=1,
        planner_plans=[],
        applied_ops=[
            EditOp("Structure", "body_1", "convert-to-bullets"),
            EditOp("Evidence", "body_2", "add-source-citation"),
        ],
        old_metrics={"overall": {"DSV-CF": 4.0}},
        new_metrics={"overall": {"DSV-CF": 5.0}},
    )

    creator_record = memory_bank.add_creator_from_trajectory(
        article_id="article_001",
        engine_id="gemini-3-pro",
        query="medical advice bullets",
        final_version_id=1,
        final_metrics={"overall": {"DSV-CF": 5.0}},
        version_history=[(0, {"overall": {"DSV-CF": 4.0}}), (1, {"overall": {"DSV-CF": 5.0}})],
        step_records=[step_record],
        summary="Bullets plus explicit citation improved DSV-CF.",
    )

    assert creator_record.best_edit_patterns
    assert creator_record.best_edit_patterns[0].target_metrics
    shutil.rmtree(test_dir, ignore_errors=True)
