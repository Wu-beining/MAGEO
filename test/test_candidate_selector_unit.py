from evaluation.candidate_selector import (
    is_safe_enough,
    net_improvement,
    objective_score,
    select_best_candidate,
)
from evaluation.metrics import compute_dsv_cf_score, compute_wlv_dpa_for_answer


def test_compute_wlv_dpa_for_answer_tracks_target_citation():
    answer = "Sentence one about the target [1]。 Sentence two about another source [2]。"
    metrics = compute_wlv_dpa_for_answer(answer)

    assert 1 in metrics["wlv"]
    assert 1 in metrics["dpa"]
    assert metrics["wlv"][1] > 0
    assert metrics["dpa"][1] > 0


def test_objective_score_matches_formula():
    metric_dict = {
        "wlv": 4.0,
        "dpa": 4.0,
        "cp": 7.0,
        "si": 7.0,
        "aa": 8.0,
        "fa": 8.0,
        "kc": 7.0,
        "ad": 7.0,
    }

    assert objective_score(metric_dict) == compute_dsv_cf_score(metric_dict)


def test_is_safe_enough_enforces_fidelity_gate():
    old_metrics = {"isi.AA": 8.0, "isi.FA": 7.5}
    new_metrics = {"aa": 8.1, "fa": 5.8}

    assert not is_safe_enough(new_metrics, old_metrics)


def test_select_best_candidate_prefers_highest_dsv_cf_gain():
    current_metrics = {
        "ssv.WLV": 1.0,
        "ssv.DPA": 1.0,
        "ssv.CP": 6.0,
        "ssv.SI": 6.0,
        "isi.AA": 8.0,
        "isi.FA": 7.0,
        "isi.KC": 6.0,
        "isi.AD": 6.0,
    }
    candidates = [
        {
            "candidate_id": "V1",
            "revised_content": "candidate one",
            "applied_edit_ops": [],
            "predicted_scores": {
                "wlv": 2.0,
                "dpa": 2.0,
                "cp": 6.5,
                "si": 6.3,
                "aa": 8.1,
                "fa": 7.2,
                "kc": 6.2,
                "ad": 6.1,
            },
        },
        {
            "candidate_id": "V2",
            "revised_content": "candidate two",
            "applied_edit_ops": [],
            "predicted_scores": {
                "wlv": 3.2,
                "dpa": 3.0,
                "cp": 6.7,
                "si": 6.5,
                "aa": 8.2,
                "fa": 7.3,
                "kc": 6.4,
                "ad": 6.2,
            },
        },
    ]

    result = select_best_candidate(candidates, current_metrics)

    assert result is not None
    assert result.candidate_id == "V2"
    assert result.improvement > 0


def test_net_improvement_uses_dsv_cf_delta():
    old_metrics = {"wlv": 1.0, "dpa": 1.0, "cp": 6.0, "si": 6.0, "aa": 8.0, "fa": 7.0, "kc": 6.0, "ad": 6.0}
    new_metrics = {"wlv": 2.0, "dpa": 2.0, "cp": 6.5, "si": 6.5, "aa": 8.0, "fa": 7.2, "kc": 6.2, "ad": 6.1}

    assert net_improvement(new_metrics, old_metrics) > 0
