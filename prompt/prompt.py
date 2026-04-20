import json


def query_rewriter_system_prompt() -> str:
    return (
        "You rewrite a natural-language user request into a compact search query. "
        "Keep the user's intent unchanged, remove filler language, and return JSON only."
    )


def query_rewriter_user_prompt(user_query: str) -> str:
    return (
        "Rewrite the following user request into a search-ready query.\n"
        "Return JSON with this schema only:\n"
        '{\n  "main_query": "...",\n  "alternative_queries": ["...", "..."]\n}\n'
        f"User request:\n{user_query}"
    )


def qa_system_prompt() -> str:
    return (
        "You are a retrieval-grounded QA assistant. "
        "Answer only from the provided documents, and place citation markers like [1] or [1][3] "
        "at the end of every sentence."
    )


def qa_user_prompt(user_query: str, documents: str) -> str:
    return (
        f"User query:\n{user_query}\n\n"
        "Documents:\n"
        f"{documents}\n\n"
        "Write a concise answer grounded only in the documents. "
        "Every sentence must end with at least one citation marker."
    )


def preference_system_prompt() -> str:
    return (
        "You are the Preference Agent in MAGEO. "
        "Transform raw engine observations or rules into a reusable engine preference profile. "
        "Return JSON only."
    )


def preference_user_prompt(engine_id: str, engine_rules: str) -> str:
    schema = {
        "engine_id": engine_id,
        "preference_profile": {
            "format_preferences": ["bullet points", "compact structure"],
            "content_preferences": ["evidence density", "authoritative tone"],
            "risk_constraints": ["avoid unsupported claims"],
            "style_preferences": ["direct", "didactic"],
        },
        "summary": "Short natural-language summary of the engine's preferred editing style.",
    }
    return (
        "Normalize the following raw engine rules into a reusable Preference Profile.\n"
        "Return JSON only in this schema:\n"
        f"{json.dumps(schema, ensure_ascii=False, indent=2)}\n\n"
        f"Engine ID:\n{engine_id}\n\n"
        f"Raw rules:\n{engine_rules}"
    )


def planner_system_prompt() -> str:
    return (
        "You are the Planner Agent in MAGEO. "
        "Given a frozen retrieval setting, a current document, an engine preference profile, "
        "and retrieved memory, produce a small set of high-level revision steps. "
        "Do not rewrite the document. Return JSON only."
    )


def planner_user_prompt(
    query: str,
    document_with_spans: str,
    engin_rules: str,
    retrieved_memory_example: str,
) -> str:
    return (
        f"Query:\n{query}\n\n"
        f"Current document:\n{document_with_spans}\n\n"
        f"Engine preference profile:\n{engin_rules}\n\n"
        f"Retrieved memory examples:\n{retrieved_memory_example}\n\n"
        "Return JSON only:\n"
        '{\n'
        '  "plan_steps": [\n'
        "    {\n"
        '      "step_id": "step_1",\n'
        '      "target_span": "intro",\n'
        '      "edit_type": "Structure",\n'
        '      "target_metrics": ["WLV", "DPA"],\n'
        '      "risk_constraints": ["preserve attribution fidelity"],\n'
        '      "rationale": "Why this edit matters under the current engine profile.",\n'
        '      "suggested_operations": ["Add ...", "Reorder ..."],\n'
        '      "inspired_by_examples": ["memory_id_1"]\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def editor_system_prompt() -> str:
    return (
        "You are the Editor Agent in MAGEO. "
        "Execute the planner's instructions and generate diverse candidate variants. "
        "Each candidate should remain faithful to the source document while exploring "
        "different structure, evidence, and style choices. Return JSON only."
    )


def editor_user_prompt(
    document_with_spans: str,
    revision_plan: str,
    engine_rules: str,
    k: int,
) -> str:
    return (
        f"Current document:\n{document_with_spans}\n\n"
        f"Revision plan:\n{revision_plan}\n\n"
        f"Engine preference profile:\n{engine_rules}\n\n"
        f"Generate {k} candidate variants.\n\n"
        "Return JSON only:\n"
        '{\n'
        '  "candidates": [\n'
        "    {\n"
        '      "candidate_id": "V1",\n'
        '      "description": "Short summary of the variant strategy.",\n'
        '      "applied_edit_ops": [\n'
        '        {"edit_type": "Evidence", "target_span": "body_1", "op_pattern": "add-credible-stat"}\n'
        "      ],\n"
        '      "revised_content": "Full revised document text."\n'
        "    }\n"
        "  ]\n"
        "}"
    )


def evaluation_system_prompt() -> str:
    return (
        "You are the Evaluator Agent in MAGEO. "
        "Judge each candidate under the DSV-CF framework and return JSON only. "
        "Use the following 8 metrics on a 1-10 scale: "
        "wlv, dpa, cp, si, aa, fa, kc, ad. "
        "The candidate must be penalized when attribution is weak or semantic faithfulness is poor."
    )


def evaluation_user_prompt(
    user_query: str,
    baseline_content: str,
    candidates: str,
    engine_rules: str,
) -> str:
    return (
        f"User query:\n{user_query}\n\n"
        f"Baseline content:\n{baseline_content}\n\n"
        f"Candidate list:\n{candidates}\n\n"
        f"Engine preference profile:\n{engine_rules}\n\n"
        "For each candidate, predict DSV-CF-related scores and short comments.\n"
        "Return JSON only with this schema:\n"
        '{\n'
        '  "evaluations": [\n'
        "    {\n"
        '      "candidate_id": "V1",\n'
        '      "predicted_scores": {\n'
        '        "wlv": 6.5,\n'
        '        "dpa": 6.2,\n'
        '        "cp": 7.0,\n'
        '        "si": 6.8,\n'
        '        "aa": 8.5,\n'
        '        "fa": 8.0,\n'
        '        "kc": 7.2,\n'
        '        "ad": 6.9\n'
        "      },\n"
        '      "metric_critic_comment": "...",\n'
        '      "safety_critic_comment": "...",\n'
        '      "preference_critic_comment": "...",\n'
        '      "overall_comment": "..."\n'
        "    }\n"
        "  ]\n"
        "}"
    )
