"""
Metrics calculation module for MAGEO.

This implementation follows the paper's method section:
- SSV: WLV, DPA, CP, SI
- ISI: AA, FA, KC, AD
- Overall objective: DSV-CF
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any


CITATION_PATTERN = re.compile(r"\[(\d+)\]")
SENT_SPLIT_PATTERN = re.compile(r"(?<=[。！？!?；;\.])\s+")

DEFAULT_LAMBDA = 0.5
DEFAULT_GAMMA = 0.5


@dataclass
class SentenceSpan:
    text: str
    tokens: int
    index: int
    citations: list[int]


def tokenize_len(text: str) -> int:
    ascii_chars = []
    non_ascii_chars = []
    for ch in text:
        if ord(ch) < 128:
            ascii_chars.append(ch)
            non_ascii_chars.append(" ")
        else:
            ascii_chars.append(" ")
            non_ascii_chars.append(ch)

    ascii_text = "".join(ascii_chars)
    non_ascii_text = "".join(non_ascii_chars)
    ascii_tokens = re.findall(r"[A-Za-z0-9]+", ascii_text)
    punct = "，。！？、；：（）()【】[]《》,.!?;:\"'"
    non_ascii_tokens = [
        ch
        for ch in non_ascii_text
        if (not ch.isspace()) and (ch not in punct)
    ]
    return len(ascii_tokens) + len(non_ascii_tokens)


def extract_sentences(answer: str) -> list[SentenceSpan]:
    answer = answer.strip()
    if not answer:
        return []

    raw_parts = SENT_SPLIT_PATTERN.split(answer)
    spans: list[SentenceSpan] = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        citation_ids = [int(m.group(1)) for m in CITATION_PATTERN.finditer(part)]
        clean_text = CITATION_PATTERN.sub("", part).strip()
        if not clean_text:
            continue

        spans.append(
            SentenceSpan(
                text=clean_text,
                tokens=tokenize_len(clean_text),
                index=len(spans),
                citations=citation_ids,
            )
        )
    return spans


def compute_wlv_dpa_for_answer(answer: str) -> dict[str, dict[int, float]]:
    spans = extract_sentences(answer)
    if not spans:
        return {"wlv": {}, "dpa": {}, "wlv_raw": {}, "dpa_raw": {}}

    total_tokens = sum(s.tokens for s in spans) or 1
    num_sent = len(spans) or 1

    wlv_raw = defaultdict(float)
    dpa_raw = defaultdict(float)

    for span in spans:
        if not span.citations:
            continue
        share_per_citation = span.tokens / len(span.citations)
        decay = math.exp(-(span.index + 1) / num_sent)
        for citation_id in span.citations:
            wlv_raw[citation_id] += share_per_citation
            dpa_raw[citation_id] += share_per_citation * decay

    wlv = {cid: value / total_tokens for cid, value in wlv_raw.items()}
    dpa = {cid: value / total_tokens for cid, value in dpa_raw.items()}

    wlv_sum = sum(wlv.values()) or 1.0
    dpa_sum = sum(dpa.values()) or 1.0
    wlv = {cid: value / wlv_sum for cid, value in wlv.items()}
    dpa = {cid: value / dpa_sum for cid, value in dpa.items()}

    return {
        "wlv": dict(wlv),
        "dpa": dict(dpa),
        "wlv_raw": dict(wlv_raw),
        "dpa_raw": dict(dpa_raw),
    }


def compute_wc_pwc_for_answer(answer: str) -> dict[str, dict[int, float]]:
    """Backward-compatible alias used by older helper code."""
    return compute_wlv_dpa_for_answer(answer)


def compute_wc_pwc_for_record(record: dict[str, Any]) -> dict[str, dict[int, float]]:
    answer = record.get("content", {}).get("response", "")
    return compute_wlv_dpa_for_answer(answer)


@dataclass
class UnifiedMetrics:
    article_id: str
    version_id: int
    engine_id: str
    query: str
    wlv: float = 0.0
    dpa: float = 0.0
    cp: float = 0.0
    si: float = 0.0
    aa: float = 0.0
    fa: float = 0.0
    kc: float = 0.0
    ad: float = 0.0
    lambda_weight: float = DEFAULT_LAMBDA
    gamma_penalty: float = DEFAULT_GAMMA

    def ssv_score(self) -> float:
        return (self.wlv + self.dpa + self.cp + self.si) / 4.0

    def isi_score(self) -> float:
        return (self.aa + self.fa + self.kc + self.ad) / 4.0

    def dsv_cf_score(self) -> float:
        return (
            self.lambda_weight * self.ssv_score()
            + (1.0 - self.lambda_weight) * self.isi_score()
            - self.gamma_penalty * (10.0 - self.aa)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "meta": {
                "article_id": self.article_id,
                "version_id": self.version_id,
                "engine_id": self.engine_id,
                "query": self.query,
            },
            "ssv": {
                "WLV": self.wlv,
                "DPA": self.dpa,
                "CP": self.cp,
                "SI": self.si,
            },
            "isi": {
                "AA": self.aa,
                "FA": self.fa,
                "KC": self.kc,
                "AD": self.ad,
            },
            "overall": {
                "DSV-CF": self.dsv_cf_score(),
            },
        }

    def get_primary_vector(self) -> dict[str, float]:
        return {
            "ssv.WLV": self.wlv,
            "ssv.DPA": self.dpa,
            "ssv.CP": self.cp,
            "ssv.SI": self.si,
            "isi.AA": self.aa,
            "isi.FA": self.fa,
            "isi.KC": self.kc,
            "isi.AD": self.ad,
            "overall.DSV-CF": self.dsv_cf_score(),
        }


def compute_delta_metrics(
    old_metrics: UnifiedMetrics, new_metrics: UnifiedMetrics
) -> dict[str, float]:
    old_vec = old_metrics.get_primary_vector()
    new_vec = new_metrics.get_primary_vector()
    return {
        key: new_vec.get(key, 0.0) - old_vec.get(key, 0.0)
        for key in set(old_vec) | set(new_vec)
    }


def compute_dsv_cf_score(
    metrics: dict[str, Any],
    lambda_weight: float = DEFAULT_LAMBDA,
    gamma_penalty: float = DEFAULT_GAMMA,
) -> float:
    def pick(*keys: str) -> float:
        for key in keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return 0.0

    wlv = pick("wlv", "WLV", "ssv.WLV")
    dpa = pick("dpa", "DPA", "ssv.DPA")
    cp = pick("cp", "CP", "ssv.CP")
    si = pick("si", "SI", "ssv.SI")
    aa = pick("aa", "AA", "isi.AA")
    fa = pick("fa", "FA", "isi.FA")
    kc = pick("kc", "KC", "isi.KC")
    ad = pick("ad", "AD", "isi.AD")

    ssv = (wlv + dpa + cp + si) / 4.0
    isi = (aa + fa + kc + ad) / 4.0
    return lambda_weight * ssv + (1.0 - lambda_weight) * isi - gamma_penalty * (10.0 - aa)
