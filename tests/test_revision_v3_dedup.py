import pandas as pd

from rankcloak.revision_v3_dedup import (
    build_strict_deduplicated_corpus,
    leave_one_model_partitions,
    normalize_visible_text,
)


def _row(row_id, pair_id, payload, label, text, model="model-a"):
    return {
        "row_id": row_id,
        "pair_id": pair_id,
        "payload_group_id": payload,
        "label": label,
        "text": text,
        "prompt_template_id": "template-{}".format(int(payload.split("-")[-1]) % 3),
        "model_id": model,
        "codec_id": "codec-{}".format(int(payload.split("-")[-1]) % 2),
        "payload_class": "class-{}".format(int(payload.split("-")[-1]) % 2),
    }


def test_normalization_uses_nfkc_casefold_and_whitespace_collapse():
    assert normalize_visible_text("  STRASSE\t\n text  ") == "strasse text"
    assert normalize_visible_text("\u212b") == normalize_visible_text("\u00c5")


def test_exact_duplicates_remove_complete_pair_before_split():
    rows = []
    for index in range(12):
        payload = "payload-{}".format(index)
        pair = "pair-{}".format(index)
        negative = "ordinary control {} with distinctive prose {}".format(
            index, chr(65 + index) * 20
        )
        if index == 0:
            negative = "  DUPLICATE\tvisible text  "
        elif index == 1:
            negative = "duplicate visible   text"
        rows.append(_row("positive-{}".format(index), pair, payload, 1, "rankcloak {} {}".format(index, chr(97 + index) * 30)))
        rows.append(_row("negative-{}".format(index), pair, payload, 0, negative))
    result = build_strict_deduplicated_corpus(
        pd.DataFrame(rows), threshold=1.0, seed=20260831
    )
    assert result.summary["original_rows"] == 24
    assert result.summary["exact_duplicate_groups"] == 1
    assert result.summary["removed_observations"] == 2
    assert result.summary["removed_pairs"] == 1
    assert len(result.frame) == 22
    assert result.frame["normalized_text_sha256"].is_unique
    assert result.leakage_audit["status"] == "pass"
    assert set(result.frame["partition"]) == {"train", "validation", "test"}


def test_near_duplicate_payload_components_cannot_cross_partitions():
    rows = []
    for index in range(15):
        payload = "payload-{}".format(index)
        pair = "pair-{}".format(index)
        positive = "rank constrained passage {} {}".format(index, chr(65 + index) * 40)
        if index == 3:
            positive = "A careful discussion of robust systems and exact replay under controlled conditions."
        if index == 4:
            positive = "A careful discussion of robust systems and exact replay under controlled condition."
        rows.append(_row("positive-{}".format(index), pair, payload, 1, positive))
        rows.append(_row("negative-{}".format(index), pair, payload, 0, "clean prose {} {}".format(index, chr(97 + index) * 35)))
    result = build_strict_deduplicated_corpus(
        pd.DataFrame(rows), threshold=0.95, seed=17
    )
    assert len(result.near_pairs) >= 1
    partitions = result.frame.set_index("payload_group_id")["partition"].to_dict()
    assert partitions["payload-3"] == partitions["payload-4"]
    assert result.leakage_audit["checks"]["near_duplicate_pair_cross_partition"] == 0


def test_leave_one_model_uses_only_other_families_for_fit_and_disjoint_clusters():
    rows = []
    models = ["model-a", "model-b", "model-c"]
    for payload_index in range(18):
        payload = "payload-{}".format(payload_index)
        for model_index, model in enumerate(models):
            pair = "pair-{}-{}".format(payload_index, model_index)
            stem = chr(65 + payload_index) * 18 + chr(97 + model_index) * 18
            rows.append(_row("p-{}-{}".format(payload_index, model), pair, payload, 1, "stego {} {}".format(payload_index, stem), model))
            rows.append(_row("n-{}-{}".format(payload_index, model), pair, payload, 0, "clean {} {}".format(payload_index, stem[::-1]), model))
    result = build_strict_deduplicated_corpus(
        pd.DataFrame(rows), threshold=1.0, seed=23
    )
    split = leave_one_model_partitions(result.frame, "model-c")
    assert "model-c" not in set(split["train"]["model_id"])
    assert "model-c" not in set(split["validation"]["model_id"])
    assert set(split["test"]["model_id"]) == {"model-c"}
    groups = [set(split[name]["dedup_cluster_id"]) for name in ("train", "validation", "test")]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
