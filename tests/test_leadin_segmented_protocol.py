from rankcloak.paper_suite import decode_forced_span_token_ids, segmented_variant_specs


def test_decode_forced_span_ignores_leadin_and_tail_tokens():
    full_token_ids = [10, 11, 20, 21, 30, 31]
    assert decode_forced_span_token_ids(full_token_ids, 2, 2) == [20, 21]


def test_leadin_variant_is_registered():
    variants = {spec["protocol_variant"]: spec for spec in segmented_variant_specs()}
    leadin = variants["segmented_hex_multi_topic_leadin8_sentence_tail_filtered"]
    assert leadin["leadin_token_count"] == 8
    assert leadin["leadin_policy"] == "greedy_leadin8"
