from rankcloak.segmented_protocol import likely_sentence_boundary, should_stop_sentence_tail


def test_sentence_tail_does_not_stop_before_minimum():
    assert not should_stop_sentence_tail("This is done.", 19, minimum_tail_tokens=20, maximum_tail_tokens=60)


def test_sentence_tail_stops_after_boundary_once_minimum_reached():
    assert should_stop_sentence_tail("This is done.", 20, minimum_tail_tokens=20, maximum_tail_tokens=60)
    assert should_stop_sentence_tail('This is done."', 20, minimum_tail_tokens=20, maximum_tail_tokens=60)
    assert likely_sentence_boundary("Ready!")


def test_sentence_tail_stops_at_maximum_without_boundary():
    assert should_stop_sentence_tail("still continuing without boundary", 60, minimum_tail_tokens=20, maximum_tail_tokens=60)
