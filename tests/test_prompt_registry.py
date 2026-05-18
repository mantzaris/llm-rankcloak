from rankcloak.prompts import cover_prompt_dictionary, prompt_family


def test_dialogue_prompt_names_exist():
    prompts = cover_prompt_dictionary()
    expected_names = [
        "recipe_dialogue_specific",
        "recipe_forum_exchange_specific",
        "car_buying_dialogue_specific",
        "biology_tutor_dialogue_specific",
    ]
    for prompt_name in expected_names:
        assert prompt_name in prompts
        assert len(prompts[prompt_name]) > 250


def test_dialogue_prompt_families_are_registered():
    assert prompt_family("recipe_dialogue_specific") == "recipe_dialogue"
    assert prompt_family("recipe_forum_exchange_specific") == "recipe_forum_exchange"
    assert prompt_family("car_buying_dialogue_specific") == "car_buying_dialogue"
    assert prompt_family("biology_tutor_dialogue_specific") == "biology_dialogue"


def test_dialogue_prompts_avoid_sensitive_terms():
    blocked_terms = ["secret", "credential", "private key", "api key", "steganography", "cipher"]
    prompts = cover_prompt_dictionary()
    for prompt_name in [
        "recipe_dialogue_specific",
        "recipe_forum_exchange_specific",
        "car_buying_dialogue_specific",
        "biology_tutor_dialogue_specific",
    ]:
        lowered = prompts[prompt_name].lower()
        assert not any(term in lowered for term in blocked_terms)

