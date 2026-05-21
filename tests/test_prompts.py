from rankcloak.prompts import cover_prompt_dictionary, prompt_family


def test_strong_prompt_names_exist():
    prompts = cover_prompt_dictionary()
    for prompt_name in [
        "recipe_long_specific",
        "biology_long_specific",
        "car_buying_long_specific",
    ]:
        assert prompt_name in prompts
        assert len(prompts[prompt_name]) > 300


def test_strong_prompts_avoid_sensitive_terms():
    blocked_terms = ["secret", "credential", "private key", "api key", "steganography", "cipher"]
    prompts = cover_prompt_dictionary()
    for prompt_name in [
        "recipe_long_specific",
        "biology_long_specific",
        "car_buying_long_specific",
    ]:
        lowered = prompts[prompt_name].lower()
        assert not any(term in lowered for term in blocked_terms)


def test_prompt_families_are_registered():
    assert prompt_family("recipe_long_specific") == "recipe"
    assert prompt_family("biology_long_specific") == "biology"
    assert prompt_family("car_buying_long_specific") == "car_buying"


def test_quality_control_prompt_names_exist():
    prompts = cover_prompt_dictionary()
    for prompt_name in [
        "grocery_planning_note_specific",
        "plant_care_note_specific",
    ]:
        assert prompt_name in prompts
        assert len(prompts[prompt_name]) > 250


def test_quality_control_prompt_families_are_registered():
    assert prompt_family("grocery_planning_note_specific") == "grocery_planning"
    assert prompt_family("plant_care_note_specific") == "plant_care"
