"""Original cover prompts used by RankCloak experiments."""

from __future__ import annotations

from typing import Dict


def cover_prompt_dictionary() -> Dict[str, str]:
    """Return original, non-copyrighted cover prompts keyed by prompt name."""

    return {
        "play_dialogue": (
            "Write a short original stage dialogue between two lighthouse keepers during a calm "
            "maintenance shift. Keep the dialogue grounded, mundane, and naturally paced.\n"
        ),
        "original_fantasy_fiction": (
            "Continue an original low-fantasy travel scene about a cartographer crossing a foggy "
            "salt marsh. Use quiet sensory detail and no named copyrighted worlds.\n"
        ),
        "recipe_blog": (
            "Write a practical recipe blog paragraph about preparing lentil stew for a rainy "
            "weekday dinner. Use warm but ordinary cooking language.\n"
        ),
        "recipe_long_specific": (
            "Continue a detailed but ordinary recipe-blog essay for home cooks making a simple "
            "weeknight vegetable and lentil stew. The audience is a busy household cook who wants "
            "clear reassurance, not performance cooking. The voice should be warm, practical, and "
            "explanatory, with coherent prose paragraphs rather than bullets or numbered steps. "
            "Continue as if the introduction has already described rinsing lentils, warming onions, "
            "and building flavor slowly in a heavy pot. Mention sensory cues such as the smell of "
            "garlic softening, the color of tomato deepening, steam on the lid, and the way lentils "
            "turn tender without becoming mushy. Work in natural transitions, ingredient substitutions, "
            "and cooking observations. Keep the tone plain, useful, and natural, as though the writer "
            "is calmly guiding a friend through dinner on a rainy evening.\n"
        ),
        "recipe_dialogue_specific": (
            "Continue a two-person kitchen conversation between Mara and Jon while they make lentil "
            "stew on a rainy evening. The tone is mundane, practical, and calm. Mara and Jon should "
            "use short natural speaker turns, with their names repeated naturally at the start of each "
            "turn. They are talking about whether the onions are soft enough, how the garlic smells, "
            "when the tomato paste darkens, how to judge lentil tenderness, what substitutions are fine, "
            "and when to add salt. After this opening, avoid stage directions and continue only with "
            "ordinary dialogue. Do not use lists, links, markup, formulas, or dramatic narration.\n"
        ),
        "recipe_forum_exchange_specific": (
            "Continue a casual cooking forum exchange about making lentil stew with ordinary ingredients "
            "already at home. One person is asking whether their pantry ingredients will work, and another "
            "person replies warmly with practical reassurance. Use short paragraphs in a normal forum style, "
            "not bullet lists. Mention simple substitutions, cooking observations, onion softness, garlic "
            "smell, tomato paste color, lentil tenderness, and adjusting salt near the end. Avoid links, "
            "hashtags, code blocks, markdown lists, and exaggerated internet slang.\n"
        ),
        "biology_long_specific": (
            "Continue a safe educational biology explanation for undergraduate students in a natural "
            "science course. The topic is careful observation of pond water under a classroom microscope "
            "and the related idea that small structures can reveal larger ecological patterns. Use calm "
            "explanatory prose, not a protocol, not numbered lab instructions, and not operational wet-lab "
            "directions. Define terms gently when they appear, describe visible observations such as green "
            "algal cells, drifting particles, tiny swimming organisms, light, focus, and uncertainty, and "
            "model cautious interpretation. The tone should be patient, concrete, and educational, like "
            "field notes rewritten for students who are learning how to distinguish what they see from what "
            "they infer. Continue in coherent paragraphs with natural transitions.\n"
        ),
        "car_buying_long_specific": (
            "Continue a casual but coherent forum-style message from someone thinking through buying a "
            "new car. The speaker is practical and undecided, writing to a friend or community group in "
            "normal paragraphs. Discuss budget, reliability, fuel economy, insurance, used versus new, "
            "test drives, maintenance history, commuting needs, family space, parking, financing, and the "
            "tradeoffs between comfort and long-term cost. Keep the tone conversational and grounded. The "
            "writer should sound like a real person sorting through priorities rather than making a polished "
            "advertisement. Continue naturally with specific considerations and ordinary transitions.\n"
        ),
        "car_buying_dialogue_specific": (
            "Continue a casual conversation between two friends about buying a new car. They should speak "
            "in short grounded turns, without placeholders and without brand names. The friends discuss "
            "budget, insurance, maintenance history, used versus new, fuel economy, test drives, parking, "
            "family space, and reliability. Keep the tone ordinary and conversational, like two practical "
            "people sorting out tradeoffs after work. Avoid lists, links, markup, slogans, and sales language.\n"
        ),
        "biology_tutor_dialogue_specific": (
            "Continue a safe educational conversation between a biology tutor and a student observing pond "
            "water under a classroom microscope. The exchange should be conceptual and observational, not a "
            "wet-lab protocol. Use short natural speaker turns. They discuss what can be seen, uncertainty, "
            "algal cells, drifting particles, tiny swimming organisms, focus, light, and careful interpretation. "
            "Avoid formulas, LaTeX, markdown, links, and procedural lab instructions.\n"
        ),
        "forum_reply": (
            "Write a helpful forum reply to someone asking how to organize a small community "
            "workshop. Be specific, friendly, and concise.\n"
        ),
        "technical_documentation": (
            "Write a technical documentation paragraph explaining how to rotate application logs "
            "on a local server. Use clear operational prose without commands that affect real systems.\n"
        ),
        "code_review_comment": (
            "Write a code review comment about simplifying a validation helper and improving test "
            "coverage. Be constructive and precise.\n"
        ),
        "meeting_minutes": (
            "Write concise meeting minutes for a neighborhood garden planning meeting. Include "
            "neutral agenda-style phrasing.\n"
        ),
    }


def prompt_family_dictionary() -> Dict[str, str]:
    """Return coarse prompt-family labels used in comparison tables."""

    return {
        "play_dialogue": "dialogue",
        "original_fantasy_fiction": "fiction",
        "recipe_blog": "recipe",
        "recipe_long_specific": "recipe",
        "recipe_dialogue_specific": "recipe_dialogue",
        "recipe_forum_exchange_specific": "recipe_forum_exchange",
        "biology_long_specific": "biology",
        "car_buying_long_specific": "car_buying",
        "car_buying_dialogue_specific": "car_buying_dialogue",
        "biology_tutor_dialogue_specific": "biology_dialogue",
        "forum_reply": "forum",
        "technical_documentation": "technical",
        "code_review_comment": "code_review",
        "meeting_minutes": "meeting",
    }


def prompt_family(prompt_name: str) -> str:
    return prompt_family_dictionary().get(prompt_name, "unknown")
