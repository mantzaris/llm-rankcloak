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

