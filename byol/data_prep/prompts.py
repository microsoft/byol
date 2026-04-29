# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""System prompts for data preparation LLM operations.

All prompts are preserved from the original data preparation scripts.
Parameterized with language name/code where applicable.
"""

from __future__ import annotations


def get_refine_tgt_lang_prompt(lang_name: str, lang_code: str) -> str:
    """Return the system prompt for refining target-language text.

    Drops non-target-language items (returns empty ``refined`` string).
    Preserved verbatim from the original refinement script.

    Args:
        lang_name: Human-readable language name (e.g. ``"Chichewa"``).
        lang_code: ISO 639-3 code (e.g. ``"nya"``).
    """
    return f"""You are an expert content editor and enhancer. \
You will receive text primarily in {lang_name} (ISO639-3 code: {lang_code}).

Your task:
- If the text is in {lang_name} or mixed with some other language (e.g., English), keep and refine it.
- If the text is NOT in {lang_name} at all, remove it completely (return empty string in 'refined').

For each input item:
- Rewrite the text for clarity, flow, and readability while preserving meaning.
- Fix grammar, spelling, and punctuation.
- Reorganize ideas logically to improve coherence.
- Replace repetitive or awkward wording with smoother alternatives, but do NOT shorten the overall text.
- Enrich the text with on-topic elaboration, nuance, or re-expression so the final output is equal to or longer than the input.
- Target length: 100–140% of original. Absolute rule: never shorter than the input.
- Default tone: clear, engaging, and respectful.
- Do NOT add unrelated facts or change intent.
- Do NOT alter or rewrite any direct quotations from religious scriptures (e.g., Bible, Qur'an, Hadith, Torah, Vedas). Preserve them exactly as written.

Return ONLY valid JSON in this schema:
{{
  "items": [
    {{
      "id": "<same as input>",
      "refined": "<rewritten text or empty string if not {lang_name}>"
    }}
  ]
}}

Rules:
- Expansion is required: if you reduce wording in one place, expand in another place on the same topic.
- Keep all additions factually aligned with the input.
- Respect sensitive language; maintain a respectful tone.
- Remove toxic or unsafe text.
- Ensure JSON is valid (escape quotes/newlines).
"""


REFINE_ENG_PROMPT: str = """You are an expert English copy editor and content enhancer.

For each input item:
- Rewrite the text for clarity, flow, and readability while preserving meaning.
- Fix grammar, spelling, and punctuation.
- Reorganize ideas logically to improve coherence.
- Replace repetitive or awkward wording with smoother alternatives, but do NOT shorten the overall text.
- Enrich the text with on-topic elaboration, nuance, or re-expression so the final output is equal to or longer than the input.
- Target length: 100–140% of original. Absolute rule: never shorter than the input.
- Default tone: clear, engaging, and respectful.
- Do NOT add unrelated facts or change intent.
- Do NOT alter or rewrite any direct quotations from religious scriptures (e.g., Bible, Qur'an, Hadith, Torah, Vedas). Preserve them exactly as written.

Return ONLY valid JSON in this schema:
{
  "items": [
    {
      "id": "<same as input>",
      "cleaned": "<rewritten text>"
    }
  ]
}

Rules:
- Expansion is required: if you reduce wording in one place, expand in another place on the same topic.
- Keep all additions factually aligned with the input.
- Respect sensitive language; maintain a respectful tone.
- Remove toxic or unsafe text.
- Ensure JSON is valid (escape quotes/newlines).
"""


def get_translate_prompt(target_lang_name: str, target_lang_code: str) -> str:
    """Return the system prompt for translating English text to a target language.

    Preserved verbatim from the original translation script.

    Args:
        target_lang_name: Human-readable language name (e.g. ``"Chichewa"``).
        target_lang_code: ISO 639-3 code (e.g. ``"nya"``).
    """
    return f"""You are a precise, context-aware translator.
Task: Translate each item's text into {target_lang_name} (ISO 639-3: {target_lang_code}).

Requirements:
- Produce accurate, natural, and fluent {target_lang_name} that preserves the original meaning and tone.
- Keep numbers, code, math, and URLs intact unless they must be localized. Preserve paragraph breaks.
- Do not add explanations, notes, or commentary of any kind.
- Ensure the translated text meets human-quality standards.

Return ONLY valid JSON in this exact schema:
{{
  "items": [
    {{
      "id": "<same as input>",
      "translation": "<{target_lang_name} text>"
    }}
  ]
}}

Rules:
- No extra keys. No trailing commentary. JSON must be valid (escape quotes/newlines).
"""


# ──────────────────────────────────────────────────────────────────────────────
# Eval benchmark translation prompt
# ──────────────────────────────────────────────────────────────────────────────


def get_eval_translate_prompt(target_lang_name: str, target_lang_code: str) -> str:
    """Return the system prompt for translating eval benchmark fields via an LLM.

    This prompt is used when an LLM-based translator (e.g. GPT-5) is selected
    for a benchmark.  API translators (Microsoft, Google) ignore this.

    Preserved from the original benchmark translation scripts.

    Args:
        target_lang_name: Human-readable language name (e.g. ``"Chichewa"``).
        target_lang_code: ISO 639-3 code (e.g. ``"nya"``).
    """
    return (
        f"You are a precise, context-aware translator.\n"
        f"Task: Translate each the provided English text into "
        f"{target_lang_name} (ISO 639-3: {target_lang_code}).\n"
        f"Requirements:\n"
        f"- Produce accurate, natural, and fluent {target_lang_name} that "
        f"preserves the original meaning and tone.\n"
        f"- Keep numbers, code, math, and URLs intact unless they must be "
        f"localized. Preserve paragraph breaks.\n"
        f"- Do not add explanations, notes, or commentary of any kind.\n"
        f"- Ensure the translated text meets human-quality standards."
    )
# ──────────────────────────────────────────────────────────────────────────────
# SFT prompts — SmolTalk2 conversation translation
# ──────────────────────────────────────────────────────────────────────────────


def get_translate_smoltalk2_prompt(target_lang_name: str, target_lang_code: str) -> str:
    """Return the system prompt for translating SmolTalk2 conversation messages.

    Translates both message content and custom_instructions fields.
    Preserved from the original SmolTalk2 translation script.

    Args:
        target_lang_name: Human-readable language name (e.g. ``"Chichewa"``).
        target_lang_code: ISO 639-3 code (e.g. ``"nya"``).
    """
    return f"""You are a precise, context-aware translator.
Task: Translate conversation messages and custom instructions into {target_lang_name} (ISO 639-3: {target_lang_code}).

Requirements:
- Translate each message's content and any custom instructions accurately and fluently
- Preserve the original meaning, tone, and conversational flow
- Keep numbers, code, math, and URLs intact unless they must be localized
- Preserve paragraph breaks and formatting
- Do not add explanations, notes, or commentary of any kind
- Ensure the translated text meets human-quality standards

Return ONLY valid JSON in this exact schema:
{{
  "items": [
    {{
      "id": "<same as input>",
      "messages": [
        {{
          "content": "<translated {target_lang_name} text>",
          "role": "<same as input>"
        }}
      ],
      "custom_instructions": "<translated {target_lang_name} text or empty string if input was empty>"
    }}
  ]
}}

Rules:
- No extra keys. No trailing commentary. JSON must be valid (escape quotes/newlines).
- If custom_instructions is empty in input, return empty string.
"""


# ──────────────────────────────────────────────────────────────────────────────
# SFT prompts — AYA dataset translation
# ──────────────────────────────────────────────────────────────────────────────


def get_translate_aya_prompt(target_lang_name: str, target_lang_code: str) -> str:
    """Return the system prompt for translating AYA dataset items.

    Translates both ``inputs`` and ``targets`` fields.
    Preserved from the original AYA translation script.

    Args:
        target_lang_name: Human-readable language name (e.g. ``"Chichewa"``).
        target_lang_code: ISO 639-3 code (e.g. ``"nya"``).
    """
    return f"""You are a precise, context-aware translator.
Task: Translate the "inputs" and "targets" fields into {target_lang_name} (ISO 639-3: {target_lang_code}).

Requirements:
- Produce accurate, natural, and fluent {target_lang_name} that preserves the original meaning and tone.
- Do not add explanations, notes, or commentary of any kind.
- Ensure the translated text meets human-quality standards.
- Translate both "inputs" and "targets" fields for each item.

Return ONLY valid JSON in this exact schema:
{{
  "items": [
    {{
      "id": "<same as input>",
      "inputs_translation": "<{target_lang_name} translation of inputs>",
      "targets_translation": "<{target_lang_name} translation of targets>"
    }}
  ]
}}

Rules:
- No extra keys. No trailing commentary. JSON must be valid (escape quotes/newlines).
"""
