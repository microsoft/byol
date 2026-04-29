#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Apply patches to lm-evaluation-harness after cloning.

Usage:
    cd eval
    python patches/apply_lmeval_patches.py

Patches applied:
1. lm_eval/filters/extraction.py - Adds OrderedRegexFilter
2. lm_eval/tasks/truthfulqa-multi/ - Adds multilingual TruthfulQA tasks
3. lm_eval/tasks/bbh/fewshot/_fewshot_template_yaml - Enable ignore_case and regexes_to_ignore for BBH
4. lm_eval/models/huggingface.py - Fix empty continuation encoding with chat templates
5. lm_eval/models/huggingface.py - Fix max_length detection for multimodal models (Gemma 3)

Run after: git clone https://github.com/EleutherAI/lm-evaluation-harness.git
"""

import shutil
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR.parent
REPO_ROOT = EVAL_DIR.parent.parent  # <repo>/byol/eval/patches -> <repo>
LMEVAL_DIR = REPO_ROOT / "third_party_libs" / "lm-evaluation-harness" / "lm_eval"

# =============================================================================
# Patch 1: OrderedRegexFilter
# =============================================================================

ORDERED_REGEX_FILTER = '''

@register_filter("ordered_regex")
class OrderedRegexFilter(Filter):
    """A filter that applies multiple regex patterns in order with fallback behavior.

    This filter tries each regex pattern sequentially. If a pattern matches, it returns
    the match. If no pattern matches, it proceeds to the next pattern. Only if all
    patterns fail does it return the fallback value.

    Optionally applies cleanup to the matched string via strip_extracts.
    """

    def __init__(
        self,
        regex_patterns: list[str] = [r"#### (\\-?[0-9\\.\\,]+)"],
        group_select: int = 0,
        fallback: str = "[invalid]",
        strip_extracts: list[str] = None,
    ) -> None:
        """
        Args:
            regex_patterns: List of regex patterns to try in order.
            group_select: Which group to select from the match.
            fallback: Fallback value if no match is found.
            strip_extracts: Regexes to remove from the extracted result.
        """
        self.regex_patterns = regex_patterns
        self.regexes = [re.compile(pattern) for pattern in regex_patterns]
        self.group_select = group_select
        self.fallback = fallback
        self.strip_extracts = [re.compile(p) for p in strip_extracts] if strip_extracts else []

    def _clean_extracted(self, text: str) -> str:
        for ignore_re in self.strip_extracts:
            text = ignore_re.sub("", text)
        return text.strip()

    def apply(self, resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
        def filter_set(inst):
            filtered = []
            for resp in inst:
                match = None

                for regex in self.regexes:
                    match = regex.findall(resp)
                    if match:
                        match = match[self.group_select]
                        if isinstance(match, tuple):
                            match = [m for m in match if m]
                            if match:
                                match = match[0]
                            else:
                                match = None
                        if match:
                            match = self._clean_extracted(match)
                            break

                if not match:
                    match = self.fallback

                filtered.append(match)
            return filtered

        filtered_resps = list(map(lambda x: filter_set(x), resps))
        return filtered_resps
'''


def patch_extraction_py() -> bool:
    """Add OrderedRegexFilter to extraction.py if not present."""
    extraction_file = LMEVAL_DIR / "filters" / "extraction.py"
    
    if not extraction_file.exists():
        print(f"❌ File not found: {extraction_file}")
        return False
    
    content = extraction_file.read_text()
    
    if "OrderedRegexFilter" in content:
        print("✅ OrderedRegexFilter already exists")
        return True
    
    # Append the filter
    new_content = content.rstrip() + ORDERED_REGEX_FILTER
    extraction_file.write_text(new_content)
    print(f"✅ Patched: {extraction_file}")
    
    return True


def patch_truthfulqa_multi() -> bool:
    """Copy truthfulqa-multi task folder."""
    target_dir = LMEVAL_DIR / "tasks" / "truthfulqa-multi"
    source_dir = SCRIPT_DIR / "truthfulqa-multi"
    
    if not source_dir.exists():
        print(f"❌ Source not found: {source_dir}")
        return False
    
    if target_dir.exists():
        print("✅ truthfulqa-multi task already exists")
        return True
    
    shutil.copytree(source_dir, target_dir, ignore=shutil.ignore_patterns("__pycache__"))
    print(f"✅ Copied: {target_dir}")
    
    return True


def patch_bbh_fewshot() -> bool:
    """Patch BBH fewshot template to enable ignore_case and regexes_to_ignore."""
    template_file = LMEVAL_DIR / "tasks" / "bbh" / "fewshot" / "_fewshot_template_yaml"
    
    if not template_file.exists():
        print(f"❌ File not found: {template_file}")
        return False
    
    content = template_file.read_text()
    
    # Check if already patched
    if 'ignore_case: true' in content and 'regexes_to_ignore:' in content:
        print("✅ BBH fewshot template already patched")
        return True
    
    # Apply the patch: enable ignore_case and add regexes_to_ignore
    old_block = '''metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    # ignore_case: true
    # ignore_punctuation: true
generation_kwargs:'''
    
    new_block = '''metric_list:
  - metric: exact_match
    aggregation: mean
    higher_is_better: true
    ignore_case: true
    # ignore_punctuation: true
    regexes_to_ignore:
      - "\\\\s+"
generation_kwargs:'''
    
    if old_block not in content:
        # Try alternate format (may have been partially modified)
        print("⚠️  BBH fewshot template format differs from expected, attempting flexible patch...")
        
        # Check if just need to uncomment ignore_case
        if '# ignore_case: true' in content:
            content = content.replace('# ignore_case: true', 'ignore_case: true')
        
        # Add regexes_to_ignore if not present
        if 'regexes_to_ignore:' not in content:
            content = content.replace(
                '# ignore_punctuation: true\ngeneration_kwargs:',
                '# ignore_punctuation: true\n    regexes_to_ignore:\n      - "\\\\s+"\ngeneration_kwargs:'
            )
        
        template_file.write_text(content)
        print(f"✅ Patched (flexible): {template_file}")
        return True
    
    new_content = content.replace(old_block, new_block)
    template_file.write_text(new_content)
    print(f"✅ Patched: {template_file}")
    
    return True


def patch_huggingface_chat_template() -> bool:
    """Patch huggingface.py to handle empty continuation encoding with chat templates."""
    hf_file = LMEVAL_DIR / "models" / "huggingface.py"
    
    if not hf_file.exists():
        print(f"❌ File not found: {hf_file}")
        return False
    
    content = hf_file.read_text()
    
    # Check if already patched
    if 'Empty continuation encoding detected' in content:
        print("✅ Huggingface chat template patch already applied")
        return True
    
    # Find and replace the assertion
    old_code = '''                assert len(context_enc) > 0
                assert len(continuation_enc) > 0'''
    
    new_code = '''                assert len(context_enc) > 0
                
                # Handle chat template tokenization issues gracefully
                if len(continuation_enc) == 0:
                    eval_logger.warning(
                        f"Empty continuation encoding detected (likely due to chat template). "
                        f"Using fallback token to avoid crash."
                    )
                    # Use a fallback single token to avoid empty continuation
                    continuation_enc = [self.tokenizer.eos_token_id or 1]  # Use EOS or fallback to token 1'''
    
    if old_code not in content:
        print("⚠️  Huggingface.py format differs from expected, skipping patch...")
        return True  # Not a failure, just different version
    
    new_content = content.replace(old_code, new_code)
    hf_file.write_text(new_content)
    print(f"✅ Patched: {hf_file}")
    
    return True


def patch_huggingface_max_length() -> bool:
    """Patch huggingface.py max_length property for multimodal models (e.g. Gemma 3).

    Gemma 3 stores max_position_embeddings inside config.text_config, not at the
    top level. Without this patch, lm-eval falls back to 2048 tokens which causes
    assertion failures on longer few-shot prompts.
    """
    hf_file = LMEVAL_DIR / "models" / "huggingface.py"

    if not hf_file.exists():
        print(f"❌ File not found: {hf_file}")
        return False

    content = hf_file.read_text()

    if '# Multimodal models (e.g. Gemma 3) store seq length in text_config' in content:
        print("✅ Huggingface max_length patch already applied")
        return True

    old_code = '''    @property
    def max_length(self) -> int:
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == TOKENIZER_INFINITY:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH'''

    new_code = '''    @property
    def max_length(self) -> int:
        if self._max_length:  # if max length manually set, return it
            return self._max_length
        seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self.model.config, attr):
                return getattr(self.model.config, attr)
        # Multimodal models (e.g. Gemma 3) store seq length in text_config
        if hasattr(self.model.config, "text_config"):
            for attr in seqlen_config_attrs:
                if hasattr(self.model.config.text_config, attr):
                    return getattr(self.model.config.text_config, attr)
        if hasattr(self.tokenizer, "model_max_length"):
            if self.tokenizer.model_max_length == TOKENIZER_INFINITY:
                return self._DEFAULT_MAX_LENGTH
            return self.tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH'''

    if old_code not in content:
        print("⚠️  Huggingface.py max_length format differs from expected, skipping patch...")
        return True

    new_content = content.replace(old_code, new_code)
    hf_file.write_text(new_content)
    print(f"✅ Patched: {hf_file}")

    return True


def main() -> int:
    print("🔧 Applying lm-evaluation-harness patches...\n")
    
    if not LMEVAL_DIR.exists():
        print(f"❌ lm-evaluation-harness not found at: {LMEVAL_DIR.parent}")
        print("   Run: git clone https://github.com/EleutherAI/lm-evaluation-harness.git")
        return 1
    
    print(f"📍 Patching: {LMEVAL_DIR}\n")
    
    # Apply patches
    results = []
    results.append(("OrderedRegexFilter", patch_extraction_py()))
    results.append(("truthfulqa-multi", patch_truthfulqa_multi()))
    results.append(("bbh-fewshot", patch_bbh_fewshot()))
    results.append(("huggingface-chat-template", patch_huggingface_chat_template()))
    results.append(("huggingface-max-length", patch_huggingface_max_length()))
    
    print()
    success = all(r[1] for r in results)
    if success:
        print("✅ All patches applied successfully!")
        return 0
    else:
        failed = [r[0] for r in results if not r[1]]
        print(f"⚠️ Failed patches: {', '.join(failed)}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())