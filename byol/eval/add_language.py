# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Generate eval task YAML files and benchmark configs for a new language.

Integrates with the BYOL eval framework to automatically create all the
lm-evaluation-harness task definitions needed to evaluate models in a
new target language.

Usage::

    python -m byol.eval add-language --lang gug --name Guarani
    python -m byol.eval add-language --lang gug --name Guarani --dry-run
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from .constants import EVAL_PACKAGE_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

# =============================================================================
# Language-Specific Prompt Translations
# =============================================================================
# Keys: question, answer, step_by_step_answer, the_answer_is,
#        cause, effect (XCOPA connectors),
#        nli_entailment/neutral/contradiction/right/or (XNLI),
#        arc_chat_instruction, arc_chat_gen_prefix,
#        flores_field
#
# For unknown languages, TODO placeholders are generated.
# =============================================================================

LANG_WORDS: dict[str, dict[str, str | int | None]] = {
    "nya": {
        "question": "Funso",
        "answer": "Yankho",
        "step_by_step_answer": "Mayankho Paso ndi Paso",
        "the_answer_is": "Yankho ndi",
        "cause": "chifukwa",
        "effect": "kotero",
        "nli_entailment": "Inde",
        "nli_neutral": "Komanso",
        "nli_contradiction": "ayi",
        "nli_right": "zabwino?",
        "nli_or": "kapena?",
        "arc_chat_instruction": (
            "Tengani funso lotsatirali ndi mayankho anayi (A, B, C ndi D), "
            "sankhani yankho labwino kwambiri.\n"
            "{question_label}: {{question.strip()}}\n"
            "A. {{choices.text[0]}}\n"
            "B. {{choices.text[1]}}\n"
            "C. {{choices.text[2]}}{% if choices.text|length > 3 %}\n"
            "D. {{choices.text[3]}}{% endif %}\n"
            "Yankho lanu liyenera kumalizidwa ndi "
            '"Yankho labwino ndi [the_answer_letter]" '
            "komwe [the_answer_letter] ndi limodzi mwa A, B, C kapena D."
        ),
        "arc_chat_gen_prefix": "Mayankho abwino ndi",
        "mgsm_answer_offset": 24,
        "flores_field": "sentence_nya_Latn",
    },
    "mri": {
        "question": "Pātai",
        "answer": "Whakautu",
        "step_by_step_answer": "Whakautu Hipanga ki te Hipanga",
        "the_answer_is": "Ko te whakautu ko",
        "cause": "take",
        "effect": "paanga",
        "nli_entailment": "āe",
        "nli_neutral": "anō",
        "nli_contradiction": "kāo",
        "nli_right": "nē?",
        "nli_or": "rānei?",
        "arc_chat_instruction": (
            "He pātai kei raro me ngā whakautu e whā (A, B, C me D), "
            "kōwhiria te whakautu pai rawa.\n"
            "{question_label}: {{question.strip()}}\n"
            "A. {{choices.text[0]}}\n"
            "B. {{choices.text[1]}}\n"
            "C. {{choices.text[2]}}{% if choices.text|length > 3 %}\n"
            "D. {{choices.text[3]}}{% endif %}\n"
            'Ko tō whakautu me mutu ki te "Ko te whakautu pai rawa ko '
            '[the_answer_letter]" ā ko [the_answer_letter] ko tētahi o A, B, C, D rānei.'
        ),
        "arc_chat_gen_prefix": "Ko te whakautu pai rawa ko",
        "mgsm_answer_offset": 31,
        "flores_field": "sentence_mri_Latn",
    },
    "gug": {
        "question": "Porandu",
        "answer": "Mbohovái",
        "step_by_step_answer": "Mbohovái peteĩteĩ",
        "the_answer_is": "Mbohovái ha'e",
        "cause": "haguére",
        "effect": "upéicha",
        "nli_entailment": "héẽ",
        "nli_neutral": "avei",
        "nli_contradiction": "nahániri",
        "nli_right": "añetehápe",
        "nli_or": "térã",
        "arc_chat_instruction": (
            "Peteĩ porandu oĩ ko'ápe irundy mbohovái ndive (A, B, C ha D), "
            "eiporavo mbohovái iporãvéva.\n"
            "{question_label}: {{question.strip()}}\n"
            "A. {{choices.text[0]}}\n"
            "B. {{choices.text[1]}}\n"
            "C. {{choices.text[2]}}{% if choices.text|length > 3 %}\n"
            "D. {{choices.text[3]}}{% endif %}\n"
            "Nde mbohovái oñembohapéva ha'evaʼerã "
            '"Mbohovái iporãvéva ha\'e [the_answer_letter]" '
            "ha [the_answer_letter] ha'e peteĩ A, B, C térã D hérava."
        ),
        "arc_chat_gen_prefix": "Mbohovái iporãvéva ha'e",
        "mgsm_answer_offset": None,  # auto-computed
        "flores_field": "sentence_grn_Latn",
    },
}


def _get_words(lang_code: str) -> dict[str, str | int | None]:
    """Get language-specific words, with TODO placeholders for unknown languages."""
    if lang_code in LANG_WORDS:
        return dict(LANG_WORDS[lang_code])
    return {
        "question": "TODO_QUESTION",
        "answer": "TODO_ANSWER",
        "step_by_step_answer": "TODO_STEP_BY_STEP_ANSWER",
        "the_answer_is": "TODO_THE_ANSWER_IS",
        "cause": "TODO_CAUSE",
        "effect": "TODO_EFFECT",
        "nli_entailment": "TODO_YES",
        "nli_neutral": "TODO_ALSO",
        "nli_contradiction": "TODO_NO",
        "nli_right": "TODO_RIGHT",
        "nli_or": "TODO_OR",
        "arc_chat_instruction": "TODO",
        "arc_chat_gen_prefix": "TODO",
        "mgsm_answer_offset": None,
        "flores_field": f"sentence_{lang_code}_Latn",
    }


def _compute_mgsm_offset(words: dict) -> int:
    """Compute MGSM answer offset from step_by_step_answer string."""
    if words.get("mgsm_answer_offset") is not None:
        return words["mgsm_answer_offset"]
    return len(str(words["step_by_step_answer"])) + 2  # ": " suffix


# =============================================================================
# Task Generator
# =============================================================================


class TaskGenerator:
    """Generate YAML configs for a new language.

    Creates task definition files under ``byol/eval/tasks/`` and benchmark
    configs under ``configs/eval/``.  Aligns data-file paths with the naming
    convention used by ``byol.data_prep --stage eval``.
    """

    def __init__(
        self,
        lang_code: str,
        lang_name: str,
        data_dir_name: str,
        *,
        dry_run: bool = False,
    ) -> None:
        self.lang = lang_code
        self.lang_name = lang_name
        self.data_dir = data_dir_name
        self.tasks_root = EVAL_PACKAGE_DIR / "tasks"
        self.configs_root = REPO_ROOT / "configs" / "eval"
        self.words = _get_words(lang_code)
        self.dry_run = dry_run
        self.created_files: list[str] = []
        self.warnings: list[str] = []

    # ── Helpers ──────────────────────────────────────────────────────────

    def _write(self, filepath: Path, content: str) -> None:
        if self.dry_run:
            logger.info(f"  [DRY-RUN] Would create: {filepath}")
            self.created_files.append(str(filepath))
            return
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content, encoding="utf-8")
        self.created_files.append(str(filepath))

    def _data_file(
        self, prefix: str, split: str, suffix: str = "microsoft_translated"
    ) -> str:
        """Data file path relative to repo root (matches data_prep output)."""
        return (
            f"data/{self.lang}/eval/"
            f"{prefix}_{split}_english2{self.data_dir}_{suffix}.jsonl"
        )

    def _resolve_data_file(
        self, prefix: str, split: str, default_suffix: str = "microsoft_translated"
    ) -> str:
        """Find the actual data file, trying multiple suffixes."""
        base = REPO_ROOT / "data" / self.lang / "eval"
        pattern = f"{prefix}_{split}_english2{self.data_dir}_"
        if base.exists():
            for f in sorted(base.iterdir()):
                if f.name.startswith(pattern) and f.name.endswith(".jsonl"):
                    return f"data/{self.lang}/eval/{f.name}"
        return self._data_file(prefix, split, default_suffix)

    # ── Task Generators ──────────────────────────────────────────────────

    def gen_arc(self) -> None:
        """ARC easy + challenge + challenge_chat."""
        w = self.words
        q, a = w["question"], w["answer"]
        base = self.tasks_root / "arc"

        self._write(
            base / f"arc_easy_{self.lang}.yaml",
            f'include: arc_common_yaml\ntask: arc_easy_{self.lang}\n'
            f'dataset_name: {self.lang}\ndataset_kwargs:\n  data_files:\n'
            f'    validation: {self._resolve_data_file("ai2_arc_easy", "validation")}\n'
            f'    test: {self._resolve_data_file("ai2_arc_easy", "test")}\n'
            f'doc_to_text: "{q}: {{{{question}}}}\\n{a}:"\n',
        )

        self._write(
            base / f"arc_challenge_{self.lang}.yaml",
            f'include: arc_common_yaml\ntask: arc_challenge_{self.lang}\n'
            f'dataset_name: {self.lang}\ndataset_kwargs:\n  data_files:\n'
            f'    validation: {self._resolve_data_file("ai2_arc_hard", "validation")}\n'
            f'    test: {self._resolve_data_file("ai2_arc_hard", "test")}\n'
            f'doc_to_text: "{q}: {{{{question}}}}\\n{a}:"\n',
        )

        chat_inst = str(w["arc_chat_instruction"]).replace("{question_label}", str(q))
        gen_prefix = w["arc_chat_gen_prefix"]
        # Escape single quotes for YAML single-quoted strings
        chat_inst_yaml = str(chat_inst).replace("'", "''")
        gen_prefix_yaml = str(gen_prefix).replace("'", "''")
        self._write(
            base / f"arc_challenge_chat_{self.lang}.yaml",
            f"tag:\n  - ai2_arc\ntask: arc_challenge_chat_{self.lang}\n"
            f"dataset_path: json\ndataset_name: {self.lang}\n"
            f"output_type: generate_until\ntraining_split: null\n"
            f"validation_split: validation\ntest_split: test\nfewshot_split: validation\n"
            f"dataset_kwargs:\n  data_files:\n"
            f'    validation: {self._resolve_data_file("ai2_arc_hard", "validation")}\n'
            f'    test: {self._resolve_data_file("ai2_arc_hard", "test")}\n'
            f"doc_to_text: '{chat_inst_yaml}'\n"
            f"gen_prefix: '{gen_prefix_yaml}'\n"
            f'fewshot_delimiter: "\\n\\n"\n'
            f"doc_to_target: \"{{{{ 'ABCD'[answerKey|int - 1] if answerKey|string in '1234' else answerKey }}}}\"\n"
            f"num_fewshot: 0\ngeneration_kwargs:\n  max_gen_toks: 100\n  until:\n"
            f'    - "\\n\\n"\n    - "."\n'
            f"metric_list:\n  - metric: exact_match\n    aggregation: mean\n"
            f"    higher_is_better: true\n    ignore_case: true\n    ignore_punctuation: true\n"
            f"filter_list:\n  - name: remove_whitespace\n    filter:\n"
            f"      - function: remove_whitespace\n      - function: take_first\n"
            f"metadata:\n  version: 1.0\n",
        )

    def gen_hellaswag(self) -> None:
        base = self.tasks_root / "hellaswag"
        self._write(
            base / f"hellaswag_{self.lang}.yaml",
            f"tag:\n  - multiple_choice\ntask: hellaswag_{self.lang}\n"
            f"dataset_path: json\ndataset_kwargs:\n  data_files:\n"
            f'    validation: {self._resolve_data_file("hellaswag", "validation")}\n'
            f"  trust_remote_code: true\noutput_type: multiple_choice\n"
            f"training_split: null\nvalidation_split: validation\ntest_split: null\n"
            f'process_docs: !function utils.process_docs\ndoc_to_text: "{{{{query}}}}"\n'
            f'doc_to_target: "{{{{gold}}}}"\ndoc_to_choice: "{{{{choices}}}}"\n'
            f"metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: acc_norm\n    aggregation: mean\n    higher_is_better: true\n"
            f"metadata:\n  version: 1.0\n",
        )

    def gen_piqa(self) -> None:
        w = self.words
        q, a = w["question"], w["answer"]
        base = self.tasks_root / "piqa"
        self._write(
            base / f"piqa_{self.lang}.yaml",
            f"task: piqa_{self.lang}\ndataset_path: json\ndataset_kwargs:\n  data_files:\n"
            f'    train: {self._resolve_data_file("piqa", "train")}\n'
            f'    validation: {self._resolve_data_file("piqa", "validation")}\n'
            f"output_type: multiple_choice\ntraining_split: train\n"
            f"validation_split: validation\ntest_split: null\n"
            f'doc_to_text: "{q}: {{{{goal}}}}\\n{a}:"\n'
            f"doc_to_target: label\n"
            f'doc_to_choice: "{{{{[sol1, sol2]}}}}"\n'
            f"should_decontaminate: true\ndoc_to_decontamination_query: goal\n"
            f"metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: acc_norm\n    aggregation: mean\n    higher_is_better: true\n"
            f"metadata:\n  version: 1.0\n",
        )

    def gen_mgsm(self) -> None:
        """MGSM direct, en_cot, and native_cot."""
        w = self.words
        q, a = w["question"], w["answer"]
        sbs = w["step_by_step_answer"]
        the_answer = w["the_answer_is"]
        offset = _compute_mgsm_offset(w)

        # direct
        base = self.tasks_root / "MGSM" / "direct"
        self._write(
            base / f"mgsm_direct_{self.lang}.yaml",
            f"include: direct_yaml\ndataset_name: {self.lang}\ndataset_kwargs:\n  data_files:\n"
            f'    train: {self._resolve_data_file("mgsm", "train")}\n'
            f'    test:  {self._resolve_data_file("mgsm", "test")}\n'
            f"doc_to_target: '{{% if answer is not none %}}{{{{answer[{offset}:]}}}}"
            f"{{% else %}}{{{{answer_number|string}}}}{{% endif %}}'\n"
            f"doc_to_text: '{{% if answer is not none %}}{{{{question+\"\\n{a}:\"}}}}"
            f"{{% else %}}{{{{\"{q}: \"+question+\"\\n{a}:\"}}}}{{% endif %}}'\n"
            f"generation_kwargs:\n  do_sample: false\n  until:\n  - '{q}:'\n  - </s>\n  - <|im_end|>\n"
            f"task: mgsm_direct_{self.lang}\n",
        )

        # en_cot
        base = self.tasks_root / "MGSM" / "en_cot"
        self._write(
            base / f"mgsm_en_cot_{self.lang}.yaml",
            f"include: cot_yaml\ndataset_name: {self.lang}\ndataset_kwargs:\n  data_files:\n"
            f'    train: {self._resolve_data_file("mgsm", "train")}\n'
            f'    test:  {self._resolve_data_file("mgsm", "test")}\n'
            f"doc_to_target: '{{% if answer is not none %}}{{{{answer[{offset}:]}}}}"
            f"{{% else %}}{{{{answer_number|string}}}}{{% endif %}}'\n"
            f'doc_to_text: \'{{% if answer is not none %}}{{{{question+"\\nStep-by-Step Answer:"}}}}'
            f'{{% else %}}{{{{\"{q}: \"+question+"\\nStep-by-Step Answer:"}}}}{{% endif %}}\'\n'
            f"generation_kwargs:\n  do_sample: false\n  until:\n  - '{q}:'\n  - </s>\n  - <|im_end|>\n"
            f"task: mgsm_en_cot_{self.lang}\n",
        )

        # native_cot
        base = self.tasks_root / "MGSM" / "native_cot"
        self._write(
            base / f"mgsm_native_cot_{self.lang}.yaml",
            f"include: cot_yaml\ndataset_name: {self.lang}\ndataset_kwargs:\n  data_files:\n"
            f'    train: {self._resolve_data_file("mgsm", "train")}\n'
            f'    test:  {self._resolve_data_file("mgsm", "test")}\n'
            f"doc_to_target: '{{% if answer is not none %}}{{{{answer[{offset}:]}}}}"
            f"{{% else %}}{{{{answer_number|string}}}}{{% endif %}}'\n"
            f"doc_to_text: '{{% if answer is not none %}}{{{{question+\"\\n{sbs}:\"}}}}"
            f"{{% else %}}{{{{\"{q}: \"+question+\"\\n{sbs}:\"}}}}{{% endif %}}'\n"
            f"filter_list:\n- filter:\n  - function: regex\n    regex_pattern: {the_answer} (\\-?[0-9\\.\\,]+)\n"
            f"  - function: take_first\n  name: strict-match\n"
            f"- filter:\n  - function: regex\n    group_select: -1\n"
            f"    regex_pattern: (-?[$0-9.,]{{{{2,}}}})|(-?[0-9]+)\n"
            f"  - function: take_first\n  name: flexible-extract\n"
            f"generation_kwargs:\n  do_sample: false\n  until:\n  - '{q}:'\n  - </s>\n  - <|im_end|>\n"
            f"task: mgsm_native_cot_{self.lang}\n",
        )

    def gen_flores(self) -> None:
        flores_field = self.words["flores_field"]
        base = self.tasks_root / "flores"

        self._write(
            base / f"flores_en_{self.lang}.yaml",
            f"include: _flores_common_yaml\ntask: flores_en_{self.lang}\n"
            f"doc_to_text: 'You are a translation expert, provide translation of the "
            f"input sentence without any additional text or reasoning\n\n"
            f"  English sentence: {{{{sentence_eng_Latn}}}}\n\n"
            f"  {self.lang_name} sentence:'\n"
            f"doc_to_target: '{{{{{flores_field}}}}}'\n",
        )

        self._write(
            base / f"flores_{self.lang}_en.yaml",
            f"include: _flores_common_yaml\ntask: flores_{self.lang}_en\n"
            f"doc_to_text: 'You are a translation expert, provide translation of the "
            f"input sentence without any additional text or reasoning\n\n"
            f"  {self.lang_name} sentence: {{{{{flores_field}}}}}\n\n"
            f"  English sentence:'\n"
            f"doc_to_target: '{{{{sentence_eng_Latn}}}}'\n",
        )

        self._write(
            base / f"flores_en_{self.lang}_bidirectional.yaml",
            f"task: flores_en_{self.lang}_bidirectional\ntask_list:\n"
            f"  - flores_en_{self.lang}\n  - flores_{self.lang}_en\n",
        )

    def gen_xcopa(self) -> None:
        base = self.tasks_root / "xcopa"
        func_name = f"doc_to_text_{self.lang}"
        self._write(
            base / f"xcopa_{self.lang}.yaml",
            f"task: xcopa_{self.lang}\ndataset_path: json\ndataset_kwargs:\n  data_files:\n"
            f'    validation: {self._resolve_data_file("copa", "validation")}\n'
            f'    test: {self._resolve_data_file("copa", "test")}\n'
            f"output_type: multiple_choice\nvalidation_split: validation\ntest_split: test\n"
            f"doc_to_text: !function utils.{func_name}\ndoc_to_target: label\n"
            f"doc_to_choice: !function utils.doc_to_choice\nmetric_list:\n  - metric: acc\n"
            f"metadata:\n  version: 1.0\n",
        )

        # Update utils.py with new connector
        self._update_xcopa_utils()

    def _update_xcopa_utils(self) -> None:
        utils_path = self.tasks_root / "xcopa" / "utils.py"
        if not utils_path.exists():
            self.warnings.append(f"xcopa/utils.py not found at {utils_path}")
            return

        func_name = f"doc_to_text_{self.lang}"
        content = utils_path.read_text(encoding="utf-8")
        if func_name in content:
            logger.info(f"  xcopa/utils.py already has {func_name}, skipping")
            return

        w = self.words
        snippet = (
            f"\n\n{func_name} = partial(\n"
            f"    doc_to_text,\n"
            f"    connector={{\n"
            f'        "cause": "{w["cause"]}",\n'
            f'        "effect": "{w["effect"]}",\n'
            f"    }},\n"
            f")\n"
        )
        if self.dry_run:
            logger.info(f"  [DRY-RUN] Would append to xcopa/utils.py: {func_name}")
            self.created_files.append(f"{utils_path} (appended {func_name})")
            return
        with open(utils_path, "a", encoding="utf-8") as f:
            f.write(snippet)
        self.created_files.append(f"{utils_path} (appended {func_name})")

    def gen_xnli(self) -> None:
        w = self.words
        right = w["nli_right"]
        yes = w["nli_entailment"]
        or_word = w["nli_or"]
        also = w["nli_neutral"]
        no = w["nli_contradiction"]
        base = self.tasks_root / "xnli"
        self._write(
            base / f"xnli_{self.lang}.yaml",
            f"dataset_name: {self.lang}\ndataset_kwargs:\n  data_files:\n"
            f'    train: {self._resolve_data_file("xnli", "validation")}\n'
            f'    validation: {self._resolve_data_file("xnli", "test")}\n'
            f"doc_to_choice: '{{{{[premise+\", {right} {yes}, \"+hypothesis,"
            f"premise+\", {or_word} {also}, \"+hypothesis,"
            f'premise+",\\\n  \\ {right} {no}, "+hypothesis]}}}}\'\n'
            f"doc_to_text: ''\ninclude: xnli_common_yaml\ntask: xnli_{self.lang}\n",
        )

    def gen_xstorycloze(self) -> None:
        base = self.tasks_root / "xstorycloze"
        self._write(
            base / f"xstorycloze_{self.lang}.yaml",
            f"include: xstorycloze_common_yaml\ntask: xstorycloze_{self.lang}\n"
            f"dataset_name: {self.lang}\ndataset_kwargs:\n  data_files:\n"
            f'    train: {self._resolve_data_file("xstory_cloze", "train")}\n'
            f'    eval:  {self._resolve_data_file("xstory_cloze", "eval")}\n',
        )

    def gen_xwinograd(self) -> None:
        base = self.tasks_root / "xwinograd"
        self._write(
            base / f"xwinograd_{self.lang}.yaml",
            f"dataset_name: {self.lang}\ninclude: xwinograd_common_yaml\n"
            f"task: xwinograd_{self.lang}\ndataset_kwargs:\n  data_files:\n"
            f"    test: data/{self.lang}/eval/xwinograd_aligned_{self.data_dir}_1000.jsonl\n",
        )

    def gen_truthfulqa(self) -> None:
        base = self.tasks_root / "truthfulqa-multi-chichewa"

        # mc_common
        self._write(
            base / f"truthfulqa-multi_mc_common_{self.lang}.yaml",
            f"tag:\n  - truthfulqa-multi\ndataset_path: json\ndataset_kwargs:\n  data_files:\n"
            f'    validation: {self._resolve_data_file("HiTZ-truthfulqa-multi", "validation")}\n'
            f'    train: {self._resolve_data_file("HiTZ-truthfulqa-multi", "train")}\n'
            f"output_type: multiple_choice\ntraining_split: train\nvalidation_split: validation\n"
            f"test_split: null\nfewshot_split: train\nfewshot_config:\n  sampler: first_n\n"
            f"doc_to_target: 0\n"
            f'doc_to_choice: "{{{{mc1_targets.choices}}}}"\n'
            f"should_decontaminate: True\ndoc_to_decontamination_query: question\n"
            f"metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"metadata:\n  version: 2.0\n"
            f"doc_to_text: \"{{{{'Q: ' + question + '\\nA:'}}}}\"",
        )
        self._write(
            base / f"truthfulqa-multi_mc1_{self.lang}.yaml",
            f"include: truthfulqa-multi_mc_common_{self.lang}.yaml\n"
            f"task: truthfulqa-multi_mc1_{self.lang}\n",
        )
        self._write(
            base / f"truthfulqa-multi_mc2_{self.lang}.yaml",
            f"include: truthfulqa-multi_mc1_{self.lang}.yaml\n"
            f"task: truthfulqa-multi_mc2_{self.lang}\n"
            f'doc_to_choice: "{{{{mc2_targets.choices}}}}"\n'
            f"process_results: !function utils.process_results_mc2\n"
            f"should_decontaminate: True\ndoc_to_decontamination_query: question\n"
            f"metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"metadata:\n  version: 2.0\n",
        )

        # gen_common
        self._write(
            base / f"truthfulqa-multi_gen_common_{self.lang}.yaml",
            f"tag:\n  - truthfulqa_multi\ndataset_path: json\ndataset_kwargs:\n  data_files:\n"
            f'    validation: {self._resolve_data_file("HiTZ-truthfulqa-multi", "validation")}\n'
            f'    train: {self._resolve_data_file("HiTZ-truthfulqa-multi", "train")}\n'
            f"output_type: generate_until\ngeneration_kwargs:\n  until:\n"
            f'    - "!\\n\\n"\n    - "Q:"\n    - ".\\n\\n"\n'
            f"training_split: train\nvalidation_split: validation\ntest_split: null\n"
            f"doc_to_target: \"{{{{'A: ' + best_answer}}}}\"\n"
            f"fewshot_split: train\nfewshot_config:\n  sampler: first_n\n"
            f"process_docs: !function utils.process_docs_gen\n"
            f"process_results: !function utils.process_results_gen\n"
            f"doc_to_text: \"{{{{'Q: ' + question}}}}\"\n"
            f"should_decontaminate: True\ndoc_to_decontamination_query: question\n"
            f"metric_list:\n"
            f"  - metric: bleu_max\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: bleu_acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: bleu_diff\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rouge1_max\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rouge1_acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rouge1_diff\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rouge2_max\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rouge2_acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rouge2_diff\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rougeL_max\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rougeL_acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"  - metric: rougeL_diff\n    aggregation: mean\n    higher_is_better: true\n"
            f"metadata:\n  version: 2.0\n",
        )
        self._write(
            base / f"truthfulqa-multi_gen_{self.lang}.yaml",
            f"include: truthfulqa-multi_gen_common_{self.lang}.yaml\n"
            f"task: truthfulqa-multi_gen_{self.lang}\n",
        )

    def gen_global_mmlu(self) -> None:
        """Global-MMLU-Lite MC and gen_0shot variants."""
        categories = [
            "business", "humanities", "medical", "other", "stem", "social_sciences",
        ]

        # --- MC variant ---
        mc_base = self.tasks_root / "Global-MMLU-Lite" / self.lang
        self._write(
            mc_base / f"_{self.lang}_template.yaml",
            f"dataset_path: json\ndataset_name: null\ntest_split: test\nfewshot_split: dev\n"
            f"fewshot_config:\n  sampler: default\noutput_type: multiple_choice\n"
            f'doc_to_text: "{{{{question.strip()}}}}\\nA. {{{{option_a}}}}\\n'
            f'B. {{{{option_b}}}}\\nC. {{{{option_c}}}}\\nD. {{{{option_d}}}}\\nAnswer:"\n'
            f'doc_to_choice: ["A", "B", "C", "D"]\ndoc_to_target: answer\n'
            f"metric_list:\n  - metric: acc\n    aggregation: mean\n    higher_is_better: true\n"
            f"metadata:\n  version: 0.0\ndataset_kwargs:\n  data_files:\n"
            f"    dev: {self._resolve_data_file('mmlu_lite', 'dev')}\n"
            f"    test: {self._resolve_data_file('mmlu_lite', 'test')}\n",
        )
        for cat in categories:
            self._write(
                mc_base / f"global_mmlu_{self.lang}_{cat}.yaml",
                f"include: _{self.lang}_template.yaml\n"
                f"process_docs: !function utils.process_{cat}\n"
                f"task: global_mmlu_{self.lang}_{cat}\n",
            )

        task_list = "\n".join(f"  - global_mmlu_{self.lang}_{cat}" for cat in categories)
        self._write(
            mc_base / f"global_mmlu_{self.lang}.yaml",
            f"group: global_mmlu_{self.lang}\ntask:\n{task_list}\n"
            f"aggregate_metric_list:\n  - metric: acc\n    weight_by_size: true\n"
            f"metadata:\n  version: 0.0\n",
        )

        # Copy utils.py from ny/ if it exists
        src_utils = self.tasks_root / "Global-MMLU-Lite" / "ny" / "utils.py"
        dst_utils = mc_base / "utils.py"
        if not self.dry_run and src_utils.exists() and not dst_utils.exists():
            shutil.copy2(src_utils, dst_utils)
            self.created_files.append(str(dst_utils))
        elif self.dry_run and src_utils.exists():
            logger.info(f"  [DRY-RUN] Would copy utils.py to {mc_base}/")
            self.created_files.append(str(dst_utils))

        # --- gen_0shot variant ---
        gen_base = self.tasks_root / "Global-MMLU-Lite" / "gen_0shot" / self.lang
        self._write(
            gen_base / f"_{self.lang}_template_with_internal_data.yaml",
            f"dataset_path: json\ntest_split: test\nfewshot_split: null\n"
            f"dataset_kwargs:\n  data_files:\n"
            f"    test: {self._resolve_data_file('mmlu_lite', 'test')}\n"
            f'output_type: generate_until\ndescription: "Your response should be concise, '
            f"and you should end with \\\"The answer is (X)\\\", where X is the correct "
            f'letter choice.\\n\\n"\nnum_fewshot: 0\n'
            f"doc_to_text: |-\n  Question: {{{{ question | trim }}}}\n  Choices:\n"
            f"  (A) {{{{ option_a | trim }}}}\n  (B) {{{{ option_b | trim }}}}\n"
            f"  (C) {{{{ option_c | trim }}}}\n  (D) {{{{ option_d | trim }}}}\n  Answer:\n"
            f"doc_to_target: answer\nfilter_list:\n  - name: \"extract-answer\"\n    filter:\n"
            f"      - function: \"ordered_regex\"\n        regex_patterns:\n"
            f"          - '[Tt]he answer is\\s+\\(?([A-D])\\)?\\b'\n"
            f"          - '[Tt]he answer is:\\s+\\(?([A-D])\\)?\\b'\n"
            f"          - '[Tt]he final answer is\\s+\\(?([A-D])\\)?\\b'\n"
            f"          - 'answer is\\s+\\(?([A-D])\\)?\\b'\n"
            f"          - '[Tt]he final answer:\\s+\\(?([A-D])\\)?\\b'\n"
            f"          - '\\(([A-D])\\)'\n"
            f"          - \"([A-D])\"\n"
            f"        group_select: -1\n      - function: \"take_first\"\n"
            f"generation_kwargs:\n  until:\n    - \"</s>\"\n    - \"Question:\"\n"
            f"    - \"<|im_end|>\"\n  temperature: 0.0\n"
            f"metric_list:\n  - metric: exact_match\n    aggregation: mean\n    higher_is_better: true\n"
            f"metadata:\n  version: 1.0\n",
        )
        for cat in categories:
            self._write(
                gen_base / f"global_mmlu_{self.lang}_{cat}.yaml",
                f"include: _{self.lang}_template_with_internal_data.yaml\n"
                f"dataset_name: {self.lang}\n"
                f"process_docs: !function ../utils.process_{cat}\n"
                f"task: global_mmlu_{self.lang}_{cat}_gen_0shot\n",
            )
        gen_task_list = "\n".join(
            f"  - global_mmlu_{self.lang}_{cat}_gen_0shot" for cat in categories
        )
        self._write(
            gen_base / f"_global_mmlu_{self.lang}.yaml",
            f"group: global_mmlu_{self.lang}_gen_0shot\ntask:\n{gen_task_list}\n"
            f"aggregate_metric_list:\n  - metric: exact_match\n"
            f"    filter_list: \"extract-answer\"\nmetadata:\n  version: 0.0\n",
        )

    # ── Benchmark Configs ────────────────────────────────────────────────

    def gen_benchmark_configs(self) -> None:
        """Generate configs/eval/benchmark_{base,instruct}_{lang}.yaml."""
        lang = self.lang

        # Base config (few-shot)
        base_tasks = [
            (f"global_mmlu_{lang}", 5, "auto:4"),
            (f"xwinograd_{lang}", 0, "auto:4"),
            (f"xcopa_{lang}", 5, "auto:4"),
            (f"xnli_{lang}", 5, "auto:2"),
            (f"mgsm_direct_{lang}", 8, "auto:4"),
            (f"hellaswag_{lang}", 0, "auto:4"),
            (f"arc_challenge_{lang}", 25, "auto:2"),
            (f"arc_easy_{lang}", 0, "auto:4"),
            (f"piqa_{lang}", 0, "auto:4"),
            (f"xstorycloze_{lang}", 5, "auto:1"),
            (f"flores_en_{lang}", 1, "auto:4"),
            (f"flores_{lang}_en", 1, "auto:4"),
            (f"truthfulqa-multi_mc1_{lang},truthfulqa-multi_mc2_{lang},truthfulqa-multi_gen_{lang}", 6, "auto:4"),
        ]
        lines = [
            f"# Base Model Evaluation - {self.lang_name} ({lang})",
            f"# Few-shot evaluation (no chat template)\n",
            "evaluation:",
            '  results_dir: "results/eval"',
            '  gpus: "0"',
            "  batch_size: auto:4\n",
            "models:",
            '  - name: "model"',
            '    path: "google/gemma-3-4b-pt"',
            '    dtype: "bfloat16"',
            "    trust_remote_code: true\n",
            "lm_eval:",
            '  include_path: "tasks"',
            "  log_samples: false\n",
            "tasks:",
        ]
        for name, fewshot, batch in base_tasks:
            lines.append(f'  - name: "{name}"')
            lines.append(f"    num_fewshot: {fewshot}")
            lines.append(f"    enabled: true")
            if batch != "auto:4":
                lines.append(f"    batch_size: {batch}")
            lines.append("")
        self._write(
            self.configs_root / f"benchmark_base_{lang}.yaml",
            "\n".join(lines) + "\n",
        )

        # Instruct config (0-shot + chat template)
        instruct_tasks = [
            (f"global_mmlu_{lang}_gen_0shot", None, "auto:4"),
            (f"xwinograd_{lang}", 0, "auto:4"),
            (f"xcopa_{lang}", 0, "auto:4"),
            (f"xnli_{lang}", 0, "auto:2"),
            (f"mgsm_direct_{lang}", 0, "auto:4"),
            (f"hellaswag_{lang}", 0, "auto:4"),
            (f"arc_challenge_{lang}", 0, "auto:2"),
            (f"arc_easy_{lang}", 0, "auto:4"),
            (f"arc_challenge_chat_{lang}", 0, "auto:4"),
            (f"piqa_{lang}", 0, "auto:4"),
            (f"xstorycloze_{lang}", 0, "auto:1"),
            (f"flores_en_{lang}", 0, "auto:4"),
            (f"flores_{lang}_en", 0, "auto:4"),
            (f"truthfulqa-multi_mc1_{lang},truthfulqa-multi_mc2_{lang}", 6, "auto:4"),
            (f"truthfulqa-multi_gen_{lang}", 6, "auto:4"),
        ]
        lines = [
            f"# Instruct Model Evaluation - {self.lang_name} ({lang})",
            f"# 0-shot evaluation with chat template\n",
            "evaluation:",
            '  results_dir: "results/eval"',
            '  gpus: "0"',
            "  batch_size: auto:4",
            "  apply_chat_template: true\n",
            "models:",
            '  - name: "model"',
            '    path: "google/gemma-3-4b-it"',
            '    dtype: "bfloat16"',
            "    trust_remote_code: true\n",
            "lm_eval:",
            '  include_path: "tasks"',
            "  log_samples: false\n",
            "tasks:",
        ]
        for name, fewshot, batch in instruct_tasks:
            lines.append(f'  - name: "{name}"')
            lines.append(f"    enabled: true")
            lines.append(f"    apply_chat_template: true")
            if fewshot is not None:
                lines.append(f"    num_fewshot: {fewshot}")
            if batch != "auto:4":
                lines.append(f"    batch_size: {batch}")
            lines.append("")
        self._write(
            self.configs_root / f"benchmark_instruct_{lang}.yaml",
            "\n".join(lines) + "\n",
        )

    # ── Run All ──────────────────────────────────────────────────────────

    def generate_all(self) -> None:
        """Generate all task YAMLs and benchmark configs."""
        logger.info(f"Generating eval tasks for {self.lang_name} ({self.lang})...")
        logger.info(f"  Data dir: data/{self.lang}/eval/")
        logger.info(f"  Tasks dir: {self.tasks_root}")
        logger.info(f"  Configs dir: {self.configs_root}")
        if self.dry_run:
            logger.info("  Mode: DRY RUN\n")
        else:
            logger.info("")

        self.gen_arc()
        self.gen_hellaswag()
        self.gen_piqa()
        self.gen_mgsm()
        self.gen_flores()
        self.gen_xcopa()
        self.gen_xnli()
        self.gen_xstorycloze()
        self.gen_xwinograd()
        self.gen_truthfulqa()
        self.gen_global_mmlu()
        self.gen_benchmark_configs()

        # ── Summary ──────────────────────────────────────────────────────
        logger.info("=" * 60)
        logger.info(
            f"{'[DRY-RUN] Would create' if self.dry_run else 'Created'} "
            f"{len(self.created_files)} files for '{self.lang}'"
        )
        logger.info("=" * 60)
        for f in sorted(self.created_files):
            logger.info(f"  📄 {f}")

        if self.words.get("question", "").startswith("TODO"):
            logger.warning("")
            logger.warning("=" * 60)
            logger.warning(
                f"⚠️  Language '{self.lang}' uses TODO placeholder prompts."
            )
            logger.warning(
                "    Edit the generated YAML files and replace TODO_* values"
            )
            logger.warning("    with native-speaker-validated translations.")
            logger.warning("=" * 60)
        elif self.lang in LANG_WORDS:
            # Known language but still has AI-suggested translations
            logger.info("")
            logger.info(
                "ℹ️  Prompt translations are AI-suggested. "
                "Have a native speaker validate them."
            )


# =============================================================================
# CLI Entry Point
# =============================================================================


def run_add_language(args: argparse.Namespace) -> int:
    """Run the add-language subcommand.

    Args:
        args: Parsed CLI arguments with lang, name, data_dir, dry_run.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )
    data_dir = args.data_dir or args.lang  # Use ISO code, matching data_prep output
    gen = TaskGenerator(
        lang_code=args.lang,
        lang_name=args.name,
        data_dir_name=data_dir,
        dry_run=args.dry_run,
    )
    gen.generate_all()
    return 0
