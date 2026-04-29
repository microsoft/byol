# coding=utf-8
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Custom FLORES-200 dataset using openlanguagedata/flores_plus with English-Chichewa pairing"""

import datasets
from typing import Dict, List, Any


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process flores_plus dataset for lm-eval. This function is called by lm-eval after loading.
    Uses the dev split for processing.
    Returns a datasets.Dataset object (required by lm-eval).
    """
    paired_examples = process_flores_plus_data("dev")
    return datasets.Dataset.from_list(paired_examples)


def process_docs_devtest(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    Process flores_plus dataset for lm-eval using devtest split.
    Returns a datasets.Dataset object (required by lm-eval).
    """
    paired_examples = process_flores_plus_data("devtest")
    return datasets.Dataset.from_list(paired_examples)


def process_flores_plus_data(split: str = "dev") -> List[Dict[str, Any]]:
    """
    Process flores_plus dataset to create paired examples for English-Chichewa and English-Maori
    """
    # Load the dataset
    dataset = datasets.load_dataset("openlanguagedata/flores_plus", split=split)
    
    # Group by id and iso_639_3
    examples_by_id = {}
    
    for example in dataset:
        example_id = example['id']
        lang_code = example['iso_639_3']
        
        if example_id not in examples_by_id:
            examples_by_id[example_id] = {}
        
        examples_by_id[example_id][lang_code] = example
    
    # Create paired examples
    paired_examples = []
    
    for example_id, languages in examples_by_id.items():
        if 'eng' in languages:
            eng_example = languages['eng']
            
            # Create paired examples for available languages
            paired_example = {
                'id': example_id,
                'sentence_eng_Latn': eng_example['text'],
                'URL': eng_example['url'],
                'domain': eng_example['domain'],
                'topic': eng_example['topic'],
                'has_image': eng_example['has_image'],
                'has_hyperlink': eng_example['has_hyperlink']
            }
            
            # Add Chichewa if available
            if 'nya' in languages:
                nya_example = languages['nya']
                paired_example['sentence_nya_Latn'] = nya_example['text']
            
            # Add Maori if available  
            if 'mri' in languages:
                mri_example = languages['mri']
                paired_example['sentence_mri_Latn'] = mri_example['text']
            
            # Only add if we have at least one target language
            if 'sentence_nya_Latn' in paired_example or 'sentence_mri_Latn' in paired_example:
                paired_examples.append(paired_example)
    
    return paired_examples


def create_flores_plus_dataset():
    """
    Create a processed version of flores_plus with paired language examples
    """
    # Process both splits
    dev_examples = process_flores_plus_data("dev")
    devtest_examples = process_flores_plus_data("devtest") 
    
    # Create dataset features
    features = datasets.Features({
        'id': datasets.Value('int32'),
        'sentence_eng_Latn': datasets.Value('string'),
        'sentence_nya_Latn': datasets.Value('string'),
        'sentence_mri_Latn': datasets.Value('string'),
        'URL': datasets.Value('string'),
        'domain': datasets.Value('string'),
        'topic': datasets.Value('string'),
        'has_image': datasets.Value('string'),
        'has_hyperlink': datasets.Value('string')
    })
    
    # Create dataset dict
    dataset_dict = datasets.DatasetDict({
        'dev': datasets.Dataset.from_list(dev_examples, features=features),
        'devtest': datasets.Dataset.from_list(devtest_examples, features=features)
    })
    
    return dataset_dict

_CITATION = """
@article{nllb2022,
  author    = {NLLB Team, Marta R. Costa-jussà, James Cross, Onur Çelebi, Maha Elbayad, Kenneth Heafield, Kevin Heffernan, Elahe Kalbassi,  Janice Lam, Daniel Licht, Jean Maillard, Anna Sun, Skyler Wang, Guillaume Wenzek, Al Youngblood, Bapi Akula, Loic Barrault, Gabriel Mejia Gonzalez, Prangthip Hansanti, John Hoffman, Semarley Jarrett, Kaushik Ram Sadagopan, Dirk Rowe, Shannon Spruit, Chau Tran, Pierre Andrews, Necip Fazil Ayan, Shruti Bhosale, Sergey Edunov, Angela Fan, Cynthia Gao, Vedanuj Goswami, Francisco Guzmán, Philipp Koehn, Alexandre Mourachko, Christophe Ropers, Safiyyah Saleem, Holger Schwenk, Jeff Wang},
  title     = {No Language Left Behind: Scaling Human-Centered Machine Translation},
  year      = {2022}
}
"""

_DESCRIPTION = """
Custom FLORES dataset processed from openlanguagedata/flores_plus with English-Chichewa and English-Maori language pairs.
This dataset contains paired translations for evaluation of machine translation between English and Chichewa, and English and Maori.
"""

_HOMEPAGE = "https://github.com/facebookresearch/flores"
_LICENSE = "CC-BY-SA-4.0"


class FloresCustomConfig(datasets.BuilderConfig):
    """BuilderConfig for the custom FLORES dataset."""
    def __init__(self, **kwargs):
        super().__init__(version=datasets.Version("1.0.0"), **kwargs)


class FloresCustom(datasets.GeneratorBasedBuilder):
    """Custom FLORES dataset with English-Chichewa pairs."""
    
    BUILDER_CONFIGS = [
        FloresCustomConfig(
            name="all",
            description="FLORES English-Chichewa paired dataset from flores_plus"
        )
    ]
    
    DEFAULT_CONFIG_NAME = "all"

    def _info(self):
        features = datasets.Features({
            'id': datasets.Value('int32'),
            'sentence_eng_Latn': datasets.Value('string'),
            'sentence_nya_Latn': datasets.Value('string'),
            'sentence_mri_Latn': datasets.Value('string'),
            'URL': datasets.Value('string'),
            'domain': datasets.Value('string'),
            'topic': datasets.Value('string'),
            'has_image': datasets.Value('string'),
            'has_hyperlink': datasets.Value('string')
        })
        
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )
    
    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name="dev",
                gen_kwargs={"split": "dev"}
            ),
            datasets.SplitGenerator(
                name="devtest", 
                gen_kwargs={"split": "devtest"}
            ),
        ]
    
    def _generate_examples(self, split):
        """Generate examples from processed flores_plus data."""
        # Create the processed dataset
        dataset_dict = create_flores_plus_dataset()
        
        # Get the appropriate split
        split_data = dataset_dict[split]
        
        # Yield examples
        for idx, example in enumerate(split_data):
            yield idx, example
