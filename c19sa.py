# coding=utf-8
# Copyright 2020 The TensorFlow Datasets Authors and the HuggingFace Datasets Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Covid-19-Sentiment-Analysis."""


import csv

import datasets
from datasets import Split


_DESCRIPTION = """\
This data is used for Global Student Competition of Covid-19 Sentiment Analysis, 
which is a multi-label text classification task.

Data collection: The dataset was collected by, an open source Twitter crawler, 
called Twint, where query words like covid-19, coronavirus, covid, corona, ect. are used. 
The user-related information is removed when the data are saved.

Data annoation: 11 labels and their covered auxiliary emotions are optimistic 
(representing hopeful, proud, trusting), thankful for the efforts to combat the virus, 
empathetic (including praying), pessimistic (hopeless), anxious (scared, fearful), sad, 
annoyed (angry), denial towards conspiracy theories, surprise (unprecedented), official 
report, and joking (ironical). The IRA and Kappa coefficient are 0.904 and 0.381, respectively.

Data information: The training data contian 5000 labeled data while validation data have 
2500 piece of unlabeled data. The training data have 3 columns, containing Tweet ID, Tweet text, 
and labels consisting of Optimistic, Thankful, Empathetic, Pessimistic, Anxious, Sad, Annoyed, Denial, 
Surprise, Official report, Joking. The sentiments and indexes are mapped as follows: Optimistic (0), 
Thankful (1), Empathetic (2), Pessimistic (3), Anxious (4), Sad (5), Annoyed (6), Denial (7), Surprise 
(8), Official report (9), Joking (10). The validiaton data only have two columns Tweet ID and Tweet 
text. The ID information is disguised to protect the user privacy.
"""

_CITATION = """\
@article{Zhang2021RiseAF,
  title={Rise and fall of the global conversation and shifting sentiments during the COVID-19 pandemic},
  author={Xiangliang Zhang and Qiang Yang and Somayah Albaradei and Xiaoting Lyu and Hind Alamro and Adil Salhi and Changsheng Ma and Manal Alshehri and I. Jaber and Faroug Tifratene and Wei Wang and T. Gojobori and C. M. Duarte and Xin Gao},
  journal={Humanities and Social Sciences Communications},
  year={2021},
  volume={8},
  pages={1-10}
}
"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/WrRan/c19sa-dataset/master/processed/train.csv"
_VAL_DOWNLOAD_URL = "https://raw.githubusercontent.com/WrRan/c19sa-dataset/master/processed/val.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/WrRan/c19sa-dataset/master/processed/val_content.csv"


class C19SA(datasets.GeneratorBasedBuilder):
    """Covid-19-Sentiment-Analysis."""

    N_LABELS = 11

    INDICES_TO_LABELS = [
        'Optimistic',
        'Thankful',
        'Empathetic',
        'Pessimistic',
        'Anxious',
        'Sad',
        'Annoyed',
        'Denial',
        'Surprise',
        'Official report',
        'Joking'
    ]

    LABELS_TO_INDICES = {
        'Optimistic': 0,
        'Thankful': 1,
        'Empathetic': 2,
        'Pessimistic': 3,
        'Anxious': 4,
        'Sad': 5,
        'Annoyed': 6,
        'Denial': 7,
        'Surprise': 8,
        'Official report': 9,
        'Joking': 10
    }

    def __init__(self, *args, writer_batch_size=None, **kwargs):
        super().__init__(*args, writer_batch_size=writer_batch_size, **kwargs)
        data_files = kwargs.get('data_files')
        data_files = data_files if data_files is not None else {}
        data_files[Split.TRAIN] = data_files.get(Split.TRAIN, _TRAIN_DOWNLOAD_URL)
        data_files[Split.VALIDATION] = data_files.get(Split.VALIDATION, _VAL_DOWNLOAD_URL)
        data_files[Split.TEST] = data_files.get(Split.TEST, _TEST_DOWNLOAD_URL)
        self.data_files = data_files

    def _info(self):
        return datasets.DatasetInfo(
                description=_DESCRIPTION,
                features=datasets.Features(
                        {
                            "c19id": datasets.Value("int32"),
                            "text": datasets.Value("string"),
                            "labels": datasets.features.Sequence(datasets.Value("int32")),
                        }
                ),
                homepage="https://github.com/gitdevqiang/Covid-19-Sentiment-Analysis",
                citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        train_path = dl_manager.download_and_extract(self.data_files[Split.TRAIN])
        val_path = dl_manager.download_and_extract(self.data_files[Split.VALIDATION])
        test_path = dl_manager.download_and_extract(self.data_files[Split.TEST])
        return [
            datasets.SplitGenerator(name=Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=Split.VALIDATION, gen_kwargs={"filepath": val_path}),
            datasets.SplitGenerator(name=Split.TEST, gen_kwargs={"filepath": test_path, "test": True})
        ]

    def _generate_examples(self, filepath, test: bool = False):
        """Generate C19SA examples."""
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                    csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                if not test:
                    c19id, tweet, labels = row
                    yield id_, {"c19id": c19id, "text": tweet, "labels": labels.strip().split()}
                else:
                    c19id, tweet = row
                    yield id_, {"c19id": c19id, "text": tweet, "labels": []}

    @classmethod
    def label_indices_to_multi_hot(cls, label_indices):
        result = [0.0] * cls.N_LABELS
        for index in label_indices:
            result[index] = 1.0
        return result
