import random
import json
import pathlib
import math
import tensorflow as tf

from typing import Dict, List, Union
from collections import Counter
from transformers import T5Tokenizer, T5TokenizerFast


def prepare_data(original_dataset_path: str, output_directory: str, include_impossible: bool):
    with open(original_dataset_path) as fr:
        original_dataset = json.load(fr)["data"]

    random.seed(0)
    random.shuffle(original_dataset)

    parent_directory = pathlib.Path(output_directory)
    parent_directory.mkdir(parents=True, exist_ok=True)
    for ds_name, start_idx, end_idx in (
            ("dev", 0, 5),
            ("test", 5, 10),
            ("train", 10, len(original_dataset)),
    ):
        current_dataset = []
        for doc in original_dataset[start_idx:end_idx]:
            for paragraph in doc["paragraphs"]:
                for question in paragraph["qas"]:
                    if not include_impossible and question["is_impossible"]:
                        continue
                    current_dataset.append({
                        "id": question["id"],
                        "context": paragraph["context"],
                        "question": question["question"],
                        "answers": question["answers"],
                        "is_impossible": question["is_impossible"],
                    })
        with open(parent_directory / f"{ds_name}.json", 'w') as fw:
            json.dump(current_dataset, fw)


class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, data: List[Dict], tokenizer: Union[T5Tokenizer, T5TokenizerFast], batch_size: int,
                 has_targets: bool = True, shuffle: bool = False, seed: int = 0):
        self._data = data
        self._tokenizer = tokenizer
        self._batch_size = batch_size
        self._has_targets = has_targets
        self._shuffle = shuffle
        self._seed = seed

        if self._shuffle:
            self._shuffle_data()

    def __len__(self) -> int:
        return math.ceil(len(self._data) / self._batch_size)

    def __getitem__(self, idx: int) -> Dict:
        batch = self._data[idx * self._batch_size:(idx + 1) * self._batch_size]

        prepared_input = []
        for ex in batch:
            prepared_input.append(" ".join(f"{ex['context']} {ex['question']} <extra_id_0>".split()))

        prepared_input = self._tokenizer(prepared_input, padding=True, return_tensors="tf")

        prepared_batch = {
            "input_ids": prepared_input.input_ids,
            "attention_mask": prepared_input.attention_mask
        }

        if self._has_targets:
            selected_answers = []
            answers = []

            for ex in batch:
                current_answers = [a["text"] for a in ex["answers"]]

                if ex['is_impossible']:
                    selected_answers.append("<extra_id_0> null")
                    answers.append([])
                else:
                    selected_answer = self._select_answer(current_answers)
                    selected_answers.append(" ".join(f"<extra_id_0> {selected_answer}".split()))
                    answers.append(current_answers)

            prepared_batch["selected_answers"] = self._tokenizer(
                selected_answers,
                padding=True,
                return_tensors="np"
            ).input_ids
            pad_mask = prepared_batch["selected_answers"] == self._tokenizer.pad_token_id
            prepared_batch["selected_answers"][pad_mask] = -100

        return prepared_batch

    @staticmethod
    def _select_answer(answers: List[str]) -> str:
        answers_freq = Counter(answers)
        answers_meta = [(a, answers_freq[a], len(a)) for a in answers]
        max_freq = max(c[1] for c in answers_meta)
        return min([c for c in answers_meta if c[1] == max_freq], key=lambda c: c[2])[0]

    def on_epoch_end(self):
        if self._shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        random.seed(self._seed)
        random.shuffle(self._data)
