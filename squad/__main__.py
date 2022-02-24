import pathlib

import fire
import json
import pickle

from squad.data import prepare_data
from squad.model import SQuADModel


class CLI:
    @staticmethod
    def prepare_data(original_dataset_path: str, output_directory: str, include_impossible: bool):
        """
        Process and splits the data.
        :param original_dataset_path: The path pointing to the original dev set.
        :param output_directory: The directory where to save the dataset splits.
        :param include_impossible: Whether to include unanswerable questions or not.
        """
        prepare_data(original_dataset_path, output_directory, include_impossible)

    @staticmethod
    def train(train_dataset_path: str, dev_dataset_path: str, save_model_path: str, batch_size: int = 2,
              learning_rate=0.0001, accumulate_grad_batches=32):
        """
        Trains a T5 model on the squad task.
        :param train_dataset_path: The path pointing to the training dataset.
        :param dev_dataset_path: The path pointing to the development dataset.
        :param save_model_path: The path where to save the trained model.
        :param batch_size: The batch size.
        :param learning_rate: The learning rate.
        :param accumulate_grad_batches: After how many forward calls to perform an optimization step.
        """
        with open(train_dataset_path) as fr:
            train_data = json.load(fr)
        with open(dev_dataset_path) as fr:
            dev_data = json.load(fr)
        trained_model = SQuADModel.train(train_data, dev_data, batch_size, learning_rate, accumulate_grad_batches)
        trained_model.save_model_weights(save_model_path)

    @staticmethod
    def evaluate(eval_dataset_path: str, model_path: str, batch_size: int = 2):
        """
        Evaluates the model.
        :param eval_dataset_path: The path pointing to the evaluation dataset.
        :param model_path: The path pointing to the trained model.
        :param batch_size: The batch size.
        """
        with open(eval_dataset_path) as fr:
            eval_data = json.load(fr)
        model = SQuADModel.load_model(model_path)
        print(model.calculate_metrics(eval_data, batch_size))

    @staticmethod
    def tune(train_dataset_path: str, dev_dataset_path: str, results_save_path: str):
        """
        Tunes the learning rate and batch size.
        :param train_dataset_path: The path pointing to the training dataset.
        :param dev_dataset_path: The path pointing to the development dataset.
        :param results_save_path: The path where to save the tuning results.
        """
        with open(train_dataset_path) as fr:
            train_data = json.load(fr)
        with open(dev_dataset_path) as fr:
            dev_data = json.load(fr)
        study = SQuADModel.tune(train_data, dev_data)
        pathlib.Path(results_save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_save_path, "wb") as fw:
            pickle.dump(study, fw)


def main():
    fire.Fire(CLI)


if __name__ == "__main__":
    main()
