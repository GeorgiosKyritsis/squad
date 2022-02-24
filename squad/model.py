import transformers
import tensorflow as tf
import optuna
import pathlib

from functools import partial
from tqdm.auto import tqdm
from typing import Dict, List

from squad.data import DataLoader
from squad.metrics import calculate_squad_f1_metric, calculate_exact_match_metric


class SQuADModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("google/t5-v1_1-base")
        self._t5 = transformers.TFT5ForConditionalGeneration.from_pretrained("google/t5-v1_1-base")

        self.n_gradients = tf.Variable(1, dtype=tf.int32, trainable=False)
        self.n_gradients_float = tf.Variable(1., dtype=tf.float32, trainable=False)
        self.n_acum_step = tf.Variable(0, dtype=tf.int32, trainable=False)
        self.gradient_accumulation = [tf.Variable(tf.zeros_like(v, dtype=tf.float32), trainable=False) for v in
                                      self.trainable_variables]

    def call(self, batch: Dict):
        loss = self._t5(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["selected_answers"]
        ).loss
        loss = tf.reduce_mean(loss)
        self.add_loss(loss)
        return loss

    def train_step(self, batch: Dict) -> Dict:
        self.n_acum_step.assign_add(1)

        with tf.GradientTape() as tape:
            loss = self(batch[0]) / self.n_gradients_float

        gradients = tape.gradient(loss, self.trainable_variables)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign_add(gradients[i])
        tf.cond(tf.equal(self.n_acum_step, self.n_gradients), self.apply_accu_gradients, lambda: None)

        return {m.name: m.result() for m in self.metrics}

    def apply_accu_gradients(self):
        self.optimizer.apply_gradients(zip(self.gradient_accumulation, self.trainable_variables))
        self.n_acum_step.assign(0)
        for i in range(len(self.gradient_accumulation)):
            self.gradient_accumulation[i].assign(tf.zeros_like(self.trainable_variables[i], dtype=tf.float32))

    @classmethod
    def train(cls, train_data: List[Dict], dev_data: List[Dict], batch_size: int = 2,
              learning_rate: float = 0.0001, accumulate_grad_batches: int = 32) -> "SQuADModel":
        model = cls()
        model.n_gradients.assign(accumulate_grad_batches)
        model.n_gradients_float.assign(float(accumulate_grad_batches))
        model.compile(tf.keras.optimizers.Adam(learning_rate=learning_rate))
        train_dataloader = DataLoader(
            train_data,
            model.tokenizer,
            has_targets=True,
            shuffle=True,
            batch_size=batch_size
        )
        dev_dataloader = DataLoader(
            dev_data,
            model.tokenizer,
            has_targets=True,
            shuffle=False,
            batch_size=batch_size
        )
        model.fit(
            train_dataloader,
            validation_data=dev_dataloader,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=5,
                    mode="min",
                    restore_best_weights=True,
                )
            ],
            epochs=100
        )
        return model

    def _generate_answers_for_batch(self, batch: Dict) -> List[str]:
        outputs = self._t5.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

    def generate_answers(self, data: List[Dict], batch_size: int = 2, verbose: bool = False) -> List[str]:
        dataloader = DataLoader(
            data,
            self.tokenizer,
            has_targets=False,
            shuffle=False,
            batch_size=batch_size
        )

        predictions = []
        progress_bar = tqdm if verbose else lambda x: x
        for batch in progress_bar(dataloader):
            predictions.extend(self._generate_answers_for_batch(batch))
        return predictions

    def calculate_metrics(self, eval_data: List[Dict], batch_size: int = 2) -> Dict:
        predicted_answers = self.generate_answers(eval_data, batch_size=batch_size, verbose=True)
        target_answers = [[answer["text"] for answer in ex["answers"]] for ex in eval_data]

        return {
            "squad_f1": calculate_squad_f1_metric(predicted_answers, target_answers),
            "exact_match": calculate_exact_match_metric(predicted_answers, target_answers)
        }

    @classmethod
    def _tuning_objective(cls, trial: optuna.Trial, train_data: List[Dict], dev_data: List[Dict]) -> float:
        learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        model = cls.train(train_data, dev_data, learning_rate=learning_rate, batch_size=batch_size)
        return model.calculate_metrics(eval_data=dev_data, batch_size=batch_size)["squad_f1"]

    @classmethod
    def tune(cls, train_data: List[Dict], dev_data: List[Dict]) -> optuna.Study:
        search_space = {
            "learning_rate": [1e-4, 3e-4, 5e-4],
            "batch_size": [8, 16, 32, 64],
        }
        study = optuna.create_study(
            study_name="tuning",
            sampler=optuna.samplers.GridSampler(search_space),
            direction="maximize"
        )
        study.optimize(partial(cls._tuning_objective, train_data=train_data, dev_data=dev_data))
        return study

    def save_model_weights(self, save_path: str):
        pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        self.save_weights(save_path)

    @classmethod
    def load_model(cls, load_path: str) -> "SQuADModel":
        model = cls()
        model.load_weights(load_path).expect_partial()
        return model
