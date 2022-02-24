import pytest

from squad.model import SQuADModel


class TestSQuADModel:

    @pytest.fixture(scope="class")
    def model(self):
        return SQuADModel()

    def test_generate_answers(self, model):
        answers = model.generate_answers(
            [{"context": "This is the question!", "question": "What is the answer?"}] * 3,
            batch_size=2
        )

        assert len(answers) == 3
