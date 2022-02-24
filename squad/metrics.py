import re
import string

from collections import Counter
from typing import List


def _normalize_text(text: str) -> str:
    # lower
    text = text.lower()
    # remove punctuations
    exclude = set(string.punctuation)
    text = "".join(ch for ch in text if ch not in exclude)
    # remove articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    # white space fix
    text = " ".join(text.split())
    return text


def calculate_squad_f1_metric(predicted_answers: List[str], answers: List[List[str]]) -> float:
    f1_score = 0.0
    total = 0

    for i in range(len(answers)):
        total += 1
        if len(answers[i]) == 0:
            f1_score += 1 if predicted_answers[i] == "null" else 0
        else:
            current_f1_scores = []
            for answer in answers[i]:
                target_tokens = [] if not answer else _normalize_text(answer).split()
                predicted_tokens = [] if not predicted_answers[i] else _normalize_text(predicted_answers[i]).split()
                common = Counter(target_tokens) & Counter(predicted_tokens)
                num_same = sum(common.values())
                if num_same == 0:
                    current_f1_scores.append(0.)
                else:
                    precision = 1.0 * num_same / len(predicted_tokens)
                    recall = 1.0 * num_same / len(target_tokens)
                    current_f1_scores.append((2 * precision * recall) / (precision + recall))
            f1_score += max(current_f1_scores)
    return 100.0 * f1_score / total


def calculate_exact_match_metric(predicted_answers: List[str], answers: List[List[str]]) -> float:
    exact_match_score = 0.0
    total = 0

    for i in range(len(answers)):
        total += 1
        if len(answers[i]) == 0:
            exact_match_score += 1 if predicted_answers[i] == "null" else 0
        else:
            current_exact_match_scores = []
            for answer in answers[i]:
                normalized_target = _normalize_text(answer)
                normalized_prediction = _normalize_text(predicted_answers[i])
                current_exact_match_scores.append(normalized_target == normalized_prediction)
            exact_match_score += max(current_exact_match_scores)
    return 100.0 * exact_match_score / total
