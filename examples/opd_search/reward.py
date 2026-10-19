"""EM is used for validation only; the OPD loss never consumes task rewards."""

import re
import string


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    del data_source, extra_info, kwargs
    # An answer quoted in a retrieved passage must not earn a validation reward.
    text = re.sub(r"<information>.*?</information>", "", solution_str, flags=re.DOTALL)
    answers = re.findall(r"<answer>(.*?)</answer>", text, flags=re.DOTALL)
    targets = ground_truth.get("target", []) if isinstance(ground_truth, dict) else ground_truth
    targets = [targets] if isinstance(targets, str) else list(targets or [])
    em = float(bool(answers) and any(normalize(answers[-1]) == normalize(t) for t in targets))
    return {"score": em, "acc": em, "format_score": float(bool(answers))}
