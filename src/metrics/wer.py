from typing import List

import torch
from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer

# TODO beam search / LM versions
# Note: they can be written in a pretty way
# Note 2: overall metric design can be significantly improved


class ArgmaxWERMetric(BaseMetric):
    EMPTY_TOK = ""

    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: List[str], **kwargs
    ):
        wers = []
        predictions = torch.argmax(log_probs.cpu(), dim=-1).numpy()
        lengths = log_probs_length.detach().numpy()
        for log_prob_vec, length, target_text in zip(predictions, lengths, text):
            target_text = self.text_encoder.normalize_text(target_text)
            pred_text = self.text_encoder.ctc_decode(log_prob_vec[:length])
            wers.append(calc_wer(target_text, pred_text))
        return sum(wers) / len(wers)

    def expand_and_merge_beams(self, dp, prob, ind2char):
        new_dp = {}
        for (prefix, last_char), prob in dp.items():
            for ind, char in ind2char.items():
                if char == self.EMPTY_TOK:
                    new_prefix = prefix
                    new_last_char = last_char
                elif char == last_char:
                    new_prefix = prefix
                    new_last_char = last_char
                else:
                    new_prefix = prefix + char
                    new_last_char = char
                new_prob = prob * prob[ind]
                #new_score = -torch.log(torch.tensor(new_prob + 1e-10)).item
                key = (new_prefix, new_last_char)
                if key not in new_dp or new_dp[key] > new_prob:
                    new_dp[key] = new_prob
        return new_dp
    
    def truncate_beams(self, dp, beam_size):
        sorted_beams = sorted(dp.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_beams[:beam_size])
    
    def beam_search(self, probs, beam_size, ind2char):
        dp = {
            ("", self.EMPTY_TOK): 0
        }
        for prob in probs:
            dp = self.expand_and_merge_beams(dp, prob, ind2char)
            dp = self.truncate_beams(dp, beam_size)

        res = [(prefix, prob) for (prefix, _), prob in dp.items()]
        res = sorted(res, key=lambda x: x[1], reverse=True)
        return res[0][0]