import re
from string import ascii_lowercase

import torch

# TODO add CTC decode
# TODO add BPE, LM, Beam Search support
# Note: think about metrics and encoder
# The design can be remarkably improved
# to calculate stuff more efficiently and prettier


class CTCTextEncoder:
    EMPTY_TOK = ""

    def __init__(self, alphabet=None, **kwargs):
        """
        Args:
            alphabet (list): alphabet for language. If None, it will be
                set to ascii
        """

        if alphabet is None:
            alphabet = list(ascii_lowercase + " ")

        self.alphabet = alphabet
        self.vocab = [self.EMPTY_TOK] + list(self.alphabet)

        self.ind2char = dict(enumerate(self.vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        """
        CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            decoded_text (str): decoded text.
        """
        decoded_chars = []
        prev_ind = None
        for ind in inds:
            ind = int(ind)
            if ind != 0 and ind != prev_ind:
                decoded_chars.append(self.ind2char[ind])
            prev_ind = ind
        return "".join(decoded_chars).strip() 

    def expand_and_merge_beams(self, dp, prob):
        new_dp = {}
        for (prefix, last_char), score in dp.items():
            for ind, char in self.ind2char.items():
                if char == self.EMPTY_TOK:
                    new_prefix = prefix
                    new_last_char = last_char
                elif char == last_char:
                    new_prefix = prefix
                    new_last_char = last_char
                else:
                    new_prefix = prefix + char
                    new_last_char = char
                new_score = score - torch.log(torch.tensor(prob[ind] + 1e-10)).item()
                key = (new_prefix, new_last_char)
                if key not in new_dp or new_dp[key] > new_score:
                    new_dp[key] = new_score
        return new_dp

    def truncate_beams(self, dp, beam_size):
        sorted_beams = sorted(dp.items(), key=lambda x: x[1])
        return dict(sorted_beams[:beam_size])  

    def beam_search(self, probs, beam_size):
        dp = {
            ("", self.EMPTY_TOK): 0
        }
        for prob in probs:
            dp = self.expand_and_merge_beams(dp, prob, self.ind2char)
            dp = self.truncate_beams(dp, beam_size)

        res = [(prefix, prob) for (prefix, _), prob in dp.items()]
        res = sorted(res, key=lambda x: x[1], reverse=True)
        return res[0][0]

    @staticmethod
    def normalize_text(text: str):
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
