"""
    ctc search implementations
"""
from typing import List

import torch


class GreedyCTC(torch.nn.Module):
    # taken from torch website
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> List[str]:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        output = list()
        for batch in emission:
            indices = torch.argmax(batch, dim=-1)  # [num_seq,]
            indices = torch.unique_consecutive(indices, dim=-1)
            indices = [i for i in indices if i != self.blank]


            # joined = "".join([self.labels[i] for i in indices])
            joined = self.labels.tokens2text(indices)
            output.append(joined)

        return output

    def __str__(self):
        return "GreedyCTC"

    def __repr__(self):
        return self.__str__()