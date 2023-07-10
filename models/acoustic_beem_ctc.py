"""
    similar to acoustic_greedy_ctc but instead of greedy ctc do the following:
    1. preform a beam search and save the best k outputs
    2. append an english lexicon, or even better, a lexicon of the an4 dataset to get better results
    3. choose the output of the best k outputs that has the most word from the lexicon
"""





