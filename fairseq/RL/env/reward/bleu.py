from __future__ import division

import numpy as np
import math
import fractions
from nltk.util import ngrams
from collections import Counter

from fractions import Fraction

def sentence_bleu(references, hypothesis, weights=(0.25, 0.25, 0.25, 0.25),
                  smoothing_function=None):
    """
    :param references: reference sentences
    :type references: list(list(str))
    :param hypothesis: a hypothesis sentence
    :type hypothesis: list(str)
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :return: The sentence-level BLEU score.
    :rtype: float
    """
    return corpus_bleu([references], [hypothesis], weights, smoothing_function)

def corpus_bleu(list_of_references, hypotheses, weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=None):
    """
    :param references: a corpus of lists of reference sentences, w.r.t. hypotheses
    :type references: list(list(list(str)))
    :param hypotheses: a list of hypothesis sentences
    :type hypotheses: list(list(str))
    :param weights: weights for unigrams, bigrams, trigrams and so on
    :type weights: list(float)
    :return: The corpus-level BLEU score.
    :rtype: float
    """
    
    # Before proceeding to compute BLEU, perform sanity checks.
    p_numerators = Counter()  # Key = ngram order, and value = no. of ngram matches.
    p_denominators = Counter()  # Key = ngram order, and value = no. of ngram in ref.
    hyp_lengths, ref_lengths = 0, 0


    assert len(list_of_references) == len(hypotheses), "The numer of hypotheses and their reference(s) should be the same"

    # Iterate through each hypothesis and their corresponding references.
    for references, hypothesis in zip(list_of_references, hypotheses):
        # For each order of ngram, calculate the numerator and
        # denominator for the corpus-level modified precision.
        for i, _ in enumerate(weights, start=1):
            p_i = modified_precision(references, hypothesis, i)
            p_numerators[i] += p_i.numerator
            p_denominators[i] += p_i.denominator
        # Calculate the hypothesis length and the closest reference length.
        # Adds them to the corpus-level hypothesis and reference counts.
        hyp_len = len(hypothesis)
        hyp_lengths += hyp_len
        ref_lengths += closest_ref_length(references, hyp_len)
    
    # Calculate corpus-level brevity penalty.
    bp = brevity_penalty(ref_lengths, hyp_lengths)

    # Collects the various precision values for the different ngram orders.
    p_n = [Fraction(p_numerators[i], p_denominators[i], _normalize=False)
           for i, _ in enumerate(weights, start=1)]
    
    # Returns 0 if there's no matching n-grams
    # We only need to check for p_numerators[1] == 0, since if there's
    # no unigrams, there won't be any higher order ngrams.
    if p_numerators[1] == 0:
        return 0, 0

    # Smoothen the modified precision.
    # Note: smooth_precision() converts values into float.
    if not smoothing_function:
        smoothing_function = SmoothingFunction().method0
    p_n = smoothing_function(p_n, references=references,
                             hypothesis=hypothesis, hyp_len=hyp_len)
    
    # Calculates the overall modified precision for all ngrams.
    # By sum of the product of the weights and the respective *p_n*
    s = (w * math.log(p_i) for w, p_i in zip(weights, p_n)
         if p_i.numerator != 0)

    # return bp * math.exp(math.fsum(s))
    bleu  = math.exp(math.fsum(s))
    bleup = bleu * bp


    return bleu, bleup

def modified_precision(references, hypothesis, n):
    """
    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: A hypothesis translation.
    :type hypothesis: list(str)
    :param n: The ngram order.
    :type n: int
    :return: BLEU's modified precision for the nth order ngram.
    :rtype: Fraction
    """
    # Extracts all ngrams in hypothesis.
    counts = Counter(ngrams(hypothesis, n))

    # Extract a union of references' counts.
    ## max_counts = reduce(or_, [Counter(ngrams(ref, n)) for ref in references])
    max_counts = {}
    for reference in references:
        reference_counts = Counter(ngrams(reference, n))
        for ngram in counts:
            max_counts[ngram] = max(max_counts.get(ngram, 0),
                                    reference_counts[ngram])

    # Assigns the intersection between hypothesis and references' counts.
    clipped_counts = {ngram: min(count, max_counts[ngram])
                      for ngram, count in counts.items()}

    numerator = sum(clipped_counts.values())
    # Ensures that denominator is minimum 1 to avoid ZeroDivisionError.
    # Usually this happens when the ngram order is > len(reference).
    denominator = max(1, sum(counts.values()))

    return Fraction(numerator, denominator, _normalize=False)

def closest_ref_length(references, hyp_len):
    '''
    This function finds the reference length that is closest in length to the
    hypothesis. The closest reference length is referred to as *r* variable
    from the brevity penalty formula in Papineni et. al. (2002)

    :param references: A list of reference translations.
    :type references: list(list(str))
    :param hypothesis: The length of the hypothesis.
    :type hypothesis: int
    :return: The length of the reference that's closest to the hypothesis.
    :rtype: int
    '''
    ref_lens = (len(reference) for reference in references)
    closest_ref_len = min(ref_lens, key=lambda ref_len:
    (abs(ref_len - hyp_len), ref_len))
    return closest_ref_len

def brevity_penalty(closest_ref_len, hyp_len):
    """
    :param hyp_len: The length of the hypothesis for a single sentence OR the
    sum of all the hypotheses' lengths for a corpus
    :type hyp_len: int
    :param closest_ref_len: The length of the closest reference for a single
    hypothesis OR the sum of all the closest references for every hypotheses.
    :type closest_reference_len: int
    :return: BLEU's brevity penalty.
    :rtype: float
    """
    if hyp_len > closest_ref_len:
        return 1
    # If hypothesis is empty, brevity penalty = 0 should result in BLEU = 0.0
    elif hyp_len == 0:
        return 0
    else:
        return math.exp(1 - closest_ref_len / hyp_len)

class SmoothingFunction:
    """
    This is an implementation of the smoothing techniques
    for segment-level BLEU scores that was presented in
    Boxing Chen and Collin Cherry (2014) A Systematic Comparison of
    Smoothing Techniques for Sentence-Level BLEU. In WMT14.
    http://acl2014.org/acl2014/W14-33/pdf/W14-3346.pdf
    """
    def __init__(self, epsilon=0.1, alpha=5, k=5):
        """
        :param epsilon: the epsilon value use in method 1
        :type epsilon: float
        :param alpha: the alpha value use in method 6
        :type alpha: int
        :param k: the k value use in method 4
        :type k: int
        """
        self.epsilon = epsilon
        self.alpha = alpha
        self.k = k

    def method0(self, p_n, *args, **kwargs):
        """ No smoothing. """
        return p_n
    
    def method5(self, p_n, references, hypothesis, hyp_len):
        """
        Smoothing method 5:
        The matched counts for similar values of n should be similar. To a
        calculate the n-gram matched count, it averages the n−1, n and n+1 gram
        matched counts.
        """
        m = {}
        # Requires an precision value for an addition ngram order.
        p_n_plus1 = p_n + [modified_precision(references, hypothesis, 5)]
        m[-1] = p_n[0] + 1
        for i, p_i in enumerate(p_n):
            p_n[i] = (m[i - 1] + p_i + p_n_plus1[i + 1]) / 3
            m[i] = p_n[i]
        return p_n


### for the reward function ###

def BLEUwithForget(beta=None, discount=1., return_quality=False, **_k):
    # init
    words = _k['words'].split()  # end-of-sentence is treated as a word
    ref   = _k['reference']

    q0    = numpy.zeros((_k['steps'],))

    # check 0, 1
    maps  = [(it, a) for it, a in enumerate(_k['act']) if a < 2]
    kmap  = len(maps)
    lb    = numpy.zeros((kmap,))
    ts    = numpy.zeros((kmap,))
    q     = numpy.zeros((kmap,))

    if not beta:
        beta = kmap

    beta = 1. / float(beta)

    chencherry = SmoothingFunction()

    # compute BLEU for each Yt
    Y = []
    bleus = []
    truebleus = []

    if len(words) == 0:
        bleus = [0]
        truebleus = [0]

    for t in xrange(len(words)):
        if len(Y) > 0:
            _temp = Y[-1] + ' ' + words[t]
            _temp = _temp.replace('@@ ', '')
            Y = Y[:-1] + _temp.split()
        else:
            Y = [words[t]]

        bb = sentence_bleu(ref, Y, smoothing_function=chencherry.method5)

        bleus.append(bb[1])   # try true BLEU
        truebleus.append(bb[1])


    # print 'Latency BLEU', lbn
    bleus = [0] + bleus    # use TRUE BLEU
    bleus = numpy.array(bleus)
    temp  = bleus[1:] - bleus[:-1]

    tpos  = 0
    for pos, (it, a) in enumerate(maps):
        if (a == 1) and (tpos < len(words)):
            q[pos] = temp[tpos]
            q0[it] = q[pos]
            tpos  += 1

    # add the whole sentence balance on it
    q0[-1] = truebleus[-1]  # the last BLEU we use the real BLEU score.
    return q0



if __name__ == "__main__":
    hyps="S'il vous plaît, rendez-vous à l'aise."
    hyps=hyps.split()
    refs=[hyps]
    print(f"hyps: {refs}")
    print(f"The BLEU score for the sentence: {sentence_bleu(refs,hyps, smoothing_function=SmoothingFunction().method0)}")