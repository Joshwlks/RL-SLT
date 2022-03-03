import numpy as np
from bleu import BLEUwithForget
from latency import consecutiveWaitDelay, averageProportion

def return_rewards(**_k):
    def NewReward():
        # params

        maxsrc   = _k['maxsrc']
        target   = _k['target']
        cw       = _k['cw']
        beta     = 0.03 # 0.5

        q0 = BLEUwithForget(return_quality=True, **_k)
        d0 = averageProportion(**_k)

        # global reward signal :::>>>
        # just bleu
        bleu  = q0[-1]

        # just delay
        delay = d0[-1]

        # local reward signal :::>>>>
        # use maximum-delay + latency bleu (with final BLEU)
        q = q0
        q[-1] = 0
        if cw > 0:
            d = consecutiveWaitDelay(_max=cw, beta=beta, **_k)
        else:
            d = 0

        # s = AwardForget(_max=maxsrc, beta=0.01)
        # s = AwardForgetBi(_max=maxsrc, beta=0.01)

        r0  = q + 0.5 * d

        if target < 1:
            tar = -numpy.maximum(delay - target, 0)
        else:
            tar = 0

        rg  = bleu + tar # it is a global reward, will not be discounted.
        r      = r0
        r[-1] += rg

        R = r[::-1].cumsum()[::-1]
        return R, bleu, delay, R[0]

    return new_reward()

def reward(**_k):


    