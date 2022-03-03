import numpy as np

def averageProportion(**_k):
    d = numpy.zeros((_k['steps'],))
    # print a
    _src = 0
    _trg = 0
    _sum = 0
    for it, a in enumerate(_k['act']):
        if a == 0: # Read
            if _src < _k['source_len']:
                _src += 1
        elif a == 1: # Write
            _trg += 1
            _sum += _src
    d[-1] = _sum / (_src * _trg + 1e-6)
    return d

def consecutiveWaitDelay(_max=5, beta=0.1, **_k):
    d = numpy.zeros((_k['steps'],))
    _cur = 0
    for it, a in enumerate(_k['act']):
        if a == 0:
            _cur += 1
            if _cur > _max:
                d[it] = -0.1 * (_cur - _max)
            pass
        elif a == 1:   # only for new commit
            _cur = 0

    return d * beta


