from . import N52xx

class P9372A( N52xx.PNABase ) :
    ''' Driver for P9372A.
    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address,
                         min_freq=300e3, max_freq=9e9,
                         min_power=-43, max_power=20,
                         nports=2,
                         **kwargs)

