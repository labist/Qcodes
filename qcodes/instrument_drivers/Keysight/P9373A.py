from qcodes.utils.validators import Enum
from . import N52xx

class P9373A( N52xx.PNABase ) :
    ''' Driver for P9372A.
    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address,
                         min_freq=300e3, max_freq=14e9,
                         min_power=-43, max_power=20,
                         nports=2,
                         **kwargs)