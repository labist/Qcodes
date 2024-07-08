from qcodes.utils.validators import Enum
from . import N52xx

# class P9372A( N52xx.PNABase ) : # deprecated, moved in v0.46.0.x upgrade
class P9372A( N52xx.KeysightPNABase ) :
    
    ''' Driver for P9372A.
    '''
    def __init__(self, name, address, **kwargs):
        super().__init__(name, address,
                         min_freq=300e3, max_freq=9e9,
                         min_power=-43, max_power=20,
                         nports=2,
                         **kwargs)

        self.add_parameter('rosc_source',
                           label='Oscillator source',
                           get_cmd='SENS:ROSC:SOUR?',
                           set_cmd='SENS:ROSC:SOUR {}',
                           vals=Enum("INT", "EXT"))

        self.add_parameter('rosc_cond',
                           label='Oscillator source condition',
                           get_cmd='SENSe:ROSCillator:SOURce:CONDition?',
                           )
                           