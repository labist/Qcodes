import logging
import time
from functools import partial
from typing import Optional
import re

from pyvisa.errors import VisaIOError
from qcodes import VisaInstrument
from qcodes import ChannelList, InstrumentChannel
from qcodes.utils import validators as vals
import numpy as np
from qcodes import MultiParameter, ArrayParameter
from qcodes.utils.validators import Ints, Numbers, Enum, Bool, Strings

log = logging.getLogger(__name__)


class FrequencySweepReIm(MultiParameter):
    """
    Sweep that return Real and Imaginary part of the S parameters.
    """

    def __init__(self, name, instrument, start, stop, npts, trnm):
        """
        Args:
            name: parameter name
            instrument: instrument the parameter belongs to
            start: starting frequency of sweep
            stop: ending frequency of sweep
            npts: number of points in frequency sweep
            trnm: tracename
        """

        super().__init__(name, names=("", ""), shapes=((), ()))
        self._instrument = instrument
        self.set_sweep(start, stop, npts, trnm)
        self.names = ('real', 'imaginary')
        self.labels = ('{}_Re_'.format(instrument.short_name) +trnm,
                        '{}_Im_'.format(instrument.short_name) +trnm)
        self.units = ('', '')
        self.setpoint_units = (('Hz',), ('Hz',))
        self.setpoint_labels = (('{} frequency'.format(instrument.short_name),), ('{} frequency'.format(instrument.short_name),))
        self.setpoint_names = (('{}_re_im_frequency'.format(instrument.short_name),), ('{}_re_im_frequency'.format(instrument.short_name),))

    def set_sweep(self, start, stop, npts, trnm):
        #  needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.labels = ('{}_Re_'.format(self._instrument.short_name) +trnm,
                        '{}_Im_'.format(self._instrument.short_name) +trnm)
        self.setpoints = ((f,), (f,))
        self.shapes = ((npts,), (npts,))

    def get_raw(self):
        #old_format = self._instrument.format()
        #self._instrument.format('Complex')
        data = self._instrument._get_sweep_data()
        #self._instrument.format(old_format)
        re = data[ 0::2 ]
        im = data[ 1::2 ]
        if len(re) != len(im):
            re = data[0:-1:2]
        return re, im    

class FrequencySweepMag(ArrayParameter):

    def __init__(self, name, instrument, start, stop, npts, trnm):
        super().__init__(name, shape=(npts,),
                         instrument=instrument,
                         unit='dB',
                         label='{}_mag_'.format(instrument.short_name) +trnm,
                         setpoint_units=('Hz',),
                         setpoint_labels=('{} frequency'.format(instrument.short_name),),
                         setpoint_names=('{}_mag_frequency'.format(instrument.short_name),))
        self.set_sweep(start, stop, npts, trnm)

    def set_sweep(self, start, stop, npts, trnm):
        #  needed to update config of the software parameter on sweep change
        # freq setpoints tuple as needs to be hashable for look up
        f = tuple(np.linspace(int(start), int(stop), num=npts))
        self.setpoints = (f,)
        self.shape = (npts,)

    def get_raw(self):
        data = self._instrument._get_sweep_data()
        re = data[ 0::2 ]
        im = data[ 1::2 ]
        if len(re) != len(im):
            re = data[0:-1:2]
        s = 10 * np.log10( np.square( re ) + np.square( im ) )
        return s

class ANVNA(VisaInstrument):
    """
    Requires FrequencySweep parameter for taking a trace

    Args:
        name: instrument name
        address: Address of instrument probably in format 'TCPIP0::192.168.15.100::inst0::INSTR'
    """

    def __init__(self, name, address, min_freq=10e6, max_freq=8e9,
                         min_power=-60, max_power=-10, trace_name='S11') -> None:

        super().__init__(name=name, address=address, terminator='\n')
        self.trace_name = trace_name
        self.min_freq = min_freq
        self.max_freq = max_freq

        #Set the device in continuous sweep mode
        self.visa_handle.write(":SENS:HOLD:FUNC CONT")

        # Drive power
        self.add_parameter('power',
                           label='Power',
                           get_cmd='SOUR:POW?',
                           set_cmd='SOUR:POW {}',
                           vals=Strings())

        # IF bandwidth
        self.add_parameter('if_bandwidth',
                           label='IF Bandwidth',
                           get_cmd='SENS:BAND?',
                           get_parser=float,
                           set_cmd='SENS:BAND {:.2f}',
                           unit='Hz')

        # Number of averages (also resets averages)
        self.add_parameter('averages_enabled',
                           label='Averages Enabled',
                           get_cmd="SENS:AVER?",
                           set_cmd="SENS:AVER {}",
                           val_mapping={True: '1', False: '0'})
        self.add_parameter('averages',
                           label='Averages',
                           get_cmd='SENS:AVER:COUN?',
                           get_parser=int,
                           set_cmd='SENS:AVER:COUN {:d}',
                           unit='')

        # Setting frequency range
        self.add_parameter('start',
                           label='Start Frequency',
                           get_cmd='SENS:FREQ:STAR?',
                           get_parser=float,
                           set_cmd=self._set_start,
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))
        self.add_parameter('stop',
                           label='Stop Frequency',
                           get_cmd='SENS:FREQ:STOP?',
                           get_parser=float,
                           set_cmd=self._set_stop,
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))
        self.add_parameter('center',
                           label='Center Frequency',
                           get_cmd='SENS:FREQ:CENT?',
                           get_parser=float,
                           set_cmd=self._set_center,
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))
        self.add_parameter('span',
                           label='Frequency Span',
                           get_cmd='SENS:FREQ:SPAN?',
                           get_parser=float,
                           set_cmd=self._set_span,
                           unit='Hz',
                           vals=Numbers(min_value=min_freq,
                                        max_value=max_freq))

        # Number of points in a sweep
        self.add_parameter('points',
                           label='Points',
                           get_cmd='SENS:SWE:POIN?',
                           get_parser=int,
                           set_cmd=self._set_points,
                           unit='',
                           vals=Ints(min_value=1,max_value=10000))

        self.add_parameter('trace',
                           label='Trace',
                           get_cmd=self._Sparam,
                           set_cmd=self._set_Sparam, 
                           vals=Strings())

        self.add_parameter(name='trace_re_im',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.points(),
                           trnm=self.trace(),
                           parameter_class=FrequencySweepReIm)

        self.add_parameter(name='trace_mag',
                           start=self.start(),
                           stop=self.stop(),
                           npts=self.points(),
                           trnm=self.trace(),
                           parameter_class=FrequencySweepMag)

        self.add_parameter('output_format',
                           get_cmd='FORMAT:DATa?',
                           get_parser=str,
                           set_cmd='FORMAT:DATa {}',
                           vals=Enum("REAL", "REAL32", "ASC"))
        
        self.connect_message()

        #Default parameters, mess with it on your own risk
        self.output_format("REAL")

    def _WaitComplete(self):
        result = 0
        print("Waiting...")
        while result == 0:
            time.sleep(.1) # pause exection in seconds (s)
            result = (self.visa_handle.query("*OPC?")).rstrip()
        print("Done!")


    def _get_sweep_data(self):
        """ Start a sweep and get the data once the sweep is completed.
        """
        self.visa_handle.write(":TRIG:SING")
        log.info("Sweep started")
        try:
            result = self.visa_handle.query_binary_values('CALCulate1:PARAmeter1:DATA:SDATa?', datatype='d', is_big_endian=False)
            return result
        except VisaIOError as maybe_timeout:
            if "Timeout" in maybe_timeout.description:
                log.warning("Function timed out before the sweep could be completed. You may have to increase the timeout \
                with .timeout()")
            raise maybe_timeout

    def _set_start(self, val):
        self.write('SENS:FREQ:STAR {}'.format(val))
        stop = self.stop()
        if val >= stop:
            raise ValueError(
                "Stop frequency must be larger than start frequency.")
        # we get start as the vna may not be able to set it to the exact value provided
        start = self.start()
        if val != start:
            log.warning(
                "Could not set start to {} setting it to {}".format(val, start))
        self.update_traces()

    def _set_stop(self, val):
        start = self.start()
        if val <= start:
            raise ValueError(
                "Stop frequency must be larger than start frequency.")
        self.write('SENS:FREQ:STOP {}'.format(val))
        # we get stop as the vna may not be able to set it to the exact value provided
        stop = self.stop()
        if val != stop:
           log.warning(
                "Could not set stop to {} setting it to {}".format(val, stop))
        self.update_traces()

    def _set_center(self, val):
        self.write('SENS:FREQ:CENT {}'.format(val))
        self.update_traces()

    def _set_points(self, val):
        self.write('SENS:SWE:POIN {}'.format(val))
        self.update_traces()

    def _set_span(self, val):
        self.write('SENS:FREQ:SPAN {}'.format(val))
        self.update_traces()

    def update_traces(self):
                    """ updates start, stop and npts of all trace parameters"""
                    start = self.start()
                    stop = self.stop()
                    npts = self.points()
                    trnm = self.trace()
                    for _, parameter in self.parameters.items():
                        if isinstance(parameter, (ArrayParameter, MultiParameter)):
                            try:
                                parameter.set_sweep(start, stop, npts, trnm)
                            except AttributeError:
                                pass

    def _Sparam(self) -> str:
        """
        Extrace S_parameter from returned PNA format
        """
        self.trace_name = self.ask("CALC:PAR:DEF?")
        return self.trace_name

    def _set_Sparam(self, val: str) -> None:
        """
        Set an S-parameter, in the format S<a><b>, where a and b
        can range from 1-4
        """
        if not re.match("S[1-4][1-4]", val):
            raise ValueError("Invalid S parameter spec")
        self.write(f"CALC:PAR:DEF {val}")
        self.trace_name = val
        self.update_traces()
