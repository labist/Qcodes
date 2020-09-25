import logging
from typing import Dict, Callable, List, Optional, Sequence
from functools import partial

import numpy as np

from qcodes import VisaInstrument, validators as vals
from qcodes import InstrumentChannel, ChannelList, Instrument
from qcodes import ArrayParameter, ParameterWithSetpoints
from qcodes.utils.validators import Enum, Numbers, Arrays


log = logging.getLogger(__name__)


class TraceNotReady(Exception):
    pass


class TraceSetPointsChanged(Exception):
    pass

# class Traces(MultiParameter):
#     def __init__(self, name, instrument):
#         # only name, names, and shapes are required
#         # this version returns two scalars (shape = `()`)
#         super().__init__(name, names=('Ch1', 'Ch2', 'Ch3', 'Ch4', 'Time'), shapes=(5,),
#                          labels=('Voltage', 'Voltage', 'Voltage', 'Voltage', 'Seconds'),
#                          units=('V', 'V', 'V', 'V', 's'),
#                          # including these setpoints is unnecessary here, but
#                          # if you have a parameter that returns a scalar alongside
#                          # an array you can represent the scalar as an empty sequence.
#                          setpoints=((), ()),
#                          docstring='')
#         self._instrument = instrument

#     def get_raw(self):
#         traces = self._instrument.get_current_traces()
#         return traces

class Traces(ParameterWithSetpoints):

    def get_raw(self):
        data = self.root_instrument.get_current_traces()
        return data

class FunctionTrace(ArrayParameter):
    """
    raw_trace will return a trace from OSCIL
    """

    def __init__(self, name, instrument, channel):
        super().__init__(name,
                         shape=(1024,),
                         label='Voltage',
                         unit='V',
                         setpoint_names=('Time',),
                         setpoint_labels=(
                             'Channel {} time series'.format(channel),),
                         setpoint_units=('s',),
                         docstring='raw trace from the scope',
                         )
        self._channel = channel
        self._instrument = instrument

    def prepare_curvedata(self):
        """
        Prepare the scope for returning curve data
        """
        # To calculate set points, we must have the full preamble
        # For the instrument to return the full preamble, the channel
        # in question must be displayed

        # shorthand
        instr = self._instrument
        # number of set points
        self.npts = int(instr.ask("WAV:POIN?"))
        # first set point
        self.xorigin = float(instr.ask(":WAVeform:XORigin?"))
        # step size
        self.xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        # calculate set points
        xdata = np.linspace(self.xorigin,
                            self.npts * self.xincrem + self.xorigin, self.npts)

        # set setpoints
        self.setpoints = (tuple(xdata), )
        self.shape = (self.npts, )

        # make this on a per channel basis?
        self._instrument._parent.trace_ready = True

    def get_raw(self):
        # when get is called the setpoints have to be known already
        # (saving data issue). Therefor create additional prepare function that
        # queries for the size.
        # check if already prepared
        if not self._instrument._parent.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for acquiring a trace.')

        # shorthand
        instr = self._instrument

        # set up the instrument
        # ---------------------------------------------------------------------

        # TODO: check number of points
        # check if requested number of points is less than 500 million

        # get intrument state
        # state = instr.ask(':RSTate?')
        # realtime mode: only one trigger is used
        # instr._parent.acquire_mode('RTIMe')

        # acquire the data
        # ---------------------------------------------------------------------

        # digitize is the actual call for acquisition, blocks
        instr.write(':DIGitize FUNC{}'.format(self._channel))

        # transfer the data
        # ---------------------------------------------------------------------

        # select the channel from which to read
        instr._parent.data_source('FUNC{}'.format(self._channel))
        # specifiy the data format in which to read
        instr.write(':WAVeform:FORMat WORD')
        instr.write(":waveform:byteorder LSBFirst")
        instr.write(":WAVeform:UNSigned 0")
        instr.write(':WAVeform:POINts:MODE NORMal')


        # request the actual transfer
        data = instr._parent.visa_handle.query_binary_values(
            'WAV:DATA?', datatype='h', is_big_endian=False)
        # the Infiniium does not include an extra termination char on binary
        # messages so we set expect_termination to False

        if len(data) != self.shape[0]:
            raise TraceSetPointsChanged('{} points have been aquired and {} \
            set points have been prepared in \
            prepare_curvedata'.format(len(data), self.shape[0]))
        # check x data scaling
        Pre = instr.ask(':WAVeform:PREamble?').split(',')
        xinc = float(Pre[4])
        xorigin = float(Pre[5])
        xref = float(Pre[6])
        yinc = float(Pre[7])
        yoriging = float(Pre[8])
        yref = float(Pre[9])
        # y data scaling
        yorigin = float(instr.ask(":WAVeform:YORigin?"))
        yinc = float(instr.ask(":WAVeform:YINCrement?"))

        channel_data = np.array(data)
        channel_data = np.multiply(np.subtract(channel_data, yref), yinc) + yorigin

        # restore original state
        # ---------------------------------------------------------------------

        # switch display back on
        instr.write(':FUNC{}:DISPlay ON'.format(self._channel))
        # continue refresh
        # if state == 'RUN':
        instr.write(':RUN')

        return channel_data

class RawTrace(ArrayParameter):
    """
    raw_trace will return a trace from OSCIL
    """

    def __init__(self, name, instrument, channel):
        super().__init__(name,
                         shape=(1024,),
                         label='Voltage',
                         unit='V',
                         setpoint_names=('Time',),
                         setpoint_labels=(
                             'Channel {} time series'.format(channel),),
                         setpoint_units=('s',),
                         docstring='raw trace from the scope',
                         )
        self._channel = channel
        self._instrument = instrument

    def prepare_curvedata(self):
        """
        Prepare the scope for returning curve data
        """
        # To calculate set points, we must have the full preamble
        # For the instrument to return the full preamble, the channel
        # in question must be displayed

        # shorthand
        instr = self._instrument
        # number of set points
        self.npts = int(instr.ask("WAV:POIN?"))
        # first set point
        self.xorigin = float(instr.ask(":WAVeform:XORigin?"))
        # step size
        self.xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        # calculate set points
        xdata = np.linspace(self.xorigin,
                            self.npts * self.xincrem + self.xorigin, self.npts)

        # set setpoints
        self.setpoints = (tuple(xdata), )
        self.shape = (self.npts, )

        # make this on a per channel basis?
        self._instrument._parent.trace_ready = True

    def get_raw(self):
        # when get is called the setpoints have to be known already
        # (saving data issue). Therefor create additional prepare function that
        # queries for the size.
        # check if already prepared
        if not self._instrument._parent.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for acquiring a trace.')

        # shorthand
        instr = self._instrument

        # set up the instrument
        # ---------------------------------------------------------------------

        # TODO: check number of points
        # check if requested number of points is less than 500 million

        # get intrument state
        # state = instr.ask(':RSTate?')
        # realtime mode: only one trigger is used
        # instr._parent.acquire_mode('RTIMe')

        # acquire the data
        # ---------------------------------------------------------------------

        # digitize is the actual call for acquisition, blocks
        instr.write(':DIGitize CHANnel{}'.format(self._channel))

        # transfer the data
        # ---------------------------------------------------------------------

        # select the channel from which to read
        instr._parent.data_source('CHAN{}'.format(self._channel))
        # specifiy the data format in which to read
        instr.write(':WAVeform:FORMat WORD')
        instr.write(":waveform:byteorder LSBFirst")
        # instr.write(":WAVeform:UNSigned 1")

        # request the actual transfer
        data = instr._parent.visa_handle.query_binary_values(
            'WAV:DATA?', datatype='H', is_big_endian=False)
        # the Infiniium does not include an extra termination char on binary
        # messages so we set expect_termination to False

        if len(data) != self.shape[0]:
            raise TraceSetPointsChanged('{} points have been aquired and {} \
            set points have been prepared in \
            prepare_curvedata'.format(len(data), self.shape[0]))
        # check x data scaling
        xorigin = float(instr.ask(":WAVeform:XORigin?"))
        # step size
        xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        error = self.xorigin - xorigin
        # this is a bad workaround
        if error > xincrem:
            raise TraceSetPointsChanged('{} is the prepared x origin and {} \
            is the x origin after the measurement.'.format(self.xorigin,
                                                           xorigin))
        error = (self.xincrem - xincrem) / xincrem
        if error > 1e-6:
            raise TraceSetPointsChanged('{} is the prepared x increment and {} \
            is the x increment after the measurement.'.format(self.xincrem,
                                                              xincrem))
        # y data scaling
        yorigin = float(instr.ask(":WAVeform:YORigin?"))
        yinc = float(instr.ask(":WAVeform:YINCrement?"))
        channel_data = np.array(data)
        channel_data = np.multiply(channel_data, yinc) + yorigin

        # restore original state
        # ---------------------------------------------------------------------

        # switch display back on
        instr.write(':CHANnel{}:DISPlay ON'.format(self._channel))
        # continue refresh
        # if state == 'RUN':
        instr.write(':RUN')

        return channel_data

    def get_function(self):
        # when get is called the setpoints have to be known already
        # (saving data issue). Therefor create additional prepare function that
        # queries for the size.
        # check if already prepared
        if not self._instrument._parent.trace_ready:
            raise TraceNotReady('Please run prepare_curvedata to prepare '
                                'the scope for acquiring a trace.')

        # shorthand
        instr = self._instrument

        # set up the instrument
        # ---------------------------------------------------------------------

        # TODO: check number of points
        # check if requested number of points is less than 500 million

        # get intrument state
        # state = instr.ask(':RSTate?')
        # realtime mode: only one trigger is used
        # instr._parent.acquire_mode('RTIMe')

        # acquire the data
        # ---------------------------------------------------------------------

        # digitize is the actual call for acquisition, blocks
        instr.write(':DIGitize FUNC{}'.format(self._channel))

        # transfer the data
        # ---------------------------------------------------------------------

        # select the channel from which to read
        instr._parent.data_source('FUNC{}'.format(self._channel))
        # specifiy the data format in which to read
        instr.write(':WAVeform:FORMat WORD')
        instr.write(":waveform:byteorder LSBFirst")
        # instr.write(":WAVeform:UNSigned 0")

        # request the actual transfer
        data = instr._parent.visa_handle.query_binary_values(
            'WAV:DATA?', datatype='h', is_big_endian=False)
        # the Infiniium does not include an extra termination char on binary
        # messages so we set expect_termination to False

        if len(data) != self.shape[0]:
            raise TraceSetPointsChanged('{} points have been aquired and {} \
            set points have been prepared in \
            prepare_curvedata'.format(len(data), self.shape[0]))
        # check x data scaling
        xorigin = float(instr.ask(":WAVeform:XORigin?"))
        # step size
        xincrem = float(instr.ask(":WAVeform:XINCrement?"))
        error = self.xorigin - xorigin
        # this is a bad workaround
        if error > xincrem:
            raise TraceSetPointsChanged('{} is the prepared x origin and {} \
            is the x origin after the measurement.'.format(self.xorigin,
                                                        xorigin))
        error = (self.xincrem - xincrem) / xincrem
        if error > 1e-6:
            raise TraceSetPointsChanged('{} is the prepared x increment and {} \
            is the x increment after the measurement.'.format(self.xincrem,
                                                            xincrem))
        # y data scaling
        yorigin = float(instr.ask(":WAVeform:YORigin?"))
        yinc = float(instr.ask(":WAVeform:YINCrement?"))
        channel_data = np.array(data)
        channel_data = np.multiply(channel_data, yinc) + yorigin

        # restore original state
        # ---------------------------------------------------------------------

        # switch display back on
        instr.write(':FUNC{}:DISPlay ON'.format(self._channel))
        # continue refresh
        # if state == 'RUN':
        instr.write(':RUN')

        return channel_data

class MeasurementSubsystem(InstrumentChannel):
    """
    Submodule containing the measurement subsystem commands and associated
    parameters
    """
    # note: this is not really a channel, but InstrumentChannel does everything
    # a 'Submodule' class should do

    def __init__(self, parent: Instrument, name: str, **kwargs) -> None:
        super().__init__(parent, name, **kwargs)

        self.add_parameter(name='source_1',
                           label='Measurement primary source',
                           set_cmd=partial(self._set_source, 1),
                           get_cmd=partial(self._get_source, 1),
                           val_mapping={i: f'CHAN{i}' for i in range(1, 5)},
                           snapshot_value=False)

        self.add_parameter(name='source_2',
                           label='Measurement secondary source',
                           set_cmd=partial(self._set_source, 2),
                           get_cmd=partial(self._get_source, 2),
                           val_mapping={i: f'CHAN{i}' for i in range(1, 5)},
                           snapshot_value=False)

        self.add_parameter(name='amplitude',
                           label='Voltage amplitude',
                           get_cmd=self._make_meas_cmd('VAMPlitude'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='average',
                           label='Voltage average',
                           get_cmd=self._make_meas_cmd('VAVerage'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='base',
                           label='Statistical base',
                           get_cmd=self._make_meas_cmd('VBASe'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='frequency',
                           label='Signal frequency',
                           get_cmd=self._make_meas_cmd('FREQuency'),
                           get_parser=float,
                           unit='Hz',
                           docstring="""
                                     measure the frequency of the first
                                     complete cycle on the screen using
                                     the mid-threshold levels of the waveform
                                     """,
                           snapshot_value=False)

        self.add_parameter(name='lower',
                           label='Voltage lower',
                           get_cmd=self._make_meas_cmd('VLOWer'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='max',
                           label='Voltage maximum',
                           get_cmd=self._make_meas_cmd('VMAX'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='middle',
                           label='Middle threshold voltage',
                           get_cmd=self._make_meas_cmd('VMIDdle'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='min',
                           label='Voltage minimum',
                           get_cmd=self._make_meas_cmd('VMIN'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='overshoot',
                           label='Voltage overshoot',
                           get_cmd=self._make_meas_cmd('VOVershoot'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='vpp',
                           label='Voltage peak-to-peak',
                           get_cmd=self._make_meas_cmd('VPP'),
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='rms',
                           label='Voltage RMS',
                           get_cmd=self._make_meas_cmd('VRMS') + ' DISPlay, DC',
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

        self.add_parameter(name='rms_no_DC',
                           label='Voltage RMS',
                           get_cmd=self._make_meas_cmd('VRMS') + ' DISPlay, AC',
                           get_parser=float,
                           unit='V',
                           snapshot_value=False)

    @staticmethod
    def _make_meas_cmd(cmd: str) -> str:
        """
        Helper function to avoid typos
        """
        return f':MEASure:{cmd}?'

    def _set_source(self, rank: int, source: str) -> None:
        """
        Set the measurement source, either primary (rank==1) or secondary
        (rank==2)
        """
        sources = self.ask(':MEASure:SOURCE?').split(',')
        if rank == 1:
            self.write(f':MEASure:SOURCE {source}, {sources[1]}')
        else:
            self.write(f':MEASure:SOURCE {sources[0]}, {source}')

    def _get_source(self, rank: int) -> str:
        """
        Get the measurement source, either primary (rank==1) or secondary
        (rank==2)
        """
        sources = self.ask(':MEASure:SOURCE?').split(',')

        return sources[rank-1]


class InfiniiumChannel(InstrumentChannel):

    def __init__(self, parent, name, channel):
        super().__init__(parent, name)
        # display
        self.add_parameter(name='display',
                           label='Channel {} display on/off'.format(channel),
                           set_cmd='CHANnel{}:DISPlay {{}}'.format(channel),
                           get_cmd='CHANnel{}:DISPlay?'.format(channel),
                           val_mapping={True: 1, False: 0},
                           )
        # scaling
        self.add_parameter(name='offset',
                           label='Channel {} offset'.format(channel),
                           set_cmd='CHAN{}:OFFS {{}}'.format(channel),
                           unit='V',
                           get_cmd='CHAN{}:OFFS?'.format(channel),
                           get_parser=float
                           )

        # scale and range are interdependent, when setting one, invalidate the
        # the other.
        # Scale is commented out for this reason
        # self.add_parameter(name='scale',
        #                    label='Channel {} scale'.format(channel),
        #                    unit='V/div',
        #                    set_cmd='CHAN{}:SCAL {{}}'.format(channel),
        #                    get_cmd='CHAN{}:SCAL?'.format(channel),
        #                    get_parser=float,
        #                    vals=vals.Numbers(0,100)  # TODO: upper limit?
        #                    )

        self.add_parameter(name='range',
                           label='Channel {} range'.format(channel),
                           unit='V',
                           set_cmd='CHAN{}:RANG {{}}'.format(channel),
                           get_cmd='CHAN{}:RANG?'.format(channel),
                           get_parser=float,
                           vals=vals.Numbers()
                           )
        # trigger
        self.add_parameter(
            'trigger_level',
            label='Tirgger level channel {}'.format(channel),
            unit='V',
            get_cmd=':TRIGger:LEVel? CHANnel{}'.format(channel),
            set_cmd=':TRIGger:LEVel CHANnel{},{{}}'.format(channel),
            get_parser=float,
            vals=Numbers(),
        )

        # Acquisition
        self.add_parameter(name='trace',
                           channel=channel,
                           parameter_class=RawTrace
                           )

        self.add_parameter(name='function',
                    channel=channel,
                    parameter_class=FunctionTrace
                    )




class Infiniium(VisaInstrument):
    """
    This is the QCoDeS driver for the Keysight Infiniium oscilloscopes from the
     - tested for MSOS104A of the Infiniium S-series.
    """

    def __init__(self, name, address, timeout=60, **kwargs):
        """
        Initialises the oscilloscope.

        Args:
            name (str): Name of the instrument used by QCoDeS
        address (string): Instrument address as used by VISA
            timeout (float): visa timeout, in secs.
        """

        super().__init__(name, address, timeout=timeout,
                         terminator='\n', **kwargs)
        self.connect_message()

        # Scope trace boolean
        self.trace_ready = False

        # switch the response header off,
        # else none of our parameters will work
        self.write(':SYSTem:HEADer OFF')

        # functions

        # general parameters

        # the parameters are in the same order as the front panel.
        # Beware, he list of implemented parameters is not complete. Refer to
        # the manual (Infiniium prog guide) for an equally infiniium list.

        # time base

        # timebase_scale is commented out for same reason as channel scale
        # use range instead
        # self.add_parameter('timebase_scale',
        #                    label='Scale of the one time devision',
        #                    unit='s/Div',
        #                    get_cmd=':TIMebase:SCALe?',
        #                    set_cmd=':TIMebase:SCALe {}',
        #                    vals=Numbers(),
        #                    get_parser=float,
        #                    )

        self.add_parameter('timebase_range',
                           label='Range of the time axis',
                           unit='s',
                           get_cmd=':TIMebase:RANGe?',
                           set_cmd=':TIMebase:RANGe {}',
                           vals=Numbers(5e-12, 20),
                           get_parser=float,
                           )
        self.add_parameter('timebase_position',
                           label='Offset of the time axis',
                           unit='s',
                           get_cmd=':TIMebase:POSition?',
                           set_cmd=':TIMebase:POSition {}',
                           vals=Numbers(),
                           get_parser=float,
                           )

        self.add_parameter('timebase_roll_enabled',
                           label='Is rolling mode enabled',
                           get_cmd=':TIMebase:ROLL:ENABLE?',
                           set_cmd=':TIMebase:ROLL:ENABLE {}',
                           val_mapping={True: 1, False: 0}
                           )

        # trigger
        self.add_parameter('trigger_enabled',
                           label='Is trigger enabled',
                           get_cmd=':TRIGger:AND:ENABLe?',
                           set_cmd=':TRIGger:AND:ENABLe {}',
                           val_mapping={True: 1, False: 0}
                           )

        self.add_parameter('trigger_edge_source',
                           label='Source channel for the edge trigger',
                           get_cmd=':TRIGger:EDGE:SOURce?',
                           set_cmd=':TRIGger:EDGE:SOURce {}',
                           vals=Enum(*(
                               ['CHANnel{}'.format(i) for i in range(1, 4 + 1)] +
                               ['CHAN{}'.format(i) for i in range(1, 4 + 1)] +
                               ['DIGital{}'.format(i) for i in range(16 + 1)] +
                               ['DIG{}'.format(i) for i in range(16 + 1)] +
                               ['AUX', 'LINE']))
                           )  # add enum for case insesitivity
        self.add_parameter('trigger_edge_slope',
                           label='slope of the edge trigger',
                           get_cmd=':TRIGger:EDGE:SLOPe?',
                           set_cmd=':TRIGger:EDGE:SLOPe {}',
                           vals=Enum('positive', 'negative', 'neither')
                           )
        self.add_parameter('trigger_level_aux',
                           label='Tirgger level AUX',
                           unit='V',
                           get_cmd=':TRIGger:LEVel? AUX',
                           set_cmd=':TRIGger:LEVel AUX,{}',
                           get_parser=float,
                           vals=Numbers(),
                           )
        # Aquisition
        # If sample points, rate and timebase_scale are set in an
        # incomensurate way, the scope only displays part of the waveform
        self.add_parameter('acquire_points',
                           label='sample points',
                           get_cmd='ACQ:POIN?',
                           get_parser=int,
                           set_cmd=self._cmd_and_invalidate('ACQ:POIN {}'),
                           unit='pts',
                           vals=vals.Numbers(min_value=1, max_value=100e6)
                           )

        self.add_parameter('acquire_sample_rate',
                           label='sample rate',
                           get_cmd='ACQ:SRAT?',
                           set_cmd=self._cmd_and_invalidate('ACQ:SRAT {}'),
                           unit='Sa/s',
                           get_parser=float
                           )

        # this parameter gets used internally for data aquisition. For now it
        # should not be used manually
        self.add_parameter('data_source',
                           label='Waveform Data source',
                           get_cmd=':WAVeform:SOURce?',
                           set_cmd=':WAVeform:SOURce {}',
                           vals = Enum( *(\
                                ['CHANnel{}'.format(i) for i in range(1, 4+1)]+\
                                ['CHAN{}'.format(i) for i in range(1, 4+1)]+\
                                ['DIFF{}'.format(i) for i in range(1, 2+1)]+\
                                ['COMMonmode{}'.format(i) for i in range(3, 4+1)]+\
                                ['COMM{}'.format(i) for i in range(3, 4+1)]+\
                                ['FUNCtion{}'.format(i) for i in range(1, 16+1)]+\
                                ['FUNC{}'.format(i) for i in range(1, 16+1)]+\
                                ['WMEMory{}'.format(i) for i in range(1, 4+1)]+\
                                ['WMEM{}'.format(i) for i in range(1, 4+1)]+\
                                ['BUS{}'.format(i) for i in range(1, 4+1)]+\
                                ['HISTogram', 'HIST', 'CLOCK']+\
                                ['MTRend', 'MTR']))
                           )

        # TODO: implement as array parameter to allow for setting the other filter
        # ratios
        self.add_parameter('acquire_interpolate',
                            get_cmd=':ACQuire:INTerpolate?',
                            set_cmd=self._cmd_and_invalidate(':ACQuire:INTerpolate {}'),
                            val_mapping={True: 1, False: 0}
                            )

        self.add_parameter('acquire_mode',
                            label='Acquisition mode',
                            get_cmd= 'ACQuire:MODE?',
                            set_cmd='ACQuire:MODE {}',
                            vals=Enum('ETIMe', 'RTIMe', 'PDETect',
                                      'HRESolution', 'SEGMented',
                                      'SEGPdetect', 'SEGHres')
                            )

        self.add_parameter('acquire_timespan',
                            get_cmd=(lambda: self.acquire_points.get_latest() \
                                            /self.acquire_sample_rate.get_latest()),
                            unit='s',
                            get_parser=float
                            )

        # time of the first point
        self.add_parameter('waveform_xorigin',
                            get_cmd='WAVeform:XORigin?',
                            unit='s',
                            get_parser=float
                            )

        self.add_parameter('data_format',
                           set_cmd='SAV:WAV:FORM {}',
                           val_mapping={'csv': 'CSV',
                                        'binary': 'BIN',
                                        'asciixy': 'ASC'},
                           docstring=("Set the format for saving "
                                      "files using save_data function")
                           )
        # Acquisition
        self.add_parameter('traces',
                            unit='V',
                            setpoints=(self.acquire_points,),
                            label='Traces',
                            parameter_class=Traces,
                            vals=Arrays(shape=(self.get_current_traces,))
                            )     
        # Channels
        channels = ChannelList(self, "Channels", InfiniiumChannel,
                                snapshotable=False)

        for i in range(1,5):
            channel = InfiniiumChannel(self, 'chan{}'.format(i), i)
            channels.append(channel)
            self.add_submodule('ch{}'.format(i), channel)
        channels.lock()
        self.add_submodule('channels', channels)

        # Submodules
        meassubsys = MeasurementSubsystem(self, 'measure')
        self.add_submodule('measure', meassubsys)

    def _cmd_and_invalidate(self, cmd: str) -> Callable:
        return partial(Infiniium._cmd_and_invalidate_call, self, cmd)

    def _cmd_and_invalidate_call(self, cmd: str, val) -> None:
        """
        executes command and sets trace_ready status to false
        any command that effects the number of setpoints should invalidate the trace
        """
        self.trace_ready = False
        self.write(cmd.format(val))

    def save_data(self, filename):
        """
        Saves the channels currently shown on oscilloscope screen to a USB.
        Must set data_format parameter prior to calling this
        """
        self.write(f'SAV:WAV "{filename}"')

    def get_current_traces(self, npts=10000, channels: Optional[Sequence[int]] = None
                          ) -> Dict:
        """
        Get the current traces of 'channels' on the oscillsocope.

        Args:
            channels: default [1, 2, 3, 4]
                list of integers representing the channels.
                gets the traces of these channels.
                the only valid integers are 1,2,3,4
                will turn any other channels off

        Returns:
            a dict with keys 'ch1', 'ch2', 'ch3', 'ch4', 'time',
            and values are np.ndarrays, corresponding to the voltages
            of the four channels and the common time axis
        """

        instr = self

        ACQ_TYPE = str(instr.ask(":ACQuire:TYPE?")).strip("\n")
        ## This can also be done when pulling pre-ambles (pre[1]) or may be known ahead of time, but since the script is supposed to find everything, it is done now.
        if ACQ_TYPE == "AVER" or ACQ_TYPE == "HRES": # Don't need to check for both types of mnemonics like this: if ACQ_TYPE == "AVER" or ACQ_TYPE == "AVERage": becasue the scope ALWAYS returns the short form
            POINTS_MODE = "NORMal" # Use for Average and High Resoultion acquisition Types.
                ## If the :WAVeform:POINts:MODE is RAW, and the Acquisition Type is Average, the number of points available is 0. If :WAVeform:POINts:MODE is MAX, it may or may not return 0 points.
                ## If the :WAVeform:POINts:MODE is RAW, and the Acquisition Type is High Resolution, then the effect is (mostly) the same as if the Acq. Type was Normal (no box-car averaging).
                ## Note: if you use :SINGle to acquire the waveform in AVERage Acq. Type, no average is performed, and RAW works. See sample script "InfiniiVision_2_Simple_Synchronization_Methods.py"
        else:
            POINTS_MODE = "RAW" # Use for Acq. Type NORMal or PEAK
            ## Note, if using "precision mode" on 5/6/70000s or X6000A, then you must use POINTS_MODE = "NORMal" to get the "precision record."

        ## Note:
            ## :WAVeform:POINts:MODE RAW corresponds to saving the ASCII XY or Binary data formats to a USB stick on the scope
            ## :WAVeform:POINts:MODE NORMal corresponds to saving the CSV or H5 data formats to a USB stick on the scope

        ###########################################################################################################
        ## Find max points for scope as is, ask for desired points, find how many points will actually be returned
            ## KEY POINT: the data must be on screen to be retrieved.  If there is data off-screen, :WAVeform:POINts? will not "see it."
                ## Addendum 1 shows how to properly get all data on screen, but this is never needed for Average and High Resolution Acquisition Types,
                ## since they basically don't use off-screen data; what you see is what you get.

        ## First, set waveform source to any channel that is known to be on and have points, here the FIRST_CHANNEL_ON - if we don't do this, it could be set to a channel that was off or did not acquire data.
        instr.write(":WAVeform:SOURce CHANnel 1")

        ## The next line is similar to, but distinct from, the previously sent command ":WAVeform:POINts:MODE MAX".  This next command is one of the most important parts of this script.
        instr.write(":WAVeform:POINts MAX") # This command sets the points mode to MAX AND ensures that the maximum # of points to be transferred is set, though they must still be on screen

        ## Since the ":WAVeform:POINts MAX" command above also changes the :POINts:MODE to MAXimum, which may or may not be a good thing, so change it to what is needed next.
        instr.write(":WAVeform:POINts:MODE " + str(POINTS_MODE))
        ## If measurements are also being made, they are made on the "measurement record."  This record can be accessed by using:
            ## :WAVeform:POINts:MODE NORMal instead of :WAVeform:POINts:MODE RAW
            ## Please refer to the progammer's guide for more details on :WAV:POIN:MODE RAW/NORMal/MAX

        ## Now find how many points are actually currently available for transfer in the given points mode (must still be on screen)
        MAX_CURRENTLY_AVAILABLE_POINTS = int(instr.ask(":WAVeform:POINts?")) # This is the max number of points currently available - this is for on screen data only - Will not change channel to channel.
        ## NOTES:
            ## For getting ALL of the data off of the scope, as opposed to just what is on screen, see Addendum 1
            ## For getting ONLY CERTAIN data points, see Addendum 2
            ## The methods shown in these addenda are combinable
            ## The number of points can change with the number of channels that have acquired data, the Acq. Mode, Acq Type, time scale (they must be on screen to be retrieved),
                ## number of channels on, and the acquisition method (:RUNS/:STOP, :SINGle, :DIGitize), and :WAV:POINts:MODE
        USER_REQUESTED_POINTS = npts

        if ACQ_TYPE == "PEAK":
            USER_REQUESTED_POINTS = MAX_CURRENTLY_AVAILABLE_POINTS
            ## Note: for Peak Detect, it is always suggested to transfer the max number of points available so that narrow spikes are not missed.
            ## If the scope is asked for more points than :ACQuire:POINts? (see below) yields, though, not necessarily MAX_CURRENTLY_AVAILABLE_POINTS, it will throw an error, specifically -222,"Data out of range"

        ## If one wants some other number of points...
        ## Tell it how many points you want
        instr.write(":WAVeform:POINts " + str(USER_REQUESTED_POINTS))

        ## Then ask how many points it will actually give you, as it may not give you exactly what you want.
        NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE = int(instr.ask(":WAVeform:POINts?"))
        ## Warn user if points will be less than requested, if desired...
        ## Note that if less than the max is set, it will stay at that value (or whatever is closest) until it is changed again, even if the time base is changed.
        ## What does the scope return if less than MAX_CURRENTLY_AVAILABLE_POINTS is returned?
            ## It depends on the :WAVeform:POINts:MODE
            ## If :WAVeform:POINts:MODE is RAW
                ## The scope decimates the data, only returning every Nth point.
                ## The points are NOT re-mapped; the values of the points, both vertical and horizontal, are preserved.
                ## Aliasing, lost pulses and transitions, are very possible when this is done.
            ## If :WAVeform:POINts:MODE is NORMal
                ## The scope re-maps this "measurement record" down to the number of points requested to give the best representation of the waveform for the requested number of points.
                ## This changes both the vertical and horizontal values.
                ## Aliasing, lost pulses and transitions, are definitely possible, though less likely for well displayed waveforms in many, but not all, cases.

        ## This above method always works w/o errors.  In summary, after an acquisition is complete:
                ## Set POINts to MAX
                ## Set :POINts:MODE as desired/needed
                ## Ask for the number of points available.  This is the MOST the scope can give for current settings/timescale/Acq. Type
                ## Set a different number of points if desired and if less than above
                ## Ask how many points it will actually return, use that

        ## What about :ACQUIRE:POINTS?
        ## The Programmers's Guide says:
            ## The :ACQuire:POINts? query returns the number of data points that the
            ## hardware will acquire from the input signal. The number of points
            ## acquired is not directly controllable. To set the number of points to be
            ## transferred from the oscilloscope, use the command :WAVeform:POINts. The
            ## :WAVeform:POINts? query will return the number of points available to be
            ## transferred from the oscilloscope.

        ## It is not a terribly useful query. It basically only gives the max amount of points available for transfer if:
                ## The scope is stopped AND has acquired data the way you want to use it and the waveform is entirely on screen
                    ## In other words, if you do a :SINGle, THEN turn on, say digital chs, this will give the wrong answer for digital chs on for the next acquisition.
                ## :POINts:MODE is RAW or MAX - thus it DOES NOT work for Average or High Res. Acq. Types, which need NORMal!
                ## and RUN/STOP vs SINGle vs :DIG makes a difference!
                ## and Acq. Type makes a difference! (it can be misleading for Average or High Res. Acq. Types)
                ## and all of the data is on screen!
                ## Thus it is not too useful here.
        ## What it is good for is:
            ## 1. determining if there is off screen data, for Normal or Peak Detect Acq. Types, after an acquisition is complete, for the current settings (compare this result with MAX_CURRENTLY_AVAILABLE_POINTS).
            ## 2. finding the max possible points that could possibly be available for Normal or Peak Detect Acq. Types, after an acquisition is complete, for the current settings, if all of the data is on-screen.

        #####################################################################################################################################
        #####################################################################################################################################
        ## Get timing pre-amble data and create time axis
        ## One could just save off the preamble factors and #points and post process this later...

        Pre = instr.ask(":WAVeform:PREamble?").split(',') # This does need to be set to a channel that is on, but that is already done... e.g. Pre = instr.ask(":WAVeform:SOURce CHANnel" + str(FIRST_CHANNEL_ON) + ";PREamble?").split(',')
        ## While these values can always be used for all analog channels, they need to be retrieved and used separately for math/other waveforms as they will likely be different.
        #ACQ_TYPE    = float(Pre[1]) # Gives the scope Acquisition Type; this is already done above in this particular script
        X_INCrement = float(Pre[4]) # Time difference between data points; Could also be found with :WAVeform:XINCrement? after setting :WAVeform:SOURce
        X_ORIGin    = float(Pre[5]) # Always the first data point in memory; Could also be found with :WAVeform:XORigin? after setting :WAVeform:SOURce
        X_REFerence = float(Pre[6]) # Specifies the data point associated with x-origin; The x-reference point is the first point displayed and XREFerence is always 0.; Could also be found with :WAVeform:XREFerence? after setting :WAVeform:SOURce
        ## This could have been pulled earlier...
        del Pre
            ## The programmer's guide has a very good description of this, under the info on :WAVeform:PREamble.
            ## This could also be reasonably be done when pulling the vertical pre-ambles for any channel that is on and acquired data.
            ## This is the same for all channels.
            ## For repetitive acquisitions, it only needs to be done once unless settings change.

        DataTime = ((np.linspace(0,NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE-1,NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE)-X_REFerence)*X_INCrement)+X_ORIGin
        if ACQ_TYPE == "PEAK": # This means Peak Detect Acq. Type
            DataTime = np.repeat(DataTime,2)
            ##  The points come out as Low(time1),High(time1),Low(time2),High(time2)....
            ### SEE IMPORTANT NOTE ABOUT PEAK DETECT AT VERY END, specific to fast time scales

        ###################################################################################################
        ###################################################################################################
        ## Determine number of bytes that will actually be transferred and set the "chunk size" accordingly.

            ## When using PyVisa, this is in fact completely unnecessary, but may be needed in other leagues, MATLAB, for example.
            ## However, the benefit in Python is that the transfers can take less time, particularly longer ones.

        ## Get the waveform format
        WFORM = str(instr.ask(":WAVeform:FORMat?"))
        if WFORM == "BYTE":
            FORMAT_MULTIPLIER = 1
        else: #WFORM == "WORD"
            FORMAT_MULTIPLIER = 2

        if ACQ_TYPE == "PEAK":
            POINTS_MULTIPLIER = 2 # Recall that Peak Acq. Type basically doubles the number of points.
        else:
            POINTS_MULTIPLIER = 1

        TOTAL_BYTES_TO_XFER = POINTS_MULTIPLIER * NUMBER_OF_POINTS_TO_ACTUALLY_RETRIEVE * FORMAT_MULTIPLIER + 11
            ## Why + 11?  The IEEE488.2 waveform header for definite length binary blocks (what this will use) consists of 10 bytes.  The default termination character, \n, takes up another byte.
                ## If you are using mutliplr termination characters, adjust accordingly.
            ## Note that Python 2.7 uses ASCII, where all characters are 1 byte.  Python 3.5 uses Unicode, which does not have a set number of bytes per character.

        ## Set chunk size:
            ## More info @ http://pyvisa.readthedocs.io/en/stable/resources.html
        if TOTAL_BYTES_TO_XFER >= 400000:
            instr.chunk_size = TOTAL_BYTES_TO_XFER
        ## else:
            ## use default size, which is 20480

        ## Any given user may want to tweak this for best throughput, if desired.  The 400,000 was chosen after testing various chunk sizes over various transfer sizes, over USB,
            ## and determined to be the best, or at least simplest, cutoff.  When the transfers are smaller, the intrinsic "latencies" seem to dominate, and the default chunk size works fine.

        ## How does the default chuck size work?
            ## It just pulls the data repeatedly and sequentially (in series) until the termination character is found...

        ## Do I need to adjust the timeout for a larger chunk sizes, where it will pull up to an entire 8,000,000 sample record in a single IO transaction?
            ## If you use a 10s timeout (10,000 ms in PyVisa), that will be good enough for USB and LAN.
            ## If you are using GPIB, which is slower than LAN or USB, quite possibly, yes.
            ## If you don't want to deal with this, don't set the chunk size, and use a 10 second timeout, and everything will be fine in Python.
                ## When you use the default chunk size, there are repeated IO transactions to pull the total waveform.  It is each individual IO transaction that needs to complete within the timeout.

        #####################################################
        #####################################################
        ## Pull waveform data, scale it

        if channels is None:
            channels = [1, 2, 3, 4]
        # check that channels are valid
        try:
            assert all([ch in [1, 2, 3, 4] for ch in channels])
        except:
            raise Exception("invalid channel in %s, integers"
                            " must be 1,2,3 or 4" % channels)

        self.write('DIGitize')
        all_data = {}
        self.write(':SYSTem:HEADer OFF')

        for i in channels:
            self.data_source('CHAN%s' % i)
            self.write(':WAVeform:FORMat WORD')
            self.write(":waveform:byteorder LSBFirst")
            self.write(':WAVeform:STReaming OFF')
            y_incr = float(self.ask(":WAVeform:YINCrement?"))
            y_origin = float(self.ask(":WAVeform:YORigin?"))
            y_ref = float(self.ask(":WAVeform:YREFerence?"))

            data = self.visa_handle.query_binary_values(
                'WAV:DATA?', datatype='h', is_big_endian=False)
            all_data['ch%d' % i] = (np.array(data)- y_ref)* y_incr + y_origin
            self.write(':CHANnel{}:DISPlay ON'.format(i))

        x_incr = float(self.ask(":WAVeform:XINCrement?"))   
        all_data['time'] = np.arange(0, len(all_data['ch%s' % channels[0]])) \
            * x_incr

        self.write(':RUN')
        # turn the channels that were not requested off
        for ch in [i for i in [1, 2, 3, 4] if i not in channels]:
            self.write(':CHANnel{}:DISPlay OFF'.format(ch))

        if TOTAL_BYTES_TO_XFER >= 400000:
            instr.chunk_size = 20480
        return all_data
