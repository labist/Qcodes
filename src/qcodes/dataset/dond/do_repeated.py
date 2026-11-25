from qcodes.dataset.dond.do_nd import dond
from qcodes.dataset.dond.sweeps import LinSweep
from qcodes.parameters import ElapsedTimeParameter, Parameter

def do_repeated(npts, delay, *params, before_meas_actions=(), **kwargs):
    """
    Repeatedly measure *params using dond. Also measures elapsed time
    Args:
        npts: number of points in sweep
        delay: time between points
        *params: parameters to measure
        before_meas_actions: actions to do before measurement
        **kwargs: passed on to dond.
    """
    duration = ElapsedTimeParameter("duration")
    iteration = Parameter("iteration", label='Iteration', unit='#', set_cmd=None, get_cmd=None)

    return dond(
        LinSweep(iteration, 0, npts, npts+1, delay=delay),
        *params,
        duration,
        before_meas_actions=before_meas_actions,
        **kwargs
    )