# -*- coding: utf-8 -*-

"""
Information and synchrony measures for oscillator systems.

(C) Pedro A.M. Mediano, 2018.
"""
import numpy as np
import jpype as jp

def OInformation(series):
    """
    O-information measure of redundancy-minus-synergy, as described in

    Rosas, Mediano, Gastpar, and Jensen,
    Quantifying high-order effects via multivariate extensions of the mutual
    information.

    Input:
      theta -- T-by-N matrix of phases

    Output: scalar value
    """
    if series.shape[1] < 2:
        raise ValueError('Cannot calculate MI on a 1D time series.')
    elif series.shape[1] == 2:
        return 0

    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=infodynamics.jar')

    oCalc = jp.JClass('infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov')()
#    oCalc = jp.JClass('infodynamics.measures.continuous.gaussian.OInfoCalculatorGaussian')()
    actual_value = oCalc.compute(series)

    return actual_value

def OInfoPh(theta):
    """
    O-information measure of redundancy-minus-synergy, as described in

    Rosas, Mediano, Gastpar, and Jensen,
    Quantifying high-order effects via multivariate extensions of the mutual
    information.

    Input:
      theta -- T-by-N matrix of phases

    Output: scalar value
    """
    if theta.shape[1] < 2:
        raise ValueError('Cannot calculate MI on a 1D time series.')
    elif theta.shape[1] == 2:
        return 0

    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=infodynamics.jar')

    oCalc = jp.JClass('infodynamics.measures.continuous.kraskov.OInfoCalculatorKraskov')()
#    oCalc = jp.JClass('infodynamics.measures.continuous.gaussian.OInfoCalculatorGaussian')()
    actual_value = oCalc.compute(np.mod(theta, 2*np.pi))

    return actual_value


def TC(series):
    """
    Input:
      theta -- T-by-N matrix of phases

    Output: scalar value
    """
    if series.shape[1] < 2:
        raise ValueError('Cannot calculate MI on a 1D time series.')
    elif series.shape[1] == 2:
        return 0

    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=infodynamics.jar')

    TCcalc = jp.JClass('infodynamics.measures.continuous.kraskov.MultiInfoCalculatorKraskov1')()
#    TCcalc = jp.JClass('infodynamics.measures.continuous.gaussian.MultiInfoCalculatorGaussian')()

    TCcalc.initialise(series.shape[1])
    TCcalc.setObservations(series)
    actual_value = TCcalc.computeAverageLocalOfObservations()
#    surrogate_mean = TCcalc.computeSignificance(100).getMeanOfDistribution()
    return actual_value

def TCPh(theta):
    """
    Input:
      theta -- T-by-N matrix of phases

    Output: scalar value
    """
    if theta.shape[1] < 2:
        raise ValueError('Cannot calculate MI on a 1D time series.')
    elif theta.shape[1] == 2:
        return 0

    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=infodynamics.jar')

    TCcalc = jp.JClass('infodynamics.measures.continuous.kraskov.MultiInfoCalculatorKraskov1')()
#    TCcalc = jp.JClass('infodynamics.measures.continuous.gaussian.MultiInfoCalculatorGaussian')()

    TCcalc.initialise(theta.shape[1])
    TCcalc.setObservations(np.mod(theta, 2*np.pi))
    actual_value = TCcalc.computeAverageLocalOfObservations()
#    surrogate_mean = TCcalc.computeSignificance(100).getMeanOfDistribution()
    return actual_value


def H(series):
    """
    Input:
      theta -- T-by-N matrix of phases

    Output: scalar value
    """
    dims=series.shape[1]

    if not jp.isJVMStarted():
        jp.startJVM(jp.getDefaultJVMPath(), '-ea', '-Djava.class.path=infodynamics.jar')

    hcalc = jp.JClass('infodynamics.measures.continuous.kozachenko.EntropyCalculatorMultiVariateKozachenko')()
#    TCcalc = jp.JClass('infodynamics.measures.continuous.gaussian.MultiInfoCalculatorGaussian')()

    hcalc.initialise(dims)
    hcalc.setObservations(series[:,:2],series[:,2:])#np.mod(theta, 2*np.pi))
    actual_value = hcalc.computeAverageLocalOfObservations()
#    surrogate_mean = TCcalc.computeSignificance(100).getMeanOfDistribution()
    return actual_value


