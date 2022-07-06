import agts
from statsmodels.tsa.seasonal import seasonal_decompose, STL


def simple_decomposition(x: agts.TimeSeries,
                         mode: str='additive'):
    return seasonal_decompose(x, mode)


def stl_decomposition(x: agts.TimeSeries):
    return STL(x)
