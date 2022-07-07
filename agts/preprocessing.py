import pandas as pd
from pandas import DataFrame, Series
import numpy as np

def interpolate(data, method):
    """
    interpolation 함수는 SciPy와 Pandas에서 찾을 수 있음.
    특히, SciPy는 다양한 interpolation 함수가 있음. 함수 간에
    어떤 차이가 있는지 구현을 해야하는지 확인할 필요가 있음.

    interpolation의 경우 결측 값 및 구간의 앞, 뒤 값과 인덱스 필요함 따라서 데이터의 첫번째, 마지막이 nan value인 경우 interpolation이 안됨
    :param data : data(ndarry) of data(Dataframe)
    :param method : interpolation 방법 method = ["linear", "spline]
    "linear" : 1차 선형식 기반 interpolation
    "spline" : 다차 방정식 기반 interpolation -> 3차 방정식 기반 "cubic spline"
    :return:
    """
    data = data
    method = method

    if isinstance(data, np.ndarry) == False:  # ndarray로 변경
        data_array = TimeSeries.to_array(data)
    data_nan = np.argwhere(np.isnan(data_array))  # 주어진 데이터에서 nan value 위치를 array 형태로 반환
    nan_column = data_nan[:, 1]  # nan values 위치에서 열 추출
    nan_column = np.unique(nan_column)  # 중복 열 제외

    "linear interpolation"
    if method == "linear":
        for i in nan_column: # nan values 있는 column 만 탐색
            for j in range(len(data_array)):
                """ nan value 가 실측 값 사이 1개인 경우"""
                if np.isnan(data_array[j, i]) == True and np.isnan(data_array[j+1, i]) == False:
                    value1_index = j-1
                    value1 = data_array[j-1, i]
                    nan_index = j
                    value2_index = j+1
                    value2 = data_array[j+1, i]
                    data_array[j,i] = linear(value1, value1_index, value2, value2_index, nan_index)
                    """ nan values 가 실측 값 사이 구간으로 존재"""
                elif np.isnan(data_array[j, i]) == True and np.isnan(data_array[j+1, i]) == False:
                    num = 1
                    while np.isnan(data_array[j+num, i]) == True:
                        num += 1
                    value1_index = j-1
                    value1 = data_array[j-1, i]
                    nan_index = j
                    value2_index = j+num
                    value2 = data_array[j+num, i]
                    data_array[j,i] = linear(value1, value1_index, value2, value2_index, nan_index)
    else :
        pass
    pass


def linear(value1, value1_index, value2, value2_index, nan_index ):
    """
    :param value1: 결측 값 및 구간 이전 실측 값
    :param value_index: 결측 값 및 구간 이전 실측 값 인덱스
    :param value2: 결측 값 이후의 첫번째 실측 값
    :param value2_index: 결측 값 이후의 첫번째 실측 값 인덱스
    :param nan_index: nan value의 인덱스
    :return:
    """
    "실측 값 index = x, value = y로 표현 "
    a_x, a_y = value1_index, value1
    b_x, b_y = value2_index, value2
    c_x = nan_index

    "거리 비율 계산(1차 함수의 기울기)"
    distance = b_x - a_x
    distance1 = c_x - a_x
    distance_ratio = distance1/distance

    new_value = a_y + (b_y-a_y)*distance_ratio

    return new_value


def impute():
    """
    imputation 함수는 sklearn에서 찾을 수 있음.
    다양한 imputation 함수를 지원하고 있음.
    TODO: 함수 간 차이점 확인
    TODO: 함수별 구현 필요성 확인

    :return:
    """
    pass


def scale():
    """
    scaling 함수는 sklearn.preprocessing에서 찾을 수 있음.
    다양한 scaling 함수를 지원하고 있음.
    TODO: 함수 간 차이점 확인
    TODO: 함수별 구현 필요성 확인

    :return:
    """
    pass


def minmax_scale():
    pass
