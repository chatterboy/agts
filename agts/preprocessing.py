import sklearn
import typing
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


def interpolate():
    """
    interpolation 함수는 SciPy와 Pandas에서 찾을 수 있음.
    특히, SciPy는 다양한 interpolation 함수가 있음. 함수 간에
    어떤 차이가 있는지 구현을 해야하는지 확인할 필요가 있음.

    :return:
    """
    pass


def Imputation_single(data: pd.DataFrame, input=None, missing_values=np.nan, strategy='mean', copy=True,
                      add_indicator=False) -> pd.DataFrame:
    """
    data : 결측치가 들어있는 데이터프레임
    input : Imputation 적용할 데이터프레임
    class sklearn.impute.SimpleImputer(
      missing_values=nan, (결측치 기준)
      strategy=’mean’, (mean, median, most_frequent, constant)
      fill_value=None, (constant 사용자 지정일 때 값 지정)
      copy=True, (복사본 생성 여부, False일때도 add_indicator = True이거나, 데이터가 float가 아니거나 희소행렬일 경우에는 복사본 생성 함)
      add_indicator=False (결측치 생성 여부 출력, 데이터에 열을 추가해서 보여줌)
      )
    """
    imputer = sklearn.impute.SimpleImputer(missing_values=missing_values, strategy=strategy, copy=copy,
                                           add_indicator=False)
    if input == None:
        return pd.DataFrame(imputer.fit_transform(data))
    else:
        imputer.fit(data)
        return pd.DataFrame(imputer.transform(input))

def Imputation_multi(data: pd.DataFrame, estimator=None, missing_values=np.nan, sample_posterior=False,
                     max_iter=10, tol=0.001, n_nearest_features=None, initial_strategy='mean',
                     imputation_order='ascending', skip_complete=False, min_value=-np.inf, max_value=np.inf,
                     verbose=0, random_state=None, add_indicator=False) -> pd.DataFrame:
    """
    결측값이 있는 피쳐를 라운드로빈 방식으로 다른 피쳐의 함수로 모델링하여 결측값 귀납
    class sklearn.impute.IterativeImputer(
      estimator=None, (라운드 로빈의 각 스탭에서 사용할 estimator, sample_posterior가 True일때 estimator의 predict는 반드시 return_std를 반환해야됨
                      default=BayesainRidge(), RandomForestRegressor, KNeighborsRegressor 등 사용 가능)
      missing_values=nan, (결측치 기준)
      sample_posterior=False, (각 스탭에서 estimator를 학습할 때 가우시안 사후 예측 샘플링을 사용할 것인지, True일때 predict에서 반드시 return_std를 반환해야됨, 다변량 imputation 할꺼면 반드시 True 설정)
      max_iter=10, (마지막 라운드에서 결측치 imputation 반환 전 수행할 최대 imputation 라운드 수. 하나의 라운드는 결측치가 있는 각각의 피쳐의 single imputation을 한다.)
                  (stopping criterion은 max(abs(X_t - X_{t-1}))/max(abs(X[known_vals])) < tol를 충족할 때이며, X_t는 t번째의 X를 의미한다. early stopping은 sample_posterior=False일때만 적용된다.)
      tol=0.001, (stopping condition의 허용 오차)
      n_nearest_features=None, (결측치를 측정할 때 사용할 다른 피쳐들의 수. None일때 모든 피쳐를 사용한다. 피쳐간 Nearness는 각 피쳐간 절대 상관 계수를 사용하여 측정한다.)
                              (Imputation 전체에 걸쳐 피처의 적용 범위를 보장하기 위해, 인접 피쳐가 반드시 가장 가까운 것은 아니지만, 각 imputation하는 대상 피쳐에 대한 상관 관계에 비례하는 확률로 그려진다.)
                              (이는 피쳐 수가 많을 때 상당한 속도 향상을 제공한다.)
      initial_strategy='mean', {'mean', 'median', 'most_frequent', 'constant'}, SimpleImputer와 변수 동일
      imputation_order='ascending',{'ascending', 'descending', 'roman', 'arabic', 'random'}, 피쳐가 impute되는 순서
          ascending : 결측값이 가장 적은 피쳐부터 많은 순서
          descending : 결측값이 가장 많은 피쳐부터 적는 순서
          roman : 왼쪽에서 오른쪽
          arabic : 오른쪽에서 왼쪽
          random : 각 라운드마다 랜덤한 순서
      skip_complete=False, (True일 때 fit에서 결측치가 없고 transform에서 결측치가 있다면 초기 imputation에서만 사용한다.
                            True일 때 많은 피쳐가 fit과 transform에서 결측치가 없다면 시간을 아낄 수 있음)
      min_value=- inf, max_value=inf, (imputation된 최소, 최대 값)
      verbose=0,
      random_state=None, (랜덤 난수 생성, n_neaest_features가 None이 아니고 imputation_order이 random이면서 sample_posterior가 True일 때 피쳐 추출기의 선택을 랜덤화한다.)
      add_indicator=False (결측치가 있는 경우 출력)
      )
    """
    imputer = sklearn.impute.IterativeImputer(estimator=estimator, missing_values=missing_values,
                                              sample_posterior=sample_posterior, max_iter=max_iter, tol=tol,
                                              n_nearest_features=n_nearest_features,
                                              initial_strategy=initial_strategy,
                                              imputation_order=imputation_order, skip_complete=skip_complete,
                                              min_value=min_value, max_value=max_value,
                                              verbose=verbose, random_state=random_state,
                                              add_indicator=add_indicator)

    return pd.DataFrame(imputer.fit_transform(data))

def Imputation_fill(data: pd.DataFrame, value=None, method=None, axis=None, inplace=False, limit=None,
                    downcast=None) -> pd.DataFrame:
    """
    data : 결측치가 들어있는 데이터프레임
    value : 대체할 값, method와 함께 못씀
    method
      backfill(bfill) : 결측치 구간 직후의 값으로 결측치 채움
      pad(ffill) : 결측치 구간 직전의 값으로 결측치 채움
    axis : 0 index, 1 columns
    inplace : True일 때 복사본 대신 직접 대입됨
    limit : 앞/뒤로 채울 최대 결측치의 수. 설정된 값보다 더 많은 결측치가 있으면 냅두고 설정 값만큼만 결측치 채움
    downcast : dict, float64->int64 처럼 형변환시켜줌
    """
    return data.fillna(value=value, method=method, axis=axis, inplace=inplace, limit=limit, downcast=downcast)

def Imputation_knn(data: pd.DataFrame, input=None, missing_values=np.NaN, n_neighbors=5, weights='uniform',
                   metric='nan_euclidean', copy=True, add_indicator=False) -> pd.DataFrame:
    """
    class sklearn.impute.KNNImputer(
      missing_values=nan, (결측치 기준)
      n_neighbors=5, (imputation에 사용할 인접 샘플 수)
      weights='uniform', (예측에 사용되는 가중치 함수, uniform, distance, callable)
        uniform : 각 이웃의 모든 포인트는 동일하게 가중치 부여
        distance : 각 이웃의 거리의 역수로 가중치 부여, 가까운 이웃이 멀리있는 이웃보다 더 큰 영향을 미침
        callable : 사용자정의 함수, 거리의 배열을 받아서 배열과 같은 크기의 가중치 반환
      metric='nan_euclidean', (이웃간 거리 측정 방법, callable)
        callable : x, y의 두 배열과 키워드를 받아 스칼라 거리값 반환하는 사용자함수
      copy=True, (새로운 복사본 생성, False일때 inplace 역할)
      add_indicator=False) (결측치 생성 여부 출력, 데이터에 열을 추가해서 보여줌)
    """
    imputer = sklearn.impute.KNNImputer(missing_values=missing_values, n_neighbors=n_neighbors, weights=weights,
                                        metric=metric, copy=copy, add_indicator=add_indicator)
    if input == None:
        return pd.DataFrame(imputer.fit_transform(data))
    else:
        imputer.fit(data)
        return pd.DataFrame(imputer.transform(input))

def Impute_test(): # 정상 작동하는지 테스트하는 함수
    df = pd.DataFrame([[1, 2, 3], [2, 2, 2], [np.NaN, 3, 9], [5, np.NaN, 4], [np.NaN, 4, 4], [7, 8, np.NaN]])
    print("data")
    print(df)
    print()
    print("single")
    print(Imputation_single(data=df))
    print()
    print("multi")
    print(Imputation_multi(df, sample_posterior=True))
    print()
    print("fill")
    print(Imputation_fill(df, method='pad'))
    print()
    print("knn")
    print(Imputation_knn(df))

# Impute_test()

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

# impute 테스트
# imputer = Impute()
# imputer.test()
