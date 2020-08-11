# %%
from re import L
from sklearn.experimental import enable_hist_gradient_boosting  # noqa

import holoviews as hv
import hvplot.pandas  # noqa
import numpy as np
import pandas as pd
from holoviews.core.data import Columns
from holoviews.streams import Pipe
from numpy.lib.function_base import piecewise
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.mixture import GaussianMixture
from sklearn.base import TransformerMixin
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    PolynomialFeatures,
    PowerTransformer,
    StandardScaler, preprocessing,
)

hv.extension('bokeh')

# %%
weather = context.catalog.load('weather')

pca = Pipeline([('scale', StandardScaler()), ('pca', PCA(2))])
Z = (pd.DataFrame(pca.fit_transform(weather.set_index('date')), index=weather.date, columns=['component_1', 'component_2'])
     .resample('D').mean()
     .sort_index()
     .pipe(lambda df: df.append(df.tail(1).assign()))
     .ffill())

pd.Series(pca.named_steps['pca'].explained_variance_ratio_).hvplot.bar(title='Explained Variance Ratio')

(
    Z
    .reset_index()
    .assign(date = lambda df: (df.date - df.date.min()).dt.total_seconds())
    .hvplot.scatter(x='component_1', y='component_2', color='date')
)



# %%
latent_climate = (Z)
# .append(
# pd.Series([ -4.330974, -1.633813], index=['component_1', 'component_2'], name=pd.to_datetime('2019-12-31'))
# ))

def climate(df, weather):
    return (df
            .assign(date = lambda s: s.Timestamp.dt.date.astype('datetime64[ns]'))
            .merge(weather, on='date', how='left')
            .drop(columns=['date']))


def distance(df):
    return df.assign(distance = lambda f: f.apply(lambda x: np.sqrt( (x.Origin_lat - x.Destination_lat)**2 + 	(x.Origin_lon - x.Destination_lon)**2), axis=1))


# %%
train = context.catalog.load('train').sort_values('Timestamp')
test = context.catalog.load('test')

train.sample(frac=0.1).hvplot.scatter(x='Origin_lat', y='Origin_lon')


# %%
history = (train.Timestamp
.append(test.Timestamp)
.to_frame()
.set_index('Timestamp')
.sort_index()
.assign(counter =1)
.rolling('H').count()
.pipe(lambda df: df.where(lambda x: x > df.counter.quantile(0.05), other=df.counter.median()))
.reset_index().groupby('Timestamp').mean())

def traffic(df, history):
    return df.merge(history, left_on='Timestamp', right_index=True, how='left') 

class Density(GaussianMixture, TransformerMixin):
    def transform(self, X):
        return super().score_samples(X).reshape(-1, 1)



# %%
X_train, y_train = (train
                    .drop(columns=['ETA'])
                    # .pipe(traffic, history=history)
                    # .pipe(climate, weather=latent_climate)
                    .pipe(distance),
                    train.ETA.pipe(lambda df: df.clip(0, df.quantile(0.99))))

X_test = (test
.reset_index()
# .pipe(traffic, history=history)
# .pipe(climate, weather=latent_climate)
.pipe(distance)
.set_index('ID'))



# %%
def hour(x):
    return x.dt.hour

def day(x):
    return x.dt.day

def week(x):
    return x.dt.week

def month(x):
    return x.dt.month

def dayofweek(x):
    return x.dt.dayofweek

def decompose(X):
    return X.transform([month, week, dayofweek, day, hour])

def cosine(x):
    return np.cos(2 * np.pi * x)

time = Pipeline([('decompose', FunctionTransformer(decompose)),
                 ('scale', MinMaxScaler())])

trig = Pipeline([('time', time),
                 ('trig', FunctionTransformer(cosine))])

# %%
cv = TimeSeriesSplit(2)

 # %%
## Gaussian Processes

# %%
# from sklearn.cluster import KMeans
# from sklearn.model_selection import train_test_split
# from sklearn.gaussian_process.kernels import RBF, WhiteKernel


# kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2)) \
#     + 2*np.pi * WhiteKernel(noise_level=1., noise_level_bounds=(1e-2, 1e+2))

# gp_engineering = ColumnTransformer([('time', time, make_column_selector(dtype_include='datetime64[ns, UTC]')),
#                                  ('trig', trig, make_column_selector(dtype_include='datetime64[ns, UTC]')),
#                                  ('other', FunctionTransformer(), make_column_selector(dtype_exclude='datetime64[ns, UTC]'))])

# region = Pipeline([('engineering', gp_engineering),
#                      ('rescale', StandardScaler()),
#                      ('means', KMeans(50))])

# train_bins = region.fit_predict(X_train)

# X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=0.05, stratify= train_bins)

# gp_param_dist = {'gp__alpha': [1e-1]}

# gp = Pipeline([('engineering', gp_engineering),
#                      ('rescale', StandardScaler()),
#                      ('whiten', PCA(whiten=True)),
#                      ('gp', GaussianProcessRegressor(kernel=kernel, normalize_y=True))])

# search = RandomizedSearchCV(gp,  param_distributions=gp_param_dist, cv=cv, return_train_score=True)
# search.fit(X_sample, y_sample)
# pd.DataFrame(search.cv_results_)

# # %%
# predictions = (X_test
# .assign(group = lambda df: np.random.randint(10, size=(df.shape[0], 1)))
# .groupby('group').apply(lambda df: df.reset_index(drop=False).assign(ETA = search.predict(df)))
# .set_index('ID')
# )




# %%
## HGB

engineering = ColumnTransformer([('time', time, make_column_selector(dtype_include='datetime64[ns, UTC]')),
                                 ('origin', Density(3), ['Origin_lat', 'Origin_lon']),
                                 ('destination', Density(3), ['Destination_lat', 'Destination_lon']),
                                 ('trig', trig, make_column_selector(dtype_include='datetime64[ns, UTC]')),
                                 ('other', FunctionTransformer(), make_column_selector(dtype_exclude='datetime64[ns, UTC]'))])



hgb_param_dist = {'hgb__learning_rate': [0.4, 0.425, 0.45]}
hgb = Pipeline([('engineering', engineering),
                     ('scale', StandardScaler()),
                     ('hgb', HistGradientBoostingRegressor(max_iter=150, max_depth=3, min_samples_leaf=100, l2_regularization=0.1, max_bins=100, monotonic_cst=[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]))])

# mlp_param_dist = {'mlp__activation': ['tanh', 'relu']}

# mlp = Pipeline([('engineering', engineering),
#                      ('scale', StandardScaler()),
#                      ('power', PowerTransformer()),
#                      ('rescale', StandardScaler()),
#                      ('whiten', PCA(whiten=True)),
#                      ('mlp', MLPRegressor(hidden_layer_sizes=(15, 5, ), activation='tanh'))])



# %%
search = RandomizedSearchCV(hgb,  param_distributions=hgb_param_dist, cv=cv, return_train_score=True)
search.fit(X_train, y_train)
pd.DataFrame(search.cv_results_)



# %%
# # %%
# pred = (X_train
# .assign(ETA = lambda df:  search.predict(df)))


# from sklearn.linear_model import LinearRegression
# T = pred.Timestamp.apply(lambda x: x - x.mean()).dt.total_seconds().to_frame()

# model = LinearRegression()
# model.fit(T, pred.ETA - y_train)


# from statsmodels.regression.linear_model import OLS


# %%
predictions = (X_test
.assign(ETA = lambda df:  search.predict(df)))



# %%
# sample_submission = context.catalog.load('sample_submission')

# %%
submission = (predictions.ETA.loc[X_test.index]
.to_frame()
.reset_index())

context.catalog.save('submission', submission)


# %%
