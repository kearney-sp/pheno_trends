## https://github.com/davemlz/GEE_TimeSeries/blob/master/GEE_TimeSeries_SavGol.py
# Use it with Google Colab
# !pip install google-api-python-client
# !pip install earthengine-api
# !earthengine authenticate

from datetime import datetime as dt
from scipy import signal
import pandas as pd
import numpy as np
import ee
import geopandas as gpd
import pandas.tseries.offsets as offsets
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit
import seaborn as sns
import scipy as sp
import time

ee.Initialize()


def cloudMask(image):
    # Quality image
    qa = image.select('QA60')
    # Thick and thin clouds bitmasks
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    maskCloud = qa.bitwiseAnd(cloudBitMask).eq(0)
    maskCirrus = qa.bitwiseAnd(cirrusBitMask).eq(0)
    # Blue band threshold mask
    maskThreshold = image.select('B2').lte(2000)
    return image.updateMask(maskCloud).updateMask(maskCirrus).updateMask(maskThreshold).select("B.*").copyProperties(
        image, ["system:time_start"])


def modis9Mask(image):
    qa = image.select('StateQA')
    cloudMask = (1 << 0)
    cloudShadowMask = (1 << 2)
    cirrusMask = (1 << 8)
    internalCloudMask = (1 << 10)
    snowMask = (1 << 12)
    internalSnowMask = (1 << 15)
    maskCloud = qa.bitwiseAnd(cloudMask).eq(0)
    maskCloudShadow = qa.bitwiseAnd(cloudShadowMask).eq(0)
    maskCirrus = qa.bitwiseAnd(cirrusMask).eq(0)
    maskInternalCloud = qa.bitwiseAnd(internalCloudMask).eq(0)
    maskSnow = qa.bitwiseAnd(snowMask).eq(0)
    maskInternalSnow = qa.bitwiseAnd(internalSnowMask).eq(0)
    return image.updateMask(maskCloud).updateMask(maskCloudShadow).updateMask(maskCirrus).updateMask(maskInternalCloud) \
        .updateMask(maskSnow).updateMask(maskInternalSnow) \
        .copyProperties(image, ["system:time_start"])


def avhrrMask(image):
    qa = image.select('QA')
    cloudMask = (1 << 0)
    cloudShadowMask = (1 << 2)
    waterMask = (1 << 3)
    glintMask = (1 << 4)
    nightMask = (1 << 6)
    channelMask = (1 << 7)
    rhoMask = (1 << 13)
    brdfMask = (1 << 14)
    maskCloud = qa.bitwiseAnd(cloudMask).eq(0)
    maskCloudShadow = qa.bitwiseAnd(cloudShadowMask).eq(0)
    maskWater = qa.bitwiseAnd(waterMask).eq(0)
    maskGlint = qa.bitwiseAnd(glintMask).eq(0)
    maskNight = qa.bitwiseAnd(nightMask).eq(0)
    maskNDVI = image.gt(0)
    return image.updateMask(maskCloud).updateMask(maskCloudShadow).updateMask(maskWater).updateMask(maskGlint) \
        .updateMask(maskNight).updateMask(maskNDVI) \
        .copyProperties(image, ["system:time_start"])
    # .updateMask(maskChannels).updateMask(maskRho).updateMask(maskBRDF) \


def timeSeriesData(imageCollection, func, name, band_name=None, date_range=None, **kwargs):
    if date_range is None:
        imgs = imageCollection.map(func, **kwargs)
    elif len(date_range) == 2:
        imgs = imageCollection.filterDate(date_range[0], date_range[1]).map(func)
    # Get features
    features = imgs.getInfo()["features"]
    index = []
    date = []
    # Creating date and vegetation index vectors
    for i in range(len(features)):
        fp = features[i]["properties"]
        # There is data
        if "nd" in fp:
            index.append(fp["nd"])
            date.append(fp["system:time_start"])
        # There is no data
        elif band_name in fp:
            index.append(fp[band_name])
            date.append(fp["system:time_start"])
        else:
            index.append(np.nan)
            date.append(fp["system:time_start"])
    # Dictionary
    data = {"Date": date, name: index}
    # Data frame
    df = pd.DataFrame(data, columns=["Date", name])
    # Date to datetime
    df["Date"] = pd.to_datetime(df["Date"] * 1000000)
    # Change format
    df["Date"] = df["Date"].map(lambda x: x.strftime('%Y-%m-%d'))
    # Interpolate missing data in time series
    df[name] = pd.Series(df[name]).interpolate(limit=len(date))
    # Drop duplicates
    df = df.drop_duplicates()
    # Convert date to datetime
    # df = pd.to_datetime(df['Date'])
    # df = df.sort_values(by='Date')
    # Get mean of date with overlapping tiles
    # df = df.groupby('Date').mean()
    # Apply Savitzky-Golay filter
    # df["SavGol"] = signal.savgol_filter(df[name], window, order, mode="interp")
    return df


def setData(image, reducer=ee.Reducer.mean()):
    img = image
    dict_nd = img.reduceRegion(reducer, fc, bestEffort=True)
    # Add GNDVI mean as new property
    return img.set(dict_nd).copyProperties(image, ["system:time_start"])


# defined polygon
def getFeatures(gdf):
    """Function to parse features from GeoDataFrame in such a manner that rasterio wants them"""
    import json
    # return [json.loads(gdf.to_json())['features'][0]['geometry']]
    return [i['geometry'] for i in json.loads(gdf.to_json())['features']]


def non_uniform_savgol(x, y, window, polynom):
    """
  Applies a Savitzky-Golay filter to y with non-uniform spacing
  as defined in x

  This is based on https://dsp.stackexchange.com/questions/1676/savitzky-golay-smoothing-filter-for-not-equally-spaced-data
  The borders are interpolated like scipy.signal.savgol_filter would do

  Parameters
  ----------
  x : array_like
      List of floats representing the x values of the data
  y : array_like
      List of floats representing the y values. Must have same length
      as x
  window : int (odd)
      Window length of datapoints. Must be odd and smaller than x
  polynom : int
      The order of polynom used. Must be smaller than the window size

  Returns
  -------
  np.array of float
      The smoothed y values
  """
    if len(x) != len(y):
        raise ValueError('"x" and "y" must be of the same size')

    if len(x) < window:
        raise ValueError('The data size must be larger than the window size')

    if type(window) is not int:
        raise TypeError('"window" must be an integer')

    if window % 2 == 0:
        raise ValueError('The "window" must be an odd integer')

    if type(polynom) is not int:
        raise TypeError('"polynom" must be an integer')

    if polynom >= window:
        raise ValueError('"polynom" must be less than "window"')

    half_window = window // 2
    polynom += 1

    # Initialize variables
    A = np.empty((window, polynom))  # Matrix
    tA = np.empty((polynom, window))  # Transposed matrix
    t = np.empty(window)  # Local x variables
    y_smoothed = np.full(len(y), np.nan)

    # Start smoothing
    for i in range(half_window, len(x) - half_window, 1):
        # Center a window of x values on x[i]
        for j in range(0, window, 1):
            t[j] = x[i + j - half_window] - x[i]

        # Create the initial matrix A and its transposed form tA
        for j in range(0, window, 1):
            r = 1.0
            for k in range(0, polynom, 1):
                A[j, k] = r
                tA[k, j] = r
                r *= t[j]

        # Multiply the two matrices
        tAA = np.matmul(tA, A)

        # Invert the product of the matrices
        tAA = np.linalg.inv(tAA)

        # Calculate the pseudoinverse of the design matrix
        coeffs = np.matmul(tAA, tA)

        # Calculate c0 which is also the y value for y[i]
        y_smoothed[i] = 0
        for j in range(0, window, 1):
            y_smoothed[i] += coeffs[0, j] * y[i + j - half_window]

        # If at the end or beginning, store all coefficients for the polynom
        if i == half_window:
            first_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    first_coeffs[k] += coeffs[k, j] * y[j]
        elif i == len(x) - half_window - 1:
            last_coeffs = np.zeros(polynom)
            for j in range(0, window, 1):
                for k in range(polynom):
                    last_coeffs[k] += coeffs[k, j] * y[len(y) - window + j]

    # Interpolate the result at the left border
    for i in range(0, half_window, 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += first_coeffs[j] * x_i
            x_i *= x[i] - x[half_window]

    # Interpolate the result at the right border
    for i in range(len(x) - half_window, len(x), 1):
        y_smoothed[i] = 0
        x_i = 1
        for j in range(0, polynom, 1):
            y_smoothed[i] += last_coeffs[j] * x_i
            x_i *= x[i] - x[-half_window - 1]

    return y_smoothed


def savitzky_golay_filtering(xdate, timeseries, wnds=[11, 7], orders=[3, 3], lwr=False, debug=False):
    # https://gis.stackexchange.com/questions/173721/reconstructing-modis-time-series-applying-savitzky-golay-filter-with-python-nump/173747
    interp_ts = pd.Series(timeseries)
    interp_ts = interp_ts.interpolate(method='linear', limit=31)
    smooth_ts = interp_ts
    wnd, order = wnds[0], orders[0]
    F = 1e8
    W = None
    it = 0
    while True:
        smoother_ts = non_uniform_savgol(xdate, smooth_ts, window=wnd, polynom=order)
        diff = smoother_ts - interp_ts
        if lwr:
            sign = diff < 0
        else:
            sign = diff > 0
        if W is None:
            W = 1 - np.abs(diff) / np.max(np.abs(diff)) * sign
            wnd, order = wnds[1], orders[1]
        fitting_score = np.sum(np.abs(diff) * W)
        print(it, ' : ', fitting_score)
        if fitting_score > F:
            break
        else:
            F = fitting_score
            it += 1
        if it > 100:
            break
        smooth_ts = smoother_ts * sign + interp_ts * (1 - sign)
    if debug:
        return smooth_ts, interp_ts
    return smooth_ts


def apply_savgol(ts, window=31, polynom=3, limit=61):
    ts_tmp = ts.copy()
    ts_interp = pd.Series(ts_tmp)
    ts_interp = ts_interp.interpolate(method='linear', limit_area='inside', limit=limit)
    ts_interp = ts_interp.interpolate(method='linear', limit=None, limit_direction='both',
                                      limit_area='outside')
    try:
        ts_smooth = savgol_filter(ts_interp, window_length=window, polyorder=polynom)
    except np.linalg.LinAlgError:
        ts_smooth = ts_interp
    return ts_smooth


def modified_z_score(ts):
    # see https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
    median_int = np.nanmedian(ts)
    mad_int = np.nanmedian([np.abs(ts - median_int)])
    modified_z_scores = 0.6745 * (ts - median_int) / mad_int
    return modified_z_scores


def mask_ts_outliers(ts, threshold=3.5):
    # see https://towardsdatascience.com/removing-spikes-from-raman-spectra-8a9fdda0ac22
    ts_masked = ts.copy()
    ts_modz_robust = np.array(abs(modified_z_score(ts_masked)))
    if not np.all(np.isnan(ts_modz_robust)):
        spikes1 = ts_modz_robust > threshold
        ts_masked[spikes1] = np.nan
    return ts_masked


def despike_ts(dat_ts, dat_thresh, days_thresh, z_thresh=3.5, mask_outliers=False, iters=2):
    dat_ts_cln = dat_ts.copy()
    if mask_outliers:
        dat_ts_cln = mask_ts_outliers(dat_ts_cln, threshold=z_thresh)
    dat_mask = np.zeros_like(dat_ts_cln)
    for i in range(iters):
        for idx in range(len(dat_ts_cln)):
            if not np.isnan(dat_ts_cln[idx]):
                idx_clear = np.where(~np.isnan(dat_ts_cln))[0]
                if idx == np.min(idx_clear):
                    continue
                elif idx == np.max(idx_clear):
                    continue
                else:
                    idx_pre = idx_clear[idx_clear < idx][-1]
                    idx_post = idx_clear[idx_clear > idx][0]
                    y = np.array([dat_ts_cln[idx_pre], dat_ts_cln[idx_post]])
                    x = np.array([idx_pre, idx_post])
                    dx = np.diff(x)
                    dy = np.diff(y)
                    slope = dy / dx
                    dat_interp = dat_ts_cln[idx_pre] + slope[0] * (idx - idx_pre)
                    dat_diff = dat_interp - dat_ts_cln[idx]
                    shadow_val = dat_diff / (dat_ts_cln[idx_post] - dat_ts_cln[idx_pre])
                    if (idx_post - idx_pre < days_thresh) & (np.abs(dat_diff) > dat_thresh) & (np.abs(shadow_val) > 2):
                        dat_ts_cln[idx] = np.nan
                        dat_mask[idx] = 1
                    else:
                        continue
            else:
                continue
    dat_ts_cln[np.where(dat_mask == 1)] = np.nan
    return dat_ts_cln


def double_logistic(x, vmin, vmax, sos, scaleS, eos, scaleA):
    y = vmin + vmax * ((1 / (1 + np.exp(-scaleS * (x - sos)))) + (1 / (1 + np.exp(scaleA * (x - eos)))) - 1)
    return y


def ndvi_to_fpar(ndvi_ts):
    SR = (1 + ndvi_ts) / (1 - ndvi_ts)
    # SR_min = 1.55  # this is based on Grigera et al., 2007
    SR_min = 1.11  # this is based on an NDVI of 0.05
    SR_max = 11.62
    fPAR = SR / (SR_max - SR_min) - SR_min / (SR_max - SR_min)
    fPAR = fPAR.where((fPAR < 0.95) | fPAR.isnull(), 0.95)
    fPAR = fPAR.where((fPAR > 0.0) | fPAR.isnull(), 0.0)
    return fPAR.astype('float32')


def fpar_to_apar(fpar_ts, par_ts):
    apar = fpar_ts * par_ts
    return apar.astype('float32')


def annotate(data, exog, endog, **kws):
    r, p = sp.stats.pearsonr(data[endog], data[exog])
    ax = plt.gca()
    if np.isnan(r):
        pass
    elif data[exog].max() > ax.get_ylim()[1] * 0.8:
        ax.text(.05, .2, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)
    else:
        ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                transform=ax.transAxes)


############################################################################
cper_2017_f = "C:/SPK_local/data/vectors/Pasture_Boundaries/Shapefiles/cper_pastures_2017_clip.shp"
par_f = 'C:/SPK_local/data/climate/PAR/PAR_max_daily_avg_CPER.csv'
cper = gpd.read_file(cper_2017_f)
cper = cper.to_crs("EPSG:3857")
cper['PAST_NEW'] = cper['Past_Name_']
cper = cper.dissolve(by='PAST_NEW')
cper['area_ha'] = cper['geometry'].area / 10 ** 4

# read in PAR data
par_df = pd.read_csv(par_f, engine='python')

cper_past = cper[cper.index == '15E']
polygon = getFeatures(cper.to_crs(epsg=4326))

[min_x, min_y, max_x, max_y] = cper.buffer(50).to_crs(epsg=4326).total_bounds
polygon = ee.Geometry.Polygon([
    [[min_x, max_y], [max_x, max_y], [max_x, min_y], [min_x, min_y], [min_x, max_y]]
])

# polygon to feature collection
fc = ee.FeatureCollection(polygon)
df_ndvi_avhrr_fnl = pd.DataFrame(columns=['Date', 'ndvi', 'ndvi_smooth', 'ndvi_season_dl'])
for yr in range(1982, 2021):
    print(yr)
    # AVHRR image collection over the feature collection and cloudmask
    ndvi_avhrr = ee.ImageCollection("NOAA/CDR/AVHRR/NDVI/V5").filterBounds(fc).map(avhrrMask)
    df_ndvi_avhrr = timeSeriesData(ndvi_avhrr, setData, name='ndvi', band_name='NDVI',
                                   date_range=[str(yr) + '-01-01',
                                               str(yr) + '-12-31'])
    df_ndvi_avhrr['Date'] = pd.to_datetime(df_ndvi_avhrr['Date'])
    df_ndvi_avhrr['ndvi_smooth'] = apply_savgol(
        despike_ts(
            np.array(
                savitzky_golay_filtering(
                    df_ndvi_avhrr.index, df_ndvi_avhrr['ndvi'], wnds=[21, 13])),
            dat_thresh=1000, days_thresh=30),
        window=31, polynom=3)

    try:
        p0 = [df_ndvi_avhrr['ndvi_smooth'].quantile(q=0.01),
              np.max(df_ndvi_avhrr['ndvi_smooth']),
              int(np.percentile(df_ndvi_avhrr.index, 25)),
              1.0,
              int(np.percentile(df_ndvi_avhrr.index, 75)),
              1.0]  # this is a mandatory initial guess
        popt, pcov = curve_fit(double_logistic, df_ndvi_avhrr.index, df_ndvi_avhrr['ndvi_smooth'],
                               p0, method='lm', maxfev=20000)
        df_ndvi_avhrr['ndvi_season_dl'] = double_logistic(df_ndvi_avhrr.index, *popt)
    except RuntimeError:
        try:
            p0 = [df_ndvi_avhrr['ndvi_smooth'].quantile(q=0.01),
                  np.max(df_ndvi_avhrr['ndvi_smooth']),
                  120,
                  1.0,
                  int(np.percentile(df_ndvi_avhrr.index, 75)),
                  1.0]  # this is a mandatory initial guess
            popt, pcov = curve_fit(double_logistic, df_ndvi_avhrr.index, df_ndvi_avhrr['ndvi_smooth'],
                                   p0, method='lm', maxfev=20000)
            df_ndvi_avhrr['ndvi_season_dl'] = double_logistic(df_ndvi_avhrr.index, *popt)
        except RuntimeError:
            print(RuntimeError)

    df_ndvi_avhrr_fnl = df_ndvi_avhrr_fnl.append(df_ndvi_avhrr)


df_ndvi_avhrr_update = pd.DataFrame(columns=df_ndvi_avhrr_fnl.columns)
df_pheno_yrly = pd.DataFrame(columns=['YEAR', 'ndvi_max_gap', 'SOS', 'MGU', 'MAT', 'POS', 'POS2', 'SEN', 'MGD', 'EOS',
                                      'LGU', 'LGS', 'LGD', 'LOS', 'num_pks'])
figure, axs = plt.subplots(figsize=(16, 10),
                           nrows=len(df_ndvi_avhrr_fnl.Date.dt.year.unique())//3 + 1,
                           ncols=3,
                           sharey=True,
                           sharex=True)
for idx, yr in enumerate(df_ndvi_avhrr_fnl.Date.dt.year.unique()):
    print(yr)
    df_ndvi_avhrr = df_ndvi_avhrr_fnl[df_ndvi_avhrr_fnl.Date.dt.year == yr]
    df_ndvi_avhrr.loc[df_ndvi_avhrr['ndvi'].diff().rolling(window=4, center=False).sum().round(4) == 0.0, ['ndvi']] = np.nan
    df_ndvi_avhrr.loc[df_ndvi_avhrr['ndvi'] < 300, ['ndvi']] = np.nan
    df_ndvi_avhrr['ndvi_smooth'] = apply_savgol(
        despike_ts(
            np.array(
                savitzky_golay_filtering(
                    df_ndvi_avhrr.index, df_ndvi_avhrr['ndvi'], wnds=[21, 13])),
            dat_thresh=1000, days_thresh=30),
        window=59, polynom=3)
    try:
        p0 = [df_ndvi_avhrr['ndvi_smooth'].quantile(q=0.01),
              np.max(df_ndvi_avhrr['ndvi_smooth']),
              int(np.percentile(df_ndvi_avhrr.index, 25)),
              1.0,
              int(np.percentile(df_ndvi_avhrr.index, 75)),
              1.0]  # this is a mandatory initial guess
        popt, pcov = curve_fit(double_logistic, df_ndvi_avhrr.index, df_ndvi_avhrr['ndvi_smooth'],
                               p0, method='lm', maxfev=20000)
        df_ndvi_avhrr['ndvi_season_dl'] = double_logistic(df_ndvi_avhrr.index, *popt)
    except RuntimeError:
        try:
            p0 = [df_ndvi_avhrr['ndvi_smooth'].quantile(q=0.01),
                  np.max(df_ndvi_avhrr['ndvi_smooth']),
                  120,
                  1.0,
                  int(np.percentile(df_ndvi_avhrr.index, 75)),
                  1.0]  # this is a mandatory initial guess
            popt, pcov = curve_fit(double_logistic, df_ndvi_avhrr.index, df_ndvi_avhrr['ndvi_smooth'],
                                   p0, method='lm', maxfev=20000)
            df_ndvi_avhrr['ndvi_season_dl'] = double_logistic(df_ndvi_avhrr.index, *popt)
        except RuntimeError:
            print(RuntimeError)

    ndvi_max_gap = (df_ndvi_avhrr['ndvi'].isnull() *
                    (df_ndvi_avhrr['ndvi'].isnull().groupby(
                        (df_ndvi_avhrr['ndvi'].isnull() != df_ndvi_avhrr['ndvi'].isnull().shift())
                            .cumsum()).cumcount() + 1)).max()
    day_ndvi_pk = df_ndvi_avhrr['ndvi_smooth'].idxmax()
    ndvi_base_gu = df_ndvi_avhrr.loc[:day_ndvi_pk, 'ndvi_season_dl'].quantile(q=0.025)
    ndvi_base_gd = df_ndvi_avhrr.loc[day_ndvi_pk:, 'ndvi_season_dl'].quantile(q=0.025)
    ndvi_amp_max_gu = df_ndvi_avhrr['ndvi_smooth'].max() - ndvi_base_gu
    ndvi_amp_max_gd = df_ndvi_avhrr['ndvi_smooth'].max() - ndvi_base_gd
    ndvi_amp_gu = df_ndvi_avhrr['ndvi_smooth'].loc[:day_ndvi_pk] - ndvi_base_gu
    doys_arr = pd.Series(np.ones_like(df_ndvi_avhrr['ndvi_smooth']) * (df_ndvi_avhrr.index + 1))
    doys_arr_gu = doys_arr.loc[:day_ndvi_pk]
    sos = doys_arr_gu.loc[(ndvi_amp_gu.shift(1) < (0.15 * ndvi_amp_max_gu)) &
                       (ndvi_amp_gu > (0.15 * ndvi_amp_max_gu))].idxmax() + 1
    mgu = doys_arr_gu.loc[(ndvi_amp_gu.shift(1) < (0.50 * ndvi_amp_max_gu)) &
                          (ndvi_amp_gu > (0.50 * ndvi_amp_max_gu))].idxmax() + 1
    mat = doys_arr_gu.loc[(ndvi_amp_gu.shift(1) < (0.90 * ndvi_amp_max_gu)) &
                          (ndvi_amp_gu > (0.90 * ndvi_amp_max_gu))].idxmax() + 1
    pos = df_ndvi_avhrr['ndvi_smooth'].idxmax() + 1
    ndvi_amp_gd = df_ndvi_avhrr['ndvi_smooth'].loc[day_ndvi_pk:, ] - ndvi_base_gd
    doys_arr_gd = doys_arr.loc[day_ndvi_pk:]
    sen = doys_arr_gd.loc[(ndvi_amp_gd.shift(1) > (0.90 * ndvi_amp_max_gd)) &
                          (ndvi_amp_gd < (0.90 * ndvi_amp_max_gd))].idxmax() + 1
    mgd = doys_arr_gd.loc[(ndvi_amp_gd.shift(1) > (0.50 * ndvi_amp_max_gd)) &
                          (ndvi_amp_gd < (0.50 * ndvi_amp_max_gd))].idxmax() + 1
    eos = doys_arr_gd.loc[(ndvi_amp_gd.shift(1) > (0.15 * ndvi_amp_max_gd)) &
                          (ndvi_amp_gd < (0.15 * ndvi_amp_max_gd))].idxmax() + 1
    pks = find_peaks(df_ndvi_avhrr['ndvi_smooth'], prominence=500, width=10)[0]
    pks = [x for x in pks if (x > sos) & (x < eos)]
    num_pks = len(pks)
    if num_pks == 2:
        pos2 = pks[1]
    else:
        pos2 = np.nan
    df_pheno_yrly_tmp = pd.DataFrame(dict(
        YEAR=yr,
        ndvi_max_gap=ndvi_max_gap,
        SOS=sos,
        MGU=mgu,
        MAT=mat,
        POS=pos,
        POS2=pos2,
        SEN=sen,
        MGD=mgd,
        EOS=eos,
        LGU=mat-sos,
        LGS=mgd-mgu,
        LGD=eos-sen,
        LOS=eos-sos,
        num_pks=num_pks,
    ), index=[idx], dtype='int')
    df_pheno_yrly = df_pheno_yrly.append(df_pheno_yrly_tmp)
    df_ndvi_avhrr_update = df_ndvi_avhrr_update.append(df_ndvi_avhrr)
    if ndvi_max_gap <= 90:
        axs.flatten()[idx].plot(df_ndvi_avhrr['ndvi'], c='grey', alpha=0.5)
        axs.flatten()[idx].plot(df_ndvi_avhrr['ndvi_smooth'], c='orange')
        axs.flatten()[idx].plot(df_ndvi_avhrr['ndvi_season_dl'], c='green')
        axs.flatten()[idx].scatter(x=pks, y=df_ndvi_avhrr['ndvi_smooth'][[x + 1 for x in pks]], c='red')
        axs.flatten()[idx].set_title(yr)
    else:
        axs.flatten()[idx].set_title(yr)
        continue
for ax in axs.flatten():
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(8)
plt.tight_layout()
figure.canvas.draw()

rename_dict = {
    'ndvi_max_gap': 'Max NDVI gap',
    'SOS': 'Start of season',
    'MGU': 'Mid-greenup',
    'MAT': 'Maturity',
    'POS': 'Peak of season',
    'POS2': 'Peak of season #2 (if appl.)',
    'SEN': 'Start of sensence',
    'MGD': 'Mid-greendown',
    'EOS': 'End of season',
    'LGU': 'Green-up',
    'LGS': 'Mid-greenup to mid-greendown',
    'LGD': 'Green-down',
    'LOS': 'Growing season',
    'num_pks': 'Number of peaks'
}
df_pheno_yrly_long = df_pheno_yrly.melt(id_vars='YEAR', var_name='PhenoPhase', value_name='DOY')
df_pheno_yrly_long[['YEAR', 'DOY']] = df_pheno_yrly_long[['YEAR', 'DOY']].astype('int32')
df_pheno_yrly_long['Phase'] = [rename_dict[n] for n in df_pheno_yrly_long['PhenoPhase']]
df_pheno_yrly_long['Period'] = df_pheno_yrly_long['Phase']
df_pheno_yrly_long['Number of days'] = df_pheno_yrly_long['DOY']

yrs_cln = df_pheno_yrly['YEAR'][df_pheno_yrly['ndvi_max_gap'] <= 90]
yrs_one_pk = df_pheno_yrly['YEAR'][df_pheno_yrly['num_pks'] == 1]

df_pheno_yrly_long_sub = df_pheno_yrly_long[df_pheno_yrly_long['YEAR'].isin(yrs_cln) & df_pheno_yrly_long['YEAR'].isin(yrs_one_pk)]
df_pheno_yrly_long_out = df_pheno_yrly_long[df_pheno_yrly_long['YEAR'].isin(yrs_cln) & ~(df_pheno_yrly_long['YEAR'].isin(yrs_one_pk))]


g1 = sns.lmplot(x='YEAR', y='DOY', col='Phase', col_wrap=4,
           data=df_pheno_yrly_long_sub[df_pheno_yrly_long_sub.PhenoPhase.isin(
               ['SOS', 'MGU', 'MAT', 'POS', 'POS2', 'SEN', 'MGD', 'EOS'])],
           fit_reg=True, robust=False, height=3.5, aspect=1)
for idx, var in enumerate(['SOS', 'MGU', 'MAT', 'POS', 'POS2', 'SEN', 'MGD', 'EOS']):
    sns.scatterplot(x='YEAR', y='DOY', data=df_pheno_yrly_long_out[df_pheno_yrly_long_out['PhenoPhase'] == var],
                    ax=g1.axes[idx])
    if idx == 4:
        x_tmp = df_pheno_yrly_long_out['YEAR'][df_pheno_yrly_long_out['PhenoPhase'] == var]
        y_tmp = df_pheno_yrly_long_out['DOY'][df_pheno_yrly_long_out['PhenoPhase'] == var]
        r, p = sp.stats.pearsonr(x_tmp, y_tmp)
        m, b = np.polyfit(x_tmp, y_tmp, 1)
        g1.axes[idx].plot(x_tmp, m * x_tmp + b, color='orange')
        if df_pheno_yrly_long_out['DOY'][df_pheno_yrly_long_out['PhenoPhase'] == var].max() > g1.axes[idx].get_ylim()[1] * 0.8:
            g1.axes[idx].text(.05, .2, 'r={:.2f}, p={:.2g}'.format(r, p),
                    transform=ax.transAxes)
        else:
            g1.axes[idx].text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
                    transform=g1.axes[idx].transAxes)
g1.map_dataframe(annotate, **dict(endog='YEAR', exog='DOY'))
g1.axes[0].set_ylabel('Day of year')
g1.axes[4].set_ylabel('Day of year')
plt.tight_layout(h_pad=3, w_pad=3, pad=1)
plt.savefig('C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_phenophase_DOY_v2.png',
            dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.close()

g2 = sns.lmplot(x='YEAR', y='Number of days', col='Period', col_wrap=2,
           data=df_pheno_yrly_long_sub[df_pheno_yrly_long_sub.PhenoPhase.isin(
               ['LGU', 'LGS', 'LGD', 'LOS'])],
           fit_reg=True, robust=False, height=3.5, aspect=1.25)
for idx, var in enumerate(['LGU', 'LGS', 'LGD', 'LOS']):
    sns.scatterplot(x='YEAR', y='DOY', data=df_pheno_yrly_long_out[df_pheno_yrly_long_out['PhenoPhase'] == var],
                    ax=g2.axes[idx])
g2.map_dataframe(annotate, **dict(endog='YEAR', exog='Number of days'))
g2.axes[0].set_ylabel('Number of days')
g2.axes[2].set_ylabel('Number of days')
plt.tight_layout(h_pad=3, w_pad=3, pad=1)
plt.savefig('C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_phenophase_periods_v2.png',
            dpi=200, bbox_inches='tight', pad_inches=0.2)
plt.close()

df_ndvi_avhrr_plt = df_ndvi_avhrr_update
df_ndvi_avhrr_plt['DOY'] = df_ndvi_avhrr_plt['Date'].dt.dayofyear
df_ndvi_avhrr_plt['Year'] = df_ndvi_avhrr_plt['Date'].dt.year
df_ndvi_avhrr_plt = df_ndvi_avhrr_plt[df_ndvi_avhrr_plt['Year'].isin(df_pheno_yrly[df_pheno_yrly['num_pks'] == 1]['YEAR'].unique())]
df_ndvi_avhrr_plt['decade'] = df_ndvi_avhrr_plt['Year']//10*10
df_ndvi_avhrr_plt['decade'][df_ndvi_avhrr_plt['Year'] == 2020] = 2010


plt.figure()
sns.lineplot(y='ndvi_smooth', x='DOY', hue='decade', estimator='mean',
             palette=sns.color_palette("deep", 4), data=df_ndvi_avhrr_plt)

plt.figure()
sns.lmplot(y='ndvi_smooth', x='DOY', hue='decade', lowess=True,
             palette=sns.color_palette("deep", 4), data=df_ndvi_avhrr_plt)


df_ndvi_avhrr_update.to_csv('C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_ndvi_ts_1982-2020.csv')
df_pheno_yrly.to_csv('C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_pheno_phases_1982-2020.csv')
df_pheno_yrly_long.to_csv('C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_pheno_phases_long_1982-2020.csv')

df_ndvi_avhrr_update = pd.read_csv(
    'C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_ndvi_ts_1982-2020.csv')
df_pheno_yrly = pd.read_csv(
    'C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_pheno_phases_1982-2020.csv')
df_pheno_yrly_long = pd.read_csv(
    'C:/SPK_local/for_others/Hoover_longterm_phenophase/CPER_pheno_phases_long_1982-2020.csv')

