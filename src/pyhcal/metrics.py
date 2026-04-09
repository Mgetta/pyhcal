# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 09:44:14 2024

@author: mfratki
"""
import numpy as np
import pandas as pd
try:
    import baseflow as bf
except:
    pass

'''
monthly
  cfs - mean, median, std
  in -  sum
  mg/l - mean, median, std
  lb - sum

annual







'''

def aggregate(df,units):
    assert units in ['lb','mg/l','in','cfs','degC']
    if units in ['mg/l','cfs','degC']:
        agg_func = 'mean'
    else:
        agg_func = 'sum'
        
    df_agg = pd.DataFrame(np.ones((12,3))*np.nan,index = range(1,13),columns = ['simulated','observed','ratio'])
    df_agg.index.name = 'month'
    df = df.groupby(df.index.month).agg(agg_func)[['simulated','observed']]
    df.columns = ['simulated','observed']
    df['ratio'] = df['observed']/df['simulated']
    df_agg.loc[df.index,df.columns] = df.values
    
    df_agg.loc['Mean'] = df_agg.agg('mean')
    df_agg['ratio'] = df_agg['observed']/df_agg['simulated']
    return df_agg

def NSE(df_daily,agg_func = 'mean'):
    #df_monthly = df_daily.resample('MS').sum()
    df_monthly = df_daily.groupby(df_daily.index.month).agg(agg_func)
    
    NSE_daily = round(1 - np.sum((df_daily['observed'] - df_daily['simulated'])**2)/np.sum((df_daily['observed'] - df_daily['observed'].mean())**2),3)
    NSE_garrick = round(1 - np.sum(np.absolute(df_daily['observed'] - df_daily['simulated']))/np.sum(np.absolute(df_daily['observed'] - df_daily['observed'].mean())),3)
    NSE_monthly = round(1 - np.sum((df_monthly['observed'] - df_monthly['simulated'])**2)/np.sum((df_monthly['observed'] - df_monthly['observed'].mean())**2),3)
    metric = pd.DataFrame(data = [NSE_daily, NSE_garrick, NSE_monthly]).transpose()
    metric.columns = ['NSE','Garrick','Monthly']
    return metric


#%% Hydrology
def hydro_stats(df_daily,drg_area): 
    '''
    Parameters
    ----------
    df_daily : TYPE
        Daily flow timeseries in cfs. A dataframe with a datetime index and columns labeled 'observed','simulated'
    drg_area : TYPE
        drainage area in acres.

    Returns
    -------
    df : TYPE
        Dataframe of various metrics used to evaluate model hydrology simulations.
        Outputs are in inches.

    '''
    df_daily = hydro_sep(drg_area*0.0015625,df_daily) # in cfs
    df_daily_inches = df_daily*60*60*24*12/(drg_area*43560) # in inches
    dfs = [total(df_daily_inches,'in'),
           annual(df_daily_inches,'in'),
           monthly(df_daily_inches,'in'),
           season(df_daily_inches,'in'),
           low_50(df_daily_inches),
           high_10(df_daily_inches),
           storm(df_daily_inches),
           storm_summer(df_daily_inches),
           baseflow(df_daily_inches)]
    
    
    dfs[0] ['interval'] = 'Total'
    dfs[1] ['interval'] = 'Annual'
    dfs[2].rename(columns = {'month':'interval'},inplace = True)
    dfs[3].rename(columns = {'season':'interval'},inplace = True)
    dfs[4] ['interval'] = 'Low 50'
    dfs[5] ['interval'] = 'High 10'
    dfs[6] ['interval'] = 'Storm'
    dfs[7] ['interval'] = 'Summer Storm'
    dfs[8] ['interval'] = 'Baseflow'
    
    
    df = pd.concat(dfs)
    df.set_index('interval',inplace=True)
    nse = NSE(df_daily)
    df.loc['Daily NSE',:] = pd.NA
    df.loc['Daily Garrick',:] = pd.NA
    df.loc['Monthly NSE',:] = pd.NA
    df.loc['Daily NSE','observed'] = nse['NSE'].values
    df.loc['Daily Garrick','observed'] = nse['Garrick'].values
    df.loc['Monthly NSE','observed'] = nse['Monthly'].values
    df.loc['Daily NSE','goal'] = .7
    df.loc['Daily Garrick','goal'] = .55
    df.loc['Monthly NSE','goal'] = .8
    return df




def hydro_sep_baseflow(df_daily,drng_area = None,method = 'Boughton'):
    #dfs,df_kge = bf.single(df_daily['observed'],area = drg_area)
    #df = bf.separation(df_daily['observed'],method = 'Boughton')
    #method = df_kge.idxmax()

    
    df_daily['observed_baseflow'] = bf.single(df_daily['observed'],area = drng_area, method = method,return_kge = False)[0][method]
    df_daily['simulated_baseflow'] = bf.single(df_daily['simulated'],area = drng_area, method = method,return_kge = False)[0][method]
    df_daily['observed_runoff'] = df_daily['observed'] - df_daily['observed_baseflow']
    df_daily['simulated_runoff'] = df_daily['simulated'] - df_daily['simulated_baseflow']
    return df_daily

    
    

def hydro_sep(drg_area,df_daily):
    try:
        df_daily = hydro_sep_baseflow(df_daily,drg_area)
    except:
        twonstar = 2*drg_area**0.2
        twonstar = round(twonstar,0)
        if twonstar%2==0:
            twonstar = twonstar - 1
        twonstar = np.median([3,11,twonstar])
        ndays = (twonstar - 1)/2
        ndays = int(ndays)
    
        df_daily['observed_baseflow'] = df_daily['observed'].rolling(2*ndays+1,center=True,min_periods=ndays+1).apply(np.nanmin)
        df_daily['observed_runoff'] = df_daily['observed'] - df_daily['observed_baseflow']
        df_daily['simulated_baseflow'] = df_daily['simulated'].rolling(2*ndays+1,center=True,min_periods=ndays+1).apply(np.nanmin)
        df_daily['simulated_runoff'] = df_daily['simulated'] - df_daily['simulated_baseflow']
    return df_daily

# def aggregate(df,period = None,agg_func = 'mean'):
#     if periond is None:
#         df = df.agg(agg_func)
#     elif period == 'Y':
#         grouper = df.index.year
#     elif period == 'M':
#         grouper = df.index.month
#     elif period == 'D':
#         grouper = df.index.dayofyear
        
    
def annual(df_daily,units):
    assert units in ['cfs','in','lb','mg/l','degC']
    if units in ['in','lb']:    
        # Total flow volume
        n_years = df_daily.groupby(df_daily.index.year).count()/365.25
        df_daily = ((df_daily.groupby(df_daily.index.year).sum())/n_years).mean()# (df_daily.groupby(df_daily.index.year).count()/365.25) .(agg_func)
    else:
        df_daily = (df_daily.groupby(df_daily.index.year).mean()).mean()# (df_daily.groupby(df_daily.index.year).count()/365.25) .(agg_func)

    
    observed = df_daily['observed']
    simulated = df_daily['simulated']
    error = round((simulated - observed)/observed*100,2)
    observed = float(round(observed,2))
    simulated = float(round(simulated,2))
    metric = { 'observed': [observed], 
             'simulated': [simulated], 
             'per_error': [error], 
             'abs_error': [simulated-observed],
             'goal': [10]}
    return pd.DataFrame(metric)

def total(df_daily,units):
    if units in ['in','lb']:
        agg_func = 'sum'
    else:
        agg_func = 'mean'
    # Calculate monthly flow in inches
    #df_monthly = df_daily.resample('MS').sum()
    #df_monthly_grouped = df_monthly.groupby(df_monthly.index.month).mean()
    #df_monthly_grouped = df_daily.groupby(df_daily.index.month).agg(agg_func)

    # Total flow volume
    df_daily = df_daily.agg(agg_func)
    observed = df_daily['observed']
    simulated = df_daily['simulated']
    error = round((simulated - observed)/observed*100,2)
    observed = float(round(observed,2))
    simulated = float(round(simulated,2))
    metric = { 'observed': [observed], 
             'simulated': [simulated], 
             'per_error': [error], 
             'abs_error': [simulated-observed],
             'goal': [10]}
    return pd.DataFrame(metric)

def monthly(df_daily,units):
    assert units in ['cfs','in','lb','mg/l','degC']
    #df_monthly = df_daily.resample('MS').sum()
    #df_monthly_grouped = df_monthly.groupby(df_monthly.index.month).agg(agg_func)
    n_months = df_daily.groupby(df_daily.index.month).count()/30  
    if units in ['in','lb']:
        df_monthly_grouped = df_daily.groupby(df_daily.index.month).sum()/n_months #.agg(agg_func)
    else:
        df_monthly_grouped = df_daily.groupby(df_daily.index.month).mean()#.agg(agg_func)

    # Monthly average flow volumes
    mont_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'] 
    metrics = []
    for month in range(1,13):
        if month in df_monthly_grouped.index:
            observed = df_monthly_grouped.loc[month,'observed']
            simulated = df_monthly_grouped.loc[month,'simulated']
            error = round((simulated-observed)/observed*100,2)
            metric = {'month' : mont_names[month-1],
                     'observed': [observed], 
                     'simulated': [simulated], 
                     'per_error': [error], 
                     'abs_error': [simulated-observed],
                     'goal': [30]}
            
        else:
            metric = {'month' : mont_names[month-1],
                     'observed': [pd.NA], 
                     'simulated': [pd.NA], 
                     'per_error': [pd.NA], 
                     'abs_error': [pd.NA],
                     'goal': [pd.NA]}
        
        metrics.append(pd.DataFrame(metric))
    return pd.concat(metrics).reset_index(drop=True)


def exceedence(df):
        
    df['simulated_rank'] = df['simulated'].rank(method = 'average', ascending = False)
    df['simulated_exceed'] = df['simulated_rank'] / (len(df) + 1) * 100
    df['observed_rank'] = df['observed'].rank(method = 'average', ascending = False)
    df['observed_exceed'] = df['observed_rank'] / (len(df) + 1) * 100
    
    if 'simulated_flow' in df.columns:
        df['simulated_flow_rank'] = df['simulated_flow'].rank(method = 'average', ascending = False)
        df['simulated_flow_exceed'] = df['simulated_flow_rank'] / (len(df) + 1) * 100
        df['observed_flow_rank'] = df['observed_flow'].rank(method = 'average', ascending = False)
        df['observed_flow_exceed'] = df['observed_flow_rank'] / (len(df) + 1) * 100
    

def low_50(df_daily):

     num_years = len(df_daily.index)/365.25
     num_bottom50 =  int(len(df_daily.index)/2)
     # 50% lowest flow volume
     observed = df_daily.sort_values('observed',ascending=False).tail(num_bottom50).sum()/num_years
     simulated = df_daily.sort_values('simulated',ascending=False).tail(num_bottom50).sum()/num_years
     error = round((simulated['simulated'] - observed['observed'])/observed['observed']*100,2)
     observed = float(round(observed['observed'],2))
     simulated = float(round(simulated['simulated'],2))
     metric = { 'observed': [observed], 
              'simulated': [simulated], 
              'per_error': [error], 
              'abs_error': [simulated-observed],
              'goal': [10]}
     return pd.DataFrame(metric)

def high_10(df_daily):
     num_years = len(df_daily.index)/365.25
     num_top10 =  int(len(df_daily.index)/10)
     # 10% highest flow volume
     observed = df_daily.sort_values('observed',ascending=False).head(num_top10).sum()/num_years
     simulated = df_daily.sort_values('simulated',ascending=False).head(num_top10).sum()/num_years
     error = round((simulated['simulated'] - observed['observed'])/observed['observed']*100,2)
     observed = float(round(observed['observed'],2))
     simulated = float(round(simulated['simulated'],2))
     metric = { 'observed': [observed], 
              'simulated': [simulated], 
              'per_error': [error], 
              'abs_error': [simulated-observed],
              'goal': [15]}

     return pd.DataFrame(metric)
    
    

def season(df_daily,units):
    assert units in ['cfs','in','lb','mg/l','degC']

    # Calculate monthly flow in inches
    #df_monthly = df_daily.resample('MS').sum()
    #df_monthly_grouped = df_monthly.groupby(df_monthly.index.month).mean()
    #df_monthly_grouped = df_daily.groupby(df_daily.index.month).agg(agg_func)
    n_months = df_daily.groupby(df_daily.index.month).count()/30  
    
    if units in ['in','lb']:
        agg_func = 'sum'
        df_monthly_grouped = df_daily.groupby(df_daily.index.month).sum()/n_months #.agg(agg_func)
    else:
        agg_func = 'mean'
        df_monthly_grouped = df_daily.groupby(df_daily.index.month).mean() #.agg(agg_func)


    metrics = []
    for season_start in [7,10,1,4]:
        #summer,fall,winter,spring
        # define seasons to allow for descriptive output
        if season_start == 7:
            season = 'summer '
        elif season_start == 10:
            season = 'fall'
        elif season_start == 1:
            season = 'winter'
        elif season_start == 4:
            season = 'spring'
        observed = df_monthly_grouped['observed'].loc[season_start:season_start + 2].agg(agg_func)
        simulated = df_monthly_grouped['simulated'].loc[season_start:season_start + 2].agg(agg_func)
        error = round((simulated - observed)/observed*100,2)
        observed = float(round(observed,2))
        simulated = float(round(simulated,2))
        metrics.append({'season': season,
                  'observed': observed, 
                  'simulated': simulated, 
                  'per_error': error, 
                  'abs_error': simulated-observed,
                  'goal': [30]}) 
    return pd.DataFrame(metrics).reset_index(drop=True)


def storm(df_daily,agg_func = 'mean'):
    # # Calculate monthly flow in inches
    # df_monthly = df_daily.resample('MS').sum()
    # df_monthly_grouped = df_monthly.groupby(df_monthly.index.month).mean()
    
    # # Storm flow volume
    # observed = df_monthly_grouped['observed_runoff'].sum()
    # simulated = df_monthly_grouped['simulated_runoff'].sum()
    
    #df_daily = df_daily.resample('Y').sum().mean()
    #df_daily = df_daily.groupby('Y').agg(agg_func)
    num_years = len(df_daily.index)/365.25
    df_daily = df_daily.sum()/num_years
    observed = df_daily['observed_runoff']
    simulated = df_daily['simulated_runoff']
    
    error = round((simulated - observed)/observed*100,2)
    observed = float(round(observed,2))
    simulated = float(round(simulated,2))
    metric = { 'observed': [observed], 
             'simulated': [simulated], 
             'per_error': [error], 
             'abs_error': [simulated-observed],
             'goal': [20]}
    
    return pd.DataFrame(metric)

def storm_summer(df_daily,agg_func = 'mean'):
    # average annual summer volume
    # Calculate monthly flow in inches
    #df_monthly = df_daily.resample('MS').sum()
    #df_monthly_grouped = df_monthly.groupby(df_monthly.index.month).mean()
    n_months = df_daily.groupby(df_daily.index.month).count()/30  
    df_monthly_grouped = df_daily.groupby(df_daily.index.month).sum()/n_months #.agg(agg_func)


    # Summer storm flow volume
    observed = df_monthly_grouped['observed_runoff'].loc[7:9].sum()
    simulated = df_monthly_grouped['simulated_runoff'].loc[7:9].sum()
    error = round((simulated - observed)/observed*100,2)
    observed = float(round(observed,2))
    simulated = float(round(simulated,2))
    metric = { 'observed': [observed], 
             'simulated': [simulated], 
             'per_error': [error], 
             'abs_error': [simulated-observed],
             'goal': [50]}
    
    return pd.DataFrame(metric)    

def baseflow(df_daily,agg_func = 'mean'):
    # average annual baseflow
    num_years = len(df_daily.index)/365.25
    #df_monthly = df_daily.resample('MS').sum()
    #df_monthly_grouped = df_monthly.groupby(df_monthly.index.month).mean()
    #df_monthly_grouped = df_daily.groupby(df_daily.index.month).agg(agg_func)

    observed = df_daily['observed_baseflow'].sum()/num_years
    simulated = df_daily['simulated_baseflow'].sum()/num_years
    error = round((simulated - observed)/observed*100,2)
    observed = float(round(observed,2))
    simulated = float(round(simulated,2))
    metric = { 'observed': [observed], 
             'simulated': [simulated], 
             'per_error': [error], 
             'abs_error': [simulated-observed],
             'goal': [20]}
    
    return pd.DataFrame(metric)    



def stats(df_daily,units):
    dfs = [total(df_daily,units),
           annual(df_daily,units),
           monthly(df_daily,units),
           season(df_daily,units)]
        
 
    dfs[0] ['interval'] = 'Total'
    dfs[1] ['interval'] = 'Annual'
    dfs[2].rename(columns = {'month':'interval'},inplace = True)
    dfs[3].rename(columns = {'season':'interval'},inplace = True)
    df = pd.concat(dfs)
    df.set_index('interval',inplace=True)

    return df

# =============================================================================
# STANDARDIZED METRICS FOR OBSERVED VS SIMULATED COMPARISON
# =============================================================================
# These functions provide simple, standardized interfaces for calculating
# common hydrological metrics. They accept pandas Series with datetime indices
# and automatically handle data overlap.

# Small offset for log transformations to avoid log(0)
LOG_OFFSET = 0.01

# Month name abbreviations
METRIC_MONTH_NAMES = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                      'jul', 'aug', 'sep', 'oct', 'nov', 'dec']

# Season definitions (month numbers)
METRIC_SEASONS = {
    'winter': [12, 1, 2],
    'spring': [3, 4, 5],
    'summer': [6, 7, 8],
    'fall': [9, 10, 11]
}

def align_series(observed, simulated):
    """
    Align observed and simulated series to ensure they have the same datetime index.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    tuple
        (obs_aligned, sim_aligned) - aligned pandas Series
    """
    common_index = observed.index.intersection(simulated.index)
    
    if len(common_index) == 0:
        raise ValueError("No overlapping dates between observed and simulated data")
    
    obs_aligned = observed.loc[common_index]
    sim_aligned = simulated.loc[common_index]
    
    return obs_aligned, sim_aligned


def _get_overlapping_data(observed, simulated):
    """
    Get overlapping data between observed and simulated series.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    tuple
        (obs_array, sim_array, common_index) - numpy arrays and pandas index
    """
    obs_clean = observed.dropna()
    sim_clean = simulated.dropna()
    common_dates = obs_clean.index.intersection(sim_clean.index)
    
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between observed and simulated data")
    
    obs_arr = obs_clean.loc[common_dates].values
    sim_arr = sim_clean.loc[common_dates].values
    
    return obs_arr, sim_arr, common_dates


def nse_metric(observed, simulated):
    """
    Calculate Nash-Sutcliffe Efficiency (NSE).
    
    NSE ranges from -infinity to 1.0, where:
    - NSE = 1.0: Perfect match
    - NSE = 0.0: Model performs as well as using observed mean
    - NSE < 0.0: Model performs worse than observed mean
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    float
        NSE value
        
    Examples
    --------
    >>> nse_value = nse_metric(observed_flow, simulated_flow)
    >>> print(f"NSE = {nse_value:.3f}")
    """
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    
    ss_res = np.sum((obs_arr - sim_arr) ** 2)
    ss_tot = np.sum((obs_arr - np.mean(obs_arr)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def log_nse_metric(observed, simulated, offset=None):
    """
    Calculate Nash-Sutcliffe Efficiency on log-transformed data.
    
    Log-NSE emphasizes low-flow performance by reducing the influence
    of high flow events.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
    offset : float, optional
        Small value added before log transform to avoid log(0).
        Default is LOG_OFFSET (0.01).
        
    Returns
    -------
    float
        Log-NSE value
    """
    if offset is None:
        offset = LOG_OFFSET
        
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    
    obs_log = np.log(obs_arr + offset)
    sim_log = np.log(sim_arr + offset)
    
    ss_res = np.sum((obs_log - sim_log) ** 2)
    ss_tot = np.sum((obs_log - np.mean(obs_log)) ** 2)
    
    return 1 - (ss_res / ss_tot) if ss_tot != 0 else np.nan


def kge_metric(observed, simulated):
    """
    Calculate Kling-Gupta Efficiency (KGE).
    
    KGE combines correlation, bias ratio, and variability ratio into
    a single metric. It ranges from -infinity to 1.0.
    
    KGE = 1 - sqrt((r-1)² + (α-1)² + (β-1)²)
    
    Where:
    - r = correlation coefficient
    - α = ratio of standard deviations (sim/obs)
    - β = ratio of means (sim/obs)
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    float
        KGE value
    """
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    
    # Correlation
    r = np.corrcoef(obs_arr, sim_arr)[0, 1] if len(obs_arr) > 1 else 0
    
    # Variability ratio
    alpha = np.std(sim_arr) / np.std(obs_arr) if np.std(obs_arr) != 0 else 0
    
    # Bias ratio
    beta = np.mean(sim_arr) / np.mean(obs_arr) if np.mean(obs_arr) != 0 else 0
    
    return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)


def bias_metric(observed, simulated):
    """
    Calculate mean bias (simulated - observed).
    
    Positive bias means model over-predicts on average.
    Negative bias means model under-predicts on average.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    float
        Mean bias value
    """
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    return np.mean(sim_arr) - np.mean(obs_arr)


def pbias_metric(observed, simulated):
    """
    Calculate percent bias.
    
    PBIAS = 100 * (Σsim - Σobs) / Σobs
    
    Positive PBIAS indicates over-prediction bias.
    Negative PBIAS indicates under-prediction bias.
    Optimal value is 0.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    float
        Percent bias value (in percent, e.g., 10.5 means 10.5%)
    """
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    
    obs_sum = np.sum(obs_arr)
    if obs_sum == 0:
        return np.nan
    
    return 100 * (np.sum(sim_arr) - obs_sum) / obs_sum


def rmse_metric(observed, simulated):
    """
    Calculate Root Mean Square Error.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    float
        RMSE value (same units as input data)
    """
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    return np.sqrt(np.mean((obs_arr - sim_arr) ** 2))


def mae_metric(observed, simulated):
    """
    Calculate Mean Absolute Error.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    float
        MAE value (same units as input data)
    """
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    return np.mean(np.abs(obs_arr - sim_arr))


def correlation_metric(observed, simulated):
    """
    Calculate Pearson correlation coefficient.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
        
    Returns
    -------
    float
        Correlation coefficient (-1 to 1)
    """
    obs_arr, sim_arr, _ = _get_overlapping_data(observed, simulated)
    
    if len(obs_arr) < 2:
        return np.nan
    
    return np.corrcoef(obs_arr, sim_arr)[0, 1]


def monthly_means_metric(observed, simulated, return_both=True):
    """
    Calculate monthly mean values from overlapping data.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
    return_both : bool, default True
        If True, return both observed and simulated monthly means.
        If False, return only simulated monthly means.
        
    Returns
    -------
    dict or tuple of dicts
        Dictionary mapping month names (e.g., 'jan', 'feb') to mean values.
        If return_both=True, returns (obs_monthly, sim_monthly) tuple.
        
    Examples
    --------
    >>> obs_monthly, sim_monthly = monthly_means_metric(observed, simulated)
    >>> print(f"January observed mean: {obs_monthly['jan']:.2f}")
    >>> print(f"January simulated mean: {sim_monthly['jan']:.2f}")
    """
    obs_arr, sim_arr, common_dates = _get_overlapping_data(observed, simulated)
    
    df = pd.DataFrame({
        'observed': observed.loc[common_dates],
        'simulated': simulated.loc[common_dates]
    })
    
    monthly_obs = df.groupby(df.index.month)['observed'].mean()
    monthly_sim = df.groupby(df.index.month)['simulated'].mean()
    
    obs_dict = {METRIC_MONTH_NAMES[m-1]: monthly_obs.get(m, np.nan) for m in range(1, 13)}
    sim_dict = {METRIC_MONTH_NAMES[m-1]: monthly_sim.get(m, np.nan) for m in range(1, 13)}
    
    if return_both:
        return obs_dict, sim_dict
    return sim_dict


def seasonal_means_metric(observed, simulated, return_both=True):
    """
    Calculate seasonal mean values from overlapping data.
    
    Seasons are defined as:
    - Winter: December, January, February
    - Spring: March, April, May
    - Summer: June, July, August
    - Fall: September, October, November
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
    return_both : bool, default True
        If True, return both observed and simulated seasonal means.
        
    Returns
    -------
    dict or tuple of dicts
        Dictionary mapping season names to mean values.
    """
    obs_arr, sim_arr, common_dates = _get_overlapping_data(observed, simulated)
    
    df = pd.DataFrame({
        'observed': observed.loc[common_dates],
        'simulated': simulated.loc[common_dates]
    })
    
    obs_dict = {}
    sim_dict = {}
    
    for season_name, months in METRIC_SEASONS.items():
        mask = df.index.month.isin(months)
        if mask.any():
            obs_dict[season_name] = df.loc[mask, 'observed'].mean()
            sim_dict[season_name] = df.loc[mask, 'simulated'].mean()
        else:
            obs_dict[season_name] = np.nan
            sim_dict[season_name] = np.nan
    
    if return_both:
        return obs_dict, sim_dict
    return sim_dict


def exceedance_values_metric(data, probs=None):
    """
    Calculate flow values at specified exceedance probabilities.
    
    Exceedance probability is the probability that a given flow will be
    equaled or exceeded. Used for flow duration curves (FDC).
    
    Parameters
    ----------
    data : pd.Series or array-like
        Flow data
    probs : list of float, optional
        Exceedance probabilities (0-100). Default: [10, 25, 50, 75, 90]
        
    Returns
    -------
    dict
        Dictionary mapping exceedance probability to flow value.
        Keys are like 'q10', 'q50', etc.
        
    Examples
    --------
    >>> fdc = exceedance_values_metric(observed_flow)
    >>> print(f"Q10 (high flow): {fdc['q10']:.2f}")
    >>> print(f"Q50 (median): {fdc['q50']:.2f}")
    >>> print(f"Q90 (low flow): {fdc['q90']:.2f}")
    """
    if probs is None:
        probs = [10, 25, 50, 75, 90]
    
    if isinstance(data, pd.Series):
        data = data.dropna().values
    
    sorted_data = np.sort(data)[::-1]  # Descending order
    n = len(sorted_data)
    
    result = {}
    for prob in probs:
        idx = int(np.ceil(prob / 100.0 * n)) - 1
        idx = max(0, min(idx, n - 1))
        result[f'q{int(prob):02d}'] = sorted_data[idx]
    
    return result


def calculate_all_statistics(observed, simulated, include_monthly=True, 
                             include_seasonal=True, include_exceedance=True,
                             exceedance_probs=None):
    """
    Calculate all statistics from overlapping observed and simulated data.
    
    This is the main function for getting a comprehensive set of metrics
    for model evaluation or PEST++ calibration.
    
    Parameters
    ----------
    observed : pd.Series
        Observed data with datetime index
    simulated : pd.Series
        Simulated data with datetime index
    include_monthly : bool, default True
        Include monthly average statistics
    include_seasonal : bool, default True
        Include seasonal average statistics
    include_exceedance : bool, default True
        Include flow duration curve statistics
    exceedance_probs : list of float, optional
        Exceedance probabilities. Default: [10, 25, 50, 75, 90]
        
    Returns
    -------
    dict
        Dictionary of all computed statistics. Keys include:
        - 'nse', 'log_nse', 'kge', 'bias', 'pbias', 'rmse', 'mae', 'r'
        - 'jan_avg', 'feb_avg', ... 'dec_avg' (if include_monthly)
        - 'winter_avg', 'spring_avg', 'summer_avg', 'fall_avg' (if include_seasonal)
        - 'q10', 'q25', 'q50', 'q75', 'q90' (if include_exceedance)
        - 'n_obs': Number of overlapping observations
        
    Examples
    --------
    >>> stats = calculate_all_statistics(observed, simulated)
    >>> print(f"NSE: {stats['nse']:.3f}")
    >>> print(f"Bias: {stats['bias']:.3f}")
    >>> print(f"January mean: {stats['jan_avg']:.2f}")
    """
    if exceedance_probs is None:
        exceedance_probs = [10, 25, 50, 75, 90]
    
    # Get overlapping data
    obs_arr, sim_arr, common_dates = _get_overlapping_data(observed, simulated)
    
    results = {
        'n_obs': len(common_dates),
        'nse': nse_metric(observed, simulated),
        'log_nse': log_nse_metric(observed, simulated),
        'kge': kge_metric(observed, simulated),
        'bias': bias_metric(observed, simulated),
        'pbias': pbias_metric(observed, simulated),
        'rmse': rmse_metric(observed, simulated),
        'mae': mae_metric(observed, simulated),
        'r': correlation_metric(observed, simulated),
        'annual_avg': np.mean(sim_arr)
    }
    
    # Monthly averages
    if include_monthly:
        _, sim_monthly = monthly_means_metric(observed, simulated)
        for month, val in sim_monthly.items():
            results[f'{month}_avg'] = val
    
    # Seasonal averages
    if include_seasonal:
        _, sim_seasonal = seasonal_means_metric(observed, simulated)
        for season, val in sim_seasonal.items():
            results[f'{season}_avg'] = val
    
    # Exceedance values (from simulated data)
    if include_exceedance:
        sim_series = simulated.loc[common_dates]
        exc = exceedance_values_metric(sim_series, exceedance_probs)
        results.update(exc)
    
    return results