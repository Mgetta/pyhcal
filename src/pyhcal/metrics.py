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
# @dataclass
# class Timeseries:
    
    
# @dataclass
# class HydroData:
    
#     sim
#     obs
#     units
    



# class Metrics():
    
#     data
    




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


#%% Hydrograph Limb Identification

def identify_limbs(series, min_duration=1):
    '''
    Identify rising and falling limbs of a hydrograph.
    
    Parameters
    ----------
    series : pd.Series
        Timeseries of flow values with a datetime index. Can have gaps.
    min_duration : int, optional
        Minimum number of consecutive timesteps for a limb to be valid. Default is 1.
        
    Returns
    -------
    pd.Series
        Series with same index as input, containing limb classifications:
        'rising' - flow is increasing
        'falling' - flow is decreasing
        'peak' - local maximum (transition from rising to falling)
        'trough' - local minimum (transition from falling to rising)
        'stable' - no change in flow
        NaN - insufficient data to classify
    '''
    if not isinstance(series, pd.Series):
        raise TypeError("Input must be a pandas Series")
    
    if len(series) < 2:
        return pd.Series(index=series.index, dtype=object)
    
    # Calculate change between consecutive values
    diff = series.diff()
    
    # Initialize limb classification
    limbs = pd.Series(index=series.index, dtype=object)
    
    # Classify based on direction of change
    limbs[diff > 0] = 'rising'
    limbs[diff < 0] = 'falling'
    limbs[diff == 0] = 'stable'
    
    # Identify peaks and troughs
    # Shift the limb classifications to compare previous and next
    prev_limb = limbs.shift(1)
    next_limb = limbs.shift(-1)
    
    # Peak: was rising, will be falling
    peak_mask = (prev_limb == 'rising') & (next_limb == 'falling')
    limbs[peak_mask] = 'peak'
    
    # Trough: was falling, will be rising
    trough_mask = (prev_limb == 'falling') & (next_limb == 'rising')
    limbs[trough_mask] = 'trough'
    
    # Filter by minimum duration if specified
    if min_duration > 1:
        limbs = _filter_by_duration(limbs, min_duration)
    
    return limbs


def _filter_by_duration(limbs, min_duration):
    '''
    Filter limb classifications by minimum duration.
    
    Parameters
    ----------
    limbs : pd.Series
        Limb classifications from identify_limbs.
    min_duration : int
        Minimum number of consecutive timesteps for a limb to be valid.
        
    Returns
    -------
    pd.Series
        Filtered limb classifications with short segments set to NaN.
    '''
    result = limbs.copy()
    
    # Group consecutive values
    groups = (limbs != limbs.shift()).cumsum()
    
    for limb_type in ['rising', 'falling']:
        mask = limbs == limb_type
        group_counts = groups[mask].value_counts()
        short_groups = group_counts[group_counts < min_duration].index
        
        for group_id in short_groups:
            result[(groups == group_id) & mask] = np.nan
    
    return result


def get_rising_limb(series, min_duration=1):
    '''
    Extract the rising limb portions of a hydrograph.
    
    Parameters
    ----------
    series : pd.Series
        Timeseries of flow values with a datetime index.
    min_duration : int, optional
        Minimum number of consecutive timesteps for a rising limb to be included.
        Default is 1.
        
    Returns
    -------
    pd.Series
        Series containing only the rising limb values, with NaN elsewhere.
    '''
    limbs = identify_limbs(series, min_duration)
    rising_mask = limbs == 'rising'
    result = series.copy()
    result[~rising_mask] = np.nan
    return result


def get_falling_limb(series, min_duration=1):
    '''
    Extract the falling limb portions of a hydrograph.
    
    Parameters
    ----------
    series : pd.Series
        Timeseries of flow values with a datetime index.
    min_duration : int, optional
        Minimum number of consecutive timesteps for a falling limb to be included.
        Default is 1.
        
    Returns
    -------
    pd.Series
        Series containing only the falling limb values, with NaN elsewhere.
    '''
    limbs = identify_limbs(series, min_duration)
    falling_mask = limbs == 'falling'
    result = series.copy()
    result[~falling_mask] = np.nan
    return result


def limb_statistics(series, min_duration=1):
    '''
    Calculate statistics for rising and falling limbs of a hydrograph.
    
    Parameters
    ----------
    series : pd.Series
        Timeseries of flow values with a datetime index.
    min_duration : int, optional
        Minimum number of consecutive timesteps for a limb to be included.
        Default is 1.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with statistics for rising and falling limbs including:
        - count: number of limb events
        - mean_duration: average duration of limb events (in timesteps)
        - mean_magnitude: average absolute magnitude change during limb (always positive)
        - total_duration: total time spent in each limb type
    '''
    limbs = identify_limbs(series, min_duration)
    
    stats = {}
    for limb_type in ['rising', 'falling']:
        mask = limbs == limb_type
        
        # Group consecutive values to identify individual events
        groups = (limbs != limbs.shift()).cumsum()
        limb_groups = groups[mask]
        
        if len(limb_groups) == 0:
            stats[limb_type] = {
                'count': 0,
                'mean_duration': np.nan,
                'mean_magnitude': np.nan,
                'total_duration': 0
            }
            continue
        
        # Calculate duration of each event
        durations = limb_groups.value_counts()
        
        # Calculate magnitude of each event (absolute change in flow value)
        magnitudes = []
        for group_id in durations.index:
            group_data = series[groups == group_id]
            if len(group_data) > 0:
                magnitudes.append(abs(group_data.iloc[-1] - group_data.iloc[0]))
        
        stats[limb_type] = {
            'count': len(durations),
            'mean_duration': durations.mean(),
            'mean_magnitude': np.mean(magnitudes) if magnitudes else np.nan,
            'total_duration': durations.sum()
        }
    
    return pd.DataFrame(stats).T


#%% Sediment

# def sed_stats(df,units):
#     if units == 'mg/l':
#         agg_func = 'mean'
#     else:
#         agg_func = 'sum'
        
#     df_agg = pd.DataFrame(np.ones((12,3))*np.nan,index = range(1,13),columns = ['model','flux','ratio'])
#     df_agg.index.name = 'month'
#     df = df.groupby(df.index.month).agg(agg_func)[['Simflow','Obsflow']]
#     df.columns = ['model','flux']
#     df['ratio'] = df['flux']/df['model']
#     df_agg.loc[df.index,df.columns] = df.values
    
#     df_agg.loc['Mean'] = df_agg.agg('mean')
#     df_agg['ratio'] = df_agg['flux']/df_agg['model']
#     return df_agg


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


#%% Nutrients
def nutrient_stats(df_daily,units):
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
