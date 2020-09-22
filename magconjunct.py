"""
Nadine Tabucol
29 March 2020

Find and plot magnetic conjunctions between RBSP and "each" (only 'm02' as of 29-03-2020)
POES spacecraft. 

Functions: 
- findConjunct
- viewConjunct
- plotConjunct

Last updated: 
21 September 2020: 
- Finished revisions compare_flux and compare_all_flux

14 September 2020: 
- compare_all_flux and compare_flux updates to get matching equatorial pitch angle values. 

04 September 2020: 
- 

03 September 2020: 
- Debug findConjunct()

13 August 2020:
- Added additional columns for MLAT and BoverBeq for POES and MAGEIS in findConjunct()

05 August 2020:
- Get POES equatorial pitch angle from POES MagEphem ('OP77Q') files in findConjunct()
- Working on: Get POES equatorial pitch angle through MLAT in findConjunct()

18 May 2020:
- Finished debugging and rewriting findConjunct()
- Finished debugging and testing compare_all_flux() and 
  corresponding helper functions.
- Updated plotConjunctFlux()
- The notebook manual_integ_test.ipynb contains the 
  testing apparatus used to check the energy-integration
  of MagEIS FEDU (output of compare_all_flux)

14 May 2020: 
- Fixed interpolation problem in findConjunct()
- Finished debugging compare_all_flux and helper functions
- To-do: 2015-05-17 NaNs interpreted as zeros instead of NaNs

13 May 2020:
- Added MagEIS L, MLT columns in findConjunct() DataFrame output.

08 May 2020
- Finished revising compareFlux algorithm
- Now consists of get_both_flux + compare_flux + compare_all_flux
- Other functions considered deprecated. 

06 May 2020
- Revised compareFlux algorithm 
- Find corresponding POES flux values
- Integrates fine-resolution MagEIS FEDU over similar
  POES energy bins (e1, e2, e3, e4) 

28 April 2020
- Added flux-finding and plotting functions. 


"""

import numpy as np
import pandas as pd
import readsat as rs
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, timezone
from dateutil.tz import tzutc
from pandas.plotting import register_matplotlib_converters
from scipy.integrate import trapz
from numpy import errstate,isneginf,array
import subplotter

# # DEBUG - COMMENT OUT
# import warnings
# warnings.filterwarnings(action = "error")
# # END DEBUG

### FIND CONJUNCTIONS ###
def mltToAngle(mval):
    """
    Helper function for findConjunct, MLT conversion to clock angle. 
    
    Args:
        mval: MLT value in hours.
    
    Output: 
        clock_angle: 'mval' mapping to clock angle (0 to 360) in degrees.
        
    """
    
    if mval < 0 or mval > 24.0: 
        print('Invalid value for MLT.')
        clock_angle = None
        raise
    
    if mval >= 0.0 and mval < 12.0:
        clock_angle = 15 * (mval + 12.0)
        
    else:
        clock_angle = 15 * (mval - 12.0)
        
    
    return clock_angle



def angleToMLT(theta):
    """
    Helper function for findConjunct, MLT conversion to clock angle. 
    
    Args:
        theta: Clock angle in degrees.
    
    """
    if theta < 0.0 or theta > 360:
        print('Invalid value for theta.')
        mval = None
        raise
    
    if theta >= 0 and theta < 180.0:
        mval = (theta / 15.0) + 12
        
    else:
        mval = (theta / 15.0) - 12
        
    return mval



def mltProximate(mlt1, mlt2, mltTol): 
    """
    Helper function for findConjunct, determines whether two MLT values 
    are close to each other, to some tolerance value ('mltTol').
    
    Args:
      mlt1, mlt2 (float, float): Input MLT values.
      mltTol (float): Tolerance value to determine closeness between mlt1 and mlt2.
      
    Output: Bool value, determined by |mlt1 - mlt2| <= mltTol, after conversion
            to clock angles.
    """
    
    phi_tol = mltToAngle(mltTol) # Tol value conversion to clock angle.
    tol_coor = (np.cos((phi_tol - 180.0) * (np.pi/180.0)), 
                np.sin((phi_tol - 180.0) * (np.pi/180.0)))

    phi1 = mltToAngle(mlt1)
    phi2 = mltToAngle(mlt2)
    
    diff = np.absolute(phi1 - phi2) * (np.pi/180.0)
    sin_diff = np.sin(diff)
    cos_diff = np.cos(diff)

    if sin_diff <= tol_coor[1] and cos_diff >= tol_coor[0]:
        return True
    else:
        return False



def findConjunct(times, time_tol, l_tol, mlt_tol, 
                    field,
                    alpha_tol = None,
                    ltype = 'L', mlt_type='EDMAG_MLT', mlat_type='EDMAG_MLAT',
                    within = False,
                    alpha_method = 0,
                    savename='',
                    save=False):
    """
    Find L and MLT conjunction values between POES and RBSP (MagEphem) probes, 
    Notes: alpha_method = 1 doesn't work as well.
    
    Kwargs:
        ltype (str): Choose specific 'L' variable to read from RBSP MagEphem.
        mlt_type (str): Choose specific 'MLT' variable to read from RBSP MagEphem.
        mlat_type (str): Choose specific 'MLAT' variable to read from RBSP MagEphem. 
        within (bool): Count time differences from [0, time_tol] as a 
                MagEIS conjunction point.
        alpha_method (int): Eq. pitch angle mapping method indicator for POES measurements.
                            Valid values for `alpha_method` are 0 (No a_eq computation), 
                            1 (MLAT method), or 2 (Uses POES OP77Q files).
        savename (str): Name of saved conjunction file, without file type suffix.
        save (bool): Bool to save conjunctions
    
    Args:
        times (list-like): A 2-element tuple or list-like variable that holds 
                           start date and end date in string format: 'YYYY-mm-dd'.
        time_tol (float): Tolerance for datetime, in seconds. 
        l_tol (float): Tolerance for L values
        mlt_tol (float): Tolerance for MLT values
        field (str): Choose MagEphem field model (RBSP probes), either 'OP77Q' or 'TS04D'.
        
    Outputs:
        conjunctions (DataFrame): List of POES-RBSP conjunctions with their properties.
        
    """
    
    # Assert valid values for parameters
    assert alpha_method in [0, 1, 2], "Valid values for `alpha_method` are 0 (No a_eq computation), 1 (MLAT method), or 2 (Uses POES OP77Q files)."
    
    nan_fill = 1.1e30    
    phi_tol = mltToAngle(mlt_tol) - 180.0
        
    ### Read MagEIS ###
    # Only time var ('Epoch') 
    rbsp_sc = ['rbspa', 'rbspb']
    
    mg_full = [rs.read_mageis(trange=times, 
                             read_vars=[],
                             probe=i,
                             get_latest=True)[0] for i in rbsp_sc]
    
    mg_data = dict(zip(rbsp_sc, mg_full))

    ### Read MagEphem ###
    #  read_vap_ephem and read_poes returns a tuple, with the data
    #  in the first element of the tuple.
    #  Read in extra day to get values in between 23:59:00 and +1 00:00:00h 
    rbsp_end_time = datetime.strftime(datetime.strptime(times[1], '%Y-%m-%d') + timedelta(days=1), '%Y-%m-%d')
    
    rb_to_read = [ltype, mlt_type, mlat_type, 'BoverBeq']
    rbsp_list = [rs.read_vap_ephem(trange=times, 
                                   read_vars=rb_to_read, 
                                   probe=i,
                                   b_model=field)[0] for i in rbsp_sc]
    hasNextDayPoint = True
    try:
        final_rb_point = [rs.read_vap_ephem(trange=[rbsp_end_time, rbsp_end_time],
                            read_vars=rb_to_read, 
                            probe=i,
                            b_model=field)[0] for i in rbsp_sc]
    except:
        print('No next day value.')
        hasNextDayPoint = False
        
    if hasNextDayPoint:
        for i in range(2):

            # Append final timestamp
            rbsp_list[i][0] = np.append(rbsp_list[i][0], final_rb_point[i][0][0]) 
            # Append final L value
            rbsp_list[i][1] = np.concatenate((rbsp_list[i][1], final_rb_point[i][1][0].reshape(-1,18)), axis=0) 
            # Append final MLT value
            rbsp_list[i][2] = np.append(rbsp_list[i][2], final_rb_point[i][2][0])
            # Append final MLAT value
            rbsp_list[i][3] = np.append(rbsp_list[i][3], final_rb_point[i][3][0])
            # Append final BoverBeq value
            rbsp_list[i][4] = np.append(rbsp_list[i][4], final_rb_point[i][4][0])
    
    
    rbsp_data = dict(zip(rbsp_sc, rbsp_list))
        
    ### Read POES ###
    poes_sc = ['m02']  # Only 'm02' is available for now
#     poes_sc = ['m01', 'm02', 'n15', 'n18', 'n19']

    # Var inits
    l_name = 'L_IGRF'
    mlt_name = 'MLT'
    lat_name = 'mag_lat_sat'
    poes_get_eq = 'default'
    poes_var_names = [l_name, mlt_name, lat_name, 'meped_alpha_90_sat', 'meped_alpha_0_sat']
    
    final_cols = ['RBSP_SC', 'POES_SC', 'POES_DT', 
                  'POES_L', 'POES_MLT',
                  'POES_MLAT', 'POES_ALPHA_90', 'POES_ALPHA_0', 
                  'RBSP_L', 'RBSP_MLT',
                  'RBSP_MLAT', 'RBSP_BoverBeq', 
                  'mageis_pts', 'MAGEIS_L', 'MAGEIS_MLT', 'MAGEIS_BoverBeq']
    
    if alpha_method >= 1:
        l_name = 'lm'
        mlt_name = 'mlt'
        lat_name = 'mlat'
        poes_get_eq = 'magephem'
        poes_var_names = [l_name, mlt_name, lat_name, 'beq', 'blocal']
        
        final_cols = ['RBSP_SC', 'POES_SC', 'POES_DT', 
                      'POES_L', 'POES_MLT', 'POES_MLAT', 
                      'POES_ALPHA_90', 'POES_ALPHA_0', 
                      'POES_ALPHA_EQ_90', 
                      'RBSP_L', 'RBSP_MLT',
                      'RBSP_MLAT', 'RBSP_BoverBeq',
                      'mageis_pts', 'MAGEIS_L', 'MAGEIS_MLT', 'MAGEIS_MLAT', 'MAGEIS_BoverBeq']
                    
    poes_list = [rs.read_poes(trange=times,
                          sat_IDs=[i],
                          read_vars=poes_var_names,
                          filterL=True,
                          fileType=poes_get_eq)[0] for i in poes_sc]
    if alpha_method == 2: # Create a separate dictionary for MEPED local pitch angle values according to spacecraft
        poes_alpha_list = [rs.read_poes(trange=times,
                                        sat_IDs = [i],
                                        read_vars = ['meped_alpha_90_sat', 'meped_alpha_0_sat'],
                                        fileType='default')[0][1] for i in poes_sc]
        poes_alpha_90_dict = dict(zip(poes_sc, [poes_alpha_list[i][1] for i in range(len(poes_alpha_list))]))
        poes_alpha_0_dict = dict(zip(poes_sc, [poes_alpha_list[i][2] for i in range(len(poes_alpha_list))]))
        
    else:
        poes_alpha_90_dict = dict(zip(poes_sc, [poes_list[i][4] for i in range(len(poes_list))]))
        poes_alpha_0_dict = dict(zip(poes_sc, [poes_list[i][5] for i in range(len(poes_list))]))
    
    poes_data = dict(zip(poes_sc, poes_list))
    
    ### NEW ALGORITHM ###
    ### """
    ## - Convert RBSP and POES MLT to clock angles
    ## - Interpolate RBSP L, MLT to POES 2-sec time cadence
    ### """
        
    conjunctions = pd.DataFrame(columns = final_cols)
    
    for r in rbsp_sc:
        
        rb_dt = rbsp_data[r][0].copy()
        rb_dt = np.array([rt.replace(tzinfo=None) for rt in rb_dt])
        
        mlt_data = rbsp_data[r][2]
        
        # Convert MagEphem MLT angle to clock angle
        cAngle = np.array([mltToAngle(elem) for elem in mlt_data])
        rbsp_data[r].append(cAngle)
        # Append to data list
        r_mca = rbsp_data[r][-1]
        
        for p in poes_sc:
            
            print((r, p))
                        
            mlt_poes = poes_data[p][2]
            mlat_poes = poes_data[p][3]
            alpha_90_poes = poes_alpha_90_dict[p]
            alpha_0_poes = poes_alpha_0_dict[p]
            
            pcAngle = np.array([mltToAngle(elem) for elem in mlt_poes])
            poes_data[p].append(pcAngle)
            
            psin = np.vectorize(np.sin)(poes_data[p][3] * (np.pi/180.0))
            pcos = np.vectorize(np.cos)(poes_data[p][3] * (np.pi/180.0))
            
            rsin = np.vectorize(np.sin)(r_mca * (np.pi/180.0))
            rcos = np.vectorize(np.cos)(r_mca * (np.pi/180.0))
            
            conjunct = pd.DataFrame(data={'RBSP_SC': r, 
                                          'POES_SC': p,
                                          'POES_DT': poes_data[p][0],
                                          'POES_L': poes_data[p][1],
                                          'POES_MLT': mlt_poes,
                                          'POES_MLAT': mlat_poes,
                                          'POES_ALPHA_90': alpha_90_poes,
                                          'POES_ALPHA_0': alpha_0_poes }).set_index('POES_DT')
                            
            if alpha_method >= 1: # Compute POES equatorial pitch angle
              
                if alpha_method == 1: # B_ratio computed through POES MLAT
                    conjunct['Be_over_Bl'] = (np.cos(conjunct['POES_MLAT'] * (np.pi/180.0)) ** 6) / np.sqrt(1 + 3 * (np.sin(conjunct['POES_MLAT'] * (np.pi/180.0)) ** 2))    
                
                else: # B_ratio obtained from POES MagEphem files
                    poes_beq = poes_data[p][4]
                    poes_bloc = poes_data[p][5]
                    conjunct['Be_over_Bl'] = np.divide(poes_beq, poes_bloc)
                
                input_arc_90 = np.multiply(np.sqrt(conjunct['Be_over_Bl']), np.sin(np.multiply(conjunct['POES_ALPHA_90'], np.pi/180.0)))
                
                alpha_equator_deg = np.arcsin(input_arc_90) * (180.0 / np.pi) # In degrees
                
                greater_idx = np.where(conjunct['POES_ALPHA_90'] > 90.0)[0]
                
                alpha_equator_deg[greater_idx] = 180.0 - alpha_equator_deg[greater_idx]
                conjunct['POES_ALPHA_EQ_90'] = alpha_equator_deg
            
            # RBSP MagEphem DataFrame
            rb_df = pd.DataFrame(data={'RBSP_DT': rb_dt, 
                                       'RBSP_L': rbsp_data[r][1][:, 0],
                                       'RBSP_MLAT': rbsp_data[r][3],
                                       'RBSP_BoverBeq': rbsp_data[r][4],
                                       'RBSP_SIN': np.array([np.sin(x) for x in r_mca*(np.pi/180.0)]),
                                       'RBSP_COS': np.array([np.cos(x) for x in r_mca*(np.pi/180.0)]) }).set_index('RBSP_DT')
            
            # RBSP MagEphem data resampled from 1min to 1 second resolution and reindexed according to POES time resolution
            # Timestamps will sometimes be off with POES by 1 second.
            rb_df = rb_df.resample('1S').interpolate(method='linear')
            rb_df_poes = rb_df.reindex(conjunct.index)
                        
            # Combine both spacecraft data into one DataFrame           
            conjunct['RBSP_L'] = rb_df_poes['RBSP_L']
            conjunct['RBSP_MLAT'] = rb_df_poes['RBSP_MLAT']
            conjunct['RBSP_BoverBeq'] = rb_df_poes['RBSP_BoverBeq']
            
            #-- MLT 
            conjunct['RBSP_PHI'] = np.arctan2(rb_df_poes['RBSP_SIN'], rb_df_poes['RBSP_COS']) * (180.0 / np.pi)
            conjunct.loc[conjunct['RBSP_PHI'] < 0.0, 'RBSP_PHI'] += 360
            conjunct['RBSP_MLT'] = np.array([angleToMLT(i) for i in conjunct['RBSP_PHI'].values])
            
            # Compare L-values
            conjunct['L_DIFF'] = np.absolute(conjunct['RBSP_L'] - conjunct['POES_L'])
            conjunct['L_BOOL'] = conjunct['L_DIFF'] <= l_tol
            
            conjunct['MLT_DIFF'] = np.absolute(conjunct['POES_MLT'] - conjunct['RBSP_MLT'])
            
            # Compare MLT-values
            conjunct['MLT_BOOL'] = conjunct[['POES_MLT', 'RBSP_MLT']].apply(lambda x: mltProximate(*x, mlt_tol), axis=1)
            
            # Find L, MLT conjunctions on POES time cadence
            conjunct['CONJUNCT'] = conjunct['L_BOOL'] & conjunct['MLT_BOOL']
            
            # Find matching MagEIS datetimes
            mageis_conjunct = conjunct[conjunct['CONJUNCT']].copy()
            mageis_conjunct['mageis_pts'] = np.nan
            mageis_conjunct['MAGEIS_L'] = np.nan
            mageis_conjunct['MAGEIS_MLT'] = np.nan
            mageis_conjunct['MAGEIS_MLAT'] = np.nan
            mageis_conjunct['MAGEIS_BoverBeq'] = np.nan
            
            # Set column data type
            mageis_conjunct['mageis_pts'] = mageis_conjunct['mageis_pts'].astype('object')
            mageis_conjunct['MAGEIS_L'] = mageis_conjunct['MAGEIS_L'].astype('object')
            mageis_conjunct['MAGEIS_MLT'] = mageis_conjunct['MAGEIS_MLT'].astype('object')
            mageis_conjunct['MAGEIS_MLAT'] = mageis_conjunct['MAGEIS_MLT'].astype('object')
            mageis_conjunct['MAGEIS_BoverBeq'] = mageis_conjunct['MAGEIS_MLT'].astype('object')
            
            mg_df = pd.DataFrame(data={'MG_DT': mg_data[r][0],
                                       'MAGEIS_L': np.nan,
                                       'MAGEIS_MLT': np.nan,
                                       'MAGEIS_MLAT': np.nan,
                                       'MAGEIS_BoverBeq': np.nan}).set_index('MG_DT')
            
            # Reindex MagEphem data to MagEIS cadence
            rb_df_mageis = rb_df.reindex(mg_df.index)                
            
            rb_df_mageis['RBSP_PHI'] = np.arctan2(rb_df_mageis['RBSP_SIN'], rb_df_mageis['RBSP_COS']) * (180.0 / np.pi)
            rb_df_mageis.loc[rb_df_mageis['RBSP_PHI'] < 0.0, 'RBSP_PHI'] += 360
            rb_df_mageis['RBSP_MLT'] = np.array([angleToMLT(i) for i in rb_df_mageis['RBSP_PHI'].values])
            
            mg_df['MAGEIS_L'] = rb_df_mageis['RBSP_L']
            mg_df['MAGEIS_MLT'] = rb_df_mageis['RBSP_MLT']
            mg_df['MAGEIS_MLAT'] = rb_df_mageis['RBSP_MLAT']
            mg_df['MAGEIS_BoverBeq'] = rb_df_mageis['RBSP_BoverBeq']
                        
            for row in mageis_conjunct.itertuples():
                
                if within:
                    mgidx = np.where(np.absolute(mg_df.index.values - row.Index.to_datetime64()) <= np.timedelta64(time_tol, 's'))[0]
                else:
                    mgidx = np.where(np.absolute(mg_df.index.values - row.Index.to_datetime64()) == np.timedelta64(time_tol, 's'))[0]
            
                if len(mgidx) > 0:
                    mageis_conjunct.at[row.Index, 'mageis_pts'] = mg_df.index[mgidx].values
                    mageis_conjunct.at[row.Index, 'MAGEIS_L'] = mg_df['MAGEIS_L'].iloc[mgidx].values
                    mageis_conjunct.at[row.Index, 'MAGEIS_MLT'] = mg_df['MAGEIS_MLT'].iloc[mgidx].values
                    mageis_conjunct.at[row.Index, 'MAGEIS_MLAT'] = mg_df['MAGEIS_MLAT'].iloc[mgidx].values
                    mageis_conjunct.at[row.Index, 'MAGEIS_BoverBeq'] = mg_df['MAGEIS_BoverBeq'].iloc[mgidx].values
            
            mageis_conjunct = mageis_conjunct[mageis_conjunct['mageis_pts'].notna()].reset_index()
            conjunctions = conjunctions.append(mageis_conjunct[final_cols])        
        
    conjunctions.reset_index(inplace = True, drop=True)
    
    if save: # Save list of conjunctions to CSV file
        if len(savename) == 0:
            savename = times[0] + 'to' + times[1] + '_conjunct.csv'
        elif not savename.endswith('.csv'):
            savename += '.csv'
        conjunctions.to_csv(savename)
        
        # Add file properties to log
        log_name = 'findconjunct_log.csv'
        logdf = pd.DataFrame(data={'FILE_NAME':savename, 
                                   'START_DATE':times[0], 
                                   'END_DATE':times[1],
                                   'NUM_CONJUNCTIONS':len(conjunctions),
                                   'TIME_TOL':time_tol,
                                   'L_TOL':l_tol,
                                   'MLT_TOL':mlt_tol,
                                   'FIELD':field,
                                   'WITHIN':within,
                                   'ALPHA_METHOD':alpha_method,
                                   'L_TYPE':ltype,
                                   'MLT_TYPE':mlt_type}, index = [0])
        with open(log_name, 'a') as f:
            logdf.to_csv(f, header = f.tell()==0)
    
    return conjunctions


def colConversion(strInput, date_type = False):
    
    listOut = strInput.replace('\'', '').replace('[','').replace(']','')
    listOut = listOut.split(' ')
    
    if date_type:
        list_converted = [pd.to_datetime(i) for i in listOut]
    else: # Assumes elements in list are type 'float'
        list_converted = [float(elem) for elem in listOut if len(elem) > 0]
        
    return list_converted


def readConjunct(cfile):
    
    """ 
    Args:
        cfile (str): File path for CSV-format conjunction list. 
        
    Output:
        tread (DataFrame)
    """
    
    col_dtypes = dict(zip(['RBSP_SC', 'POES_SC'], [str, str]))

    tread = pd.read_csv(cfile, sep=',',
                        index_col = 0, header = 0,
                        dtype=col_dtypes, parse_dates = ['POES_DT'])
    
    tread['mageis_pts'] = np.array([colConversion(i, date_type=True) for i in tread['mageis_pts'].values])  
    
    commonfunc = lambda x: x.replace('\'', '').replace('[','').replace(']','')
    
    if 'MAGEIS_L' in tread.columns:
        tread['MAGEIS_L'] = [colConversion(i, date_type=False) for i in tread['MAGEIS_L'].values]
        
    if 'MAGEIS_MLT' in tread.columns:
        tread['MAGEIS_MLT'] = [colConversion(i, date_type=False) for i in tread['MAGEIS_MLT'].values] 
        
    if 'MAGEIS_MLAT' in tread.columns:
        tread['MAGEIS_MLAT'] = [colConversion(i, date_type=False) for i in tread['MAGEIS_MLAT'].values] 
        
    if 'MAGEIS_BoverBeq' in tread.columns:
        tread['MAGEIS_BoverBeq'] = [colConversion(i, date_type=False) for i in tread['MAGEIS_BoverBeq'].values] 
    
    return tread
         


def plotConjunct(df_conj, n_conj,
                 field = 'OP77Q',
                 ltype = 'L',
                 mlt_type = 'EDMAG_MLT',
                 mlat_type = 'EDMAG_MLAT',
                 window = 15):
    """ (OLD, might need updating)
    Kwargs: 
        field: RBSP field model variable
        ltype: RBSP L variable
        mlt_type: RBSP MLT variable
        mlat_type: RBSP MLAT variable
        window: Window to plot magnetic data, 
                in minutes from RBSP conjunction point.
        
    Args:
        df_conj: DataFrame of conjugation points, 
                 also the output of findConjunction method.
        n_conj: nth conjunction point (starting from 0)
        rb_arr: Full time range RBSP data array
        po_arr: Full time range POES data array
    
    """
    register_matplotlib_converters()
    
    dt_left = timedelta(minutes = window)
    dt_right = timedelta(minutes = window)
    
    # rb_point is a series, has:
    #   Array index = 'R_IND'
    #   Probe name  = 'R_SC'
    #   Datetime    = 'R_TIME'
    #   L           = 'R_L'
    #   MLT         = 'R_MLT'
    #   Array of 
     #  POES conjs. = 'POES_Conjunct'
    rb_point = df_conj.iloc[n_conj]
    conj_list = rb_point['POES_Conjunct']

    # Bounds for x-axis (time) of plot
    x_low = rb_point['R_TIME'] - dt_left
    x_hi = rb_point['R_TIME'] + dt_right

    x_low_date = x_low.date().strftime('%Y-%m-%d')
    x_hi_date = x_hi.date().strftime('%Y-%m-%d')
    timerange = [x_low_date, x_hi_date]

    # Set the probe dataset that we're considering
    # Array of quantities:
    # 0th row = datetime
    # 1       = L
    # 2       = MLT
    rb_arr = rs.read_vap_ephem(trange = timerange,
                                  read_vars = [ltype, mlt_type],
                                  probe = rb_point['R_SC'],
                                  b_model = field)[0]

    rb_Dts = np.vectorize(datetime.replace)(rb_arr[0], tzinfo=None) # Datetime array of probe
    rb_Ls = rb_arr[1]  # L array of probe
    rb_Ms = rb_arr[2]  # MLT array of probe
#     rb_As = rb_arr[3] # MLAT array of probe


    ## Find RBSP argument(index) of x boundaries
    rlo_idx = np.where(rb_Dts == x_low)[0]
    if len(rlo_idx) == 0:
        rlo_idx = 0  # First datetime entry
    else:
        rlo_idx = rlo_idx[0]
        
    if window <= 1: 
        rb_hi_limit = x_hi + timedelta(minutes=1)
    else:
        rb_hi_limit = x_hi
    rhi_idx = np.where(rb_Dts == rb_hi_limit)[0]
    if len(rhi_idx) == 0:
        rhi_idx = -1 # Last datetime entry
    else:
        rhi_idx = rhi_idx[0]
        
    # Set up POES data array
    # ASSUMING 'm02' is the only one available currently
    # Addition of new spacecrafts mean findConjunction would have to 
    # change how it outputs POES conjunctions

    po_arr = rs.read_poes(trange = timerange,
                          sat_IDs = ['m02'],
                          read_vars = ['L_IGRF', 'MLT'],
                          filterL = True)[0]
    po_Dts = po_arr[0]
    po_Dts_trunc = np.vectorize(datetime.replace)(po_arr[0], second=0)
    po_Ls = po_arr[1]
    po_Ms = po_arr[2]  
#     po_As = po_arr[3]  # MLAT

    # Find POES argument(index) of x boundaries
    plo_idx = np.where(po_Dts_trunc == x_low)[0]
    if len(plo_idx) == 0:
        plo_idx = 0
    else:
        plo_idx = plo_idx[0]
        
    phi_idx = np.where(po_Dts_trunc == x_hi)[0]
    if len(phi_idx) == 0:
        phi_idx = -1
    else:
        phi_idx = phi_idx[0]
    
    
    # Plot both L, MLT values over trange
    fig = plt.figure(figsize=(10,5), dpi=250)
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2, sharex = ax1)

    ax1.plot(rb_Dts[rlo_idx:rhi_idx], rb_Ls[rlo_idx:rhi_idx, 0], '--', color='darksalmon') 
    ax1.plot(po_Dts[plo_idx:phi_idx], po_Ls[plo_idx:phi_idx], color='cornflowerblue')
    ax1.set_ylabel('(McIlwain) L')
    ax1.legend([rb_point['R_SC'], 'm02'], loc='best')
    
    # Conjunction lines
    # conj_list[n][2] accesses datetime info for 
    ax1.axvline(rb_point['R_TIME'], linestyle='--', color='indianred', linewidth=0.40)
    ax1.axvline(conj_list[0][2],color='cornflowerblue', linewidth=0.40)
    ax1.axvline(conj_list[-1][2],color='cornflowerblue', linewidth=0.40)
    
    # MLT                            
    ax2.plot(rb_Dts[rlo_idx:rhi_idx], rb_Ms[rlo_idx:rhi_idx], '--', color='indianred') 
    ax2.plot(po_Dts[plo_idx:phi_idx], po_Ms[plo_idx:phi_idx], color='cornflowerblue')
    ax2.set_ylabel('MLT')
    ax2.legend([rb_point['R_SC'], 'm02'], loc='best')
    
    ax2.axvline(rb_point['R_TIME'], linestyle='--', color='indianred',linewidth=0.40)
    ax2.axvline(conj_list[0][2],color='cornflowerblue', linewidth=0.40)
    ax2.axvline(conj_list[-1][2],color='cornflowerblue', linewidth=0.40)
                          
    return rb_point


### GET AND COMPARE FLUX ###
def compare_all_flux(dfConjunct, l_min = 3.0, 
                     low_pitch_avg=True, poes_telescope='tel90', 
                     savename='', save=False):
    """
    Kwargs:
        l_min (float):
        low_pitch_avg (bool):
        poes_telescope (str, {'tel0', 'tel90'}:
        savename (str):
        save (bool): 
        
    Args:
        dfConjunct (DataFrame):
        
    Output:
        df_flux (DataFrame): 
    """
    assert poes_telescope in ['tel0', 'tel90'], "Only {'tel0', 'tel90'} are valid values for poes_telescope."
    
    # L minimum limit
    df_reduced = dfConjunct[ dfConjunct['POES_L'] > l_min ]
    
    if len(df_reduced) == 0:
        print('Error: No conjunction points with `POES_L` < ', l_min)
        return
    
    df_reduced = df_reduced.reset_index(drop=True)
        
    if type(df_reduced.at[0, 'mageis_pts']) in [np.dtype('<M8[ns]'), np.datetime64]:
        # Convert list of str datetimes to actual pd.datetime objects
        to_pd_dt = lambda x: pd.to_datetime(x)
        df_reduced['mageis_pts'] = df_reduced['mageis_pts'].apply(to_pd_dt)
    
    if low_pitch_avg:
        mepAlphas = None
        mageisB_Beq = None
    else:
        if poes_telescope == 'tel90':
            assert 'POES_ALPHA_EQ_90' in df_reduced.columns, "{'POES_ALPHA_90', 'POES_ALPHA_EQ_90'} are required as columns within dfConjunct if poes_telescope = 'tel90' and low_pitch_avg = False."
            mepAlphas = df_reduced[['POES_ALPHA_90', 'POES_ALPHA_EQ_90']].values
        else:
            assert 'POES_ALPHA_EQ_0' in df_reduced.columns, "{'POES_ALPHA_0', 'POES_ALPHA_EQ_0'} are required as columns within dfConjunct if poes_telescope = 'tel0' and low_pitch_avg = False."
            mepAlphas = df_reduced[['POES_ALPHA_0', 'POES_ALPHA_EQ_0']].values
                
        mageisB_Beq = df_reduced['MAGEIS_BoverBeq'].values
    
    zipped_params = zip(df_reduced['mageis_pts'].values, 
                        df_reduced['POES_DT'].values,
                        df_reduced['RBSP_SC'].values,
                        mepAlphas,
                        mageisB_Beq,
                        df_reduced['POES_SC'].values
                        )
    
    agg = [compare_flux(mageis_conj_list = i, 
                        poes_time = j,
                        rbsc = k,
                        meped_alphas = l,
                        mageis_BoverBeq = r,
                        poes_sc = s,
                        lowPitchAngleAvg = low_pitch_avg,
                        poes_tel=poes_telescope) for i, j, k, l, r, s in zipped_params]
    
            
    t = np.concatenate([a[0] for a in agg], axis=0)
    eq_alphas = np.concatenate([a[1] for a in agg], axis=0)
    e1 = np.concatenate([a[2] for a in agg], axis=0)
    e2 = np.concatenate([a[3] for a in agg], axis=0)
    e3 = np.concatenate([a[4] for a in agg], axis=0)
    e4 = np.concatenate([a[5] for a in agg], axis=0)
    
    poes_l = df_reduced.apply(lambda x: np.repeat(x['POES_L'], 
                                                  repeats=len(x['mageis_pts'])), axis=1)
    poes_mlt = df_reduced.apply(lambda x: np.repeat(x['POES_L'], 
                                                  repeats=len(x['mageis_pts'])), axis=1)
    
    
    pdt_idx = [np.where(df_reduced['POES_DT'] == pdt)[0][0] for pdt in t[:,0]]
        
    # Get L, MLT MagEIS values
    mageis_l = [i for list_item in df_reduced['MAGEIS_L'] for i in list_item]
    mageis_mlt = [i for list_item in df_reduced['MAGEIS_MLT'] for i in list_item]
    
    df_flux = pd.DataFrame()
    df_flux['POES_DT'] = t[:, 0]
    df_flux['MAGEIS_DT'] = t[:, 1]
    df_flux['MEPED_ALPHA_EQ'] = eq_alphas[:, 0]
    df_flux['MAGEIS_ALPHA_EQ'] = eq_alphas[:, 1]
    df_flux['POES_L'] = df_reduced.iloc[pdt_idx]['POES_L'].values 
    df_flux['POES_MLT'] = df_reduced.iloc[pdt_idx]['POES_MLT'].values
    df_flux['MAGEIS_L'] = mageis_l
    df_flux['MAGEIS_MLT'] = mageis_mlt
    df_flux['POES_e1'] = e1[:, 0] 
    df_flux['POES_e2'] = e2[:, 0]                            
    df_flux['POES_e3'] = e3[:, 0]
    df_flux['POES_e4'] = e4[:, 0]                            
    df_flux['MAGEIS_e1'] = e1[:, 1] 
    df_flux['MAGEIS_e2'] = e2[:, 1]                            
    df_flux['MAGEIS_e3'] = e3[:, 1]
    df_flux['MAGEIS_e4'] = e4[:, 1]
    
    if save:
        if len(savename) == 0:
            rand_num = np.random.randint(0, 1000)
            savename = 'flux_' + str(randnum) + '_' + poes_telescope + '.csv'
        elif not savename.endswith('.csv'):
            savename += '.csv'
        df_flux.to_csv(savename)
                                
    return df_flux


def compare_flux(mageis_conj_list, poes_time, rbsc, meped_alphas = None, mageis_BoverBeq = None, 
                 poes_sc='m02', lowPitchAngleAvg = True, poes_tel='tel90'):    
    """
    Takes get_both_flux output, and implements one more processing step 
    to computing MagEIS e-binned values.
    
    Kwargs:
        poes_sc (str, {'m02', 'm01', etc.}): POES spacecraft. Default 'm02' (metop02).
        lowPitchAngleAvg (bool): If True, will output MagEIS flux value averaged over the two lowest pitch angles. 
                                 If False, will calculate MagEIS equatorial pitch-angle values from mageis_BoverBeq
                                 and use the closest match to POES eq. alpha.
        poesTel (str, {'tel0', 'tel90'}): POES telescope indicator, used for obtaining flux values.
        meped_alphas (list-like): Array of MEPED pitch angle values. This takes in a 2-element row vector or list where the first element corresponds to 
                                  the local value, and the second column corresponds to the computed equatorial value. 
                                  Default None if lowPitchAngleAvg = True, otherwise, must pass in a valid array or list. 
                                  Use columns [['POES_ALPHA_xx', 'POES_ALPHA_EQ_xx']] from findConjunct() DataFrame output.
        mageis_BoverBeq (list-like, or None): List or array of Blocal / Beq values at each conjunction point (an element of 
                                              column 'MAGEIS_BoverBeq' from findConjunct()). 
                                              Default None if lowPitchAngleAvg=True. Otherwise, must pass in a valid list or array. 
                                              

    Args:
        mageis_conj_list (list-like): List of MagEIS conjunction points (an element of column 'mageis_pts' from 
                                                       findConjunct() output).
        poes_time (list-like): POES conjunction point (an element of column 'POES_DT' from findConjunct()
                                                output).
        rbsc (str): Van Allen probe, 'rbspa' or 'rbspb', (an element of column 'RBSP_SC' from findConjunct()).
        
                 
    Output: 
        t: 
        eq_alphas:  where N is the number of associated MagEIS conjunction points for the particular POES timestamp.
                    If 'lowPitchAngleAvg' is False, 
                    If 'lowPitchAngleAvg' is True,
        e1:
        e2:
        e3:
        e4:

    """
    
    if not lowPitchAngleAvg:
        assert meped_alphas is not None and mageis_BoverBeq is not None, "If lowPitchAngleAvg = False, then valid values must be passed in for meped_alphas and mageis_BoverBeq."
    
    t, pbin, mbin_full, mageis_alphas = get_both_flux(mageis_conj_list, 
                                                      poes_dt=poes_time, 
                                                      rb_sc=rbsc,
                                                      p_sc=poes_sc,
                                                      poesTel = poes_tel) 
    n_points = len(mageis_conj_list)
    assert n_points == mbin_full.shape[0], "Check n_points."
    
    # Debug
    assert n_points < 3, "Check n_points = len(mageis_conj_list)"
    
    # One more processing step for MagEIS flux: average over certain pitch angles
    if lowPitchAngleAvg: 
        mbin = []
        
        
        for n in range(mbin_full.shape[0]):
            mbin_n = [np.nanmean(mbin_full[n, [0, -1], col]) for col in range(4)]
            mbin.append(mbin_n)
        
        mbin = np.array(mbin)
        
        poes_alphas_eq = np.repeat(meped_alphas[1], repeat = n_points, axis = 0)
        averaged_m_alpha = np.nanmean([mageis_alphas[0, 0], 180.0 - mageis_alphas[0, -1]])
        mag_alphas_avg = np.repeat(averaged_m_alpha, repeat = n_points, axis = 0)
        
        eq_alphas = np.concatenate([poes_alphas_eq, mag_alphas_avg], axis = 1)
        
    else:
        # get_both_flux outputs MagEIS local pitch-angle bins (last output)
        
        # Create array of MEPED alphas. Second column (:,1) holds 'POES_ALPHA_EQ_x'
        poes_alphas = np.array(meped_alphas).reshape(-1, 2)
        if n_points > 1:
            poes_alphas = np.repeat(poes_alphas, repeats = n_points, axis = 0)
        
        # (N, 11) array of MagEIS pitch angle bins (equatorial values)
        mageis_alpha_eq = np.zeros(shape = (n_points, 11))
        
        # Beq/Blocal
        Be_over_Bl = [1.0/n for n in mageis_BoverBeq]
        
        # Convert to equatorial values
            # - alpha_method = 2
            
        r1 = mageis_alphas * (np.pi/180.0)
        for i in range(n_points):
            r2 = np.sqrt(Be_over_Bl[i]) * np.sin(r1)
            r3 = np.arcsin(r2) * (180.0 / np.pi) # Convert back to degrees
            r3 = r3.reshape(1, 11)
            mageis_alpha_eq[i, :] = r3
                            
        #-- Find closest matching MagEIS pitch-angle bin.
        pEq = poes_alphas[0, 1]
        alpha_diff = np.abs(mageis_alpha_eq[0, :] - pEq)
        argMinEq = np.argmin(alpha_diff)
        
        print(mbin_full) # Debug
        
        mbin = mbin_full[:, argMinEq, :]
        assert mbin.shape == (n_points, 4), "Fix mbin shape." # Debug
        
        print('mbin_full[0, :, :]:\n')  # Debug
        print(mbin_full[0, :, :]) # Debug
                
        eq_alphas = np.concatenate([poes_alphas[:, 1].reshape(-1, 1), mageis_alpha_eq[:, argMinEq].reshape(-1,1)], axis=1)
        
        #-- Map those equatorial values to the corresponding binned data.
            # -- Get pitch angle index of closest matching MagEIS eq value to POES eq value
            # -- Get mbin_full[idx, :], where the first dim. corr. to pitch angles, the second to energy bins.
    
    e1 = np.concatenate([pbin[:, 0].reshape(-1,1), mbin[:,0].reshape(-1,1)], axis=1)   
    e2 = np.concatenate([pbin[:, 1].reshape(-1,1), mbin[:,1].reshape(-1,1)], axis=1) 
    e3 = np.concatenate([pbin[:, 2].reshape(-1,1), mbin[:,2].reshape(-1,1)], axis=1)   
    e4 = np.concatenate([pbin[:, 3].reshape(-1,1), mbin[:,3].reshape(-1,1)], axis=1)
        
    return t, eq_alphas, e1, e2, e3, e4


def get_both_flux(dt_list, poes_dt, rb_sc, p_sc='m02', 
                  poesTel='tel90'):
    """
    Returns arrays of both MagEIS energy-binned flux and POES energy-binned
    flux, with respect to POES energy channels. 

    Kwargs:
        p_sc (str, {'m02', 'm01', etc.}): POES spacecraft. 
        lowPitchAngleAvg (bool): If True, will output MagEIS flux value averaged over the two lowest pitch angles. 
        poesTel (str, {'tel0', 'tel90'}): POES telescope indicator, used for obtaining flux values.

    Args:
        dt_list (list-like): List of MagEIS conjunction points (column 'mageis_pts' from 
                             findConjunction() output).
        poes_dt (Pandas Timestamp or datetime): POES conjunction point (column 'POES_DT' 
                                                from findConjunction() output).
        rb_sc (str): Van Allen probe, 'rbspa' or 'rbspb', (column 'RBSP_SC' from 
                     findConjunction()).
                 
    Output: 
        time_array (2d-array): (N, 2) array of timestamp values associated with output flux measurements, where 
                                N (as for the other variables) is the number of associated MagEIS conjunction points 
                                for the particular POES timestamp.
                                Column 1 corresponds to POES timestamps, column 2 corresponds to MagEIS 
                                timestamps.
        poes_binned (2d-array): (N, 4) array of energy-binned POES flux values for selected telescope.
        mCumulative (3d-array): (N, 11, 4) array of energy-integrated (wrt. POES' MEPED 4-energy bins)
                                 MagEIS electron flux, binned additionally by pitch-angle (11 bins). 
        mageis_fedu_alpha (1d-array):  Array of 11 local pitch-angle values for MagEIS
                                       FEDU, according to time range specified in time_array.
        
    """
    
    assert poesTel in ['tel0', 'tel90'], "Only `tel0` or `tel90` are valid inputs to poesTel."

    # VAR INITS
    mCumulative = [] # Cumulative MagEIS POES-energy-binned flux,
                     # shape: (N points, 11, 4)
    N_points = len(dt_list) # Num of MagEIS conjunction points, for full conjunction list

    # POES VAR DEFAULTS
    meped_fill = 9.969209968386869e+36 # NULL fill value
    tel0_labels = ['mep_ele_tel0_flux_e1', 
                   'mep_ele_tel0_flux_e2',
                   'mep_ele_tel0_flux_e3',
                   'mep_ele_tel0_flux_e4']
    tel90_labels = ['mep_ele_tel90_flux_e1', 
                   'mep_ele_tel90_flux_e2',
                   'mep_ele_tel90_flux_e3',
                   'mep_ele_tel90_flux_e4']
#     pitch_labels = ['meped_alpha_0_sat', 'meped_alpha_90_sat']
    full_labels = tel0_labels + tel90_labels
    poes_fill = [meped_fill for i in range(len(full_labels))]
    
    dtime = pd.to_datetime(poes_dt, format='%Y-%m-%d')
    dt_date = dtime.strftime('%Y-%m-%d')

    ###--- READ POES ---###
    poes_data = rs.read_poes(trange = [dt_date, dt_date], 
                             sat_IDs = [p_sc], 
                             read_vars = full_labels, 
                             null_vals = poes_fill, 
                             filterL = True)[0]
    matching_pidx = np.where(poes_data[0] == dtime)[0][0]
    pfluxes = [f[matching_pidx] for f in poes_data[1:]]

    # Return a dictionary with meped flux across tel0 and tel90,
    # and all energy bins, and with datetime, tel0 and tel90 pitch angle information
    poes_flux = dict(zip(['POES_DT'] + full_labels, [poes_dt] + pfluxes))

    if poesTel == 'tel90':
        poes_binned = np.array([poes_flux[f] for f in tel90_labels]).reshape(-1, 4)
#         meped_sat_alpha = np.array([poes_flux[pitch_labels[1]]]).reshape(-1, 1)
    elif poesTel == 'tel0':
        poes_binned = np.array([poes_flux[f] for f in tel0_labels]).reshape(-1, 4)
#         meped_sat_alpha = np.array([poes_flux[pitch_labels[0]]]).reshape(-1, 1)

    # Returns ... 
    poes_binned = np.repeat(poes_binned, repeats = N_points,
                            axis=0)
#     meped_sat_alpha = np.repeat(meped_sat_alpha, repeats = N_points, 
#                                 axis=0)

    ## MAGEIS VAR DEFAULTS
    mageis_fill = -9.9e+30 # MagEIS FEDU fill values
    magNumEbins = 25 # Number of energy bins for MagEIS 'FEDU'
    alpha_dim_size = 11
    
    ## e1: >30 keV
    ## e2: >100 keV
    ## e3: >300 keV
    ## e4: >900 keV
    e_bounds = {'e1': 30, 'e2': 100, 'e3': 300, 'e4': 900} # keV 
    
    mageis_dt_list = [pd.to_datetime(i).strftime('%Y-%m-%d') if type(i) == np.datetime64 
                      else i.strftime('%Y-%m-%d') for i in dt_list]
    
    if len(mageis_dt_list) == 1: 
        mageis_dt_list.append(mageis_dt_list[0])
    
    ###--- READ MAGEIS ---### 
    mg_labels = ['FEDU', 'FEDU_Energy', 'FEDU_Alpha']
    mageis_data = rs.read_mageis(trange = mageis_dt_list, 
                                 read_vars = mg_labels,
                                 probe = rb_sc)[0]
    mgdt = mageis_data[0]
    fedu = mageis_data[1]
    fedu_en = mageis_data[2]
    alphas = mageis_data[3]
    
    mageis_fedu_alpha = alphas # (OUTPUT)
    
    # For each MagEIS conjunction point
    for t in range(N_points):
        m_t = dt_list[t]
        try:
            i = np.where(mgdt == pd.to_datetime(m_t))[0][0]
        except Exception as e:
            print('mgdt', mgdt[0])
            print('m_t', m_t)
            print(pd.to_datetime(m_t))
            raise e
        en = fedu_en.copy()
        
        if en.ndim >= 2:
            try:
                en = en[i, :] # Flatten to a one-dimensional array
            except:
                print('FEDU energy shape:', fedu_en.shape)
                raise
            
        pos_ener = np.where(en > 0.0)[0]
        valid_ener = en[pos_ener]
        
        e1_idx = np.where(valid_ener > e_bounds['e1'])[0]
        e2_idx = np.where(valid_ener > e_bounds['e2'])[0]
        e3_idx = np.where(valid_ener > e_bounds['e3'])[0]
        e4_idx = np.where(valid_ener > e_bounds['e4'])[0]
        
        # x
        x_integ_1 = [valid_ener[e1_idx]] * alpha_dim_size
        x_integ_2 = [valid_ener[e2_idx]] * alpha_dim_size
        x_integ_3 = [valid_ener[e3_idx]] * alpha_dim_size        
        x_integ_4 = [valid_ener[e4_idx]] * alpha_dim_size
                
        # y 
        fedu_pt = fedu[i, :, :]
        fedu_pt = fedu_pt[:, pos_ener]

        fedu1 = fedu_pt[:, e1_idx]
        fedu2 = fedu_pt[:, e2_idx]
        fedu3 = fedu_pt[:, e3_idx]
        fedu4 = fedu_pt[:, e4_idx]

        nan_idx = list(zip(*np.where(fedu_pt < 0.0))) # Invalid points = -1e+31 and/or < 0
        
        ## AVERAGE OVER ENERGY, ACCORDING TO POES ENER. BINS
        # Update domain of integration for each pitch angle
        # (Implicitly for each MagEIS conjunction point).             
        for a_pt, e_pt in nan_idx:
            e_val = valid_ener[e_pt] # Energy value to remove from x
            
#             print('Energy val to delete: ', e_val)
                        
            if e_val in x_integ_1[a_pt]:
                new_e_pt = np.where(x_integ_1[a_pt] == e_val)[0]
                x_integ_1[a_pt] = np.delete(x_integ_1[a_pt],
                                            obj = new_e_pt)

            if e_val in x_integ_2[a_pt]:
                new_e_pt = np.where(x_integ_2[a_pt] == e_val)[0]
                x_integ_2[a_pt] = np.delete(x_integ_2[a_pt],
                                            obj = new_e_pt)
                
            if e_val in x_integ_3[a_pt]:
                new_e_pt = np.where(x_integ_3[a_pt] == e_val)[0]
                x_integ_3[a_pt] = np.delete(x_integ_3[a_pt],
                                            obj = new_e_pt)
                
            if e_val in x_integ_4[a_pt]:
                new_e_pt = np.where(x_integ_4[a_pt] == e_val)[0]
                x_integ_4[a_pt] = np.delete(x_integ_4[a_pt],
                                            obj = new_e_pt)
        
        Y1 = [x[x >= 0.0] for x in fedu1]
        Y2 = [x[x >= 0.0] for x in fedu2]
        Y3 = [x[x >= 0.0] for x in fedu3]
        Y4 = [x[x >= 0.0] for x in fedu4]
        
        # Zip each row of df1 and each row of x_e1
        # zip: y, x
        zip1 = list(zip(Y1, x_integ_1))
        zip2 = list(zip(Y2, x_integ_2))
        zip3 = list(zip(Y3, x_integ_3))
        zip4 = list(zip(Y4, x_integ_4))
        
        # Apply function trapz using map
        avg1 = np.array([trapz(f, x_int) for f, x_int in zip1]).reshape((-1,1))
        avg2 = np.array([trapz(f, x_int) for f, x_int in zip2]).reshape((-1,1))
        avg3 = np.array([trapz(f, x_int) for f, x_int in zip3]).reshape((-1,1))
        avg4 = np.array([trapz(f, x_int) for f, x_int in zip4]).reshape((-1,1))
        
        for row in range(11): # Iterate through all pitch angles
            # If flux is unavailable for a certain pitch angle across all 
            # energies, then mark its averaged value as NULL.
            if np.all(fedu1[row,:] < 0.0):
                avg1[row] = np.nan
            if np.all(fedu2[row,:] < 0.0):
                avg2[row] = np.nan
            if np.all(fedu3[row,:] < 0.0):
                avg3[row] = np.nan
            if np.all(fedu4[row,:] < 0.0):
                avg4[row] = np.nan

        averagedE = np.concatenate([avg1, avg2, avg3, avg4], axis=1)
        mCumulative.append(averagedE) 
        
    mCumulative = np.array(mCumulative)
    lenList = len(mCumulative)
    
    if lenList > 11 and lenList % 11 == 0:
        mCumulative = np.array(np.split(mCumulative, int(lenList / 11), axis = 0))
        
    # TIME ARRAY
    p_dts = np.repeat(np.array([pd.to_datetime(poes_dt)]), repeats = N_points).reshape(-1,1)
    m_dts = np.array(dt_list).reshape(-1, 1)
    time_arr = np.concatenate([p_dts, m_dts], axis=1)
    
    return time_arr, poes_binned, mCumulative, mageis_fedu_alpha


##-- PLOT --##

def plotConjunctFlux(df_flux, sci_limit = (5,5), plot_lin=True, savename='', save=False):
    """ Plots flux comparisons between POES and MagEIS for each energy bin
        {e1, e2, e3, e4}, corresponding to a 2 x 2 grid. 
    Kwargs:
        sci_limit:
        plot_lin: 
        savename:
        save:
        
    Args:
        df_flux: 
    """
    
    if save and len(savename) == 0:
        now = datetime.now()
        savename = now.strftime('%d-%m-%Y_%H%M%S')
    
    e1zip = np.dstack((df_flux['POES_e1'].values, df_flux['MAGEIS_e1'].values))[0]
    e2zip = np.dstack((df_flux['POES_e2'].values, df_flux['MAGEIS_e2'].values))[0]
    e3zip = np.dstack((df_flux['POES_e3'].values, df_flux['MAGEIS_e3'].values))[0]
    e4zip = np.dstack((df_flux['POES_e4'].values, df_flux['MAGEIS_e4'].values))[0]
    
    Ebinned = [e1zip, e2zip, e3zip, e4zip]
    
    fig = plt.figure(figsize=(12,8))
    fig.suptitle('$log$ scale')
    ax = fig.subplots(2, 2)
    axes = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]
    xlabels = ['>30keV $J_{MagEIS}$', 
              '>100keV $J_{MagEIS}$',
              '>300keV $J_{MagEIS}$',
              '>900keV $J_{MagEIS}$']
    
    ylabels = ['>30keV $J_{POES}$', 
              '>100keV $J_{POES}$',
              '>300keV $J_{POES}$',
              '>900keV $J_{POES}$']

    xlin = [] # y = x line
    x_ax = [] # Jmageis, for each energy bin
    y_ax = [] # Jpoes,   for each energy bin

    for arr in Ebinned:
        
        with np.errstate(divide='ignore'):
            res = np.log10(arr)
            
        res[np.isneginf(res)] = 0.0
        
        xmin = min(np.nanmin(res[:, 1]), np.nanmin(res[:, 0]))
        xmax = max(np.nanmax(res[:, 1]), np.nanmax(res[:, 0]))
        if plot_lin:
            xlin.append(np.linspace(xmin, xmax))
        x_ax.append(res[:, 1]) # Plot log10(flux)
        y_ax.append(res[:, 0])


    for i in range(4):
        axes[i].ticklabel_format(style='sci', axis='both', scilimits=sci_limit)
        axes[i].set_xlabel(xlabels[i], fontsize=13)
        axes[i].set_ylabel(ylabels[i], fontsize=13)
        if plot_lin:
            axes[i].plot(xlin[i], xlin[i], '--',linewidth = 0.8, color='grey')
        axes[i].plot(x_ax[i], y_ax[i], 'o', markersize=2.0)

    plt.tight_layout(pad=3.0)
    
    if save:
        if not savename.endswith('.png'):
            savename += '.png'
        plt.savefig(savename, dpi=600)

    plt.show()
