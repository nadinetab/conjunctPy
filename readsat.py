"""
Nadine Tabucol

- Read and plot Magnetic Ephemeric data from Van Allen Probes and 
POES data. 
- Find magnetic conjunctions between the two satellites (in prog.)

Last updated: 
30 May 20
- Added sorting functionality to load_rbsp() to ensure 
retrieval of latest file.

07 May 20
- Removed call to 'poes_datetime' within 'read_poes'
- For loops --> list comprehension within 'read_poes'
- np.vectorize --> list comprehension within 'iso_parser'

"""

import numpy as np
import pandas as pd
import h5py
import os
from glob import glob
import dateutil.parser
import matplotlib.pyplot as plt
import netCDF4 as ncdf
import cdflib
from datetime import datetime, timedelta
from pandas.plotting import register_matplotlib_converters
from dateutil.tz import tzutc

#### --- RBSP --- ####

## Preliminaries ##
def getAttrs(varName):
    """ (MagEphem)
    Get attributes of an H5File as a dictionary.
    """
    
    dtDay = datetime.strptime('2017-12-28', '%Y-%m-%d')
    testFile = load_rbsp([dtDay, dtDay], 
                         setprobe = 'rbspa',
                         bmodel = 'OP77Q',
                         dataset = 'MagEphem')[0]
    
    h5file = h5py.File(testFile, 'r')
    
    vardict = dict(h5file[varName].attrs.items())
    
    h5file.close()
    
    return vardict
    
def getKeys(testFile, frame = True):
    """ (MagEphem)
    Get the keys of an HDF5 file and return a dictionary
    with corresponding attributes: (units, fill value, description.)
    """
    
    h5file = h5py.File(testFile, 'r')
    
    list_keys = list(h5file.keys())
    list_desc = []
    list_null = []
    list_units = []
    list_depend = []
    
    for k in list_keys:
        
        desc = h5file[k].attrs.get('DESCRIPTION')
        null = h5file[k].attrs.get('FILLVAL')
        units = h5file[k].attrs.get('UNITS')
        
        if desc is not None:
            list_desc.append(desc[0].decode("utf-8"))
        else:
            list_desc.append(desc)
            
        if null is not None:
            list_null.append(null[0].decode("utf-8"))
        else:
            list_null.append(null)
            
        if units is not None:
            list_units.append(units[0].decode("utf-8"))
        else:
            list_units.append(units)
        
    
    h5file.close()
    
    attrs_zip = list(zip(list_units, list_null, list_desc))
    zipped = dict(zip(list_keys, attrs_zip))
    
    if frame:
    
        df = pd.DataFrame.from_dict(zipped, orient='index',
                               columns = ['UNITS', 'FILLVALUE', 'DESC'])
        with pd.option_context('display.max_colwidth', -1) and pd.option_context('display.max_rows', None):
                return df
    else:
        return zipped
    
    
def peek(day = '2017-12-28', probe_type = 'rbspa',
         field_model = 'TS04D', dataframe = True):
    """ (MagEphem)
    Returns a dictionary or DataFrame of the keys of the HDF5 file dataset, and some 
    corresponding attributes: (units, fill value, description)
    """
    
    dtDay = datetime.strptime(day, '%Y-%m-%d')
    
    h5list = load_rbsp([dtDay, dtDay], setprobe = probe_type, bmodel = field_model)
    
    print('Variable: ', '(Units, Fill Val, Description)\n')
    return getKeys(h5list[0], dataframe)
        
    
# More general plotting function
def plotvars(list_vars, var_labels, time_var, trange, 
             plot_size=(15, 7), res=600):
    """ 
    
    Kwargs:
        plot_size: Size of the entire plot. Formatted as a tuple, default
                    is (15, 7).
        res: Set the dpi/resolution of the plot. Default is 600.
        
    
    Args:
        list_vars: List of arrays of variables to plot. 
        var_labels: List of the variable names to plot. Assumes order matches
                that of list_vars.
        time_var: Time array for which to plot values. 
        trange: Range for which to plot 'list_vars' across. Format as 
                a tuple or list of strings 'YYYY-mm-dd hh:mm'. 
                If empty, plots 'list_vars' across the entire range 
                of `time_var`.
        
    Returns:
        None. Prints plot of specified variables across trange.
    """
    
    time_truncated = time_var.copy()
    time_truncated = np.vectorize(datetime.replace)(time_truncated, second = 0)
    
    N = len(list_vars) # Num. of variables to plot
    register_matplotlib_converters()
    
    fig = plt.figure(figsize=plot_size, dpi = res, tight_layout = True)

    s_idx = 0
    e_idx = time_truncated.shape[0]
    

    if len(trange) > 0:
        sdate = datetime.strptime(trange[0], '%Y-%m-%d %H:%M')
        edate = datetime.strptime(trange[1], '%Y-%m-%d %H:%M')
        
        if time_var[0].tzinfo is None: # POES 
#            print('Tzinfo none')
            start = datetime(sdate.year, sdate.month, sdate.day,
                         sdate.hour, sdate.minute)
            end = datetime(edate.year, edate.month, edate.day,
                         edate.hour, edate.minute)
                        
            try: 
                s_idx = np.where( time_truncated == start )[0][0]
                print(s_idx)
            except:
                print(s_idx)
                return
            try:
                e_idx = np.where( time_truncated == end )[0][0]
                print(e_idx)
            except: 
                print(e_idx)
                return
            
        else: # RBSP 
            start = datetime(sdate.year, sdate.month, sdate.day,
                             sdate.hour, sdate.minute, tzinfo=tzutc())
            end = datetime(edate.year, edate.month, edate.day,
                             edate.hour, edate.minute, tzinfo=tzutc())
            
            s_idx = np.where( time_truncated == start )[0][0]
            e_idx = np.where( time_truncated == end )[0][0]
                
            
    for i in range(N):
        ax = fig.add_subplot(N, 1, i + 1)
        
        # Set plot labels
        ax.set_xlabel("Time")
        ax.set_ylabel(var_labels[i])
        
        ax.plot(time_truncated[s_idx:e_idx], list_vars[i][s_idx:e_idx])        
        
        if list_vars[i].ndim > 1:
            alpha_range = np.arange(90, 0, step=-5).tolist()
            ax.legend(alpha_range, loc='best')
            
    return s_idx, e_idx

        
## LOADING ##
def load_rbsp(dtrange, setprobe, bmodel='OP77Q', dataset='MagEphem', 
              most_recent = True, local=False):
    """
    Returns a list of RBSP MagEphem definitive or MagEIS level 3 pitch angle 
    data for one date range within the same year.
    Grabs latest version of each file, by default. No implementation
    yet for selecting a specific version for specific date(s).
    
    Kwargs: 
        bmodel: Type of magnetic field model, takes in a string,
                either "OP77Q" or "TS04D". Necessary when reading MagEphem def files.
        dataset: 'MagEphem' or 'mageis'. Not case-sensitive. 
        most_recent: Grabs the latest version of each file. If 'False', 
                    grabs version 1.0.0. Set to 'True' by default.
        local: Only for debugging on local computer. Sets file paths 
                wrt local files.
    
    Args:
        dtrange: A time range (a two-element list/array of datetime.datetime format)
        setprobe: Takes in either 'rbspa' or 'rbspb', specifies RBSP probe.
                  Is a lowercase string.
    """
    dataset = dataset.lower()

    if dataset == 'magephem':
        dataset = 'MagEphem'

    if dataset not in ['mageis', 'MagEphem']:
        print('Only `MagEIS` and `definitive MagEphem` data are implemented.')
        return
    
    if type(dtrange) not in [list, np.ndarray, tuple]:
        print("Not a valid 'trange' type. Input an array, list, tuple \
               of elements of type datetime.datetime.")
        return
    
    fp_list = []
    
    # ECT data products path
    ect_path = '/data1/big/SATELLITE/rbsp/www.rbsp-ect.lanl.gov/data_pub/' + setprobe + '/'
    
    ####### LOCAL ####### 
    # DEBUGGING purposes
    if local:
        ect_path = '/Users/frootloops/Downloads/' + setprobe + '/'
    
    Year = dtrange[0].year
        
    if dataset == 'MagEphem':
        file_path = ect_path + dataset + '/definitive/' + str(Year) + '/' 
    else:     
        file_path = ect_path + dataset + '/level3/pitchangle/' + str(Year) + '/'
        
        ####### LOCAL ####### 
        if local:
            file_path = ect_path + dataset + '/' + str(Year) + '/'

    start_date = dtrange[0].date()
    end_date = dtrange[1].date()
    d = start_date
    
    while d <= end_date:
        
        datestring = d.strftime('%Y%m%d')
    
        if dataset == 'MagEphem':
            if bmodel == 'OP77Q':
                file_list = glob(file_path+'*OP77Q*'+ datestring +'*.h5')

            elif bmodel == 'TS04D':
                file_list = glob(file_path+'*TS04D*'+ datestring +'*.h5')
            else:
                raise Exception(bmodel + " is not a valid magnetic field model. Input either \
                                 'OP77Q' or 'TS04D'.")
        else: # dataset = 'mageis'
            file_list = glob(file_path + '*' + datestring + '*.cdf')
            
        # Retrieve latest version
        if len(file_list) > 1 and most_recent:
            # Default: append latest version
            file_list.sort()
            fp_list.append(file_list[-1])
            
        elif len(file_list) == 1 or not most_recent:
            fp_list.append(file_list[0])
            
        d += timedelta(days = 1)
            
    return fp_list


def read_mageis(trange, read_vars, probe, get_latest = True):
    """ Read MagEIS 
    
    Kwargs:
        get_latest (bool): Read the latest version of the file, or ver 1.0.0.
        
    Args:
        trange (list): Time range to read in, times are of type datetime.datetime.
        read_vars (list): Variables to read.
        probe (str): Specifies the which spacecraft data to read, 
                     takes in either 'rbspa' or 'rbspb'.
                     
    Outputs: 
        data_list (list): List of arrays of variable data. The first element
                          is always the time array.
        unread (list): List of files that couldn't be read.
    """
    
    trange0 = datetime.strptime(trange[0], '%Y-%m-%d')
    trange1 = datetime.strptime(trange[1], '%Y-%m-%d')
    
    flist = load_rbsp([trange0, trange1], setprobe = probe, 
                         dataset = 'mageis', most_recent = get_latest, local=False)
    
    data_list = []
    unread = []
    err_msg = []
        
    for f in flist:
        
        try:
            cfile = cdflib.CDF(f) # Read CDF file
        except Exception as e:
            print('Unable to read file: ', f)
            print('Error: ', e)
            raise
            
            
        # Get array of datetimes
        #   NB: 'Epoch' dtype is 'CDF_EPOCH' 
        #   Equivalent to milliseconds since Year 0.
        cepoch = cfile.varget('Epoch')
        csplit = cdflib.cdfepoch.breakdown(cepoch)
        c_datetime = np.array([datetime(*c).replace(microsecond=0) for c in csplit])
        
        first = flist.index(f) == 0
        
        if first:
            data_list.append(c_datetime)
        else:
            data_list[0] = np.concatenate((data_list[0], c_datetime), axis=0)

        if len(read_vars) > 0:
            for var in read_vars:
                cvar = cfile.varget(var)    
                
                if type(cvar) != np.ndarray:
                    print('NB: ', var, ' is not type numpy.ndarray but is type ', type(cvar))
                    return None, None
                                
                if first:
                    data_list.append(cvar)
                else:
                    var_index = read_vars.index(var) + 1
                    data_list[var_index] = np.concatenate((data_list[var_index], cvar), axis = 0)
                    
        # Close CDF file
        cfile.close()

    return data_list, unread


# Converting Iso to Datetime.Datetime
def iso_parser(iso_arr):
    """ (MagEphem)
    Converts time in ISO8601 format to datetime.datetime format.
    
    Args:
        iso_arr: Array of times in ISO8601 format.
    
    Output: 
        An array of times in datetime.datetime format.
    """
    
    # This line replaces the code below the dashed line.
    vfunc = np.array([dateutil.parser.isoparse(x) for x in iso_arr])

    #------------------------------------------
    
    # Please see 
    # https://stackoverflow.com/questions/35215161/most-efficient-way-to-map-function-over-numpy-array
#     parser = lambda x: dateutil.parser.isoparse(x)
#     vfunc = np.vectorize(parser)
#     return vfunc(iso_arr)

    return vfunc


def generate_datetime(files_list):
    """ (MagEphem)
    Returns an array of times (in datetime.datetime format)
    """

    dt_arr = np.empty(shape=(1,))
    dt_list = []
    
    for f in files_list:
        
        h5file = h5py.File(f, 'r') # Read file
        
#         if 'IsoTime' in list(h5file.keys()):
#             print('True')
#         else:
#             print('False')
        cur_iso = np.array(h5file['IsoTime'])
        
#         print('Iso shape:', cur_iso.shape)
        
        if len(cur_iso) not in (289, 1441):
#             print(len(cur_iso))
            h5file.close() # Close file if error
            raise Exception("The arrays should be of length 289 or 1441.")
        
        iso_array = np.array(list(filter(None, cur_iso[:-1]))) # Removes empty strings from list
#         cur_dt = iso_parser(cur_iso[:-1])
        cur_dt = iso_parser(iso_array)
        
        dt_arr = np.append(dt_arr, cur_dt, axis=0)
#        dt_list.append(cur_dt)
        
        h5file.close() # Close file
        
#    dt_arr = np.array(dt_list)
    return dt_arr[1:]    


## READING ## 
def check_file(flist):
    """ (MagEphem)
    Checks if HDF5 files in `flist` are corrupted. Readable files 
    in list `readable`, corrupted files in list `unreadable`.
    
    """
    
    readable = []
    unreadable = []
    err_msg = []
    
    for f in flist:
        try:
            h5file = h5py.File(f, 'r') # Attempt to read file 
            
            # Attempt to read some variable
            somevar = np.array(h5file['IsoTime'])
            readable.append(f)
            h5file.close()
        except OSError as err:
#             print(err)
            unreadable.append(f)
            err_msg.append(err)
        except:
            unreadable.append(f)
            err_msg.append("Some unknown error.")
            
    
    return readable, list(zip(unreadable, err_msg))
    
    

def data_array(var_name, flist, tolerance = -1e31):
    """ (MagEphem)
    file_list assumed to be in chronological order.
    
    Kwargs:
        tolerance (float): Values at or exceeding this number in magnitude
                   are omitted.
        
    Args:
        var_name (str): The variable to be obtained.
        flist (list): List of HDF5 files to read.
        
    Output:
        array: Values of the specified variable.
    
    """

    tol = np.absolute(tolerance)
    L_tol = 7.5 # Tolerance value for L parameters
    
    if var_name in ['Lsimple', 'L', 'Lstar', 'Lm_eq']:
        tol = L_tol
    
    test_file = h5py.File(flist[0], 'r')
    data_shape = test_file[var_name].shape
        
    if len(data_shape) > 1:    
        data_cols = data_shape[1]
        var_array = np.empty(shape=(1, data_cols))
    else:
        var_array = np.empty(shape=(1,))
        
    for f in flist:
        cur_file = h5py.File(f, 'r') # Read file
        cur_data = np.array(cur_file[var_name])
        
        var_array = np.append(var_array, cur_data[:-1], axis=0)
        cur_file.close() # Close file
            
    ## Close file
    test_file.close()
            
    
    var_data = var_array.copy()
    
    if len(data_shape) > 1:      
        
        ## Filter out NULL values, replace with NaN
        for col in range(data_shape[1]):
            non_nan = np.where(np.isnan(var_data[:, col]) == False)
            current_col = var_data[non_nan, col]
            
            col_idx = np.where(np.absolute(current_col) >= tol)
            var_data[col_idx, col] = np.nan
            
        # Since we are appending to an empty array, we omit the 
        # first row.
        var_data = var_data[1:,:]
        
    else:
        
        ## Filter out NULL values, replace with NaN
        non_nan = np.where(np.isnan(var_data) == False)
        idxs = np.where(np.absolute(var_data[non_nan]) >= tol)
        var_data[idxs] = np.nan
        
        var_data = var_data[1:]
        
    return var_data


def read_vap_ephem(trange, read_vars, probe, b_model, 
                   null_vals = [],
                   get_latest = True,
                   plot_data = False,
                   plot_vars = []):
    """
    Read in RBSP MagEphem data, given time range 'trange' and a list
    of variables to read ('read_vars').
    
    Loads latest version of each file, by default. No implementation
    yet for selecting a specific version for specific date(s).
    
    Kwargs: 
        null_vals (list): List of tolerances or NULL values corresponding 
                          to each variable in 'read_vars'. The first element
                          in this list corresponds to the first variable listed
                          in 'read_vars'. 
                          The default tolerance value for any variable is 
                          set to -1e31.
        get_latest (bool): Loads the latest version of each file. If 'False', 
                           loads version 1.0.0. Set to 'True' by default.
        plot_data (bool): Toggle to plot 'L' and 'MLT' across entire trange.
        plot_vars (bool): If plot_data is True, then plots the specified 
                          variables.
                          If empty, plots all variables in `read_vars`.
                          Note on usage: Only plots one 'L' data and one 'MLT' 
                          data at a time. Can only take in two elements. 
                          Plots multidimensional L data for pitch angle = 90
                                  only.         
        
    Args:
        trange (list-like): A time range (a two-element list/array of type string)
        read_vars (list-like): A list or array of variables to read in.
        probe (str): Takes in either 'rbspa' or 'rbspb', specifies RBSP probe.
                     Lowercase.
        b_model (str): Type of magnetic field model, either "OP77Q" or "TS04D". 
                
    Outputs:
        data_list (list-like): A list of `read_vars` arrays,
                               with the corresponding time as the first entry of
                               the list.
                               Entries in this list match the order of 
                               'read_vars'.
            Example: read_vap_ephem(['2016-12-30', '2018-05-07'], 
                                    read_vars=['L', 'EDMAG_MLT'],
                                    probe='rbspb', b_model='OP77Q')
                     returns 'data_list'
                     where 
                     len(data_list) = 3,
                     data_list[0] = Time range as array
                     data_list[1] = 'L' as array across 'trange'
                     data_list[2] = 'EDMAG_MLT' as array across 'trange'
                     
        unread_list (list-like): List of files that could not be read, formatted 
                                 as tuples: (file name, error message)

    """
    
    ## Get file list within the specified time range
    
    time1 = 0
    time2 = 0
    
    file_list = []
    
    # Default null vals: -1e+31
    default_null = 1e+31
    
    if not null_vals:
        for i in range(len(read_vars)):
            null_vals.append(default_null)
    else:
        for i in range(len(null_vals)):
            if i is None:
                null_vals[i] = default_null
    
    null_vals = np.array(null_vals)
    
    # Convert dates as strings to datetime.datetime format
    if type(trange) in [list, np.ndarray, tuple]:
        time1 = datetime.strptime(trange[0], '%Y-%m-%d')
        time2 = datetime.strptime(trange[1], '%Y-%m-%d')
        
        if time1.year != time2.year:
            
            start_year = time1.year
            end_year = time2.year
            
            y = start_year
            while y <= end_year:
                
                jan01 = datetime.strptime(str(y)+'0101', '%Y%m%d')
                dec31 = datetime.strptime(str(y)+'1231','%Y%m%d')
                
                dt1 = jan01
                dt2 = dec31
                
                if y == start_year:
                    dt1 = time1
                
                elif y == end_year:
                    dt2 = time2
                
                file_list.extend(load_rbsp([dt1, dt2], 
                                           probe, b_model,
                                           dataset = 'MagEphem',
                                           most_recent=get_latest))
                y += 1
        
                        
        else:
            file_list = load_rbsp([time1, time2], probe, 
                                  b_model, 
                                  dataset = 'MagEphem',
                                  most_recent=get_latest)
        
    else:
        print("`trange` must be a 2-element list/array-like of strings \
              in the format `YYYY-mm-dd`.")
        return
    
    # Check if files are readable
    read_list, unread_list = check_file(file_list)
    
    data_list = [generate_datetime(read_list)]
    
    for var in read_vars:        
        ## Extract `read_vars` variables
            # Call data_array for each variable
        idx = read_vars.index(var)
        var_array = data_array(var, read_list, tolerance=null_vals[idx])
        data_list.append(var_array)
    
    ## Plot `L` and `MLT` for preview
    if plot_data:
        if len(plot_vars) == 0:
            plot_vars = read_vars.copy()
        
        pvars = [] # Data to plot
        for elem in plot_vars:
            elem_idx = np.where(read_vars == elem)[0]
            i = elem_idx + 1
            
            if data_list[i].ndim > 1: 
                pvars.append(data_list[i][:,0]) # Automatically plots for pitch angle = 90
            else:
                pvars.append(data_list[i])
        
            
        start_idx, end_idx = plotvars(pvars, plot_vars, data_list[0], 
        [trange[0] + ' 00:00', trange[1] + ' 00:00'], plot_size=(15, 10), res=150)
    
    return data_list, unread_list
        


    #### --- POES --- ####
    #### READING DATA
def load_poes(dtrange, satIDs):
    """
    Helper function for read_poes.
    Returns a list of POES files for specified spacecraft(s) over
    one year.
    
    Kwargs:
    
    Args:
        dtrange: Time range as a tuple or list-like element where the 
                 first element corresponds to the start date, and the 
                 last element corresponds to the end date. Dates are in 
                 datetime.datetime format. 
        satIDs: List-like element indicating 
                which spacecraft(s) to read. Must be names 
                within the following list: [m01, m02, n15, n18, n19] 
                corresponding to 'metop01', 'metop02', 'noaa15', 
                'noaa18', and 'noaa19'.
        
    """
    
    poes_path = '/data1/big/SATELLITE/noaa_w/poes_seth/'
#     poes_path = '/Users/frootloops/Downloads/poes/'
    sat_dict = {'m01':'metop01', 'm02':'metop02', 
               'n15':'noaa15', 'n18':'noaa18', 'n19':'noaa19'}
    
    Year = dtrange[0].year
    
    dt0 = str(Year) + str(dtrange[0].month) + str(dtrange[0].day)
    dt1 = str(Year) + str(dtrange[1].month) + str(dtrange[1].day)
    
    start_date = dtrange[0].date()
    end_date = dtrange[1].date()
    d = start_date
    
    satfiles = []
    
    for sc in satIDs:
        
        file_path = poes_path + sat_dict[sc] + '/'

        while d <= end_date:

            datestring = d.strftime('%Y%m%d')

            file_list = glob(file_path+'*'+sc+'_'+datestring+'_proc.nc')
            satfiles.extend(file_list)

            d += timedelta(days = 1)
            
        d = start_date # Set iterator back to the start for 
                       # new spacecraft  

    return satfiles

def read_poes(trange, sat_IDs, read_vars, null_vals=[], filterL=True):
    """
    
    Kwargs:
        null_vals (list-like):  Tolerance associated with each variable. If empty,
                                default is 9.9e36 for each variable. Pass in None 
                                to a specific variable to assume default tolerance
                                value. 
        filterL (bool):  If true, every L-value > 7 and every MLT-value > 24 and 
                         < 0 is omitted. Default is True.
    
    Args:
        trange (list-like): Time range as a tuple or list-like element where the 
                            first element corresponds to the start date, and the 
                            last element corresponds to the end date. Dates are in 
                            datetime.datetime format. 
        sat_IDs (list-like): A tuple or other list-like element indicating 
                             which spacecraft(s) to read. Must be names 
                             within the following list: [m01, m02, n15, n18, n19] 
                             corresponding to 'metop01', 'metop02', 'noaa15', 
                             'noaa18', and 'noaa19'.
        read_vars (list): List of variables to read.
    """

    
    # Parameter defaults: fill values
    default_null = 9.69e+36
    
    if len(null_vals) == 0:
        null_vals = np.empty(shape=(len(read_vars),), dtype = object)
        null_vals.fill(default_null)
    else:
        for i in range(len(null_vals)):
            if null_vals[i] is None:
                null_vals[i] = default_null
       
    if filterL:
        if 'L_IGRF' in read_vars:
            idxL = read_vars.index('L_IGRF')
            null_vals[idxL] = (1.0, 7.5)
        if 'MLT' in read_vars:
            idxM = read_vars.index('MLT')
            null_vals[idxM] = (0.0, 24.0)
            
    null_vals = np.array(null_vals, dtype=object)
        
    # Convert dates as strings to datetime.datetime format
    if type(trange) in [list, np.ndarray, tuple]:
        time1 = datetime.strptime(trange[0], '%Y-%m-%d')
        time2 = datetime.strptime(trange[1], '%Y-%m-%d')
                
        filelist = load_poes([time1, time2], sat_IDs) # Generate list to read

    else:
        print("`trange` must be a 2-element list/array-like of strings \
              in the format `YYYY-mm-dd`.")
        return
    
    readable, unread = check_poes_file(filelist)
    
    # Generate timestamps
#     timestamps = poes_datetime(readable) # 1D array of timestamps associated with
#                                          # 'readable' list of POES files.
    dtimes = np.empty(shape=(1,))
#     dtimelist = []
    
    for f in readable:
        
        poes_file = ncdf.Dataset(f)
        time_var = np.array(poes_file['time'])
        time_data = [(datetime.utcfromtimestamp(t)).replace(microsecond=0) for t in time_var/1000.0]
#       # The line above replaces the two lines of code below.
#         time_data = np.vectorize(datetime.utcfromtimestamp)(time_var/1000.0)
#         time_data = np.vectorize(datetime.replace)(time_data, microsecond=0)
        time_data = np.array(list(filter(None, time_data)))
        dtimes = np.append(dtimes, time_data, axis=0)
        poes_file.close()
        
#    dtimelist = np.array(dtimelist).reshape(-1,)
    timestamps = dtimes[1:]
       
    data_list = [timestamps] # List holding all variables and assoc. timestamps
        
    # Read variable data: for each variable. There are separate files list 
    # for each spacecraft. 
#     for v in read_vars:
#     # Replaced by list comprehension below.
#         v_idx = read_vars.index(v)
#         data_list.append(poes_data(v, readable, tolerance = null_vals[v_idx]))
    vardata = [poes_data(read_vars[i], readable, tolerance=null_vals[i]) for i in range(0, len(read_vars))]
    data_list.extend(vardata)
    
    return data_list, unread
    
    
def check_poes_file(files_list):
    
    readable = []
    unreadables = []
    message_list = []
    
    for f in files_list:
        
        try:
            poes_file = ncdf.Dataset(f)
            readable.append(f)
            poes_file.close()
        except OSError as err:
            unreadables.append(f)
            message_list.append(err)
        except:
            unreadables.append(f)
            message_list.append('Unknown error: File could not be read.')
    
    unread_zip = list(zip(unreadables, message_list))
    
    return readable, unread_zip
    

# def poes_datetime(files_list):
#     """ DEPRECATED as of 07 May 20
#     For all files in files_list, return an array of all timestamps combined. 
#     """
    
#     dtimes = np.empty(shape=(1,))
#     dtimelist = []
    
#     for f in files_list:
        
#         poes_file = ncdf.Dataset(f)
#         time_var = np.array(poes_file['time'])
#         time_data = np.vectorize(datetime.utcfromtimestamp)(time_var/1000.0)
#         time_data = np.vectorize(datetime.replace)(time_data, microsecond=0)
        
#         time_data = np.array(list(filter(None, time_data)))
        
#         dtimes = np.append(dtimes, time_data, axis=0)
        
#         poes_file.close()
        
# #    dtimelist = np.array(dtimelist).reshape(-1,)
#     dtimes = dtimes[1:]
    
#     return dtimes


def poes_data(var_name, flist, tolerance = 9.9e36):
    """    
    Kwargs: 
        tolerance: Either a float (default, value = 9.9e36) to represent a fill value 
                   or a tuple that represents the upper and lower bounds of the 
                   variable, INCLUSIVE. 
                   For L var(s), tuple (1, 7) would typically be passed in.
                   For MLT var(s), tuple (0, 24) would be passed in.
    Args:
        var_name: Name of variable to be passed in.
        flist: List of files to read.
    """
    
    automask = False # Issue 206 in netCDF4 github
    test_file = ncdf.Dataset(flist[0])
    
    try:
        test_data = np.array(test_file[var_name])
    except TypeError:
        #print(te)
        test_file.set_auto_maskandscale(False)
        automask = True
        pass
    except IndexError as idxe:
        test_file.close()
        print('Variable ', var_name, ' is not found.')
        raise 
    except:
        test_file.close()
        raise
    
    if automask:
        test_data = np.array(test_file[var_name])
        
    var_shape = test_data.shape
    test_file.close()
    
    if test_data.ndim > 1:
        data_cols = var_shape[1]
        var_array = np.empty(shape=(1, data_cols))
    else:
        var_array = np.empty(shape=(1,))
    
    for f in flist:
        
        poes_file = ncdf.Dataset(f)
        
        if automask:
            poes_file.set_auto_maskandscale(False)
        
        var_data = poes_file.variables[var_name][:]
        poes_file.close()
                
        var_array = np.append(var_array, var_data.data, axis=0)
        
        # Check attributes for fill value if tolerance is the default value.
        try:
            if type(tolerance) in ['float', 'np.float32', 'float32'] and type(var_data.fill_value) in ['np.float32', 'float32']:
                tolerance = var_data.fill_value
        except:
            continue
            
    # Since we appended each file's variable data to var_array
    # including the first file, the first element is empty/zero
    # and should be removed.
    
    if test_data.ndim > 1:
        var_array = var_array[1:, :]
        
        ## Filter out NULL values, replace with NaN
        for col in range(data_shape[1]):
            
            # Precautionary... since 2012 POES files is full of NaNs
            non_nan = np.where(np.isnan(var_data[:, col]) == False)
            current_col = var_array[non_nan, col]
            
            if type(tolerance) in [tuple, list, np.ndarray]:
                idx_low = np.where(current_col < tolerance[0]) # Lower bound, inclusive
                idx_up = np.where(current_col > tolerance[1]) # Upper bound, inclusive
                var_array[idx_low, col] = np.nan
                var_array[idx_up, col] = np.nan
                
            else: # Assume tolerance is a float, just filter absolute upper bound
                col_idx = np.where(np.absolute(current_col) >= np.absolute(tolerance))
                var_array[col_idx, col] = np.nan
        
    else:
        var_array = var_array[1:]
        
        # Check for NULL values, replace with NaN
        non_nan = np.where(np.isnan(var_array) == False)
        
        if type(tolerance) in [tuple, list, np.ndarray]:
            idx_low = np.where(var_array[non_nan] < tolerance[0])
            idx_up = np.where(var_array[non_nan] > tolerance[1])
            
            var_array[idx_low] = np.nan
            var_array[idx_up] = np.nan
        
        else:
            null_idx = np.where(np.absolute(var_array[non_nan]) >= np.absolute(tolerance))
            var_array[null_idx] = np.nan
        
    return var_array



