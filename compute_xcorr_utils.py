#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: destin nziengui bâ (destin.nziengui-ba@univ-grenoble-alpes.fr / destin.nziengui@febus-optics.com)

Script for computing cross-correlation from reducted DAS data
"""


############################################################
#
#                       MAIN
#
############################################################


def compute_xcorr_from_ReductedFile(filename, couples, P_noise_processing, P_xcorr, prefix):
    """
    Compute cross-correlation from a reduced seismic data file.

    Parameters:
        filename (str): Name of the reduced seismic data file (SR_xxxx.h5).
        couples (ndarray): Array of couples to correlate.
        P_noise_processing (dict): Pre-processing parameters.
        P_xcorr (dict): Cross-correlation parameters.
        prefix (str): Prefix for output filenames.

    Returns:
        None
    """
    import os
    import h5py 
    import numpy as np
    from a1das import xcor
    
    # Load file
    file_path_in = os.path.join(P_xcorr['cc_path_in'], filename)
    f = h5py.File(file_path_in, 'r')
    
    # Extract header attributes
    hdr_attr = f['header'].attrs
    dhd = {}
    for key, value in hdr_attr.items():
        if key != 'file_type' and key != 'version':
            dhd[key] = value
    dhd['time'] = f['/time'][:]
    dhd['dist'] = f['/distance'][:]
    ntrace = dhd['nspace']
    fs = 1 / dhd['dt']
    
    # Load Xcorr parameters
    win_size = P_xcorr['cc_WinSize']
    lag = P_xcorr['cc_lag']
    stack = P_xcorr['cc_stack']
    min_to_stack = P_xcorr['cc_min_to_stack']
    
    # Register noise processing
    for method, values in P_noise_processing.items():
        values=list(P_noise_processing[method].values())
        if method == "white": 
            #print(values,fs)
            values = [v / fs for v in values]
        xcor.register_par(method, values)
    
    # Register couples to correlate
    xcor.register_couple(couples, ntrace=ntrace)
    nxc = xcor.get_nxcorr()
    
    # define windows segmentation
    _, ref_block = segR(dhd["time"], threshold=2, ws=win_size, fs=int(fs), infos=False)
    stack_md = seg_stack(ref_block, min_to_stack)
    nwin = len(stack_md)
    ntime = Len_LagTime(win_size, lag, fs)
    
    
    # Compute cross-correlation
    if not P_xcorr["cc_save"]:
        print("TO IMPLEMENT ...")
    else:
        if prefix == "SR_DS": # native file 
            xc_name = os.path.join(P_xcorr['cc_path_out'], "xcorr_R_%s.h5" % (filename[filename.find(prefix) + 6:-7]))
        else:  #reducted file 
            xc_name = os.path.join(P_xcorr['cc_path_out'], "xcorr_R_%s.h5" % (filename[filename.find(prefix) + 4:-7]))
        
        fout, xcorr_dset, ijx_dset, lags_dset, timestamp_dset, ct_dset = create_h5_file(xc_name, nwin, nxc, ntime)
        
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print("file: %s" % (filename))
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n')
        
        XC1 = np.zeros((nwin, nxc, ntime))
        
        for iwin in range(nwin):
            ind = stack_md["%02d" % iwin]
            nstack = len(ind)
            xcorr = np.zeros((nxc, ntime))
            
            tmin = ref_block[ind[0], 0]
            tmax = ref_block[ind[-1], 0]
            tmean = (tmin + tmax) / 2.
            timestamp_dset[iwin] = tmean + dhd["otime"]
            
            print('=> win %d/%d [nstack=%d] : computing xcorr @time : %s  ' % (iwin + 1, nwin, nstack, timestamp_to_datestring(tmin + dhd["otime"])))
            print('_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ ')
            
            for i in ind:
                start, end = ref_block[i, 1:3].astype(int)
                A1S = f["/section"][:, int(start):int(end)].astype("float64")
                lag_tmp = int(lag * fs) if lag else None
                xcorr_tmp, lags, ijx, ier = xcor.compute_xcorr(arrayIn=A1S, lag=lag_tmp, stack=stack, verbose=0) #MAIN !!!
                xcorr = xcorr + xcorr_tmp
            
            ct_dset[iwin] = nstack
            XC1[iwin, :, :] = xcorr / nstack
        
        xcorr_dset[:] = XC1.astype("f4")
        lags_dset[:] = (lags / fs).reshape((-1, 1))
        ijx_dset[:, :] = ijx
        fout.close()
    f.close()

        
############################################################
#
#                       H5FILE 
#
############################################################

def create_h5_file(filename,nwin,nxcorr,ntime):

    """
    création du fichier hdf5 contenant les xcorr pour le fichier 
    ------------------------------------------------------------
    filename (str): filename 
    nwin (int): number of windows
    nxcorr (int): number of xcorr
    ntime(int): number of time lags samples
    
    ------------------------------------------------------------
    !!! TO DO
    RQ:
        - Il faudra changer l'ordre plus tard pour optimiser la lecture de ces fichiers 
    """
    
    import h5py 
    
    fout = h5py.File(filename, "w")
    
    xcorr_dset = fout.create_dataset('/xcorr', (nwin,nxcorr,ntime),dtype='f4',chunks=(nwin,1,ntime))
    ijx_dset = fout.create_dataset('/ijx', (nxcorr,2),dtype='i8')
    lags_dset = fout.create_dataset('/lags', (ntime,1),dtype='f8')
    timestamp_dset = fout.create_dataset('/timestamp', (nwin,1),dtype='f8')
    ct_dset= fout.create_dataset('/stack_ct', (nwin,1),dtype='f8')
    return fout,xcorr_dset,ijx_dset,lags_dset,timestamp_dset,ct_dset



############################################################
#
#                   XCORR LAG TIME
#
############################################################
def Len_LagTime(ntime,lag,fs):
    """
    sert à déterminer la taille de lags (au cas où on enregistre le fichier)
    -----------------------------------
    ntime (int): number of time samples
    lag (float64): lag (sec)
    fs (float64): sampling frequency (Hz)
    
    ------------------------------------
    
    RQ: 
        - on n'utilise pas l'array lags 
    """
    
    
    import numpy as np 

    if lag is None:
        lag = ntime
    #lag = int(lag)
        
    ntime=int(ntime*fs)
    lag=int(lag*fs)
    
    if ntime % 2 == 0: # Even case ntime=2*k
        if lag is None or (lag == 0) or (2*lag + 1 > ntime):
            lag = ntime/2
            len = ntime
            lags = np.arange(-lag,lag)
        else:
            len = 2*lag+1
            lags = np.arange(-lag,lag+1)

    else:             # Odd case ntime=2*k+1
        if lag is None or lag == 0:
            lag = (ntime-1)/2
            len = ntime
        else:
            if (2*lag + 1 > ntime):
                lag = (ntime-1)/2
            len = 2*lag+1
        lags = np.arange(-lag, lag + 1)
    return int(len)




############################################################
#
#                       SEGMENTATION
#
############################################################





def segR(time, threshold=2, ws=4,fs=200,infos=False):
    
    """
    Performs segmentation based on the timestamps 
    
    Permet de 
    1) regrouper les blocs de données contigus (pour les fichiers qui ont des overflows))
    2) subdiviser en blocs  de taille win_size
    
    Parameters:
        time (ndarray): Timestamps of seismic data.
        threshold (float): Threshold for detecting overflow in seconds.
        ws (int): Window size in seconds.
        fs (float): Sampling frequency.
        infos (bool): Flag to print information about overflow and data loss.
    
    Returns:
        out, out2 (ndarray, ndarray): Primary segmentation of seismic data, secondary segmentation of seismic data.
    """
    
    import numpy as np 
    
    nblocks=int(time.size/fs)
       
    #case 1 : complete file => simple case 
    if nblocks==3601 or nblocks==3600 :  
        if infos:
            print("Dataloss: 0 %")
            print("------------------------------------------------")
            
        I1=np.arange(0,3601-ws,ws)*fs
        I2=I1+ws*fs#-1
        T1=time[I1]
        
        out=np.array([T1,I1,I2]).T
        
        out2=out.copy()
        
        out2[:,0]=time[np.mean(out2[:,1:3],axis=1).astype(int)]
        
        return out,out2 
    
    
    # case 2 : incomplete file ( with overflows) => complex case 
    else:
        #find Indices of Overflows
        io=np.where(np.diff(time)>threshold)[0]
        
        no=len(io) #number of overflows
        
        if infos:

            print("noverflows: %d "%(no))
            print("nblocks   : %d/%d"%(nblocks,3601))
            print("DATALOSS  : {} %".format(100*(1-nblocks/3601)))
            print("------------------------------------------------")
        
        if io[0]!=0:
            io=np.append(-1,io)
        if io[-1]<=int(nblocks*fs):
            io=np.append(io,int(nblocks*fs-1))          

            
        out=np.zeros((no+1,3))         
    
        for i in range(no+1):
            out[i,1]=io[i]+1
            out[i,2]=io[i+1]
        out[:,0]=time[out[:,1].astype(int)]
        
        out1=segR2(time,out,fs,ws)
        
        return out,out1
    
def segR2(time,seg,fs,ws):
    """
    segmentation secondaire sur les blocs continus => blocs de taille win_size
    
    IN
    --------------------------------------------------------------------------
    time (ns x 1 array, float) : timestamp. ns=number of samples
    seg (nb x 3, array, float) : time x I1 x I2. nb=number of blocks 
    fs (float)                 : sampling frequency 
    ws (int)                   : window size 
    
    OUT
    -------------------------------------------------------------------------
    
    out (nbb x 3) : seg array subdivised. nbb= number of blocks bis 
    """
    
    import numpy as np 

    
    #nombre de blocs de taille ws qu'on peut extraire 
    k=(np.diff(seg[:,1:3])+1)//(ws*fs)   
    k=k[:,0].astype(int)
    
    #nombre de subdivisions en blocs contigus  
    nsub=int(np.sum(k))
    out=np.zeros((nsub,3))
    
    c=0 #compteur 
    for i in range(len(k)):
        #print(c,seg[i,1],k[i])
        if k[i]!=0:
            out[c:c+k[i],1]=seg[i,1]+np.arange(k[i])*(ws*fs)
            out[c:c+k[i],2]=out[c:c+k[i],1]+(ws*fs)
            c+=k[i]
    #out[:,0]=time[out[:,1].astype(int)]
    out[:,0]=time[np.mean(out[:,1:3],axis=1).astype(int)]
    return out 


def seg_stack(out, min_to_stack):
    # out issue de segR
    import numpy as np 
    sec_to_stack=min_to_stack*60
    nwin=3601//(sec_to_stack)
    
    t=out[:,0] #timestamp 
    
    md={}
    c=0
    for i in range(nwin):
        md["%02d"%i]=np.where(((t>i*sec_to_stack)&(t<(i+1)*sec_to_stack)))[0]
        c+=len(md["%02d"%i])
        
    if c!=out.shape[0]:
        print("some xcorr missing ....")
    return md




############################################################
#
#                       DATETIME
#
############################################################

def datestring_to_timestamp(date_string,fmt="%Y-%m-%dT%H:%M:%S.%f"):
    #convert datestring to timestamp (default=UTC)
    
    from datetime import timezone
    
    date=datestring_to_datetime(date_string,fmt="%Y-%m-%dT%H:%M:%S.%f")
    
    return date.replace(tzinfo=timezone.utc).timestamp()
   


def datestring_to_datetime(date_string,fmt="%Y-%m-%dT%H:%M:%S.%f"):
    #convert datestring to datetime 
    
    from datetime import datetime    
   
    return datetime.strptime(date_string, fmt)

def datetime_to_datestring(datetime,fmt="%Y-%m-%dT%H:%M:%S.%f"):
    #convert datetime to datestring 
    
    return datetime.strftime(fmt)

def timestamp_to_datestring(timestamp,fmt="%Y-%m-%dT%H:%M:%S.%f"):
    #convert timestamp to datestring (default=UTC)
    
    from datetime import datetime, timezone
    
    dt=datetime.fromtimestamp(timestamp,tz=timezone.utc)
    
    return dt.strftime(fmt)