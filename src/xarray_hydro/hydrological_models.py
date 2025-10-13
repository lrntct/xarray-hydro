
import numpy as np
import xarray as xr

def HBV(
    P: xr.DataArray,
    T: xr.DataArray,
    ETP: xr.DataArray,
    parameters:xr.Dataset,
) -> xr.DataArray:
    """
    Application of the HBV-light semi-distributed hydrological model (Seibert & Vis, 2012)
    Inputs:
        P = Precipitation (mm/Δt)
        ETP = Potential evapotranspiration (mm/Δt)
        T = Temperature (°C)
        parameters = Dataset with the variables: BETA, FC, K0, K1, K2, LP, PERC, UZL, TT, CFMAX, CFR, CWH, PCORR and SFCF.
    Output:
        Qhbv = Streamflow (mm/Δt).
    - The input variables and parameters must have the same watershed ID dimension. 
    - Time must be the first dimension of the input variables.
    - Time interval (Δt) can be on a daily basis, hourly basis, or any other.
    - K0,K1,K2,PERC,CFMAX depend on time. If you possess parameter values, but their time interval 
      differs from the run time interval, you must initially convert the time unit of these parameters.
    - The values for PCORR and SFCF can be 1 if you don't have them.
    """
    # Initialize time series of model variables
    SNOWPACK = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    MELTWATER = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    SM = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    SUZ = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    SLZ = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    ETact = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    Qhbv = xr.zeros_like(P*parameters['BETA'], dtype=float)
    
    for t in range(0, P.shape[0]):
        
        # Separate precipitation into liquid and solid components
        PREC = P[t]*parameters['PCORR']
        RAIN = PREC*(T[t] >= parameters['TT'])
        SNOW =PREC*(T[t] < parameters['TT'])
        SNOW = SNOW*parameters['SFCF']
        
        # Snow
        SNOWPACK = SNOWPACK+SNOW
        melt = parameters['CFMAX']*(T[t]-parameters['TT'])
        melt = melt.clip(0.0, SNOWPACK)
        MELTWATER = MELTWATER+melt
        SNOWPACK = SNOWPACK-melt
        refreezing_meltwater = parameters['CFR'] * parameters['CFMAX'] * (parameters['TT']-T[t])
        refreezing_meltwater = refreezing_meltwater.clip(0.0, MELTWATER)
        SNOWPACK = SNOWPACK+refreezing_meltwater
        MELTWATER = MELTWATER-refreezing_meltwater
        tosoil = MELTWATER - (parameters['CWH']*SNOWPACK)
        tosoil = tosoil.clip(0.0, None)
        MELTWATER = MELTWATER-tosoil

        # Soil and evaporation
        soil_wetness = (SM/parameters['FC']) ** parameters['BETA']
        soil_wetness = soil_wetness.clip(0.0, 1.0)
        recharge = (RAIN+tosoil) * soil_wetness
        SM = SM+RAIN+tosoil-recharge
        excess = SM-parameters['FC']
        excess = excess.clip(0.0, None)
        SM = SM-excess
        evapfactor = SM / (parameters['LP']*parameters['FC'])
        evapfactor = evapfactor.clip(0.0, 1.0)
        ETact = ETP[t]*evapfactor
        ETact = np.minimum(SM, ETact)
        SM = SM-ETact

        # Groundwater boxes
        SUZ = SUZ+recharge+excess
        PERC = np.minimum(SUZ, parameters['PERC'])
        SUZ = SUZ-PERC
        Q0 = parameters['K0']*np.maximum(SUZ-parameters['UZL'], 0.0)
        SUZ = SUZ-Q0
        Q1 = parameters['K1']*SUZ
        SUZ = SUZ-Q1
        SLZ = SLZ+PERC
        Q2 = parameters['K2']*SLZ
        SLZ = SLZ-Q2
        Qhbv[t] = Q0+Q1+Q2
    return Qhbv.to_dataset(name="Q")


def HBV_snowless(
    P: xr.DataArray,
    ETP: xr.DataArray,
    parameters:xr.Dataset,
) -> xr.DataArray:
    """
    Application of the HBV-light semi-distributed hydrological model (Seibert & Vis, 2012), without the snow routine.
    Inputs:
        P = Precipitation (mm/Δt)
        ETP = Potential evapotranspiration (mm/Δt)
        parameters = Dataset with the variables: BETA, FC, K0, K1, K2, LP, PERC, UZL, TT, CFMAX, CFR, CWH and PCORR .
    Output:
        Qhbv = Streamflow (mm/Δt).
    - The input variables and parameters must have the same watershed ID dimension. 
    - Time must be the first dimension of the input variables.
    - Time interval (Δt) can be on a daily basis, hourly basis, or any other.
    - K0,K1,K2,PERC,CFMAX depend on time. If you possess parameter values, but their time interval 
      differs from the run time interval, you must initially convert the time unit of these parameters.
    - The value for PCORR can be 1 if you don't have one.    
    """
    # Initialize time series of model variables
    SM = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    SUZ = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    SLZ = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    ETact = xr.zeros_like(parameters['BETA'], dtype=float)+0.001
    Qhbv = xr.zeros_like(P*parameters['BETA'], dtype=float)
    
    for t in range(0, P.shape[0]):
        RAIN = P[t]*parameters['PCORR']
        
        # Soil and evaporation
        soil_wetness = (SM/parameters['FC']) ** parameters['BETA']
        soil_wetness = soil_wetness.clip(0.0, 1.0)
        recharge = (RAIN) * soil_wetness
        SM = SM+RAIN-recharge
        excess = SM-parameters['FC']
        excess = excess.clip(0.0, None)
        SM = SM-excess
        evapfactor = SM / (parameters['LP']*parameters['FC'])
        evapfactor = evapfactor.clip(0.0, 1.0)
        ETact = ETP[t]*evapfactor
        ETact = np.minimum(SM, ETact)
        SM = SM-ETact

        # Groundwater boxes
        SUZ = SUZ+recharge+excess
        PERC = np.minimum(SUZ, parameters['PERC'])
        SUZ = SUZ-PERC
        Q0 = parameters['K0']*np.maximum(SUZ-parameters['UZL'], 0.0)
        SUZ = SUZ-Q0
        Q1 = parameters['K1']*SUZ
        SUZ = SUZ-Q1
        SLZ = SLZ+PERC
        Q2 = parameters['K2']*SLZ
        SLZ = SLZ-Q2
        Qhbv[t] = Q0+Q1+Q2
    return Qhbv.to_dataset(name="Q")


def Q_to_cms(
    dataset: xr.DataArray,
    areas: xr.DataArray,
    dim_time: str,
) -> xr.DataArray:
    """
    Change the Streamflow unit from mm/Δt to m3/s.
    areas = Watershed area in square meters.
    Streamflow and areas must have the same watershed ID dimension.
    """
    delta_time=(dataset[dim_time][1]-dataset[dim_time][0]).astype('timedelta64[s]').astype(float)
    Qhbv_cms=dataset/(1000*delta_time)*areas
    return Qhbv_cms.to_dataset(name="Q")


def _routing_maxbas(
    Q: xr.DataArray,
    m: xr.DataArray,
) -> xr.DataArray:
    """
    Triangular weight function.
    Q = Streamflow. Must contain the dimension of maxbas.
    m = MAXBAS is the routing parameter (Δt).
    """
    Qm = np.zeros_like(Q)
    for i in range(1,m+1):     
        c = -((abs(m-((i-1)*2))-abs(m-(i*2))-4)*m-
                ((i-1)*2)*abs(m-((i-1)*2))+
                (i*2)*abs(m-(i*2)))/(2*m**2) 
        Qi=np.roll(Q,i-1,axis=0)
        Qi[0:i-1] = 0   
        Qm+=Qi*c
    return Qm

def maxbas_hbv(
    Q: xr.DataArray,
    m: xr.DataArray,
    dim_time: str,
) -> xr.DataArray:
    """
    Triangular weight function vectorized.
    Q = Streamflow
    m = MAXBAS is the routing parameter (Δt).
    """
    Q_=Q.copy()
    Q_, m_ = xr.broadcast(Q, m)
    Qmxb=xr.apply_ufunc(_routing_maxbas,Q_,m,
                            input_core_dims=[[dim_time], []],
                            output_core_dims=[[dim_time]],
                            vectorize=True,
                            dask="parallelized").compute()
    return Qmxb.to_dataset(name="Q").transpose(dim_time,...)