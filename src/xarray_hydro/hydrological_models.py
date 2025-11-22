import numpy as np
import xarray as xr
from typing import TypedDict
from typing import Optional

# Set up the dictionary structure by defining the TypedDict.
class UserProfile(TypedDict):
    dim_time: str # Time dimension name
    precipitation: str # mm/Δt
    temperature: str # °C
    pot_evapotr: str # mm/Δt
    par_BETA:str # parameter that determines the relative contribution to runoff from rain or snowmelt
    par_FC:str # maximum soil moisture storage (mm)
    par_K0:str # storage (or recession) coefficient 0 (∆t-1)
    par_K1:str # storage (or recession) coefficient 1 (∆t-1)
    par_K2:str # storage (or recession) coefficient 2 (∆t-1)
    par_LP:str # soil moisture value above which AET reaches PET
    par_PERC:str # treshold parameter (Mm ∆t-1)
    par_UZL:str # treshold parameter (mm)
    par_TT:str #  threshold temperature (°C)
    par_CFMAX:str # degree-Δt factor (°C-1 ∆t-1)
    par_CFR:str # refreezing coefficient
    par_CWH:str # water holding capacity
    par_PCORR:str # precipitation correction factor
    par_SFCF:str # snowfall correction factor

class hbvl:
    """
    Application of the HBV-light semi-distributed hydrological model (Seibert & Vis, 2012)
    input_data = Dataset with the variables:
        - Precipitation (mm/Δt)
        - Temperature (°C)
        - pot_evapotr = ETPotential evapotranspiration (mm/Δt)
    parameters = Dataset with the parameters: BETA, FC, K0, K1, K2, LP, PERC, UZL, TT, CFMAX, CFR, CWH, PCORR and SFCF.
    - The input_data and parameters must have the same watershed ID dimension. 
    - Time interval (Δt) can be on a daily basis, hourly basis, or any other.
    - K0,K1,K2,PERC,CFMAX depend on time. If you possess parameter values, but their time interval 
      differs from the run time interval, you must initially convert the time unit of these parameters.
    - The values for PCORR and SFCF can be 1 if you don't have them.
    Output:
        Qhbv = Streamflow (mm/Δt).
    """
    def __init__(self, input_data: xr.Dataset, parameters: xr.Dataset, data_name: UserProfile, snow: Optional[bool] = None):
        # Anotación de tipo para el atributo de instancia
        self.input_data = input_data
        self.parameters = parameters
        self.data_name = data_name
        self.snow = snow
    
        # Initialize time series of model variables
        self.SNOWPACK = xr.full_like(self.parameters[self.data_name['par_BETA']], 1e-6, dtype=np.float64)
        self.MELTWATER = xr.full_like(self.parameters[self.data_name['par_BETA']], 1e-6, dtype=np.float64)
        self.SM = xr.full_like(self.parameters[self.data_name['par_BETA']], 1e-6, dtype=np.float64)
        self.SUZ = xr.full_like(self.parameters[self.data_name['par_BETA']], 1e-6, dtype=np.float64)
        self.SLZ = xr.full_like(self.parameters[self.data_name['par_BETA']], 1e-6, dtype=np.float64)
        self.ETact = xr.full_like(self.parameters[self.data_name['par_BETA']], 1e-6, dtype=np.float64)
        self.Qhbvl = xr.full_like(self.parameters[self.data_name['par_BETA']]*self.input_data[self.data_name["precipitation"]], 1e-6, dtype=np.float64)
    
    def calc_Streamflow(self):
        for t in self.input_data[self.data_name["precipitation"]][self.data_name["dim_time"]].to_numpy(): 

            if self.snow:            
                # Separate precipitation into liquid and solid components
                PREC = self.input_data[self.data_name["precipitation"]].sel({self.data_name["dim_time"]:t}) * self.parameters[self.data_name['par_PCORR']]
                RAIN = PREC*(self.input_data[self.data_name["temperature"]].sel({self.data_name["dim_time"]:t}) >= self.parameters['TT'])
                SNOW = PREC*(self.input_data[self.data_name["temperature"]].sel({self.data_name["dim_time"]:t}) < self.parameters['TT'])
                SNOW = SNOW*self.parameters['par_SFCF']
                
                # Snow
                self.SNOWPACK = self.SNOWPACK+SNOW
                melt = self.parameters['par_CFMAX'] * (self.input_data[self.data_name["temperature"]].sel({self.data_name["dim_time"]:t})-self.parameters['TT'])
                melt = melt.clip(0.0, self.SNOWPACK)
                self.MELTWATER = self.MELTWATER + melt
                self.SNOWPACK = self.SNOWPACK - melt
                refreezing_meltwater = self.parameters['par_CFR'] * self.parameters['par_CFMAX'] * (self.parameters['par_TT'] - self.input_data[self.data_name["temperature"]].sel({self.data_name["dim_time"]:t}))
                refreezing_meltwater = refreezing_meltwater.clip(0.0, self.MELTWATER)
                self.SNOWPACK = self.SNOWPACK + refreezing_meltwater
                self.MELTWATER = self.MELTWATER - refreezing_meltwater
                tosoil = self.MELTWATER - (self.parameters['par_CWH']*self.SNOWPACK)
                tosoil = tosoil.clip(0.0, None)
                self.MELTWATER = self.MELTWATER-tosoil

            else:
                RAIN = self.input_data[self.data_name["precipitation"]].sel({self.data_name["dim_time"]:t}) * self.parameters[self.data_name['par_PCORR']]
                tosoil = 0

            # Soil and evaporation
            soil_wetness = (self.SM / self.parameters[self.data_name['par_FC']]) ** self.parameters[self.data_name['par_BETA']]
            soil_wetness = soil_wetness.clip(0.0, 1.0)        
            recharge = (RAIN) * soil_wetness        
            self.SM = self.SM+RAIN-recharge        
            excess = self.SM-self.parameters[self.data_name['par_FC']]
            excess = excess.clip(0.0, None)       
            self.SM = self.SM-excess
            evapfactor = self.SM / (self.parameters[self.data_name['par_LP']] * self.parameters[self.data_name['par_FC']])        
            evapfactor = evapfactor.clip(0.0, 1.0)
            self.ETact = self.input_data[self.data_name["pot_evapotr"]].sel({self.data_name["dim_time"]:t}) * evapfactor        
            self.ETact = np.minimum(self.SM, self.ETact)
            self.SM = self.SM-self.ETact
            
            # Groundwater boxes
            self.SUZ = self.SUZ + recharge+excess
            PERC = np.minimum(self.SUZ, self.parameters[self.data_name['par_PERC']])
            self.SUZ = self.SUZ - PERC
            Q0 = self.parameters[self.data_name['par_K0']] * np.maximum(self.SUZ-self.parameters[self.data_name['par_UZL']], 0.0)
            self.SUZ = self.SUZ - Q0
            Q1 = self.parameters[self.data_name['par_K1']] * self.SUZ
            self.SUZ = self.SUZ - Q1
            self.SLZ = self.SLZ + PERC
            Q2 = self.parameters[self.data_name['par_K2']] * self.SLZ            
            self.SLZ = self.SLZ-Q2
            
            self.Qhbvl.loc[dict({self.data_name["dim_time"]:t})] = Q0 + Q1 + Q2
        
        return self.Qhbvl.to_dataset(name="Q")


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
    Routing routine simulates the delay and attenuation of runoff as it travels through a basin,
    using a triangular weighting function, where c is the coefficient or weight in each time interval,
    and maxbas is the total number of time steps that define the base length of the triangle.
    Q = Streamflow. Previously, the dimensions of maxbas must be transferred to the Q dataset.
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