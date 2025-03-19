import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams['text.usetex'] = True

def func(t, t0, p0, p1, p2):
    return np.where(t<t0, p0, p0 + p1 * (1.0-np.exp(-p2*(t-t0))))

def calc_flux(
        time, 
        mixing_ratio, 
        chamber_base_area, 
        headspace_volume, 
        gas = "CO2",
        air_temperature = 300.0, 
        air_pressure = 1010.0,
        in_units = "mol mol-1",
        out_units = "kg m-2 s-1",
        water_mixing_ratio = 0.0
        ):
    """
    Computes the soil flux based on the Vented Flux Chamber Method

    Parameters:
        time: array(float)
            1D array of timestamps [s]
        mixing_ratio: array(float)
            1D array of Mole mixing ratio at every timestamp [mol mol-1]
        air_pressure: scalar(float)
            Air pressure [hPa]
    """

    gas_constant       = 287.052874247 # specific gas constant of dry air [J kg-1 K-1]
    molecular_mass_air = 28.96 # molecular weight of air [g mol-1]
    
    if gas == "CH4":
        molecular_mass_gas = 16.04 # molecular weight of CH4 [g mol-1]
    if gas == "CO2":
        molecular_mass_gas = 44.009 # molecular weight of CO2 [g mol-1]
    
    tbeg = min(time)
    tend = max(time) 
    delt = tend - tbeg

    nbeg = mixing_ratio[0]
    nend = mixing_ratio[-1]
    deln = nend - nbeg

    # scaled mole mixing ratio
    phi = (mixing_ratio - nbeg) / deln

    # scaled time
    tau = (time - tbeg) / delt
    
    # run regression
    popt, pcov = curve_fit(
            func, 
            tau, 
            phi, 
            p0=(0.0, 0.0, 1.0, 1.0))
    # print(popt)
    # print(pcov)

    mass_ratio   = molecular_mass_gas / molecular_mass_air

    # air density
    density = air_pressure * 100.0 / air_temperature / gas_constant

    # flux in SI units [kg m-2 s-1] 
    flux_si = density * mass_ratio * headspace_volume / chamber_base_area * deln / delt * popt[2] * popt[3]

    # determine flux in desired output units
    if out_units == "kg m-2 s-1":
        flux_out_units = flux_si
    if out_units == "mg m-2 h-1":
        flux_out_units = flux_si * 1e6 * 3600 

    # unscale 
    mixing_ratio_fit = nbeg + deln * func(tau, *popt)

    return flux_out_units, mixing_ratio_fit


if __name__ == "__main__":


    headspace_volume  = 0.017207387 # [m3] chamber volume for /Portable/TVC2023_2023_7_11_09_45_5_2_1.csv  
    chamber_base_area = 0.047143525 # [m2] chamber base area
    
    flist = (
        "./fixed/TVC2022__2022_07_25_18_37_16_2_4.csv", # good counter example where it wont work
        "./fixed/TVC2024_2024_6_29_10_52_3_5_2.csv",
        # "./fixed/TVC2024_2024_6_29_14_58_28_5_7.csv", # really bad data
        "./fixed/TVC2024_2024_6_29_15_04_2_5_7.csv",
        "./Portable/TVC2023_2023_7_11_09_35_16_2_1.csv", # bad data
        "./Portable/TVC2023_2023_7_11_09_45_5_2_1.csv", # best 
        "./Portable/TVC2023_2023_7_11_10_18_5_2_2.csv",
        "./Portable/TVC2024_2024_6_27_10_34_9_L1_1.csv", # bad data
        "./Portable/TVC2024_2024_6_27_12_13_7_L2_1.csv" # negative flux
            )
    

    for f in flist:
    
        df = pd.read_csv(f, skiprows=[0,2,3], na_values='NA')
        
        # compose time from dataframe
        time = df.Minute_7810*60.0 + df.Second_7810 
        
        # read mole mixing ratio of CH4 [ppb mol mol-1] from dataframe
        mixing_ratio = df.CH4_7810.to_numpy() * 1e-9
    
        # call flux calculator
        flux, mixing_ratio_fit = calc_flux(
                        time, 
                        mixing_ratio, 
                        chamber_base_area, 
                        headspace_volume, 
                        gas = "CH4",
                        out_units = "mg m-2 h-1")
    
        # plotting
        plt.plot(time, mixing_ratio_fit*1e9, 'g--')
        plt.plot(time, mixing_ratio * 1e9)
        plt.title(f+"\n Flux = %.4f mg CH4 m-2 h-1" % flux)
        plt.xlabel("time [s]")
        plt.ylabel("CH4 mixing ratio [ppb]")
        plt.savefig(f+".png")
        plt.show()
    
    
    
