import numpy as np
from scipy.optimize import curve_fit


def func(t, t0, p0, p1, p2):
    return np.where(t < t0, p0, p0 + p1 * (1.0 - np.exp(-p2 * (t - t0))))


def calc_flux(
    time,
    mixing_ratio,
    chamber_base_area,
    headspace_volume,
    gas="CO2",
    air_temperature=20.0,
    air_pressure=1010.0,
    in_units="mol mol-1",
    out_units="kg m-2 s-1",
    water_mixing_ratio=0.0,
):
    """
    Computes the soil flux based on the Non-steady-state Open Flux Chamber Method

    Parameters:
        time: array(float)
            1D array of timestamps [s]
        mixing_ratio: array(float)
            1D array of Mole mixing ratio at every timestamp [mol mol-1]
        air_pressure: scalar(float)
            Air pressure [hPa]
    """

    gas_constant = 287.052874247  # specific gas constant of dry air [J kg-1 K-1]
    molecular_mass_air = 28.96  # molecular weight of dry air [g mol-1]

    if gas == "CH4":
        molecular_mass_gas = 16.04  # molecular weight of CH4 [g mol-1]
    if gas == "CO2":
        molecular_mass_gas = 44.009  # molecular weight of CO2 [g mol-1]
    if gas == "N2O":
        molecular_mass_gas = 44.013 # molecular weight of N2O [g mol-1]

    # water dilution correction
    # according to Pérez-Priego et al., Plant Soil (2015) 393:229-244
    mixing_ratio_corr = mixing_ratio / (1.0 - water_mixing_ratio)

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

    try:
        popt, pcov = curve_fit(
                func, 
                tau.squeeze(), 
                phi.squeeze(), 
                p0=(0.0, 0.0, 1.0, 1.0), 
                check_finite=True,
		nan_policy='omit'
                )
    except RuntimeError as e:
        print(f"warning: Curve_fit failed to converge: {e}")
        return np.nan, np.full_like(time, np.nan)

    # print(popt)
    # print(pcov)

    mass_ratio = molecular_mass_gas / molecular_mass_air

    # convert air pressure to SI units and take mean
    air_pressure  = np.mean(air_pressure) * 100.0

    # convert air temperature to SI units and take mean
    air_temperature  = np.mean(air_temperature) + 273.5 

    # air density
    density = air_pressure / air_temperature / gas_constant

    # flux in SI units [kg m-2 s-1]
    factor = density * mass_ratio * headspace_volume / chamber_base_area * deln / delt
    flux_si = factor * popt[2] * popt[3]

    # standard deviation in SI units [kg m-2 s-1] by propagation of uncertainty
    stddev_si = np.abs(flux_si) * np.sqrt(
        + pcov[2, 2] / popt[2] ** 2
        + pcov[3, 3] / popt[3] ** 2
        + 2.0 * pcov[2, 3] / popt[2] / popt[3]
    )

    # Coefficient of determination R2
    residuals = phi - func(tau, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((phi-np.mean(phi))**2)
    r_squared = 1 - (ss_res / ss_tot)

    # determine flux and its standard deviation in desired output units
    if out_units == "kg m-2 s-1":
        flux_out_units = flux_si
        stddev_out_units = stddev_si

    if out_units == "mg m-2 h-1":
        flux_out_units = flux_si * 1e6 * 3600
        stddev_out_units = stddev_si * 1e6 * 3600


    # redimensionalize
    ambient_mixing_ratio = nbeg + deln * popt[1]
    mixing_ratio_fit = nbeg + deln * func(tau, *popt)
    dilution_rate = popt[3] / delt

    return {
            "flux" : flux_out_units, 
            "std_dev" : stddev_out_units, 
            "ambient_mixing_ratio" : ambient_mixing_ratio, 
            "r_squared" : r_squared,
            "dilution_rate" : dilution_rate, 
            "mixing_ratio_fit" : mixing_ratio_fit 
    }


