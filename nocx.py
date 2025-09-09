import numpy as np
from scipy.optimize import curve_fit
from scipy.special import erfc
from scipy.stats import shapiro



def func(t, t0, p0, p1, p2, p3):

    def coth(x):
        """Cotangens Hyperbolicus"""
        return np.cosh(x) / np.sinh(x)

    return np.where(t < t0, p0, p0 + p1 / (p2 + p3 * coth(p3 * (t - t0))))


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
        molecular_mass_gas = 44.013  # molecular weight of N2O [g mol-1]

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
            p0=(0.1, 0.5, 1.0, 1.0, 1.0),
            # bounds=([-0.5,0.0,-1e2,0.0,0.0],[0.5,1.0,1e2,26.0,26.0]),
            check_finite=True,
            nan_policy="omit",
        )
    except RuntimeError as e:
        print(f"warning: Curve_fit failed to converge: {e}")
        return np.nan, np.full_like(time, np.nan)

    c0 = popt[1]
    F0 = popt[2]
    L = popt[3]
    Dsqr = popt[4]
    # print(popt)
    # print(pcov)

    mass_ratio = molecular_mass_gas / molecular_mass_air

    # convert air pressure to SI units and take mean
    air_pressure = np.mean(air_pressure) * 100.0

    # convert air temperature to SI units and take mean
    air_temperature = np.mean(air_temperature) + 273.5

    # air density
    density = air_pressure / air_temperature / gas_constant

    # redimensionalize
    ambient_mixing_ratio = nbeg + deln * c0
    mixing_ratio_fit = nbeg + deln * func(tau, *popt)

    # flux in SI units [kg m-2 s-1]
    factor = density * mass_ratio * headspace_volume / chamber_base_area
    flux_si = factor * F0 * deln / delt

    # standard deviation in SI units [kg m-2 s-1]
    stddev_si = np.abs(flux_si) * np.sqrt(pcov[2, 2] / F0 ** 2)

    # Coefficient of determination R2
    residuals = phi - func(tau, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((phi - np.mean(phi)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    # Durbin Watson test of autocorrelation of the residuals
    diffs = np.diff(residuals)
    durbin_watson = np.sum(diffs**2) / ss_res

    # Shapiro-Wilk test of the normality of the residuals
    stats = shapiro(residuals)
    shapiro_wilk_w = stats[0]
    shapiro_wilk_p = stats[1]

    # Dimensional analysis, Buckingham pi terms
    first_order_pi = L / F0 
    N = (L**2 - Dsqr**2) / F0
    second_order_pi = N / F0

    # determine flux and its standard deviation in desired output units
    if out_units == "kg m-2 s-1":
        flux_out_units = flux_si
        stddev_out_units = stddev_si

    if out_units == "mg m-2 h-1":
        flux_out_units = flux_si * 1e6 * 3600
        stddev_out_units = stddev_si * 1e6 * 3600

    return {
        "flux": flux_out_units,
        "std_dev": stddev_out_units,
        "ambient_mixing_ratio": ambient_mixing_ratio,
        "r_squared": r_squared,
        "durbin_watson": durbin_watson,
        "shapiro_wilk_w": shapiro_wilk_w,
        "shapiro_wilk_p": shapiro_wilk_p,
        "first_order_pi": first_order_pi,
        "second_order_pi": second_order_pi,
        "mixing_ratio_fit": mixing_ratio_fit,
    }
