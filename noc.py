"""
NOC - Non-Steady-State Open Chamber Extended

Soil greenhouse gas flux calculation using the Non-Steady-State Open Flux Chamber method.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import shapiro


# Molecular masses [g mol-1]
MOLECULAR_MASS = {
    "AIR": 28.96,
    "H2O": 18.02,
    "CH4": 16.04,
    "CO2": 44.009,
    "N2O": 44.013,
}

# Specific gas constants [J kg-1 K-1]
GAS_CONSTANT_AIR = 287.052874247
GAS_CONSTANT_H2O = 462.0


def _coth(x):
    """Hyperbolic cotangent."""
    return np.cosh(x) / np.sinh(x)


def _fit_func(t, t0, p0, p1, p2, p3):
    """Fitting function for the NOC model."""
    return np.where(t < t0, p0, p0 + p1 / (p2 + p3 * _coth(p3 * (t - t0))))


def convert_inputs(gas_concentration, water_concentration, gas, in_units):

    if in_units == "mol mol-1 dry air":

        water_mixing_ratio_dry = MOLECULAR_MASS["H2O"] / MOLECULAR_MASS["AIR"] * water_concentration
        mass_mixing_ratio_dry =  MOLECULAR_MASS[gas] / MOLECULAR_MASS["AIR"] * gas_concentration

        water_mixing_ratio = water_mixing_ratio_dry / (1.0 + water_mixing_ratio_dry)
        mass_mixing_ratio = mass_mixing_ratio_dry / (1.0 + water_mixing_ratio_dry)

    elif in_units == "mol mol-1 wet air":

        MOLECULAR_MASS_WETAIR = (1.0 - water_concentration) * MOLECULAR_MASS["AIR"] + water_concentration * MOLECULAR_MASS["H2O"]

        water_mixing_ratio = MOLECULAR_MASS["H2O"] / MOLECULAR_MASS_WETAIR * water_concentration 
        mass_mixing_ratio = MOLECULAR_MASS[gas] / MOLECULAR_MASS_WETAIR * gas_concentration 

    else:
        raise ValueError(f"Unknown in_units: {in_units}. Choose from: mol mol-1 dry air, mol mol-1 wet air")

    return mass_mixing_ratio, water_mixing_ratio # [kg kg-1]


def nondimensionalize(time, mass_mixing_ratio):

    return tau, phi, cbeg, cdel


#def redimensionalize(tau, phi, tbeg, tdel, cbeg, cdel):
def redimensionalize(phi, cbeg, cdel):
    
    mass_mixing_ratio = cbeg + cdel * phi
    #time = tbeg + tdel * tau

    #return time, conc
    return mass_mixing_ratio


def calc_flux(
    time,
    gas_concentration,
    chamber_base_area,
    headspace_volume,
    gas="CO2",
    air_temperature=20.0,
    air_pressure=1010.0,
    in_units="mol mol-1 dry air",
    out_units="kg m-2 s-1",
    water_concentration=0.0,
):
    """
    Compute soil flux using the Non-Steady-State Open Flux Chamber method.

    Parameters
    ----------
    time : array_like
        Timestamps [s].
    gas_mole_fraction : array_like
        Mole fraction of the gas of interest [mol mol-1].
    chamber_base_area : float
        Chamber base area [m2].
    headspace_volume : float
        Chamber headspace volume [m3].
    gas : {"CH4", "CO2", "N2O"}, default "CO2"
        Target gas species.
    air_temperature : float or array_like, default 20.0
        Air temperature [deg C].
    air_pressure : float or array_like, default 1010.0
        Air pressure [hPa].
    out_units : {"kg m-2 s-1", "mg m-2 h-1"}, default "kg m-2 s-1"
        Output flux units.
    water_mixing_ratio : float or array_like, default 0.0
        Water vapor mixing ratio for dilution correction [mol mol-1].

    Returns
    -------
    dict
        flux : float
            Calculated soil flux.
        std_dev : float
            Standard deviation of flux.
        ambient_mixing_ratio : float
            Background gas concentration.
        r_squared : float
            Coefficient of determination.
        durbin_watson : float
            Durbin-Watson autocorrelation statistic.
        shapiro_wilk_w : float
            Shapiro-Wilk test statistic.
        shapiro_wilk_p : float
            Shapiro-Wilk p-value.
        first_order_pi : float
            First-order Buckingham Pi term.
        second_order_pi : float
            Second-order Buckingham Pi term.
        mixing_ratio_fit : array
            Fitted concentration curve.

    References
    ----------
    Water dilution correction: Perez-Priego et al., Plant Soil (2015) 393:229-244
    """
    if gas not in MOLECULAR_MASS:
        raise ValueError(f"Unknown gas: {gas}. Choose from: CH4, CO2, N2O")

    
    mass_mixing_ratio, water_mixing_ratio = convert_inputs(gas_concentration, water_concentration, gas, in_units)

    # Non-dimensionalize
    tbeg, tend = min(time), max(time)
    tdel = tend - tbeg
    cbeg, cend = mass_mixing_ratio[0], mass_mixing_ratio[-1]
    cdel = cend - cbeg

    phi = (mass_mixing_ratio - cbeg) / cdel  # scaled mixing ratio
    tau = (time - tbeg) / tdel  # scaled time


    # Fit model
    try:
        popt, pcov = curve_fit(
            _fit_func,
            tau.squeeze(),
            phi.squeeze(),
            p0=(0.1, 0.5, 1.0, 1.0, 1.0),
            check_finite=True,
            nan_policy="omit",
        )
    except RuntimeError as e:
        print(f"Warning: curve_fit failed to converge: {e}")
        return None

    t0, c0, F0, L, Dsqr = popt

    # Convert to SI units
    pressure_si = np.mean(air_pressure) * 100.0  # hPa -> Pa
    temperature_si = np.mean(air_temperature) + 273.15  # deg C -> K
    GAS_CONSTANT = (1.0 - np.mean(water_mixing_ratio)) * GAS_CONSTANT_AIR + GAS_CONSTANT_H2O * np.mean(water_mixing_ratio)
    density = pressure_si / temperature_si / GAS_CONSTANT

    # Compute flux [kg m-2 s-1]
    factor = density * headspace_volume / chamber_base_area
    flux_nondim = factor * F0 

    # Redimensionalize 
    ambient_mass_mixing_ratio = cbeg + cdel * c0
    mass_mixing_ratio_fit = cbeg + cdel * _fit_func(tau, *popt)
    flux_si = flux_nondim * cdel / tdel
    stddev_si = np.abs(flux_si) * np.sqrt(pcov[2, 2] / F0**2)

    # Statistics
    residuals = phi - _fit_func(tau, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((phi - np.mean(phi))**2)
    r_squared = 1 - (ss_res / ss_tot)

    diffs = np.diff(residuals, n=1, axis=0)
    durbin_watson = np.sum(diffs**2) / ss_res

    sw_stat = shapiro(residuals)
    shapiro_wilk_w, shapiro_wilk_p = sw_stat.statistic, sw_stat.pvalue

    # Buckingham Pi terms
    first_order_pi = L / F0
    second_order_pi = (L**2 - Dsqr**2) / F0**2

    # Convert to output units
    if out_units == "kg m-2 s-1":
        flux, std_dev = flux_si, stddev_si
    elif out_units == "mg m-2 h-1":
        flux = flux_si * 1e6 * 3600
        std_dev = stddev_si * 1e6 * 3600
    else:
        raise ValueError(f"Unknown out_units: {out_units}")

    return {
        "flux": flux,
        "std_dev": std_dev,
        "ambient_mass_mixing_ratio": ambient_mass_mixing_ratio,
        "r_squared": r_squared,
        "durbin_watson": durbin_watson,
        "shapiro_wilk_w": shapiro_wilk_w,
        "shapiro_wilk_p": shapiro_wilk_p,
        "first_order_pi": first_order_pi,
        "second_order_pi": second_order_pi,
        "mass_mixing_ratio": mass_mixing_ratio,
        "mass_mixing_ratio_fit": mass_mixing_ratio_fit,
    }
