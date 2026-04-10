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
GAS_CONSTANT_AIR = 287.05
GAS_CONSTANT_H2O = 462.0


def _coth(x):
    """Hyperbolic cotangent."""
    return np.cosh(x) / np.sinh(x)


def _fit_func(t, t0, p0, p1, p2, p3):
    """Fitting function for the NOC model."""
    return np.where(t < t0, p0, p0 + p1 / (p2 + p3 * _coth(p3 * (t - t0))))


def _convert_inputs(gas_concentration, water_concentration, gas, in_units):

    if in_units == "mol mol-1 dry air":

        water_mole_fraction = water_concentration
        gas_wet_mole_fraction = gas_concentration * (1.0 - water_mole_fraction)

    elif in_units == "mol mol-1 wet air":
        
        water_mole_fraction = water_concentration
        gas_wet_mole_fraction = gas_concentration

    else:
        raise ValueError(f"Unknown in_units: {in_units}. Choose from: mol mol-1 dry air, mol mol-1 wet air")

    molecular_mass_wet_air = (
            (1.0 - water_mole_fraction) * MOLECULAR_MASS['AIR']
            + water_mole_fraction * MOLECULAR_MASS['H2O']
            )

    gas_mass_fraction = MOLECULAR_MASS[gas] / molecular_mass_wet_air * gas_wet_mole_fraction
    water_mass_fraction = MOLECULAR_MASS['H2O'] / molecular_mass_wet_air * water_mole_fraction

    return gas_mass_fraction, water_mass_fraction # [kg kg-1]


def _convert_outputs(gas_mass_fraction, water_concentration, gas, in_units):

    water_mole_fraction = water_concentration
    
    molecular_mass_wet_air = (
            (1.0 - water_mole_fraction) * MOLECULAR_MASS['AIR']
            + water_mole_fraction * MOLECULAR_MASS['H2O']
            )

    gas_wet_mole_fraction = gas_mass_fraction * molecular_mass_wet_air / MOLECULAR_MASS[gas] 

    if in_units == "mol mol-1 dry air":

        gas_concentration = gas_wet_mole_fraction / (1.0 - water_mole_fraction)

    elif in_units == "mol mol-1 wet air":
        
        gas_concentration = gas_wet_mole_fraction

    else:
        raise ValueError(f"Unknown in_units: {in_units}. Choose from: mol mol-1 dry air, mol mol-1 wet air")

    return gas_concentration


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
    gas_concentration : array_like
        Gas concentration in units matching `in_units`.
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
    in_units : {"mol mol-1 dry air", "mol mol-1 wet air"}, default "mol mol-1 dry air"
        Units of `gas_concentration` and `water_concentration`.
    out_units : {"kg m-2 s-1", "mg m-2 h-1"}, default "kg m-2 s-1"
        Output flux units.
    water_concentration : float or array_like, default 0.0
        Water vapor concentration for dilution correction, in units matching `in_units`.

    Returns
    -------
    dict or None
        None if curve fitting fails to converge, otherwise a dict with:

        flux : float
            Calculated soil flux in `out_units`.
        std_dev : float
            Standard deviation of flux in `out_units`.
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
        ambient_mass_fraction : float
            Background gas mass fraction [kg kg-1].
        ambient_gas_concentration : float
            Background gas concentration in `in_units`.
        gas_mass_fraction : array
            Input gas concentration converted to mass fraction [kg kg-1].
        gas_mass_fraction_fit : array
            Fitted gas mass fraction curve [kg kg-1].
        gas_concentration_fit : array
            Fitted gas concentration curve in `in_units`.
    """

    # Test availability
    if gas not in MOLECULAR_MASS:
        raise ValueError(f"Unknown gas: {gas}. Choose from: CH4, CO2, N2O")

    # convert input concentrations to mass fractions 
    gas_mass_fraction, water_mass_fraction = _convert_inputs(
            gas_concentration, water_concentration, gas, in_units
            )

    # Non-dimensionalize
    tbeg, tend = min(time), max(time)
    tdel = tend - tbeg
    cbeg, cend = gas_mass_fraction[0], gas_mass_fraction[-1]
    cdel = cend - cbeg

    phi = (gas_mass_fraction - cbeg) / cdel  # scaled mixing ratio
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

    # Estimators
    t0, c0, F0, L, Dsqr = popt

    # Convert to SI units
    pressure_si = air_pressure * 100.0  # hPa -> Pa
    temperature_si = air_temperature + 273.15  # deg C -> K
    gas_constant = (
            (1.0 - water_mass_fraction) * GAS_CONSTANT_AIR
            + water_mass_fraction * GAS_CONSTANT_H2O
            )
    density = np.mean(
            pressure_si / temperature_si / gas_constant
            )
 
    # Compute flux
    factor = density * headspace_volume / chamber_base_area
    flux_nondim = factor * F0 

    # Redimensionalize to SI units
    ambient_mass_fraction = cbeg + cdel * c0
    gas_mass_fraction_fit = cbeg + cdel * _fit_func(tau, *popt)
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

    # Convert back to input units for concentrations
    ambient_gas_concentration = _convert_outputs(
            ambient_mass_fraction, np.mean(water_concentration), gas, in_units)

    gas_concentration_fit = _convert_outputs(
            gas_mass_fraction_fit, water_concentration, gas, in_units)

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
        "r_squared": r_squared,
        "durbin_watson": durbin_watson,
        "shapiro_wilk_w": shapiro_wilk_w,
        "shapiro_wilk_p": shapiro_wilk_p,
        "first_order_pi": first_order_pi,
        "second_order_pi": second_order_pi,
        "ambient_mass_fraction": ambient_mass_fraction,
        "ambient_gas_concentration": ambient_gas_concentration,
        "gas_mass_fraction": gas_mass_fraction,
        "gas_mass_fraction_fit": gas_mass_fraction_fit,
        "gas_concentration_fit": gas_concentration_fit,
    }
