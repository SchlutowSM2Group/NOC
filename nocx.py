"""
NOCX - Non-Steady-State Open Chamber Extended

Soil greenhouse gas flux calculation using the Non-Steady-State Open Flux Chamber method.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import shapiro


# Molecular masses [g mol-1]
MOLECULAR_MASS = {
    "air": 28.96,
    "CH4": 16.04,
    "CO2": 44.009,
    "N2O": 44.013,
}

# Specific gas constant of dry air [J kg-1 K-1]
GAS_CONSTANT_AIR = 287.052874247


def _coth(x):
    """Hyperbolic cotangent."""
    return np.cosh(x) / np.sinh(x)


def _fit_func(t, t0, p0, p1, p2, p3):
    """Fitting function for the NOC model."""
    return np.where(t < t0, p0, p0 + p1 / (p2 + p3 * _coth(p3 * (t - t0))))


def calc_flux(
    time,
    mixing_ratio,
    chamber_base_area,
    headspace_volume,
    gas="CO2",
    air_temperature=20.0,
    air_pressure=1010.0,
    out_units="kg m-2 s-1",
    water_mixing_ratio=0.0,
):
    """
    Compute soil flux using the Non-Steady-State Open Flux Chamber method.

    Parameters
    ----------
    time : array_like
        Timestamps [s].
    mixing_ratio : array_like
        Mole mixing ratio [mol mol-1].
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

    molecular_mass_gas = MOLECULAR_MASS[gas]
    molecular_mass_air = MOLECULAR_MASS["air"]

    # Water dilution correction (Perez-Priego et al., Plant Soil 2015)
    mixing_ratio_corr = mixing_ratio / (1.0 - water_mixing_ratio)

    # Scale time and mixing ratio to [0, 1]
    tbeg, tend = min(time), max(time)
    delt = tend - tbeg
    nbeg, nend = mixing_ratio[0], mixing_ratio[-1]
    deln = nend - nbeg

    phi = (mixing_ratio - nbeg) / deln  # scaled mixing ratio
    tau = (time - tbeg) / delt  # scaled time

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
    mass_ratio = molecular_mass_gas / molecular_mass_air

    # Convert to SI units
    pressure_si = np.mean(air_pressure) * 100.0  # hPa -> Pa
    temperature_si = np.mean(air_temperature) + 273.15  # deg C -> K
    density = pressure_si / temperature_si / GAS_CONSTANT_AIR

    # Compute flux [kg m-2 s-1]
    factor = density * mass_ratio * headspace_volume / chamber_base_area
    flux_si = factor * F0 * deln / delt
    stddev_si = np.abs(flux_si) * np.sqrt(pcov[2, 2] / F0**2)

    # Redimensionalize fitted curve
    ambient_mixing_ratio = nbeg + deln * c0
    mixing_ratio_fit = nbeg + deln * _fit_func(tau, *popt)

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
        "ambient_mixing_ratio": ambient_mixing_ratio,
        "r_squared": r_squared,
        "durbin_watson": durbin_watson,
        "shapiro_wilk_w": shapiro_wilk_w,
        "shapiro_wilk_p": shapiro_wilk_p,
        "first_order_pi": first_order_pi,
        "second_order_pi": second_order_pi,
        "mixing_ratio_fit": mixing_ratio_fit,
    }
