"""
NOC - Non-Steady-State Open Chamber

Soil greenhouse gas flux calculation using the Non-Steady-State Open
Flux Chamber method.

Copyright (C) 2025-2026 Mark Schlutow

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Author:  Mark Schlutow <mschlutow@bgc-jena.mpg.de>
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import shapiro

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
    window_offset=10.0,
    window_slope=1.5
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
    window_offset : float, default 10.0
        Offset for optimal fitting window size [s]
    window_slope : float, default 1.5
        Slope for optimal fitting window size

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
        residual_autocorr : float
            Lag-1 residual autocorrelation (rho ~ 1 - durbin_watson / 2), clipped
            to [0, 0.99]; used to inflate std_dev for autocorrelated noise.
        aic : float
            Akaike Information Criterion of the fit.
        first_order_pi : float
            First-order Buckingham Pi term.
        second_order_pi : float
            Second-order Buckingham Pi term.
        onset_time : float
            Onset time (lag before the relaxation begins) [s].
        transience_time : float
            Transience (relaxation) time scale [s]; inf if not resolved within the record.
        ambient_mass_fraction : float
            Background gas mass fraction [kg kg-1].
        ambient_gas_concentration : float
            Background gas concentration in `in_units`.
        gas_mass_concentration : array
            Input gas concentration converted to mass concentration [kg m-3].
        gas_mass_fraction_fit : array
            Fitted gas mass fraction curve [kg kg-1].
        gas_concentration_fit : array
            Fitted gas concentration curve in `in_units`.
        gas_mass_concentration_fit : array
            Fitted gas mass concentration curve [kg m-3].
    """

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

    # Minimum lag-1 residual autocorrelation below which the noise is treated as
    # white and the flux uncertainty is left uncorrected.
    RHO_MIN = 0.05
    
    
    ###################################################################
    ###               Auxiliary Functions                           ###
    ###################################################################
    def _fit_func(t, t0, p0, p1, p2, p3):
        """
        Fitting function for the NOC model.
        """
        u = np.maximum(t - t0, 0.0)
        g = np.tanh(p3 * u)
        return np.where(t < t0, p0, p0 + p1 * g / (p2 * g + p3))
    
    def _density(air_pressure, air_temperature, water_mass_fraction):
    
        # Convert to SI units
        pressure_si = air_pressure * 100.0  # hPa -> Pa
        temperature_si = air_temperature + 273.15  # deg C -> K
        gas_constant = (
                (1.0 - water_mass_fraction) * GAS_CONSTANT_AIR
                + water_mass_fraction * GAS_CONSTANT_H2O
                )
        total_air_density = pressure_si / temperature_si / gas_constant
    
        dry_air_density = total_air_density * (1.0 - water_mass_fraction)
    
        return total_air_density, dry_air_density
    
    
    def _convert_inputs(gas_concentration, water_concentration, gas, in_units):
    
        if in_units == "mol mol-1 dry air":
    
            water_mole_fraction = water_concentration
            gas_dry_mole_fraction = gas_concentration
            gas_wet_mole_fraction = gas_dry_mole_fraction * (1.0 - water_mole_fraction)
    
        elif in_units == "mol mol-1 wet air":
            
            water_mole_fraction = water_concentration
            gas_wet_mole_fraction = gas_concentration
            gas_dry_mole_fraction = gas_wet_mole_fraction / (1.0 - water_mole_fraction)
    
        else:
            raise ValueError(f"Unknown in_units: {in_units}. Choose from: mol mol-1 dry air, mol mol-1 wet air")
    
        molecular_mass_wet_air = (
                (1.0 - water_mole_fraction) * MOLECULAR_MASS['AIR']
                + water_mole_fraction * MOLECULAR_MASS['H2O']
                )
    
        water_mass_fraction = MOLECULAR_MASS['H2O'] / molecular_mass_wet_air * water_mole_fraction
        gas_dry_mass_fraction = MOLECULAR_MASS[gas] /  MOLECULAR_MASS['AIR'] * gas_dry_mole_fraction
        gas_wet_mass_fraction = MOLECULAR_MASS[gas] / molecular_mass_wet_air * gas_wet_mole_fraction
    
        return gas_dry_mass_fraction, gas_wet_mass_fraction, water_mass_fraction # [kg kg-1]
    
    
    def _convert_outputs(gas_mass_fraction, water_concentration, gas, in_units, wet=True):
    
        water_mole_fraction = water_concentration
        
        molecular_mass_wet_air = (
                (1.0 - water_mole_fraction) * MOLECULAR_MASS['AIR']
                + water_mole_fraction * MOLECULAR_MASS['H2O']
                )
    
        if wet:
            gas_wet_mass_fraction = gas_mass_fraction
            gas_wet_mole_fraction = gas_wet_mass_fraction * molecular_mass_wet_air / MOLECULAR_MASS[gas] 
        else:
            gas_dry_mass_fraction = gas_mass_fraction
            gas_dry_mole_fraction = gas_dry_mass_fraction * MOLECULAR_MASS["AIR"] / MOLECULAR_MASS[gas] 
    
        if in_units == "mol mol-1 dry air":
            if wet:
                gas_concentration = gas_wet_mole_fraction / (1.0 - water_mole_fraction)
            else:
                gas_concentration = gas_dry_mole_fraction
    
        elif in_units == "mol mol-1 wet air":
            if wet:
                gas_concentration = gas_wet_mole_fraction
            else:
                gas_concentration = gas_dry_mole_fraction * (1.0 - water_mole_fraction)
    
        else:
            raise ValueError(f"Unknown in_units: {in_units}. Choose from: mol mol-1 dry air, mol mol-1 wet air")
    
        return gas_concentration # in SI units


    def _fit(tau_data, phi_data):
        return curve_fit(
            _fit_func,
            tau_data.squeeze(),
            phi_data.squeeze(),
            p0=(0.1, 0.5, 1.0, 1.0, 1.0),
            bounds=((-1.0, -np.inf, -np.inf, -np.inf, 0.0), (1.0, np.inf, np.inf, np.inf, np.inf)),
            check_finite=True,
            nan_policy="omit",
            method="trf"
        )


    ###################################################################
    ###               Converting                                    ###
    ###################################################################

    # Test availability
    if gas not in MOLECULAR_MASS:
        raise ValueError(f"Unknown gas: {gas}. Choose from: CH4, CO2, N2O")

    # convert input concentrations to mass fractions 
    gas_dry_mass_fraction, gas_wet_mass_fraction, water_mass_fraction = _convert_inputs(
            gas_concentration, water_concentration, gas, in_units
            )

    total_air_density, dry_air_density = _density(air_pressure, air_temperature, water_mass_fraction)

    # Choose model density
    # density = total_air_density
    # density = dry_air_density
    density = np.mean(dry_air_density)
    # density = np.mean(total_air_density)

    # Non-dimensionalize time
    tbeg, tend = min(time), max(time)
    tdel = tend - tbeg

    # Definition of prognostic variable (gas_mass_concentration)
    # c = gas_dry_mass_fraction
    # c = gas_wet_mass_fraction
    # c = density * gas_wet_mass_fraction # gas mass concentration
    c = density * gas_dry_mass_fraction

    # Non-dimensionalize progostic variable
    #cbeg, cend = c[0], c[-1]
    cmax, cmin = max(c), min(c)

    #cdel = cend - cbeg
    cdel = cmax - cmin


    #phi = (c - cbeg) / cdel  # scaled mixing ratio
    phi = (c - cmin) / cdel  # scaled mixing ratio
    tau = (time - tbeg) / tdel  # scaled time


    ###################################################################
    ###         Fit model                                           ###
    ###################################################################
    try:
        # First pass: fit the full record to estimate the timescales
        popt, pcov = _fit(tau, phi)
        t0, c0, F0, L, Dsqr = popt

        # Empirically optimal fitting window: onset + 1.5 * transience + 10 s
        onset_time = t0 * tdel
        transience_time = np.inf if Dsqr == 0.0 else 1.0 / Dsqr * tdel

        window_end = onset_time + window_slope * transience_time + window_offset  # [s] measured from tbeg

        # Second pass: refit within the optimal window (unless it spans the whole record)
        window = ((time - tbeg) <= window_end).squeeze()
        if np.isfinite(window_end) and not np.all(window) and np.count_nonzero(window) >= 5:
            tau_fit, phi_fit = tau[window], phi[window]
            popt, pcov = _fit(tau_fit, phi_fit)
            t0, c0, F0, L, Dsqr = popt
        else:
            tau_fit, phi_fit = tau, phi
    except RuntimeError as e:
         print(f"Warning: curve_fit failed to converge: {e}")
         return None
  
    ###################################################################
    ###         Redimensionalize to SI units
    ###################################################################

    # factor = density * headspace_volume / chamber_base_area  * cdel / tdel
    factor = headspace_volume / chamber_base_area  * cdel / tdel
    flux_si = factor * F0
    # ambient_mass_fraction = cbeg + cdel * c0
    # gas_mass_fraction_fit = cbeg + cdel * _fit_func(tau, *popt)
    #ambient_mass_fraction = (cbeg + cdel * c0) / np.mean(density)
    #gas_mass_concentration_fit = (cbeg + cdel * _fit_func(tau, *popt))
    ambient_mass_fraction = (cmin + cdel * c0) / np.mean(density)
    gas_mass_concentration_fit = (cmin + cdel * _fit_func(tau, *popt))
    gas_mass_fraction_fit = gas_mass_concentration_fit / density
    stddev_si = np.abs(flux_si) * np.sqrt(pcov[2, 2] / F0**2)
    transience_time = 1.0 / Dsqr * tdel
    if transience_time > 10 * tdel:
        transience_time = np.inf
    onset_time = t0 * tdel

    ###################################################################
    ###          Statistics
    ###################################################################

    residuals = phi_fit - _fit_func(tau_fit, *popt)
    rss = np.sum(residuals**2)
    ss_tot = np.sum((phi_fit - np.mean(phi_fit))**2)
    r_squared = 1 - (rss / ss_tot)

    # Durbin Watson test for autocorrelation of the residuals
    diffs = np.diff(residuals, n=1, axis=0)
    durbin_watson = np.sum(diffs**2) / rss

    # Correct the flux uncertainty for slowly-varying, autocorrelated noise
    # (drifting background + wind). The flux estimate is left unchanged; only its
    # standard deviation is inflated by the effective-sample-size factor for AR(1)
    # noise, sqrt((1 + rho) / (1 - rho)). The lag-1 residual autocorrelation is
    # read straight from the Durbin-Watson statistic, rho ~ 1 - durbin_watson / 2.
    # Skipped when the noise is effectively white (rho < RHO_MIN).
    residual_autocorr = 1.0 - durbin_watson / 2.0
    residual_autocorr = min(max(float(residual_autocorr), 0.0), 0.99)
    if residual_autocorr >= RHO_MIN:
        stddev_si *= np.sqrt((1.0 + residual_autocorr) / (1.0 - residual_autocorr))

    # Shapiro Wilk test for normality of the residuals
    sw_stat = shapiro(residuals)
    shapiro_wilk_w, shapiro_wilk_p = sw_stat.statistic, sw_stat.pvalue

    # Akaike Information Criterion for least square method
    n = len(phi_fit)
    p = 5 # number of estimated parameters plus
    aic = n * np.log(rss / n) + 2 * (p + 1)

    # Buckingham Pi terms
    first_order_pi = L / F0
    second_order_pi = (L**2 - Dsqr**2) / F0**2

    ###################################################################
    ###       Convert back to input units for concentrations
    ###################################################################

    ambient_gas_concentration = _convert_outputs(
            ambient_mass_fraction, np.mean(water_concentration), gas, in_units, wet=False)

    gas_concentration_fit = _convert_outputs(
            gas_mass_fraction_fit, water_concentration, gas, in_units, wet=False)

    ###################################################################
    ###       Convert flux to selected output units
    ###################################################################

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
        "residual_autocorr": residual_autocorr,
        "aic": aic,
        "first_order_pi": first_order_pi,
        "second_order_pi": second_order_pi,
        "onset_time": onset_time,
        "transience_time": transience_time,
        "ambient_mass_fraction": ambient_mass_fraction,
        "ambient_gas_concentration": ambient_gas_concentration,
        "gas_mass_concentration": c,
        "gas_mass_fraction_fit": gas_mass_fraction_fit,
        "gas_concentration_fit": gas_concentration_fit,
        "gas_mass_concentration_fit": gas_mass_concentration_fit,
    }
