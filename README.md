# NOC - Non-Steady-State Open Chamber

Library for calculating soil greenhouse gas flux using the Non-Steady-State Open Flux Chamber method. Can be used in Python or R.

## Installation

### Python

```bash
pip install numpy scipy pandas matplotlib
```

### R

```r
install.packages("reticulate")
```

## Usage

### Python

```python
import noc

results = noc.calc_flux(
    time,                              # timestamps [s]
    gas_concentration,                 # gas concentration (see in_units)
    chamber_base_area,                 # [m2]
    headspace_volume,                  # [m3]
    gas="CH4",                         # "CH4", "CO2", or "N2O"
    air_temperature=20.0,              # [deg C]
    air_pressure=1010.0,               # [hPa]
    in_units="mol mol-1 dry air",      # "mol mol-1 dry air" or "mol mol-1 wet air"
    out_units="mg m-2 h-1",            # "kg m-2 s-1" or "mg m-2 h-1"
    water_concentration=0.0,           # water vapor concentration (see in_units)
)
```

### R

```r
library(reticulate)
source_python('noc.py')

results <- calc_flux(
    time,                              # timestamps [s]
    gas_concentration,                 # gas concentration (see in_units)
    chamber_base_area,                 # [m2]
    headspace_volume,                  # [m3]
    gas="CH4",                         # "CH4", "CO2", or "N2O"
    air_temperature=20.0,              # [deg C]
    air_pressure=1010.0,               # [hPa]
    in_units="mol mol-1 dry air",      # "mol mol-1 dry air" or "mol mol-1 wet air"
    out_units="mg m-2 h-1",            # "kg m-2 s-1" or "mg m-2 h-1"
    water_concentration=0.0            # water vapor concentration (see in_units)
)
```

## Return Values

Returns `None` if curve fitting fails to converge, otherwise a dict with:

| Key | Description |
|-----|-------------|
| `flux` | Calculated soil flux in `out_units` |
| `std_dev` | Standard deviation of flux in `out_units` |
| `r_squared` | Coefficient of determination |
| `durbin_watson` | Durbin-Watson autocorrelation statistic |
| `shapiro_wilk_w` | Shapiro-Wilk W statistic |
| `shapiro_wilk_p` | Shapiro-Wilk p-value |
| `first_order_pi` | First-order Buckingham Pi term |
| `second_order_pi` | Second-order Buckingham Pi term |
| `ambient_mass_fraction` | Background gas mass fraction [kg kg-1] |
| `ambient_gas_concentration` | Background gas concentration in `in_units` |
| `gas_mass_fraction` | Input gas concentration converted to mass fraction [kg kg-1] |
| `gas_mass_fraction_fit` | Fitted gas mass fraction curve [kg kg-1] |
| `gas_concentration_fit` | Fitted gas concentration curve in `in_units` |

## Tutorial

Run the example scripts with sample data:

```bash
python run_tvc.py
```

```bash
Rscript run_tvc.R
```

## License

GPL-3.0
