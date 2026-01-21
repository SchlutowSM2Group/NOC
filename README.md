# NOC - Non-Steady-State Open Chamber

Python library for calculating soil greenhouse gas flux using the Non-Steady-State Open Flux Chamber method. To be used in python or R.

## Installation

```bash
pip install numpy scipy pandas matplotlib
```

## Usage

```python
import nocx as noc

results = noc.calc_flux(
    time,                    # timestamps [s]
    mixing_ratio,            # mole mixing ratio [mol mol-1]
    chamber_base_area,       # [m2]
    headspace_volume,        # [m3]
    gas="CH4",               # "CH4", "CO2", or "N2O"
    air_temperature=20.0,    # [deg C]
    air_pressure=1010.0,     # [hPa]
    out_units="mg m-2 h-1",
)

print(f"Flux: {results['flux']:.6f} +/- {results['std_dev']:.6f} mg m-2 h-1")
```

## Return Values

| Key | Description |
|-----|-------------|
| `flux` | Calculated soil flux |
| `std_dev` | Standard deviation |
| `ambient_mixing_ratio` | Background gas concentration |
| `r_squared` | Coefficient of determination |
| `durbin_watson` | Durbin-Watson statistic |
| `shapiro_wilk_w` | Shapiro-Wilk W statistic |
| `shapiro_wilk_p` | Shapiro-Wilk p-value |
| `first_order_pi` | First-order Buckingham Pi term |
| `second_order_pi` | Second-order Buckingham Pi term |
| `mixing_ratio_fit` | Fitted concentration curve |

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
