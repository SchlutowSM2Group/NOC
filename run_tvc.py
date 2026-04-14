import matplotlib.pyplot as plt
import pandas as pd
import noc

plt.rcParams["text.usetex"] = True



headspace_volume = 0.017207387  # [m3] chamber volume for /Portable/TVC2023_2023_7_11_09_45_5_2_1.csv
chamber_base_area = 0.047143525  # [m2] chamber base area

flist = (
    "./data/TVC2022__2022_07_25_18_37_16_2_4.csv",  # good counter example where it wont work
    "./data/TVC2024_2024_6_29_10_52_3_5_2.csv",
    "./data/TVC2024_2024_6_29_14_58_28_5_7.csv",  # really bad data
    "./data/TVC2024_2024_6_29_15_04_2_5_7.csv",
    "./data/TVC2023_2023_7_11_09_35_16_2_1.csv",  # bad data
    "./data/TVC2023_2023_7_11_09_45_5_2_1.csv",  # best
    "./data/TVC2023_2023_7_11_10_18_5_2_2.csv",
    "./data/TVC2024_2024_6_27_10_34_9_L1_1.csv",  # bad data
    "./data/TVC2024_2024_6_27_12_13_7_L2_1.csv",  # negative flux
)

for f in flist:

    df = pd.read_csv(f, skiprows=[0, 2, 3], na_values="NA")

    # compose time from dataframe
    time = df.Minute_7810 * 60.0 + df.Second_7810

    # read mole mixing ratio of CH4 [ppb mol mol-1] from dataframe
    gas_mole_fraction = df.CH4_7810.to_numpy() * 1e-9
    # H2O mixing ratio in ppm
    water_mole_fraction = df.H2O_7810.to_numpy() * 1e-6
    # Air temperature in deg C
    air_temperature = df.AirTemp.to_numpy() 
    # Air pressure in hPa
    air_pressure = df.AirPres.to_numpy()

    # call flux calculator
    results = noc.calc_flux(
        time,
        gas_mole_fraction,
        chamber_base_area,
        headspace_volume,
        water_concentration=water_mole_fraction,
        air_temperature=air_temperature,
        air_pressure=air_pressure,
        gas="CH4",
        out_units="mg m-2 h-1",
    )


    if results is not None:
        print('Flux calculation')
        print('----------------')
        print("flux          = %.8f mg m-2 h-1"% results['flux'])
        print("std deviation = %.8f mg m-2 h-1"% results['std_dev'])
        print()
    
        print('Statistics')
        print('----------')
        print("R squared      = %.4f "% results['r_squared'])
        print("durbin watson  = %.2f "% results['durbin_watson'])
        print("shapiro wilk w = %.2f "% results['shapiro_wilk_w'])
        print("shapiro wilk p = %.2e "% results['shapiro_wilk_p'])
        print()
    
        ambient_gas_concentration = results['ambient_gas_concentration'] * 1e9
        print('ambient mole fraction = %.0f ppb'% ambient_gas_concentration)
        print()
    
        print('Dimensional analysis, Buckingham Pi Terms')
        print('-----------------------------------------')
        print('first-order pi  = %.3f'% results['first_order_pi'])
        print('second-order pi = %.3f'% results['second_order_pi'])
        print()
    
        # plotting
        plt.plot(time, results['gas_concentration_fit'] * 1e9, "g--")
        plt.plot(time, gas_mole_fraction * 1e9,'.')
        plt.title(f + "\n Flux = %.6f $\\pm$ %.6f mg CH4 m-2 h-1" %(results['flux'], results['std_dev']))
        plt.xlabel("time [s]")
        plt.ylabel("CH4 mole fraction [ppb]")
        plt.savefig(f + ".png")
        plt.show()

        plt.plot(time, results['gas_mass_concentration'], '.')
        plt.plot(time, results['gas_mass_concentration_fit'], '-')
        plt.xlabel("time [s]")
        plt.ylabel("gas mass concetration [kg m-3]")
        plt.show()

        plt.plot(time, results['total_density'], '.')
        plt.xlabel("time [s]")
        plt.ylabel("total air density [kg m-3]")
        plt.show()
