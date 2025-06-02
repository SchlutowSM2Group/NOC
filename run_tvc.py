import matplotlib.pyplot as plt
import pandas as pd
import noc

plt.rcParams["text.usetex"] = True



headspace_volume = 0.017207387  # [m3] chamber volume for /Portable/TVC2023_2023_7_11_09_45_5_2_1.csv
chamber_base_area = 0.047143525  # [m2] chamber base area

flist = (
    #"./fixed/TVC2022__2022_07_25_18_37_16_2_4.csv",  # good counter example where it wont work
    #"./fixed/TVC2024_2024_6_29_10_52_3_5_2.csv",
    #"./fixed/TVC2024_2024_6_29_14_58_28_5_7.csv",  # really bad data
    #"./fixed/TVC2024_2024_6_29_15_04_2_5_7.csv",
    #"./Portable/TVC2023_2023_7_11_09_35_16_2_1.csv",  # bad data
    "./Portable/TVC2023_2023_7_11_09_45_5_2_1.csv",  # best
    #"./Portable/TVC2023_2023_7_11_10_18_5_2_2.csv",
    #"./Portable/TVC2024_2024_6_27_10_34_9_L1_1.csv",  # bad data
    #"./Portable/TVC2024_2024_6_27_12_13_7_L2_1.csv",  # negative flux
)

for f in flist:

    df = pd.read_csv(f, skiprows=[0, 2, 3], na_values="NA")

    # compose time from dataframe
    time = df.Minute_7810 * 60.0 + df.Second_7810

    # read mole mixing ratio of CH4 [ppb mol mol-1] from dataframe
    mixing_ratio = df.CH4_7810.to_numpy() * 1e-9

    # call flux calculator
    results = noc.calc_flux(
        time,
        mixing_ratio,
        chamber_base_area,
        headspace_volume,
        gas="CH4",
        out_units="mg m-2 h-1",
    )


    print("flux          = %.8f mg m-2 h-1"% results['flux'])
    print("std deviation = %.8f mg m-2 h-1"% results['std_dev'])
    print("R squared     = %.4f "% results['r_squared'])
    print()

    ambient_mixing_ratio = results['ambient_mixing_ratio']*1e9
    print('ambient mixing ratio   = %.0f ppb'% ambient_mixing_ratio)
    print('dilution exchange rate = %.6f s-1'% results['dilution_rate'])
    print()

    # plotting
    plt.plot(time, results['mixing_ratio_fit'] * 1e9, "g--")
    plt.plot(time, mixing_ratio * 1e9,'.')
    plt.title(f + "\n Flux = %.6f $\\pm$ %.6f mg CH4 m-2 h-1" %(results['flux'], results['std_dev']))
    plt.xlabel("time [s]")
    plt.ylabel("CH4 mixing ratio [ppb]")
    plt.savefig(f + ".png")
    plt.show()
