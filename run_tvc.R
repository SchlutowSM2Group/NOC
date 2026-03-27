library(reticulate)

source_python('noc.py')

out_units="mg m-2 h-1"

headspace_volume = 0.017207387  # [m3] chamber volume for /Portable/TVC2023_2023_7_11_09_45_5_2_1.csv
chamber_base_area = 0.047143525  # [m2] chamber base area

flnm = "./data/TVC2023_2023_7_11_09_45_5_2_1.csv"

headers <- read.csv(flnm, skip = 1, header = FALSE, nrows = 1)
df <- read.csv(flnm, skip=4, header = FALSE)
colnames(df)= headers

minutes <- matrix(df$Minute_7810, ncol=1)
seconds <- matrix(df$Second_7810, ncol=1)
ch4 <- matrix(df$CH4_7810, ncol=1)
air_temperature <- matrix(df$AirTemp, ncol=1)
air_pressure <- matrix(df$AirPres, ncol=1)
h2o <- matrix(df$H2O_7810, ncol=1)


# compose time from dataframe
time <- minutes * 60.0 + seconds

# read mole mixing ratio of CH4 [ppb mol mol-1] from dataframe
mixing_ratio <- ch4 * 1e-9

# read mole mixing ratio of H2O [ppm mol mol-1] from dataframe
water_mixing_ratio <- h2o * 1e-6

results <- calc_flux(
            time,
            mixing_ratio,
            chamber_base_area,
            headspace_volume,
	    water_mixing_ratio=water_mixing_ratio,
	    air_temperature=air_temperature,
	    air_pressure=air_pressure,
            gas="CH4",
            out_units=out_units
)

sprintf("---------------------------------------------")
sprintf("    Flux calculation")
sprintf("---------------------------------------------")
sprintf("flux                 = %f %s",results$flux, out_units)
sprintf("standard deviation   = %f %s",results$std_dev, out_units)
sprintf("---------------------------------------------")
sprintf("    Statistics")
sprintf("---------------------------------------------")
sprintf("R squared            = %f",results$r_squared)
sprintf("durbin watson        = %f",results$durbin_watson)
sprintf("shapiro wilk w       = %f",results$shapiro_wilk_w)
sprintf("shapiro wilk p       = %f",results$shapiro_wilk_p)
sprintf("ambient mixing ratio = %f ppb",results$ambient_mixing_ratio * 1e9)
sprintf("---------------------------------------------")
sprintf("    Dimensional analysis, Buckingham Pi Terms")
sprintf("---------------------------------------------")
sprintf("first-order pi term  = %f",results$first_order_pi)
sprintf("second-order pi term = %f",results$second_order_pi)

