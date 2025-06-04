library(reticulate)

source_python('noc.py')

out_units="mg m-2 h-1"

headspace_volume = 0.017207387  # [m3] chamber volume for /Portable/TVC2023_2023_7_11_09_45_5_2_1.csv
chamber_base_area = 0.047143525  # [m2] chamber base area

flnm = "./Portable/TVC2023_2023_7_11_09_45_5_2_1.csv"

headers <- read.csv(flnm, skip = 1, header = FALSE, nrows = 1)
df <- read.csv(flnm, skip=4, header = FALSE)
colnames(df)= headers

minutes <- matrix(df$Minute_7810, ncol=1)
seconds <- matrix(df$Second_7810, ncol=1)
ch4 <- matrix(df$CH4_7810, ncol=1)

# compose time from dataframe
time <- minutes * 60.0 + seconds

# read mole mixing ratio of CH4 [ppb mol mol-1] from dataframe
mixing_ratio <- ch4 * 1e-9

results <- calc_flux(
            time,
            mixing_ratio,
            chamber_base_area,
            headspace_volume,
            gas="CH4",
            out_units=out_units
)

sprintf("flux                 = %f %s",results$flux, out_units)
sprintf("standard deviation   = %f %s",results$std_dev, out_units)
sprintf("R squared            = %f",results$r_squared)
sprintf("ambient mixing ratio = %f ppb",results$ambient_mixing_ratio * 1e9)
sprintf("dilution rate        = %f s-1",results$dilution_rate)

