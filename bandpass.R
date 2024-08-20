library(tuneR)
library(seewave)

setwd("E:/Larissa's Pickles/Background_noise")

files <- list.files()

for (file in files){
audio <- readWave(file)
band_pass <- ffilter(audio, f=44100, bandpass=T,from=1500, to=10000, output="Wave")
savewav(band_pass, 44100, filename=file)

}