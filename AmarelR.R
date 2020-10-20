sessionInfo()
library(tidyverse)
library(data.table)
library(bit64)
library(ggcorrplot)


# grab data
setwd("/scratch/rwomack/data")
parkingsample<-read.csv("Parking2014.csv")

# sample or slice data
# parkingsample2<-sample_n(parkingsample,100000)
parkingsample2014<-slice_sample(parkingsample, prop=.05)

# installing R

# currently on Amarel, need to get updated pcre2
# we can load this as a module
# module use /projects/community/modulefiles
# module load pcre2/10.35-gc563

# for packages, need java and gcc
# module load java/14.0.1
# module load gcc/5.4

# we will also need zlib for data.table
# wget https://zlib.net/zlib-1.2.11.tar.gz
# gunzip zlib-1.2.11.tar.gz
# tar -xvf zlib-1.2.11.tar
# cd zlib-1.2.11
# ./configure --prefix=$HOME/zlib-1.2.11
# make
# make install

# alternatively full install
# wget https://ftp.pcre.org/pub/pcre/pcre2-10.35.tar.gz
# gunzip pcre2-10.35.tar.gz
# tar -xvf pcre2-10.35.tar
# cd pcre2-10.35
# ./configure --prefix=$HOME/pcre2-10.35
# make
# install

# wget https://cran.r-project.org/src/base/R-4/R-4.0.3.tar.gz
# gunzip R-4.0.3.tar.gz
# tar -xvf R-4.0.3.tar
# 
# go into R directory to run
# cd R-4.0.3
# ./configure --prefix=$HOME/R-4.0.3
# make
# make install

# set paths
# export PATH=/home/rwomack/R-4.0.3/bin:$PATH
# export C_INCLUDE_PATH=/home/rwomack/R-4.0.3/include:$C_INCLUDE_PATH
# export CPLUS_INCLUDE_PATH=/home/rwomack/R-4.0.3/include:$CPLUS_INCLUDE_PATH
# export LIBRARY_PATH=/home/rwomack/R-4.0.3/lib:$LIBRARY_PATH
# export LD_LIBRARY_PATH=/home/rwomack/R-4.0.3/lib:$LD_LIBRARY_PATH
# export MANPATH=/home/rwomack/R-4.0.3/share/man:$MANPATH

# and similarly for other installs

# tidyverse requires libsodium

install.packages("data.table", dependencies=TRUE)
install.packages("tidyverse", dependencies=TRUE)
install.packages("bit64", dependencies=TRUE)


# Now let's process the data

setwd("/scratch/rwomack/data")

# data.frame
Parking2014<-read.csv("Parking2014.csv")
Parking2015<-read.csv("Parking2015.csv")
Parking2016<-read.csv("Parking2016.csv")
Parking2017<-read.csv("Parking2017.csv")

Compare
# tibble
Parking2017tibble<-read_csv("Parking2017.csv")
Parking2016tibble<-read_csv("Parking2016.csv")
Parking2015tibble<-read_csv("Parking2015.csv")
Parking2014tibble<-read_csv("Parking2014.csv")

# data.table
Parking2017table<-fread("Parking2017.csv")
Parking2016table<-fread("Parking2016.csv")
Parking2015table<-fread("Parking2015.csv")
Parking2014table<-fread("Parking2014.csv")

# homogenize data
# 2017 data has fewer variables, so truncate the other
# years to match
Parking2014table<-Parking2014table[,1:45]
Parking2015table<-Parking2015table[,1:45]
Parking2016table<-Parking2016table[,1:45]

# 2017 also has no data for Unregistered vehicle
# so we drop this variable
Parking2014table<-Parking2014table[,-35]
Parking2015table<-Parking2015table[,-35]
Parking2016table<-Parking2016table[,-35]
Parking2017table<-Parking2017table[,-35]

# There is also a problem with Vehicle Expiration Date
Parking2014table<-Parking2014table[,-13]
Parking2015table<-Parking2015table[,-13]
Parking2016table<-Parking2016table[,-13]
Parking2017table<-Parking2017table[,-13]

# And with Date First Observed
Parking2014table<-Parking2014table[,-26]
Parking2015table<-Parking2015table[,-26]
Parking2016table<-Parking2016table[,-26]
Parking2017table<-Parking2017table[,-26]


# merge in two stages
Parking14_15<-merge(Parking2014table, Parking2015table, all=TRUE)

#save some memory
rm(Parking2014table)
rm(Parking2015table)

Parking16_17<-merge(Parking2016table, Parking2017table, all=TRUE)

rm(Parking2016table)
rm(Parking2017table)

Parking14_17<-merge(Parking14_15,Parking16_17, all=TRUE)

rm(Parking14_15)
rm(Parking16_17)

# slice 1%
parkingsample<-slice_sample(Parking14_17, prop=.01)

# write
# you can use write_csv to export data
# write_csv(parkingsample, "parkingsample.csv")

# import parkingsample
parkingsample <- fread("/scratch/rwomack/data/parking_one_percent_sample.csv")

summary(parkingsample)
cor(as.matrix(parkingsample))

attach(parkingsample)
# we cannot run cor against non-numeric values (unless we recode them as factors). Most of these variables do not make sense to consider in this fashion

# so we will just run a few correlations

cor.test()

# these columns have character or logicaldata
# c(2:5,7:9,17:32, 34, 36:37)
parksample_nochar <- parkingsample[,-c(2:5,7:9,17:32,34, 36:44)]

# note the pairwise complete observations option
cor(parksample_nochar, use="pairwise.complete.obs")

# make correlation matrix
corr <- round(cor(parksample_nochar, use="pairwise.complete.obs"), 2)

# Compute a matrix of correlation p-values
p.mat <- cor_pmat(corr)

# ggcorrplot
# see this page for more info
# https://rpkgs.datanovia.com/ggcorrplot/

ggcorrplot(corr, 
           type = "upper",
           lab = TRUE,
           outline.color = "white",
           colors = c("maroon", "white", "steelblue"))
ggcorrplot(corr, method = "circle",
           type = "upper",
           p.mat = p.mat,
           outline.color = "white",
           colors = c("maroon", "white", "steelblue"))

cor(`Violation Precinct`,`Violation Location`, use = 'pairwise.complete.obs')

# interesting, we might use this information to drop a column and simplify our dataset

cor(`Issuer Precinct`,`Violation Location`, use = 'pairwise.complete.obs')

regoutput<-lm(`Violation Location`~`Issuer Precinct`)
summary(regoutput)

# by the way, here is a map of police precincts
# https://www.centralbooking.info/legal-assistance/ny-free-legal-resources/york-city-police-precincts-map/

# descriptive statistics

summary(parksample_nochar)
hist(`Summons Number`)
hist(`Violation Code`)
hist(`Street Code1`)
hist(`Street Code2`)
hist(`Street Code3`)
hist(`Violation Location`)
hist(`Violation Precinct`)
hist(`Issuer Precinct`)
hist(`Issuer Code`)
hist(`Vehicle Year`)

my_data_view<-slice_sample(parksample_nochar,n=30)

# let's examine the non-numeric data

glimpse(parkingsample)

table(`No Standing or Stopping Violation`)

# turns out the latter columns also are basically all NA's
# let's shrink the data further
parkingsample<-parkingsample[,1:37]

x <- `Violation Description`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Feet From Curb`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Meter Number`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Registration State`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Plate Type`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Issue Date`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Violation Code`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Vehicle Body Type`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Vehicle Make`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Vehicle Body Type`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Vehicle Color`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Vehicle Year`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Violation In Front Of Or Opposite`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Issuer Command`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Issuing Agency`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Violation Time`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Time First Observed`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Violation County`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Plate Type`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))

x <- `Registration State`
table(x) %>% 
  as.data.frame() %>% 
  arrange(desc(Freq))
