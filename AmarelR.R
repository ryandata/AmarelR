sessionInfo()
library(tidyverse)
library(data.table)
library(bit64)
library(ggcorrplot)
library(Hmisc)
library(factoextra)
library(ggpubr)
library(mltools)
library(caret)
library(psych)



# grab data
setwd("/scratch/rwomack/data")
parkingsample<-read.csv("Parking2014.csv")

# sample or slice data
# parkingsample2<-sample_n(parkingsample,100000)
parkingsample2014<-slice_sample(parkingsample, prop=.05)

# installing packages

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

## Machine learning techniques

# clustering

# we will illustrate k-means clustering, a widely used technique
# k-means relies on numeric distance between variables
# therefore factor variables are not appropriate for the algorithm
# we will use the nocharacter extract defined above

parksample_nochar

names(parksample_nochar)

# Summons Number and the Street Codes don't seem particularly interesting
# let's drop them

parksample_nochar <- parksample_nochar[,-c(1,3,4,5)]

rescaled_parking <- parksample_nochar %>%
  mutate(`Violation Code` = scale(`Violation Code`),
         `Violation Location` = scale(`Violation Location`),
         `Violation Precinct` = scale(`Violation Precinct`),
         `Issuer Precinct` = scale(`Issuer Precinct`),
         `Issuer Code` = scale(`Issuer Code`),
         `Vehicle Year` = scale(`Vehicle Year`),
         `Feet From Curb` = scale(`Feet From Curb`)) 

summary(rescaled_parking)

centers <- 4
kmeans(parksample_nochar, centers)
kmeans(rescaled_parking, centers)

# what happened?
# kmeans cannot work with NA cells
# we could impute the data
# using the means or median of each variable

# Hmisc does basic imputation via mean, median, etc.
# you may want to check out "mice" package for more sophisticated methods

is.na(parksample_nochar)
table(is.na(parksample_nochar$`Violation Location`))
table(is.na(parksample_nochar$`Violation Precinct`))
table(is.na(parksample_nochar$`Violation Code`))
table(is.na(parksample_nochar$`Issuer Precinct`))
table(is.na(parksample_nochar$`Issuer Code`))
table(is.na(parksample_nochar$`Vehicle Year`))
table(is.na(parksample_nochar$`Feet From Curb`))

# so it looks like our NAs stem from just one variable
# let's go ahead and drop that one
parksample_nochar <- parksample_nochar[,-2]
rescaled_parking <- rescaled_parking[,-2]

# try again
# k-means is sensitive to the random starting assignments
# we specify nstart = 5. 
# R will try 5 different random starting assignments and then select the best results.
# This number can be usually higher, but here our dataset is large
# don't want to slow things down too much

centers <- 4
kmeans(parksample_nochar, centers, nstart=5)
kmeans(rescaled_parking, centers, nstart=5)

# in order to visualize results easily, we need to limit to two dimensions
# note  .. syntax within brackets is specific to data.table

features <- c('Violation Precinct','Issuer Precinct')
centers <- 4
my_k <- kmeans(parksample_nochar[, ..features], centers, nstart=5)
my_k_scaled <- kmeans(rescaled_parking[, ..features], centers, nstart=5)

# visualizing using technique outlined at
# https://www.datanovia.com/en/blog/k-means-clustering-visualization-in-r-step-by-step-guide/

fviz_cluster(my_k, parksample_nochar[, ..features],
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)
fviz_cluster(my_k_scaled, rescaled_parking[, ..features],
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

# now try
features <- c('Violation Precinct','Feet From Curb')
my_k <- kmeans(parksample_nochar[, ..features], centers, nstart=5)
my_k_scaled <- kmeans(rescaled_parking[, ..features], centers, nstart=5)
fviz_cluster(my_k, parksample_nochar[, ..features],
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)
fviz_cluster(my_k_scaled, rescaled_parking[, ..features],
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

features <- c('Vehicle Year','Feet From Curb')
my_k <- kmeans(parksample_nochar[, ..features], centers, nstart=5)
my_k_scaled <- kmeans(rescaled_parking[, ..features], centers, nstart=5)
fviz_cluster(my_k, parksample_nochar[, ..features],
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)
fviz_cluster(my_k_scaled, rescaled_parking[, ..features],
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

# we can also plot across combined dimensions,
# although the interpretation in this case is less clear
my_k <- kmeans(parksample_nochar, centers, nstart=5)
my_k_scaled <- kmeans(rescaled_parking, centers, nstart=5)
fviz_cluster(my_k, parksample_nochar,
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)
fviz_cluster(my_k_scaled, rescaled_parking[, ..features],
             palette = c("blue", "red", "green","black"), 
             geom = "point",
             ellipse.type = "convex", 
             ggtheme = theme_bw()
)

# we can also iterate through different numbers of centers

# there is an art to finding the right number

# See this page for more
# https://www.guru99.com/r-k-means-clustering.html

# create a convenience function to generate sum of squares
# for any given k number of clusters
# withinss is the sum of squares within the model

kmean_withinss <- function(k) {
  cluster <- kmeans(rescaled_parking, k)
  return (cluster$tot.withinss)
}

# then iterate over many cluster values

# Set maximum cluster 
max_k <-20 

# Run algorithm over a range of k 
wss <- sapply(2:max_k, kmean_withinss)

# Create a data frame to plot the graph
elbow <-data.frame(2:max_k, wss)

# Then plot the graph to visualize where the "elbow" is
# this represent the point of diminishing returns
# Ultimately, a judgement call

ggplot(elbow, aes(x = X2.max_k, y = wss)) +
  geom_point() +
  geom_line() +
  scale_x_continuous(breaks = seq(1, 20, by = 1))

# Training and Testing (Prediction)

# We have been running our models up to now with the intent
# to describe existing data
# If we want to focus on prediction
# that is a different approach

# We must split our data into training and testing sets
# We build the model off of training data
# Then train it on the testing data, which the model 
# has not "seen" yet

# see, for example,
# https://www.r-bloggers.com/2019/10/evaluating-model-performance-by-building-cross-validation-from-scratch/



# let's restrict to a subset for this section
parkingsample <- parkingsample[,-c(1:2,5,10:13,16:20,22:31,34,36:44)]
names(parkingsample)
summary(parkingsample)

# we can one-hot encode using function from mltools
# make sure we have a factor
parkingsample$County2<-as.factor(parkingsample$`Violation County`)

parkingsample_1 <- one_hot(parkingsample, col="County2")

# a quick built-in example
# caret iterates and optimizes the model itself

inTrain <- createDataPartition(y = iris$Species, p = .8, list = FALSE)
createFolds(parksample_nochar, k=5)
iris.train <- iris[inTrain, ]
iris.test <- iris[- inTrain, ]
fit.control <- caret::trainControl(method = "cv", number = 10)
rf.fit <- caret::train(Species ~ .,
                       data = iris.train,
                       method = "rf",
                       trControl = fit.control)

# https://quantdev.ssri.psu.edu/sites/qdev/files/CV_tutorial.html

data_ctrl <- trainControl(method = "cv", number = 5)
model_caret <- train(ACT ~ gender + age + SATV + SATQ,   # model to fit
                     data = sat.act,                        
                     trControl = data_ctrl,              # folds
                     method = "lm")  
                     
# using caret
# https://www.machinelearningplus.com/machine-learning/caret-package/

# define a split of your data for training
trainRowNumbers <- createDataPartition(rescaled_parking$`Feet From Curb`, p=0.01, list=FALSE)


# Create the training  dataset
trainData <- rescaled_parking[trainRowNumbers,]

# Create the test dataset
testData <- rescaled_parking[-trainRowNumbers,]

# Store X and Y for later use.
x = trainData[, 5]
y = trainData$`Feet From Curb`


# See available algorithms in caret
modelnames <- paste(names(getModelInfo()), collapse=',  ')
modelnames

modelLookup('knn')

my_model <- train(x,y, method='knn')
fitted <- predict(my_model)
my_model
plot(my_model)

# predictions

predicted <- predict(my_model, testData)
head(predicted)

# Root Mean Square Error (RMSE)

rmse(predicted-testData$`Feet From Curb`, na.rm=TRUE)

# can also compute confusion matrix for categoricals

confusionMatrix(reference = testData$`Feet From Curb`, data = predicted, mode='everything')
     
# feature plots -- y needs to be a factor

featurePlot(x,y,
            plot = "scatter",
            strip=strip.custom(par.strip.text=list(cex=.7)),
            scales = list(x = list(relation="free"), 
                          y = list(relation="free")))
                     
                     
                     