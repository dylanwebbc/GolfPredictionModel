#Created by Dylan Webb
#March 23, 2021

#set working directory and import data files
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
golf <- read.csv("golf.csv", fileEncoding = "UTF-8-BOM")
prediction.rf <- read.csv("prediction_rf.csv", fileEncoding = "UTF-8-BOM")

#create yobs vector from golf.csv
yobs <- c()
for (name in prediction.rf$Name) {
  individual.data <- subset(golf, Name == name, select = c(Name, Score))
  yobs <- rbind(yobs, tail(individual.data, 10))
}

#define predictive distribution for gamma-poisson
pred.dist <- function(a, b, ynew) {
  
  lp <- -log(factorial(ynew)) + a*log(b) - lgamma(a) + 
    lgamma(ynew + a) - (ynew + a)*log(b + 1)
  
  exp(lp)
}

#calculate predicted score for each player
PredictedScore <- c()
for (name in prediction.rf$Name) {
  individual.data <- subset(yobs, Name == name, select = Score)
  
  a <- 3*as.integer(sqrt(prediction.rf$Odds[match(name, prediction.rf$Name)])/6 - 5)
  b <- 3
  
  astar <- a + sum(individual.data)
  bstar <- b + length(t(individual.data))
  
  PredictedScore <- rbind(PredictedScore, 
                          which.max(pred.dist(astar, bstar, 0:30)) - 1)
}

#combine prediction results into one dataframe
prediction.gp <- data.frame(PredictedScore)
prediction.gp$Name <- prediction.rf$Name
prediction.gp <- prediction.gp[order(PredictedScore),]

#output results to csv and display
write.csv(prediction.gp, "prediction_gp.csv", row.names = FALSE)
prediction.gp