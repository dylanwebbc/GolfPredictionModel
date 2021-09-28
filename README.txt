Golf Prediction Model
Written by Dylan Webb 
V 1.0 -- 1/7/2021
V 1.1 -- 1/8/2021
  score now represents total strokes over the top player's strokes in a given tournament
  random shuffling of names corrects bias toward players at the beginning of the alphabet
  golf_predict no longer necessary
  cosmetic changes
V 1.2 -- 1/19/2021
  added formatting functionality to convert pga field name format to pga stat name format
  separated tournament IDs into a separate csv file to streamline addition of new tournaments
V 1.3 -- 2/2/2021
  pgaData function is no longer run by prediction model but remains accessible for reconstructing golf.csv
  updating is now performed automatically and corresponding variables have been removed from prediction model function call
  two new statistics added to dataframe
  
V 2.0 -- 2/23/2021
  separated data creation and modification functions into golfdatahandler.py
  golf.csv and golf_train.csv now updated with each new tournament
  reduced code duplication by combining getPredictionData and getTrainingData
  renamed functions for clarity
  
V 3.0 -- 3/23/2021
  prediction model now scrapes betting odds for top 10 predicted players and outputs result to prediction_rf.csv
  new R file runs a Bayesian analysis on predicted players and outputs a second prediction to prediction_gp.csv
V 3.1 -- 4/28/2021
  corrected prediction_rf.csv file compatibility in R code when betting odds are entered manually
  included project write-up for the Bayesian portion of the model
V 3.2 -- 5/25/2021
  fixed an error with the createTrainingCSV function in golfdatahandler.py
  incorporated the statistic consecutive cuts into the random forest model
V 3.3 -- 6/8/2021
  fixed pga site connection retry feature compatibility with new tournaments
V 3.4 -- 7/13/2021
  user now manually inputs verification of existing data for the same tournament from the previous year in golfpredictionmodel.py
  corrected csv encoding for player name comparison in golfgammapoisson.r
V 3.5 -- 8/18/2021
  added tournament id to golf.csv for reference
  golf_tournaments.csv can now have columns of arbitrary length without previous editing
  golfer names file is now modified within the program to prevent bugs
  added failsafe against adding duplicate data
  removed year as an argument and program automatically infers year
V 3.6 -- 9/27/21
  fixed new year functionality

A random forest golf prediction model which forecasts the top 10 players in the upcoming tournament on the pga tour and then ranks them using Bayesian analysis
Built using python and R
Utilizes data scraped from pgatour.com and vegasinsider.com

Methodology:
  Data is scraped from pgatour.com and stored in the golf.csv file
  Training data is created from the golf.csv file and stored in the golf_train.csv file
  A random forest is trained on golf_train.csv and then used to predict the scores for each player in the upcoming tournament
  The predicted scores are weighted by player performance in the same tournament held the previous season
  The last two steps are looped 200 times to minimize randomness, and the predicted top ten players are displayed
  A Bayesian analysis can then be run in R to order the top ten players with higher accuracy using their past performance and current betting odds

First-time Setup:
  Following the pattern of golf_tournaments.csv, create a csv file of IDs for all past tournaments you would like to train the model with
    (IDs are displayed in the URL on pgatour.com/stats when "tournament only" is selected for a given stat)
  Run createGolfCSV and createTrainingCSV from the golfdatahandler.py file
  Add the ID of the tournament you would like to predict to your tournaments csv file
  Follow the instructions for prediction

Instructions for Prediction:
  Create a new csv file with a single column titled "Name" and copy and paste field data from pgatour.com into that column
    (names must be pasted in the form "last, first")
  Run predictionModel from the golfpredictionmodel.py file
    (arguments: name of the csv file of names, tournament ID, boolean for weighting by past year)
  Manually fill any missing values (marked by 0s) for betting odds in the prediction_rf.csv file
  Run golfgammapoisson.R
  
Note: when you want to predict for tournaments in a new year, simply name a new column in the tournaments.csv file and proceed like normal
