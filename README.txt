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
V 1.4 -- 2/23/2021
  separated data creation and modification functions into golfdatahandler.py
  training data is now updated with each new tournament to save resources
  renamed functions for clarity

A random forest golf prediction model which forecasts the top 10 players in the upcoming tournament on the pga tour
Built using the scikitlearn module in python
Utilizes data scraped from pgatour.com

Methodology:
  Data is scraped from pgatour.com and stored in the golf.csv file
  Training data is created from the golf.csv file and stored in the golf_train.csv file
  A random forest is trained on golf_train.csv and then used to predict the scores for each player in the upcoming tournament
  The predicted scores are weighted by player performance in the same tournament held the previous season
  The last two steps are looped 200 times to minimize randomness, and the predicted top ten players are displayed

First-time Setup:
  Following the pattern of golf_tournaments.csv, create a csv file of IDs for all past tournaments you would like to train the model with
    (IDs are displayed in the URL on pgatour.com/stats when "tournament only" is selected for a given stat)
  Run createGolfCSV and createTrainingCSV from the golfdatahandler.py file
  Add the ID of the tournament you would like to predict to your tournaments csv file
  Follow the instructions for prediction

Instructions for Prediction:
  Create a new csv file with a single column titled "Name" and copy and paste field data from pgatour.com into that column
    (names will be pasted in the form "last, first" and later automatically formatted to "first last")
  Run predictionModel from the golfpredictionmodel.py file
    (arguments: name of the csv file of names, year, tournament ID)
  
