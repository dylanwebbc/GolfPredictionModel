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
  Separated data creation and modification functions into golfdatahandler.py
  Training data is now updated with each new tournament to save resources
  Renamed functions for clarity

A random forest golf prediction model which forecasts the top 10 players in the upcoming tournament on the pga tour
Built using the scikitlearn module in python
Utilizes data scraped from pgatour.com

The golf prediction model functions as follows:
  +Data is first scraped from pgatour.com for tournaments from the 2019, 2020 and 2021 seasons and stored in the golf.csv file
  +Training data is created from the golf.csv file and stored in the golf_train.csv file
  +The same metrics are created for players in the upcoming tournament and this data is stored in the golf_predict.csv file
  +A random forest is trained on golf_train.csv and then used on golf_predict.csv to predict the scores for each player in the upcoming tournament
  +The predicted scores are weighted by player performance in the same tournament held the previous season
  +The last two steps are looped 100 times to minimize randomness
