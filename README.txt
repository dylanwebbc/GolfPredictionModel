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
