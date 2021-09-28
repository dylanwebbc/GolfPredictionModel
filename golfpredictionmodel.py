#golfpredictionmodel.py
"""
Created by Dylan Webb
January 7, 2021
"""

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

import golfdatahandler as gdh


def updateGolf(year, tourneyID):
  """Update the golf.csv file with data from the most recent tournament in the
  tournament.csv file

  Parameters:
      year (str) the year of the tournament
      tourneyID (str) the ID of the tournament (found on pgatourn.com/stats)
  Raises:
      ValueError: if the tourneyID is a duplicate of the most recent entry
  """
  
  print("Scraping Data from pgatour.com...\n")
  df_total = pd.read_csv("golf.csv")

  # Check for duplicate tournament data
  if (df_total["TourneyID"].iloc[len(df_total["TourneyID"]) - 1] == int(tourneyID)):
    raise ValueError("Duplicate Entry Detected for Tournament " + tourneyID + "\nProcess Aborted")

  # Get tournament data
  df_tourney = gdh.getPgaData(year, tourneyID)
  df_tourney["TourneyNum"] = df_total["TourneyNum"].iloc[len(df_total["TourneyNum"]) - 1] + 1
        
  # Combine dataframe to the rest and output
  df_total = pd.concat([df_total, df_tourney], axis = 0)
  df_total.to_csv("golf.csv", index = False)


def updateTrain():
  """Update the golf_train.csv file with data from the most recent tournament
  found in the golf.csv file
  """
  
  print("Updating Training Data...\n")
  df_train = pd.read_csv("golf_train.csv")

  # Get training data for the latest tournament
  df_tourney = gdh.getTrainingData(0)

  # Remove entries with missing data and export to csv
  df_train = pd.concat([df_tourney, df_train], axis = 0)
  df_train.to_csv("golf_train.csv", index = False)


def formatField(names):
  """Reformat the names from "last, first" (from pga field list)
  to be "first last" (from pga stats site)

  Parameters:
      names (DataFrame) the names of the players in the upcoming tournament
  Returns:
      (DataFrame) the reformatted names
  """

  # Loop through each name and reformat
  for i in range(len(names["Name"])):
      name = names["Name"].iloc[i]
      last_name = name[0:name.find(',')]
      first_name = name[(name.find(',') + 2):len(name)]
      names.loc[i, "Name"] = first_name + " " + last_name
  return names


def forestRegress(inputData):
  """Train and run a random forest regression on the data from the players
  in the upcoming tournament

  Parameters:
      inputData (DataFrame) data on each player describing their performance
                            in the past two tournaments
  Returns:
      (DataFrame) random forest prediction based on the player's stats
  """

  # Get the data and split it into independent (X) and dependent (y) variables
  df = pd.read_csv("golf_train.csv")
  X = df.drop(["Name", "Score"], axis = 1)
  y = df["Score"]

  # Split the data into training data and testing data
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

  # Setup the random forest
  forest = RandomForestRegressor(warm_start = False, oob_score = True, 
                                min_samples_leaf = 4, max_depth = 80,
                                n_estimators = 80, n_jobs = -1)

  # Fit the model to the training data
  rf = forest.fit(X_train, y_train.values.ravel())

  # Uncomment to check the relative importance of the features
  """importance = permutation_importance(rf, X_train, y_train)["importances_mean"]
  features = sorted(zip(importance, X.columns), reverse=True)
  print()
  for i in range(len(X.columns)):
    print("{:>24}\t{:<24}".format(features[i][1], features[i][0]))
  print()"""

  # Create output DataFrame and use the forest to predict scores
  output = pd.DataFrame()
  output["Predicted Score"] = forest.predict(inputData.drop(["Name"], axis = 1))
  
  return output


def predictionModel(fileName, tourneyID, weightPast = False):
  """Run the prediction by updating data from the previous tournament,
  updating the training data with the new data, creating data used for
  prediction, predicting using random forest regression, and outputing
  the final prediction of the top 10 accompanied by associated betting odds

  Parameters:
      fileName (str) the name of the file containing the tournament field
      tourneyID (str) the three-digit tournament ID associated with this
                      tournament (found on pgatourn.com/stats)
      weightPast (boolean) whether or not the tournament was held last year
  """
  
  # Get info from golf_tournaments.csv
  ids = pd.read_csv("golf_tournaments.csv")
  lastColumn = np.shape(ids)[1] - 1
  year = int(ids.columns[lastColumn])

  # Detect if a new year has been added
  newYear = False
  if np.isnan(ids.iloc[0, lastColumn]):
    year -= 1
    lastColumn -= 1
    newYear = True

  # Reformat the most recent tournament ID
  lastRow = np.count_nonzero(ids.iloc[:, lastColumn].notnull()) - 1
  lastID = str(int(ids.iloc[lastRow, lastColumn]))
  while len(lastID) < 3:
    lastID = "0" + lastID

  # Update golf and training data if analyzing a new tournament
  # Throw an error if duplicate entry detected
  if lastID != tourneyID:
    try:
      updateGolf(str(year), lastID)
    except ValueError as error:
      print(error)
      return
    updateTrain()
    
    # Add the new tournament ID to golf_tournaments file if it isn't there
    # Create a new row if necessary
    if newYear:
      year += 1
      ids.iloc[0, lastColumn + 1] = int(tourneyID)
    elif lastRow == np.shape(ids)[0] - 1:
      newRow = np.empty((1, lastColumn + 1))
      newRow[:] = np.nan
      newRow[0, lastColumn] = int(tourneyID)
      newRow = pd.DataFrame(newRow, columns=ids.columns)
      ids = ids.append(newRow, ignore_index = True)
    else:
      ids.iloc[lastRow + 1, lastColumn] = int(tourneyID)
    ids.to_csv("golf_tournaments.csv", index = False)

  # Scrape data for the same tournament from the previous year
  if weightPast:
    print("Scraping More Data from pgatour.com...\n")
    past = gdh.scrapeStats("Past Score", "108", str(year - 1), tourneyID, 3)
    past["Past Score"] -= past["Past Score"].iloc[0]

  # Get prediction data to input into the random forest
  print("Creating Prediction Data...\n")
  names = formatField(pd.read_csv(fileName))
  golf_predict = gdh.getTrainingData(-1, names)

  # Create final prediction dataframe
  finalPrediction = pd.DataFrame()
  finalPrediction["Name"] = golf_predict["Name"]
  finalPrediction["Rank"] = pd.DataFrame(np.zeros((len(finalPrediction["Name"]), 1)))

  # Run prediction model multiple times and combine results in final prediction
  print("Prediction Progress...")
  numReps = 200
  for k in range(numReps):
    print(k, "/", str(numReps))

    # Run random forest on the shuffled prediction data
    golf_predict = golf_predict.sample(frac = 1).reset_index(drop = True)
    p = forestRegress(golf_predict)
    predicted = pd.concat([golf_predict["Name"], p], axis = 1)
    
    # Weight prediction by performance in the same tournament a year earlier
    if weightPast:
      for i in range(len(predicted["Name"])):
        for j in range(len(past["Name"])):
          if predicted["Name"].iloc[i] == past["Name"].iloc[j]:
            predicted.loc[i, "Predicted Score"] = (9*predicted["Predicted Score"].iloc[i] + past["Past Score"].iloc[j])/10
            break
          if j == len(past["Name"]) - 1:
            predicted.loc[i, "Predicted Score"] = (9*predicted["Predicted Score"].iloc[i] + np.mean(past["Past Score"]))/10

    # Rearrange prediction dataframe by predicted score
    predicted.dropna(how = "any", inplace = True)
    predicted.sort_values(by = "Predicted Score", inplace = True)
    predicted.reset_index(drop = True, inplace = True)

    # Add rank to prediction dataframe
    rank = np.zeros((len(predicted),1))
    for i in range(len(predicted)):
      rank[i] = i + 1
    predicted["Rank"] = pd.DataFrame(rank)

    # Combine recent prediction ranking with finalPrediction
    for i in range(len(predicted["Name"])):
      for j in range(len(finalPrediction["Name"])):
        if predicted["Name"].iloc[i] == finalPrediction["Name"].iloc[j]:
          finalPrediction.loc[j, "Rank"] += predicted["Rank"].iloc[i]

  # Sort final prediction and remove missing values
  print(str(numReps), "/", str(numReps))
  finalPrediction.sort_values(by = "Rank", inplace = True)
  mask = finalPrediction["Rank"] != 0
  finalPrediction = finalPrediction[mask]
  finalPrediction.reset_index(drop = True, inplace = True)

  # Get betting odds for predicted top 10 and display
  print("\nScraping Data from vegasinsider.com...")
  finalOutput = gdh.scrapeOdds(finalPrediction[:10])
  finalOutput.index += 1
  print("\nRandom Forest Prediction:")
  pd.set_option('display.max_rows', None)
  print(finalOutput)
  finalOutput.to_csv("prediction_rf.csv", index = False)

predictionModel("Sanderson.csv", "054", True)
