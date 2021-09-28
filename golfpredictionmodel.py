#Created by Dylan Webb
#January 7, 2021

import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

import golfdatahandler as gdh

#UPDATE GOLF DATAFRAME WITH ONE TOURNAMENT

def updateGolf(year, tourneyID):
  print("Scraping Data from pgatour.com...\n")
  
  df_total = pd.read_csv("golf.csv")

  #check for duplicate tournament data
  if (df_total["TourneyID"].iloc[len(df_total["TourneyID"]) - 1] == int(tourneyID)):
    return False

  #get tournament data
  df_tourney = gdh.getPgaData(year, tourneyID)
  df_tourney["TourneyNum"] = df_total["TourneyNum"].iloc[len(df_total["TourneyNum"]) - 1] + 1
        
  #combine dataframe to the rest and output
  df_total = pd.concat([df_total, df_tourney], axis = 0)
  df_total.to_csv("golf.csv", index = False)

  return True


#UPDATE TRAINING DATAFRAME WITH ONE TOURNAMENT

def updateTrain():
  print("Updating Training Data...\n")
  df = pd.read_csv("golf.csv")
  df_train = pd.read_csv("golf_train.csv")

  #get training data for the latest tournament
  df_tourney = gdh.getTrainingData(0)

  #remove entries with missing data and export to csv
  df_train = pd.concat([df_tourney, df_train], axis = 0)
  df_train.to_csv("golf_train.csv", index = False)


#FORMAT FIELD DATA FROM PGA WEBSITE
#Each name is "last, first" in field, but "first last" in stats scrape
def formatField(names):
  for i in range(len(names["Name"])):
      name = names["Name"].iloc[i]
      last_name = name[0:name.find(',')]
      first_name = name[(name.find(',') + 2):len(name)]
      names.loc[i, "Name"] = first_name + " " + last_name
  return names


#RANDOM FOREST REGRESSOR

def forestRegress(inputData):
  df = pd.read_csv("golf_train.csv")
  X = df.drop(["Name", "Score"], axis = 1)
  y = df["Score"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

  forest = RandomForestRegressor(warm_start = False, oob_score = True, 
                                min_samples_leaf = 4, max_depth = 80,
                                n_estimators = 80, n_jobs = -1)

  rf = forest.fit(X_train, y_train.values.ravel())

  #uncomment to check the relative importance of the features
  """importance = permutation_importance(rf, X_train, y_train)["importances_mean"]
  features = sorted(zip(importance, X.columns), reverse=True)
  print()
  for i in range(len(X.columns)):
    print("{:>24}\t{:<24}".format(features[i][1], features[i][0]))
  print()"""

  output = pd.DataFrame()
  output["Predicted Score"] = forest.predict(inputData.drop(["Name"], axis = 1))
  
  return output


#PREDICTION MODEL

def predictionModel(fileName, tourneyID, weightPast = False):

  #check most recent id for
  ids = pd.read_csv("golf_tournaments.csv")
  lastColumn = np.shape(ids)[1] - 1
  year = int(ids.columns[lastColumn])

  #detect if a new year has been added
  newYear = False
  if np.isnan(ids.iloc[0, lastColumn]):
    year -= 1
    lastColumn -= 1
    newYear = True

  #reformat the most recent tournament ID
  lastRow = np.count_nonzero(ids.iloc[:, lastColumn].notnull()) - 1
  lastID = str(int(ids.iloc[lastRow, lastColumn]))
  while len(lastID) < 3:
    lastID = "0" + lastID

  #update golf and training data if analyzing a new tournament
  if lastID != tourneyID:
    if not updateGolf(str(year), lastID):
      print("Duplicate Data Detected for Tournament " + lastID + "\nProcess Aborted")
      return
    updateTrain()
    
    #add the new tournament ID to golf_tournaments file if it isn't there
    #create a new row if necessary
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

  #scrape data for the same tournament from the previous year
  if weightPast:
    print("Scraping More Data from pgatour.com...\n")
    past = gdh.scrapeStats("Past Score", "108", str(year - 1), tourneyID, 3)
    past["Past Score"] -= past["Past Score"].iloc[0]

  #get prediction data to input into the random forest
  print("Creating Prediction Data...\n")
  names = formatField(pd.read_csv(fileName))
  golf_predict = gdh.getTrainingData(-1, names)

  #create final prediction dataframe
  finalPrediction = pd.DataFrame()
  finalPrediction["Name"] = golf_predict["Name"]
  finalPrediction["Rank"] = pd.DataFrame(np.zeros((len(finalPrediction["Name"]), 1)))

  #run prediction model multiple times and combine results in final prediction
  print("Prediction Progress...")
  numReps = 200
  for k in range(numReps):
    print(k, "/", str(numReps))

    #run random forest on the shuffled prediction data
    golf_predict = golf_predict.sample(frac = 1).reset_index(drop = True)
    p = forestRegress(golf_predict)
    predicted = pd.concat([golf_predict["Name"], p], axis = 1)
    
    #weight prediction by performance in the same tournament a year earlier
    if weightPast:
      for i in range(len(predicted["Name"])):
        for j in range(len(past["Name"])):
          if predicted["Name"].iloc[i] == past["Name"].iloc[j]:
            predicted.loc[i, "Predicted Score"] = (9*predicted["Predicted Score"].iloc[i] + past["Past Score"].iloc[j])/10
            break
          if j == len(past["Name"]) - 1:
            predicted.loc[i, "Predicted Score"] = (9*predicted["Predicted Score"].iloc[i] + np.mean(past["Past Score"]))/10

    #rearrange prediction dataframe by predicted score
    predicted.dropna(how = "any", inplace = True)
    predicted.sort_values(by = "Predicted Score", inplace = True)
    predicted.reset_index(drop = True, inplace = True)

    #add rank to prediction dataframe
    rank = np.zeros((len(predicted),1))
    for i in range(len(predicted)):
      rank[i] = i + 1
    predicted["Rank"] = pd.DataFrame(rank)

    #combine recent prediction ranking with finalPrediction
    for i in range(len(predicted["Name"])):
      for j in range(len(finalPrediction["Name"])):
        if predicted["Name"].iloc[i] == finalPrediction["Name"].iloc[j]:
          finalPrediction.loc[j, "Rank"] += predicted["Rank"].iloc[i]

  #sort final prediction and remove missing values
  print(str(numReps), "/", str(numReps))
  finalPrediction.sort_values(by = "Rank", inplace = True)
  mask = finalPrediction["Rank"] != 0
  finalPrediction = finalPrediction[mask]
  finalPrediction.reset_index(drop = True, inplace = True)

  #get betting odds for predicted top 10 and display
  print("\nScraping Data from vegasinsider.com...")
  finalOutput = gdh.scrapeOdds(finalPrediction[:10])
  finalOutput.index += 1
  print("\nRandom Forest Prediction:")
  pd.set_option('display.max_rows', None)
  print(finalOutput)
  finalOutput.to_csv("prediction_rf.csv", index = False)

predictionModel("Sanderson.csv", "054", True)
