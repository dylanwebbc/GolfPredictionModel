import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

#FORMATS FIELD DATA FROM PGA WEBSITE
#Each name is "last, first" in field, but "first last" in stats scrape
def formatField(fileName):
  df = pd.read_csv(fileName)
  print("Formatting Field...\n")
  
  #loop through every name and reformat
  for i in range(len(df["Name"])):
      name = df["Name"].iloc[i]
      last_name = name[0:name.find(',')]
      first_name = name[(name.find(',') + 2):len(name)]
      df.loc[i, "Name"] = first_name + " " + last_name
  df.to_csv(fileName, index = False)

#SCRAPE STATS FROM PGA WEBSITE

def scrapeStats(col, statID, year, tourneyID, pos = 2):

  #create soup object from url.
  url = "https://www.pgatour.com/stats/stat."+statID+".y"+year+".eon.t"+tourneyID+".html"
  soup = BeautifulSoup(requests.get(url).text, 'lxml')

  #get players from html tags and create dataframe
  players = []
  players_html = soup.select("td a")[1:]
  for player in players_html:
    players.append(player.get_text())
  df = pd.DataFrame()
  df["Name"] = players

  #check for missing data
  if len(df) == 0:
    return df
  
  #get stats from html tags
  stats = []
  stats_html = soup.find_all("td")
  for stat in stats_html:
    if stat.get_text() == "E":
      stats.append("0")
    else:
      stats.append(stat.get_text())

  #add relevant stats to dataframe
  colVals = np.zeros((len(df),1))
  numName = 0
  for i in range(len(stats)):
    if stats[i].find(players[numName]) != -1:
      colVals[numName] = float(stats[i + pos].replace(',',''))
      numName += 1
      if numName == len(players):
        break
  df[col] = colVals

  return df

#CREATE DATAFRAME OF PGA STATS

def pgaData():
  print("Scraping Data from pgatour.com...")
  tourneys = []
  tourneyNum = 1
  ids = pd.read_csv("golf_tournaments.csv")

  #iterate through each year of data
  for i in range(len(ids.columns)):
    year = str(ids.columns[i])
    tourneys = ids[year].values
    tourneys = tourneys[tourneys != 0]
    print("--" + year + "--")

    df_year = pd.DataFrame()

    #iterate through each tournament in the given year
    num = 0
    for tourney in tourneys:
      #reformat tourneyID
      tourney = str(tourney)
      while len(tourney) < 3:
        tourney = "0" + tourney
        
      print(num, "/", len(tourneys))
      num += 1

      df1 = scrapeStats("Score", "108", year, tourney, 3) #Strokes
      df2 = scrapeStats("Driving Accuracy", "102", year, tourney)
      df3 = scrapeStats("Greens In Regulation", "103", year, tourney)
      df4 = scrapeStats("Putting Average", "104", year, tourney)
      df5 = scrapeStats("Stroke Differential", "02417", year, tourney)
      df6 = scrapeStats("Scrambling", "130", year, tourney)

      dataframes = [df1, df2, df3, df4, df5, df6]

      #skip tournaments with missing values
      skip = False
      for i in range(len(dataframes)):
        if len(dataframes[i]) == 0:
          if skip == False:
            print(tourney, "missing:")
          print("  ", i + 1)
          skip = True

      if skip == False:
        #merge all stats into one dataframe
        df_tourney = pd.DataFrame()
        df_tourney = dataframes[0]
        dataframes.pop(0)
        for df in dataframes:
          df_tourney = pd.merge(df_tourney, df, on = "Name")
        df_tourney["Tourney"] = tourneyNum
        tourneyNum += 1

        #normalize score
        df_tourney["Score"] -= df_tourney["Score"].iloc[0]
        
        #combine dataframes from different tournaments
        if tourney == tourneys[0]:
          df_year = df_tourney
        else:
          df_year = pd.concat([df_year, df_tourney], axis = 0)

    print(len(tourneys), "/", len(tourneys),"\n")

    #combine dataframes from different years
    df_year["Year"] = int(year)
    if year == str(ids.columns[0]):
      df_total = df_year
    else:
      df_total = pd.concat([df_total, df_year], axis = 0)

  df_total.to_csv("golf.csv", index = False)

#CREATE DATAFRAME FOR TRAINING

def trainingData():
  print("Creating Training Data...")
  df = pd.read_csv("golf.csv")
  numTourneys = df["Tourney"].iloc[len(df["Tourney"]) - 1]
  df_total = pd.DataFrame() #dataframe to store ouput data
  numPredict = 2 #number of tournaments to predict on

  for i in range(numTourneys - numPredict):
    print(i, "/", numTourneys - numPredict)
    #set tourney dataframe to next highest tournament and create stats dataframe
    mask = df["Tourney"] == numTourneys - i
    df_tourney = pd.DataFrame()
    df_tourney["Name"] = df[mask]["Name"]
    df_tourney["Score"] = df[mask]["Score"]
    df_stats = pd.DataFrame()

    #loop through each player and find most recent numPredict tournament stats
    for j in range(len(df_tourney["Name"])):
      numEntries = 0
      df_statsRow = pd.DataFrame(np.zeros((1, numPredict*(len(df.columns) - 2))))
      for k in range(i + 1, numTourneys):
        mask1 = df["Tourney"] == numTourneys - k
        mask2 = df["Name"] == df_tourney["Name"].iloc[j]
        mask = mask1 & mask2

        #record stats if available
        if len(df[mask]) == 1:
          for l in range(len(df.columns) - 3):
            df_statsRow[l + (numEntries)*(len(df.columns) - 2)] = df[mask].values[0][l+1]
          numEntries += 1
          df_statsRow[(numEntries)*(len(df.columns) - 2) - 1] = k

        #record up until numPredict entries
        if numEntries == numPredict or k == numTourneys - 1:
          #marks missing values
          if k == numTourneys - 1 and numEntries < numPredict:
            df_statsRow[0] = float("NaN")
          #appends player data to df_stats
          if len(df_stats) == 0:
            df_stats = df_statsRow
          else:
            df_stats = pd.concat([df_stats, df_statsRow], axis = 0)
          break

    #combine the tourney and stats dataframes and then add to total dataframe
    df_stats.reset_index(drop = True, inplace = True)
    df_tourney.reset_index(drop = True, inplace = True)
    df_tourney = pd.concat([df_tourney, df_stats], axis = 1)

    if len(df_total) == 0:
      df_total = df_tourney
    else:
      df_total = pd.concat([df_total, df_tourney], axis = 0)

  print(numTourneys - numPredict, "/", numTourneys - numPredict, "\n")

  #name columns of df_total
  cols = ["Name", "Score"]
  for i in range(numPredict):
    for j in range(len(df.columns) - 3):
      cols.append(str(i + 1) + "_" + df.columns[j + 1])
    cols.append(str(i + 1) + "_Time")

  df_total.columns = cols

  #remove entries with missing data and export to csv
  df_total.dropna(how = "any", inplace = True)
  df_total.to_csv("golf_train.csv", index = False)

#CREATE DATAFRAME FOR PREDICTION

def predictionData(names):
  print("Creating Prediction Data...\n")
  df = pd.read_csv("golf.csv")
  numTourneys = df["Tourney"].iloc[len(df["Tourney"]) - 1]
  df_stats = pd.DataFrame() #dataframe to store player stats
  numPredict = 2 #number of tournaments to predict on

  #loop through each player and find most recent numPredict tournament stats
  for j in range(len(names)):
    numEntries = 0
    df_statsRow = pd.DataFrame(np.zeros((1, numPredict*(len(df.columns) - 2))))
    for k in range(numTourneys):
      mask1 = df["Tourney"] == numTourneys - k
      mask2 = df["Name"] == names["Name"].iloc[j]
      mask = mask1 & mask2

      #record stats if available
      if len(df[mask]) == 1:
        for l in range(len(df.columns) - 3):
          df_statsRow[l + numEntries*(len(df.columns) - 2)] = df[mask].values[0][l+1]
        numEntries += 1
        df_statsRow[numEntries*(len(df.columns) - 2) - 1] = float(k + 1)

      #record up until numPredict entries
      if numEntries == numPredict or k == numTourneys - 1:
        #marks missing values
        if k == numTourneys - 1 and numEntries < numPredict:
          df_statsRow[0] = float("NaN")
        #appends player data to df_stats
        if len(df_stats) == 0:
          df_stats = df_statsRow
        else:
          df_stats = pd.concat([df_stats, df_statsRow], axis = 0)
        break

  #combine the tourney and stats dataframes and then add to total dataframe
  df_stats.reset_index(drop = True, inplace = True)
  names = pd.concat([names, df_stats], axis = 1)

  #name columns of names
  cols = ["Name"]
  for i in range(numPredict):
    for j in range(len(df.columns) - 3):
      cols.append(str(i + 1) + "_" + df.columns[j + 1])
    cols.append(str(i + 1) + "_Time")

  names.columns = cols

  #remove entries with missing data and export to csv
  names.dropna(how = "any", inplace = True)
  names.reset_index(drop = True, inplace = True)
  return names

#RANDOM FOREST REGRESSOR

def forestRegress(input):
  df = pd.read_csv("golf_train.csv")
  X = df.drop(["Name", "Score"], axis = 1)
  y = df["Score"]

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .3)

  forest = RandomForestRegressor(warm_start = False, oob_score = True, 
                                min_samples_leaf = 4, max_depth = 80,
                                n_estimators = 80, n_jobs = -1)

  forest.fit(X_train, y_train.values.ravel())

  output = pd.DataFrame()
  output["Predicted Score"] = forest.predict(input.drop(["Name"], axis = 1))
  
  return output

#PREDICTION MODEL

def predictionModel(fileName, year, tourneyID, update = False, formatF = True):

  #update the golf and golf_train files for a new tournament
  if update:
    pgaData()
    trainingData()
    if formatF:
      formatField(fileName)

    #add current tournament ID for next update
    ids = pd.read_csv("golf_tournaments.csv")
    i = 0
    while ids.loc[i, str(year)] != 0:
      i += 1
    ids.loc[i, str(year)] = tourneyID
    ids.to_csv("golf_tournaments.csv", index = False)

  names = pd.read_csv(fileName)
  golf_predict = predictionData(names)

  #create final prediction dataframe
  finalPrediction = pd.DataFrame()
  finalPrediction["Name"] = golf_predict["Name"]
  finalPrediction["Rank"] = pd.DataFrame(np.zeros((len(finalPrediction["Name"]), 1)))

  #run prediction model multiple times and combine results in final prediction
  print("Prediction Progress...")
  numReps = 100
  for k in range(numReps):
    print(k, "/", str(numReps))

    #run random forest on the shuffled prediction data
    golf_predict = golf_predict.sample(frac = 1).reset_index(drop = True)
    p = forestRegress(golf_predict)
    predicted = pd.concat([golf_predict["Name"], p], axis = 1)
    
    #weight prediction by past performance in the same tournament
    past = scrapeStats("Past Score", "108", str(year - 1), tourneyID, 3)
    past["Past Score"] -= past["Past Score"].iloc[0]
    pastMean = np.mean(past["Past Score"])
    for i in range(len(predicted["Name"])):
      for j in range(len(past["Name"])):
        if predicted["Name"].iloc[i] == past["Name"].iloc[j]:
          predicted.loc[i, "Predicted Score"] = (9*predicted["Predicted Score"].iloc[i] + past["Past Score"].iloc[j])/10
          break
        if j == len(past["Name"]) - 1:
          predicted.loc[i, "Predicted Score"] = (9*predicted["Predicted Score"].iloc[i] + pastMean)/10

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

  #sort final prediction, remove missing values and display top 10
  finalPrediction.sort_values(by = "Rank", inplace = True)
  mask = finalPrediction["Rank"] != 0
  finalPrediction = finalPrediction[mask]
  finalPrediction.reset_index(drop = True, inplace = True)
  finalPrediction.index += 1
  finalPrediction.drop(["Rank"], axis = 1, inplace = True)
  print(str(numReps), "/", str(numReps), "\n\nTournament Prediction:")
  print(finalPrediction[:10])

#predictionModel("TournamentOfChampions.csv", 2021, "016")
#predictionModel("SonyOpen.csv", 2021, "006")
predictionModel("AmericanExpress.csv", 2021, "002")
