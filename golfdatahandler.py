import pandas as pd
import numpy as np

import requests
from bs4 import BeautifulSoup

#SCRAPE STATS FROM PGA WEBSITE

def scrapeStats(col, statID, year, tourneyID, pos = 2):

  #create soup object from url and retrieve players
  url = "https://www.pgatour.com/stats/stat."+statID+".y"+year+".eon.t"+tourneyID+".html"

  retry = False
  while True:
    soup = BeautifulSoup(requests.get(url).text, 'lxml')

    #get players from html tags and create dataframe
    players = []
    players_html = soup.select("td a")[1:]
    for player in players_html:
      players.append(player.get_text())
    df = pd.DataFrame()
    df["Name"] = players

    #retry when data is missing
    if len(df) > 0:
      break
    elif retry == False:
      print("Connection failed, retrying...")
      retry = True
  
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

#COLLECT SINGLE TOURNAMENT DATA

def getPgaData(year, tourney):
  df1 = scrapeStats("Score", "108", year, tourney, 3) #Strokes
  df2 = scrapeStats("Halfway Score", "116", year, tourney)
  df3 = scrapeStats("Driving Accuracy", "102", year, tourney)
  df4 = scrapeStats("Greens In Regulation", "103", year, tourney)
  df5 = scrapeStats("Putting Average", "104", year, tourney)
  df6 = scrapeStats("Stroke Differential", "02417", year, tourney)
  df7 = scrapeStats("Scrambling", "130", year, tourney)
  df8 = scrapeStats("Birdie/Bogey", "02415", year, tourney)

  #normalize scores
  df1["Score"] -= df1["Score"].iloc[0]
  df2["Halfway Score"] -= min(df2["Halfway Score"])

  #merge all stats into one dataframe
  dataframes = [df1, df2, df3, df4, df5, df6, df7, df8]
  df_tourney = pd.DataFrame()
  df_tourney = dataframes[0]
  dataframes.pop(0)
  for df in dataframes:
    df_tourney = pd.merge(df_tourney, df, on = "Name")

  return df_tourney

#COLLECT TRAINING DATA FOR EACH PLAYER IN A GIVEN TOURNAMENT

def getTrainingData(df, numTourneys, index):
  numPredict = 2 #number of tournaments to predict on
  #set train dataframe to next highest tournament and create stats dataframe
  mask = df["Tourney"] == numTourneys - index
  df_train = pd.DataFrame()
  df_train["Name"] = df[mask]["Name"]
  df_train["Score"] = df[mask]["Score"]
  df_stats = pd.DataFrame()

  #loop through each player and find most recent numPredict tournament stats
  for j in range(len(df_train["Name"])):
    numEntries = 0
    df_statsRow = pd.DataFrame(np.zeros((1, numPredict*(len(df.columns) - 2))))
    for k in range(index + 1, numTourneys):
      mask1 = df["Tourney"] == numTourneys - k
      mask2 = df["Name"] == df_train["Name"].iloc[j]
      mask = mask1 & mask2

      #record stats if available
      if len(df[mask]) == 1:
        for l in range(len(df.columns) - 3):
          df_statsRow[l + (numEntries)*(len(df.columns) - 2)] = df[mask].values[0][l+1]
        numEntries += 1
        df_statsRow[(numEntries)*(len(df.columns) - 2) - 1] = k - index

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
      
  #combine the train and stats dataframes
  df_stats.reset_index(drop = True, inplace = True)
  df_train.reset_index(drop = True, inplace = True)
  df_train = pd.concat([df_train, df_stats], axis = 1)
  df_train.dropna(how = "any", inplace = True)

  #name columns of df_train
  cols = ["Name", "Score"]
  for i in range(numPredict):
    for j in range(len(df.columns) - 3):
      cols.append(str(i + 1) + "_" + df.columns[j + 1])
    cols.append(str(i + 1) + "_Time")
  df_train.columns = cols
      
  return df_train

#CREATE DATAFRAME FOR PREDICTION

def getPredictionData(names):
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

#CREATE DATAFRAME OF PGA STATS

def createGolfCSV():
  print("Scraping All Data from pgatour.com...")
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

      df_tourney = getPgaData(year, tourney)

      df_tourney["Tourney"] = tourneyNum
      tourneyNum += 1
        
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

def createTrainingCSV():
  print("Creating Training Data...")
  df = pd.read_csv("golf.csv")
  numTourneys = df["Tourney"].iloc[len(df["Tourney"]) - 1]
  df_total = pd.DataFrame() #dataframe to store ouput data
  numPredict = 2 #number of tournaments to predict on

  for i in range(numTourneys - numPredict):
    print(i, "/", numTourneys - numPredict)

    #get training data for one tourney from playerData
    df_tourney = getTrainingData(df, numTourneys, i)
    if len(df_total) == 0:
      df_total = df_tourney
    else:
      df_total = pd.concat([df_total, df_tourney], axis = 0)

  print(numTourneys - numPredict, "/", numTourneys - numPredict, "\n")

  #remove entries with missing data and export to csv
  df_total.dropna(how = "any", inplace = True)
  df_total.to_csv("golf_train.csv", index = False)
