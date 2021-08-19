#Created by Dylan Webb
#February 23, 2021

import pandas as pd
import numpy as np
import math

import requests
from bs4 import BeautifulSoup

#SCRAPE STATS FROM PGA WEBSITE

def scrapeStats(col, statID, year, tourneyID, pos = 2, e = "on"):

  #create soup object from url and retrieve players
  url = "https://www.pgatour.com/stats/stat."+statID+".y"+year+".e"+e+".t"+tourneyID+".html"

  players = []
  retry = False
  while True:
    soup = BeautifulSoup(requests.get(url).text, 'lxml')

    #get players from html tags and create dataframe
    players_html = soup.select("td a")[1:]
    for player in players_html:
      players.append(player.get_text())

    #retry when data is missing
    if len(players) > 0:
      break
    elif col == "Past Score":
      return pd.DataFrame()
    elif retry == False:
      print("Connection failed, retrying...")
      retry = True

  df = pd.DataFrame()
  df["Name"] = players
    
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

#SCRAPE ODDS FROM BETTING ODDS WEBSITE

def scrapeOdds(names):
  df = pd.DataFrame()
  
  #create soup object from url and retrieve players
  url = "https://www.vegasinsider.com/golf/odds/futures/"

  retry = False
  data = []
  while True:
    soup = BeautifulSoup(requests.get(url).text, 'lxml')

    #get data from html tags and create dataframe
    data_html = soup.select("td")[1:]
    for datum_html in data_html:
      datum = datum_html.get_text()
      if "\n" not in datum:
          data.append(datum)

    #retry when data is missing
    if len(data) > 0:
      break
    elif retry == False:
      print("Connection failed, retrying...")
      retry = True

  #loop through data and append corresponding odds after players in names
  warning = False
  odds = []
  for name in names["Name"]:
    add = False
    for datum in data:
      if add:
        odds.append(int(datum[1:]))
        break
      elif name == datum:
        add = True
    if add == False:
      odds.append(int(0))
      warning = True

  #output error if values are missing
  if warning:
    print("\nError retrieving odds\nManually enter missing values")

  result = pd.DataFrame()
  result["Name"] = names["Name"]
  result["Odds"] = odds
  return result

#COLLECT SINGLE TOURNAMENT DATA

def getPgaData(year, tourneyID):
  df1 = scrapeStats("Score", "108", year, tourneyID, 3) #Strokes
  df2 = scrapeStats("Halfway Score", "116", year, tourneyID)
  df3 = scrapeStats("Driving Accuracy", "102", year, tourneyID)
  df4 = scrapeStats("Greens In Regulation", "103", year, tourneyID)
  df5 = scrapeStats("Putting Average", "104", year, tourneyID)
  df6 = scrapeStats("Stroke Differential", "02417", year, tourneyID)
  df7 = scrapeStats("Scrambling", "130", year, tourneyID)
  df8 = scrapeStats("Birdie/Bogey", "02415", year, tourneyID)
  df9 = scrapeStats("Consecutive Cuts", "122", year, tourneyID, 1, "off")

  #normalize scores
  df1["Score"] -= df1["Score"].iloc[0]
  df2["Halfway Score"] -= min(df2["Halfway Score"])

  #merge all stats into one dataframe
  dataframes = [df1, df2, df3, df4, df5, df6, df7, df8, df9]
  df_tourney = pd.DataFrame()
  df_tourney = dataframes[0]
  dataframes.pop(0)
  for df in dataframes:
    df_tourney = pd.merge(df_tourney, df, on = "Name", how = "outer")

  #remove/replace NaNs associated with consecutive cuts
  for i in range(len(df["Name"])):
    if math.isnan(df_tourney["Consecutive Cuts"].iloc[i]):
      df_tourney.loc[i, "Consecutive Cuts"] = 1
  df_tourney.dropna(how = "any", inplace = True)

  df_tourney["Year"] = int(year)
  df_tourney["TourneyID"] = tourneyID

  return df_tourney

#COLLECT TRAINING DATA FOR EACH PLAYER IN A GIVEN TOURNAMENT
#Also used to create prediction data

def getTrainingData(index, df_train = pd.DataFrame()):
  numPredict = 2 #number of tournaments to predict on
  df_stats = pd.DataFrame()
  df = pd.read_csv("golf.csv")
  df = df.drop(columns = ["Year", "TourneyID"])
  numTourneys = df["TourneyNum"].iloc[len(df["TourneyNum"]) - 1]
  predict = True

  if df_train.empty:
    #if not predicting, take names and score from golf.csv
    predict = False
    mask = df["TourneyNum"] == numTourneys - index
    df_train["Name"] = df[mask]["Name"]
    df_train["Score"] = df[mask]["Score"]

  #loop through each player and find most recent numPredict tournament stats
  for j in range(len(df_train["Name"])):
    numEntries = 0
    df_statsRow = pd.DataFrame(np.zeros((1, numPredict*(len(df.columns) - 1))))

    for k in range(index + 1, numTourneys):
      mask1 = df["TourneyNum"] == numTourneys - k
      mask2 = df["Name"] == df_train["Name"].iloc[j]
      mask = mask1 & mask2
      
      #record stats if available
      if len(df[mask]) == 1:
        for l in range(len(df.columns) - 2):
          df_statsRow[l + numEntries*(len(df.columns) - 1)] = df[mask].values[0][l+1]
        numEntries += 1
        df_statsRow[numEntries*(len(df.columns) - 1) - 1] = k - index

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
  df_train.reset_index(drop = True, inplace = True)

  #name columns of df_train
  cols = ["Name"]
  if predict == False:
    cols.append("Score")
  for i in range(numPredict):
    for j in range(len(df.columns) - 2):
      cols.append(str(i + 1) + "_" + df.columns[j + 1])
    cols.append(str(i + 1) + "_Time")
  df_train.columns = cols
      
  return df_train

#CREATE DATAFRAME OF PGA STATS

def createGolfCSV():
  print("Scraping All Data from pgatour.com...\n")
  tourneys = []
  tourneyNum = 1
  ids = pd.read_csv("golf_tournaments.csv")

  #iterate through each year of data
  for i in range(len(ids.columns)):
    year = str(ids.columns[i])
    tourneyIDs = ids[year][ids[year].notnull()]
    print("--" + year + "--")

    df_year = pd.DataFrame()

    #iterate through each tournament in the given year
    num = 0
    for tourneyID in tourneyIDs:
      #reformat tourneyID
      tourneyID = str(tourneyID)
      while len(tourneyID) < 3:
        tourneyID = "0" + tourneyID
        
      print(num, "/", len(tourneyIDs))
      num += 1

      df_tourney = getPgaData(year, tourneyID)
      df_tourney["TourneyNum"] = tourneyNum
      tourneyNum += 1
        
      #combine dataframes from different tournaments
      if tourneyID == tourneyIDs[0]:
        df_year = df_tourney
      else:
        df_year = pd.concat([df_year, df_tourney], axis = 0)

    print(len(tourneyIDs), "/", len(tourneyIDs),"\n")

    #combine dataframes from different years
    if year == str(ids.columns[0]):
      df_total = df_year
    else:
      df_total = pd.concat([df_total, df_year], axis = 0)

  print("Done")

  df_total.to_csv("golf.csv", index = False)

#CREATE DATAFRAME FOR TRAINING

def createTrainingCSV():
  print("Creating Training Data...\n")
  df = pd.read_csv("golf.csv")
  numTourneys = df["TourneyNum"].iloc[len(df["TourneyNum"]) - 1]
  df_total = pd.DataFrame() #dataframe to store ouput data
  numPredict = 2 #number of tournaments to predict on

  for i in range(numTourneys - numPredict):
    print(i, "/", numTourneys - numPredict)

    #get training data for one tourney from playerData
    df_tourney = getTrainingData(i, pd.DataFrame())
    if len(df_total) == 0:
      df_total = df_tourney
    else:
      df_total = pd.concat([df_total, df_tourney], axis = 0)

  print(numTourneys - numPredict, "/", numTourneys - numPredict, "\n")
  print("Done")

  #remove entries with missing data and export to csv
  df_total.dropna(how = "any", inplace = True)
  df_total.to_csv("golf_train.csv", index = False)
