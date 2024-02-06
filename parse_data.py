# Twitter Analysis Project
# parse_data.py
# Converts tweets.js from a downloaded Twitter archive into a
#   dataset readable for scikitlearn.

import csv
import dateparser
import json
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

raw_tweets = open('tweets.js', 'r', encoding="cp866")
# raw_tweets = open('tweets_test.js', 'r')
json_tweets = json.load(raw_tweets)
raw_tweets.close()

formatted_tweets = []

analyzer = SentimentIntensityAnalyzer()

for i in range(900, 976):
#for i in range(0, len(json_tweets)):
  tweet = json_tweets[i]['tweet']
  
  tweet_text = word_tokenize(tweet['full_text'].lower())
  print("==========================================================================")
  print("Raw tweet: " + tweet['full_text'])
  filtered_tokens = [token for token in tweet_text if token not in stopwords.words('english')]
  lemmatizer = WordNetLemmatizer()
  lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
  processed_text = ' '.join(lemmatized_tokens)  
  print("Score:", analyzer.polarity_scores(processed_text))
  
  
  
  
  
  tweet_len = int(tweet['display_text_range'][1])
  
  tweet_dt = dateparser.parse(tweet['created_at'], date_formats=['%a %b %d %H:%M:%S %z %Y'])
  tweet_year = tweet_dt.year
  tweet_month = tweet_dt.month
  tweet_day = tweet_dt.day
  tweet_hour = tweet_dt.hour
  tweet_day_of_week = tweet_dt.weekday()
  
  tweet_is_reply = 1 if 'in_reply_to_screen_name' in tweet else 0
  
  tweet_user_mentions = len(tweet['entities']['user_mentions'])
  tweet_hashtags = len(tweet['entities']['hashtags'])
  
  tweet_photos = 0
  tweet_videos = 0
  
  if 'extended_entities' in tweet:
    tweet_media = tweet['extended_entities']['media']
    for m in range(0, len(tweet_media)):
      media = tweet_media[m]['type']
      if media == 'photo':
        tweet_photos += 1
      elif media == 'video':
        tweet_videos += 1
  
  tweet_faves = int(tweet['favorite_count'])
  tweet_rts = int(tweet['retweet_count'])
    
  formatted_tweets.append([tweet_text, tweet_len, tweet_year, tweet_month, tweet_day, tweet_hour, tweet_day_of_week, tweet_is_reply, tweet_user_mentions, tweet_hashtags, tweet_photos, tweet_videos, tweet_faves, tweet_rts])
  # print(formatted_tweets[i])

quit()

with open('dataset.csv', 'w', newline='') as f:
  wr = csv.writer(f)
  wr.writerow(['Tweet Text', 'Tweet Length', 'Year', 'Month', 'Day', 'Hour', 'Day of week', 'Is reply', 'User mentions', 'Hashtags', 'Pictures', 'Videos', 'Favorites', 'Retweets'])
  wr.writerows(formatted_tweets)

#input('\nProgram complete. ENTER to quit.')