import botometer
import tweepy
import time
import csv
import sys
import json





mashape_key = "D3nryZcw89mshsQjIOnj0Fn8eLUUp18bHYvjsnbqds68V2vKMU"
twitter_app_auth = {
    'consumer_key': 'ia2S7T0A9MPK25DScr7TGqbHK',
    'consumer_secret': 'LJdkFRGDO5tOLt5f6ciVcz7PVodJeQwBEZqH8AG1eFIAeu37ry',
    'access_token': '634823262-xISVePQ5EqITbeMs6FgMu9sUfANnQVHL2YPMlGZD',
    'access_token_secret': 'H1JnJ0i0VF3HMcZoarSP3fQa5vwLjBIwZ1zERPfEVDYbC',
  }
bom = botometer.Botometer(wait_on_ratelimit=True,
                          mashape_key=mashape_key,
                          **twitter_app_auth)


result = bom.check_account('@clayadavis')
print result['scores']['universal']
