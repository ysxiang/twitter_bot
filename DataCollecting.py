import tweepy
import time
import csv
import sys


# Consumer key and access token, used for OAuth
consumer_key = 'ia2S7T0A9MPK25DScr7TGqbHK'
consumer_secret = 'LJdkFRGDO5tOLt5f6ciVcz7PVodJeQwBEZqH8AG1eFIAeu37ry'

access_token = '634823262-xISVePQ5EqITbeMs6FgMu9sUfANnQVHL2YPMlGZD'
access_token_secret = 'H1JnJ0i0VF3HMcZoarSP3fQa5vwLjBIwZ1zERPfEVDYbC'

# OAuth process, using the key and token
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

userID = 'realDonaldTrump'

outputfilecsv = userID + "followers.csv"
fc = csv.writer(open(outputfilecsv, 'w'))
fc.writerow(["screen_name", "url", "verified", "location", "description", "followers_count", "friends_count", "created_at", "favourites_count", "statuses_count"])

users = tweepy.Cursor(api.followers, id=userID).items()
count = 0
errorCount = 0

while True:
    try:
        user = next(users)
        count += 1
    except tweepy.TweepError:
        # catches TweepError when rate limiting occurs, sleeps, then restarts.
        # 15 minutes, make a bit longer to avoid attention.
        print("15 minutes break, sleeping....")
        time.sleep(60 * 16)
        user = next(users)
    except StopIteration:
        break

    try:
        print("@" + user.screen_name + " has " + str(user.followers_count) +\
              " followers, has made "+str(user.statuses_count)+" tweets and location=" +\
              user.location+" geo_enabled="+str(user.geo_enabled)+" count="+str(count))
        fc.writerow([user.screen_name, user.url, user.verified, user.location, user.description, str(user.followers_count), str(user.friends_count), user.created_at, str(user.favourites_count), str(user.statuses_count)])
    except UnicodeEncodeError:
        errorCount += 1
        print("UnicodeEncodeError,errorCount =" + str(errorCount))

print("completed, errorCount = " + errorCount + " total followers = " + count)

