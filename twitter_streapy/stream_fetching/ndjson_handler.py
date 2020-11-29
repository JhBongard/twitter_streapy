

import jsonlines
import os
from tqdm import tqdm
from datetime import datetime
from collections import Counter




def import_tweets(ndjson_file, avoidRT=None):
    """
    Imports a single ndjson_file. If the optional parameter "avoidRT" is not "None"
    retweets are ignored if their "RT" parameter is "true"

    :param ndjson_file:
    :param avoidRT:
    :return:
    """
    tweet_list = []
    with jsonlines.open(ndjson_file) as reader:
        for obj in tqdm(reader.iter(type=dict, skip_invalid=True)):

            # Check for Retweets
            if avoidRT:
                if obj.get("RT") == "true":
                    continue

            # Change "created_at" parameter to datetime object if necessary
            if "created_at" in obj:
                try:
                    # convert time stamp into datetime object
                    obj["created_at"] = str(datetime.strptime(obj["created_at"], "%a %b %d %H:%M:%S +0000 %Y"))
                    tweet_list.append(obj)
                except:
                    tweet_list.append(obj)
    return tweet_list




def import_interval(ndjson_file, interval, start_tweet):

    """
    Returns the tweets that are within a certain time interval.
    The starting point can be specified by the index.
    When the end of the interval is reached, the stop-index is returned as well.
    The stop-index is suitable as input for the subsequent run.


    Tweets should be sorted by their IDs or time-stamps!
    But sometimes it isn`t enough to sort the tweets by their ID to achieve increasing time-stamps!


    :param ndjson_file: path specification (str)
    :param interval: seconds (int)
    :param start_tweet: index containing the starting point (int)
    :return: tweets and stop-index
    """

    tweet_list = []
    time_list = []
    count = 0

    with jsonlines.open(ndjson_file) as reader:
        for obj in reader.iter(type=dict, skip_invalid=False):

            if count >= start_tweet:
                current_time = datetime.strptime(obj["created_at"], '%Y-%m-%d %H:%M:%S')
                time_list.append(current_time)
                obj["created_at"] = str(current_time)
                delta = time_list[-1] - time_list[0]
                #if delta.seconds >= interval:
                if delta.seconds == interval:
                    break
                else:
                    tweet_list.append(obj)

            count += 1
    return tweet_list, count




def export_tweets(tweets, ndjson_file, mode):
    """
    If the file already exists, the data can be appended (mode="a") or overwritten (mode="w")
    :param tweets: list of dictionaries (list)
    :param ndjson_file: path specification (str)
    :return:
    """
    with jsonlines.open(ndjson_file, mode=mode) as writer:
        writer.write_all(tweets)




def ndjson_export_daily(nd_json_list, output_dir, basename):
    """
    Reads a list of multiple ndjson files and splits the tweets over the different days.
    For each day a ndjson file is created.

    :param nd_json_list: path specifications to the .ndjson (list)
    :param output_dir: path specificaiton (str)
    :param basename: e.g.: "Event2012" (str)
    :return: "Event2012_2012-11-07.ndjson"
    """

    for jsons in nd_json_list:
        tweets = import_tweets(jsons, avoidRT=True)

        # find all days in current json_file
        counting = Counter(tok["created_at"][:10] for tok in tweets)
        counting = sorted(counting.items())
        days = [i[0] for i in counting]

        # export/or expand files
        for day in days:
            daily_tweet = []
            for tweet in tweets:
                if tweet["created_at"][:10] == day:
                    daily_tweet.append(tweet)
            filename = os.path.join(output_dir, basename + "_" + day + ".ndjson")
            export_tweets(tweets=daily_tweet, ndjson_file=filename, mode="a")




def ndjson_sort_tweets(nd_json_list, dic_key):
    """
    Sort tweets in each ndjson_file based on the dic_key

    :param nd_json_list: list of ndjson_files
    :param dic_key: dictionary key
    :return: exports ndjson to same directory
    """

    for jsons in nd_json_list:
        tweets = import_tweets(jsons)
        tweets = sorted(tweets, key=lambda k: k[dic_key])
        export_tweets(tweets=tweets, ndjson_file=jsons[:-7]+"_"+str(dic_key)+".ndjson", mode="w")




def count_tweets(ndjson_file_list):
    """
    Examines ndjson-files and prints the number of valid tweets, missing tweets (presented just with their ID)
    and the number of retweets.

    :param ndjson_file_list: containing one or multiple ndjson files (list)
    :return:
    """
    missing = 0
    tweets = 0
    retweets = 0

    for file in ndjson_file_list:
        with jsonlines.open(file) as reader:
            for obj in tqdm(reader.iter(type=dict, skip_invalid=True)):  # Hier evtl false besser??
                if "created_at" in obj:
                    tweets += 1
                    if obj["RT"] == "true":
                        retweets += 1
                else:
                    missing += 1

    print("Fetched Tweets: " + str(tweets))
    print("Retweets:     " + str(retweets) + " (" + str(int((100 / tweets) * retweets)) + "%)")
    print("Missing:      " + str(missing) + " (" + str(int((100 / (tweets + missing) * missing))) + "%)")