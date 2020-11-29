
from __future__ import absolute_import, print_function
import tweepy
import pandas as pd
import json
import time
import jsonlines
import tqdm
import os
from tqdm import tqdm








def splitting_tsv(tsv_path, col, parts, output_dir):
    """
    Imports a tsv-file as pandas dataframe, selects a column and splits the df in equal parts. Each part is exported as
    tsv-file

    :param tsv_path: location of the .tsv-file (str)
    :param col: Defining the pandas column containing the IDs (int)
    :param parts: Defining the number of parts by (int)
    :param output_dir: Name of the directory (str)
    :return: .tsv-files
    """

    to_download = pd.read_csv(tsv_path, sep="\t", header=None)[col]  # 1 or 0
    amount = int(len(to_download) /parts) + (len(to_download) % parts > 0)

    for enu, i in enumerate(range(0, len(to_download), amount)):
        part = pd.DataFrame(to_download[i:i + amount])
        # export as tsv
        filename = str(os.path.basename(tsv_path).split(".")[0]) + "_part" + str(enu) + ".tsv"
        part.to_csv(os.path.join(output_dir, filename), sep="\t", index=False, header=False)





def define_query(json_file, tsv_file, col, query_tsv=None):
    """
    This function is useful if the tweet download breaks down or to check if all tweets have been downloaded.

    It compares the downloaded tweets (ndjson file) with the all the tweet IDs (tsv_file)
    and defines a new query list.

    If query_tsv is specified the query is written to a new .tsv-file.

    :param json_file: location of the .ndjson file (str)
    :param tsv_file: location of the .tsv-file (str)
    :param col: Defining the pandas column containing the IDs (int)
    :param query_tsv: Optional: Path to the new query tsv file (str) (Default=None)
    :return: A list object containing the IDs that are still to be downloaded
    """

    # Read ids from downloaded tweets
    downloaded = []
    with jsonlines.open(json_file) as reader:
        for obj in reader.iter(type=dict, skip_invalid=True):
            downloaded.append(obj["id"])

    # Read list with all tweets ids
    to_download = list(pd.read_csv(tsv_file, sep='\t', header=None)[col])
    # Compare both lists
    query = list(set(to_download) - set(downloaded))
    # Export query to tsv
    query_df = pd.DataFrame(query)
    if query_tsv:
        query_df.to_csv(query_tsv, sep="\t", index=False, header=False)

    return(query)




def append_json(tweet_list, json_file):

    """
    Creates or appends a list of tweets to a ndjson file.
    At the same time the tweets are shortened to selected parameters

    Since there is no compact Boolean parameter available to identify retweets, the parameter "RT" was added,
    confirming the presence of the parameter "retweeted_status" with true or false
    More information: https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object

    Attention: no long-text included! Valid for Tweets before 2017
    :param tweet_list: fetched tweets from Twitter stream (list)
    :param json_file: output directory  (str)
    :return:
    """
    with open(json_file, "a") as f:
        for i in tweet_list:
            try:
                tweet = {k: i._json[k] for k in ("id", "created_at", "text", "geo", "coordinates", "place",
                                                 #"retweeted", "favorite_count",
                                                 "retweet_count",  "lang")}
                tweet.update({"user_location": i._json["user"]["location"]})

                if "retweeted_status" in i._json:
                    tweet.update({"RT": "true"})
                else:
                    tweet.update({"RT": "false"})

            except:
                # If the tweet isn't accessible, just print the ID
                tweet = i._json
            json.dump(tweet, f)
            f.write("\n")




def id_fetching(key_path, tsv_file, col, json_file):
    """
    Using Tweepy's "AppAuthHandler" instead of the commonly used "AccessTokenAuth"
    because of higher limits (resulting in ca. 30000 tweets / 12 minutes)
    - limitation of maximum 100 tweets per request
    - 300 of these bulk requests are possible in 12 minutes
    - limits are automatically defined by tweepy (But if error messages occur the function waits 15 minutes)

    :param key_path: location of the .txt key file
                        containing: APIkey,APIsecretkey,Accesstoken,Accesstokensecret
                        (separated by a comma without space)
    :param tsv_file: location of the .tsv file (str)
    :param col: Defining the pandas column containing the IDs (int)
    :param json_file: location of the json_file file (str)

    :return: Creates or appends fetched tweets to the json_file.
    """

    # if tweets are already downloaded to the json_file, create a new query
    if os.path.isfile(json_file):
        query = define_query(json_file=json_file, tsv_file=tsv_file, col=col)
    # otherwise load query from tsv file
    else:
        query = list(pd.read_csv(tsv_file, sep='\t', header=None)[col])

    # Tweepy settings
    f = open(key_path, "r")
    keys = f.read().split(",")
    f.close()
    auth = tweepy.AppAuthHandler(keys[0], keys[1])

    # Create an API object
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    # Create bulk request with 100 Tweets
    for i in tqdm(range(0, len(query), 100)):
        try:
            # Fetch tweets 100 per request
            fetched = api.statuses_lookup(query[i:i + 100], include_entities=False, map_=True)

            # Append the fetched list to json file
            append_json(fetched, json_file)

        except tweepy.TweepError as e:
            # If an error occurs wait 15 min
            print("Error occured continuing in 15 Min")
            time.sleep(60 * 15)
            continue

    print("Finished!")












