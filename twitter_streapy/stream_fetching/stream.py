
#from __future__ import absolute_import, print_function
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import jsonlines





# Use either geobox or keywords!!!

def twitter_stream(key_path, outfile, languages, geobox=None, keywords=None):
    """
    Connects to the Twitter Streaming API and stores Tweets locally.
    The variety of tweets can be determined by language, location or keywords
    Streaming API doesn't allow to filter by location AND keyword simultaneously!!! --> Use either keywords or geobox


    :param key_path: path to .txt-file containing:
                        consumer_key,consumer_secret,access_token,access_token_secret
                        (oneliner with comma, without space!)
                        create developer account: https://developer.twitter.com/en/apply-for-access

    :param outfile: path specification .ndjson format (str)

    :param languages: list containing language code identifier
                        (see: https://github.com/libyal/libfwnt/wiki/Language-Code-identifiers)

    :param geobox: WGS84 geobox (LNG/LAT) (e.g.: http://bboxfinder.com)
    :param keywords: list of keywords
    :return:
    """


    # Authenticate with Twitter
    # load .txt file & extract the keys
    f = open(key_path, "r")
    keys = f.read().split(",")
    f.close()

    auth = OAuthHandler(keys[0], keys[1])
    auth.set_access_token(keys[2], keys[3])
    # Create an API object
    api = tweepy.API(auth)


    # Twitter Stream requires a user-defined listener class
    class StdOutListener(StreamListener):
        """
        A listener handles tweets that are received from the stream.
        All objects of a tweet are listed here:
        https://developer.twitter.com/en/docs/tweets/data-dictionary/overview/tweet-object
        """

        def on_status(self, status):
            # discarding retweets
            #if retweets != None:
            #    if status.retweeted:
            #        return

            # get the tweet text
            try:
                # Check if there is a full/long text
                # "full_text" contains tweets with 140<280 characters (since 2017)
                text = status.extended_tweet["full_text"]
            except AttributeError:
                # if no full_text is available get standard Tweet
                text = status.text

            # get further important tweet objects
            id_str = status.id_str
            created_at = str(status.created_at)
            coords = str(status.coordinates)
            #place = status.place.country
            # name = status.user.screen_name
            # fav = status.favorite_count
            # description = status.user.description
            #loc = status.user.location
            # user_created = status.user.created_at
            # followers = status.user.followers_count

            tweet = {"id": id_str,
                     "created_at": created_at,
                     "text": text,
                     "coordinates": coords}


            # write into file using jsonlines
            with jsonlines.open(outfile, mode='a') as writer:
                writer.write(tweet)

            #print(tweet)
            print(status.text)

        def on_error(self, status_code):
            # disconnect rate limit is reached (Error 420)
            if status_code == 420:
                return False

    stream = Stream(auth=api.auth, listener=StdOutListener())
    # doesnt allow to filter for geobox and keywords!
    if geobox:
        stream.filter(locations=geobox, languages=languages)
    else:
        stream.filter(track=keywords, languages=languages)



# Examples
#twitter_stream(key_path="D:/01_FSU_Jena/02_Master/Masterarbeit/Data/Keys/twitterkeys.txt",
#               outfile="D:/01_FSU_Jena/02_Master/Masterarbeit/Data/Keys/stream_geo.ndjson",
#               languages=[],
#               geobox=[17.363892,58.975809,18.880005,59.664907])

#twitter_stream(key_path="D:/01_FSU_Jena/02_Master/Masterarbeit/Data/Keys/twitterkeys.txt",
#               outfile="D:/01_FSU_Jena/02_Master/Masterarbeit/Data/Keys/stream_geo.ndjson",
#               languages=["en", "de", "fr"],
#               keywords=["Covid-19"])



