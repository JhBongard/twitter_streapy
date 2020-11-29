
import re
from nltk import ngrams
from collections import Counter
import string
import itertools
import numpy as np
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer



def preprocessing(tweet, word_count_min):
    """
    Function tokenizing the tweets to extract the url, mentions, hashtags & coordinates.
    While mentions and hashtags remains in the text, the url gets removed.
    If the number of words are higher than word_count_min the tweet is returned.

    param:
    param:
    """

    # tokenization
    tokens = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False).tokenize(tweet["text"])
    # remove all symbols which are stored in string.punctuation
    tokens = [token for token in tokens if not token in string.punctuation + "“”’‘..."]
    # extract URL
    tweet["url"] = [i for i in tokens if i.startswith("http")]
    # extract @
    tweet["mention"] = [i for i in tokens if i.startswith("@") and len(i)>2] # avoiding false @ with whitespace
    # extract #
    tweet["hashtag"] = [i for i in tokens if i.startswith("#") and len(i)>2] # avoiding false # with whitespace

    if tweet["coordinates"] != None:
        tweet["coordinates"] = tweet["coordinates"]["coordinates"]

    # delete extracted tokens
    tokens = [i for i in tokens if not i.startswith(("http", "#", "@"))]
    # count words
    tokens = [re.sub(r"[^a-zA-Z]+", '', i) for i in tokens]
    while '' in tokens:
        tokens.remove('')
    word_count = sum([i != "" for i in tokens])

    if word_count >= word_count_min:
        # NER
        tweet["ner"] = []#tweet["ner"] = tw_ner.ner_stanza(tweet["text"])
        # remove URL
        tweet["text"] = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', tweet["text"])

        return tweet



def count_words(text):
    """
    count words, no hashtags mentions or URLs or numbers
    :param text:
    :return:
    """
    tokens = TweetTokenizer(preserve_case=False, reduce_len=False, strip_handles=False).tokenize(text)
    tokens = [i for i in tokens if not i.startswith(("http", "#", "@"))]
    tokens = [re.sub(r"[^a-zA-Z]+", '', i) for i in tokens]
    word_count = sum([i != "" for i in tokens])
    return word_count




def remove_emojies(text):
    """
    :param text:
    :return:
    """
    text = emoji.get_emoji_regexp().sub(u"",text)
    return text



def tokenizer(text):
    """
    :param dataframe:
    :return:
    """
    corpus = text.replace("’","'") # Tweet Tokenizer can handle ' better than ’ or ‘
    corpus = corpus.replace("‘", "'")
    tokens = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=False).tokenize(str(corpus))

    # remove all symbols which are stored in string.punctuation
    tokens = [token for token in tokens if not token in string.punctuation+"“”..."]

    # Removing URLs, Hashtags, Mentions & Emojies
    tokens = [i for i in tokens if not i.startswith(("http", "#", "@"))]
    tokens = [remove_emojies(i) for i in tokens]

    # Removing stop words (like: of, the, a, an, ...)
    # nltk.download('stopwords') # download stopwords on first code excecution
    cached_stopwords = stopwords.words("english")
    tokens = [i for i in tokens if i not in cached_stopwords]

    # Removing tokens containing whitespaces resulting
    while '' in tokens:
        tokens.remove('')
    while '️' in tokens:
        tokens.remove('️')

    return tokens



def get_cluster_top_gram(tweets, clusterlabels):
    cluster_trigram_list = []

    for label in np.unique(clusterlabels):
        bool_flags = [i for i in clusterlabels == label] # finding the tweets of the current cluster
        tweets_subset = list(itertools.compress(tweets, bool_flags)) # extract them
        cluster_tokens = [tokenizer(i["text"]) for i in tweets_subset] # get tokens

        trigrams = []
        for i in cluster_tokens:
            trigrams.extend(ngrams(i,3)) # calculate trigrams of cluster token
        trigram_freq = Counter(trigrams) # count similar trigrams
        top_trigram = trigram_freq.most_common(1) # find top trigram
        cluster_trigram_list.append(top_trigram)

    return cluster_trigram_list
