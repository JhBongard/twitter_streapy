


import tensorflow_hub as hub
import numpy as np
import tf_sentencepiece
import tensorflow_text


embed = hub.KerasLayer("/media/jb/Volume/twitter_streapy/use_models/USE_large_4/universal-sentence-encoder-large_4")
#embed = hub.KerasLayer("/media/jb/Volume/twitter_streapy/use_models/USE_multilingual_largeV3")


def use_embedding(tweets):
    embeddings = embed([tweets[0]["text"]])["outputs"].numpy()

    for i in range(1,len(tweets),2500):
        embedding = embed([y["text"] for y in tweets[i:i + 2500]])["outputs"].numpy()
        embeddings = np.vstack((embeddings, embedding))
    return embeddings

def sentence_embed(sentence):
    return embed(sentence)["outputs"].numpy()

