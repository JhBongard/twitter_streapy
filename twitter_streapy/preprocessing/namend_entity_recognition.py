

import stanza

# check if pytorch is installed correctly (just re-install: conda install -c pytorch pytorch)

#nlp = stanza.Pipeline(lang='en', processors='tokenize,ner', use_gpu=True)
#def ner_stanza(text):
#    """
#    Stanford NLP
#    support for using GPU
#    """
#    # NER
#    # first excecution: download the pre-trained model language
#    #stanza.download('en')
#    ner_list=[]
#    for i in nlp(text).ents:
#        ner_list.append(i.text)
#
#    return ner_list





nlp = stanza.Pipeline(lang="en", processors="tokenize, mwt, pos", use_gpu=True)
def find_nouns(text):

    noun_list = []#

    doc = nlp(text)
    for i in doc.sentences:
        noun_list.extend([[word.text for word in i.words if word.upos=="NOUN"]])

    return [item for sublist in noun_list for item in sublist]











