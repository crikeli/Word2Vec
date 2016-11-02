# To read csv files
import pandas as pd
from bs4 import BeautifulSoup
import re
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import PunktSentenceTokenizer

trainingData = pd.read_csv("labeledTrainData.tsv", header = 0, delimiter="\t", quoting=3)
testData = pd.read_csv("testData.tsv", header=0, delimiter="\t", quoting=3)
unlabeledTrainingData = pd.read_csv("unlabeledTrainData.tsv", header=0, delimiter="\t", quoting=3)

# We verify the number of reviews read in total (100,000).
# print "Successfully read %d labeledTrainData, %d testData"\
# "and %d unlabeledTrainingData\n" %(trainingData["review"].size, testData["review"].size, unlabeledTrainingData["review"].size)

# the word2vec model of analyzing text is different from bag of words as we do not need to worry about stopwords.
# Because the algorithm relies on a broad context of the word, so stopwords matter.

def rev2wordList(review, remove_stopwords=False):
    # The document passed in is converted to a sequence of words.
    # A list of words is returned

    # We first apply BeautifulSoup to a specific review to rid it of any html tags
    reviewText = BeautifulSoup(review, "html.parser").get_text()
    # print reviewText

    # We then remove any non-letters and numbers
    reviewText = re.sub("[^a-zA-Z]", " ", review)

    # The words are then converted to lowercase & split
    words = reviewText.lower().split()

    # Optionally, we can remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    # print "Words:",words
    # Output: Words: ['it', 'must', 'be', 'assumed', 'that', 'those', 'who', 'praised', 'this', 'film', 'the', 'greatest', 'filmed', 'opera', 'ever', 'didn', 't', 'i', 'read', 'somewhere', 'either', 'don', 't', 'care', 'for', 'opera', 'don', 't', 'care', 'for', 'wagner', 'or', 'don', 't', 'care', 'about', 'anything', 'except', 'their', 'desire', 'to', 'appear', 'cultured', 'either', 'as', 'a', 'representation', 'of', 'wagner', 's', 'swan', 'song', 'or', 'as', 'a', 'movie', 'this', 'strikes', 'me', 'as', 'an', 'unmitigated', 'disaster', 'with', 'a', 'leaden', 'reading', 'of', 'the', 'score', 'matched', 'to', 'a', 'tricksy', 'lugubrious', 'realisation', 'of', 'the', 'text', 'br', 'br', 'it', 's', 'questionable', 'that', 'people', 'with', 'ideas', 'as', 'to', 'what', 'an', 'opera', 'or', 'for', 'that', 'matter', 'a', 'play', 'especially', 'one', 'by', 'shakespeare', 'is', 'about', 'should', 'be', 'allowed', 'anywhere', 'near', 'a', 'theatre', 'or', 'film', 'studio', 'syberberg', 'very', 'fashionably', 'but', 'without', 'the', 'smallest', 'justification', 'from', 'wagner', 's', 'text', 'decided', 'that', 'parsifal', 'is', 'about', 'bisexual', 'integration', 'so', 'that', 'the', 'title', 'character', 'in', 'the', 'latter', 'stages', 'transmutes', 'into', 'a', 'kind', 'of', 'beatnik', 'babe', 'though', 'one', 'who', 'continues', 'to', 'sing', 'high', 'tenor', 'few', 'if', 'any', 'of', 'the', 'actors', 'in', 'the', 'film', 'are', 'the', 'singers', 'and', 'we', 'get', 'a', 'double', 'dose', 'of', 'armin', 'jordan', 'the', 'conductor', 'who', 'is', 'seen', 'as', 'the', 'face', 'but', 'not', 'heard', 'as', 'the', 'voice', 'of', 'amfortas', 'and', 'also', 'appears', 'monstrously', 'in', 'double', 'exposure', 'as', 'a', 'kind', 'of', 'batonzilla', 'or', 'conductor', 'who', 'ate', 'monsalvat', 'during', 'the', 'playing', 'of', 'the', 'good', 'friday', 'music', 'in', 'which', 'by', 'the', 'way', 'the', 'transcendant', 'loveliness', 'of', 'nature', 'is', 'represented', 'by', 'a', 'scattering', 'of', 'shopworn', 'and', 'flaccid', 'crocuses', 'stuck', 'in', 'ill', 'laid', 'turf', 'an', 'expedient', 'which', 'baffles', 'me', 'in', 'the', 'theatre', 'we', 'sometimes', 'have', 'to', 'piece', 'out', 'such', 'imperfections', 'with', 'our', 'thoughts', 'but', 'i', 'can', 't', 'think', 'why', 'syberberg', 'couldn', 't', 'splice', 'in', 'for', 'parsifal', 'and', 'gurnemanz', 'mountain', 'pasture', 'as', 'lush', 'as', 'was', 'provided', 'for', 'julie', 'andrews', 'in', 'sound', 'of', 'music', 'br', 'br', 'the', 'sound', 'is', 'hard', 'to', 'endure', 'the', 'high', 'voices', 'and', 'the', 'trumpets', 'in', 'particular', 'possessing', 'an', 'aural', 'glare', 'that', 'adds', 'another', 'sort', 'of', 'fatigue', 'to', 'our', 'impatience', 'with', 'the', 'uninspired', 'conducting', 'and', 'paralytic', 'unfolding', 'of', 'the', 'ritual', 'someone', 'in', 'another', 'review', 'mentioned', 'the', 'bayreuth', 'recording', 'and', 'knappertsbusch', 'though', 'his', 'tempi', 'are', 'often', 'very', 'slow', 'had', 'what', 'jordan', 'altogether', 'lacks', 'a', 'sense', 'of', 'pulse', 'a', 'feeling', 'for', 'the', 'ebb', 'and', 'flow', 'of', 'the', 'music', 'and', 'after', 'half', 'a', 'century', 'the', 'orchestral', 'sound', 'in', 'that', 'set', 'in', 'modern', 'pressings', 'is', 'still', 'superior', 'to', 'this', 'film']
    return(words)

# rev2wordList(trainingData["review"][3])

# In order to run the word2vec algorithm, we need  to make sure that we provide an input of single sentences, each one as a list of words. ie. a list of a list.
# In order to format sentences in an acceptable format for word2vec, we use the punkt tokenizer.
# The punkt tokenizer is pretrained and it knows that words such as Mr. Smith and Kelin K. Christi do not mark sentence boundaries.
# The punkt tokenizer also detects sentences that don't always start with non-capitalized letters & vice versa.

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
# tokenizer = PunktSentenceTokenizer(review)

# We define a function that splits a review into parsed sentences. It returns a list of sentences where each sentence is a list of words.
def rev2Sent(review, tokenizer, remove_stopwords=False):
    # The punkt tokenizer is used to split each paragraph into sentences
    rawSentences = tokenizer.tokenize(review.strip())

    sentences=[]
    for rawSentence in rawSentences:
        if len(rawSentence) > 0:
            sentences.append(rev2wordList(rawSentence))

    # print "Sentences:", sentences
    return sentences

sentences2 = []
print "Parsing sentences from the training set"
for i in trainingData["review"]:
    sentences2 += rev2Sent(i.decode("utf-8"), tokenizer)

print "Parsing sentences from the unlabeled training data"
for i in unlabeledTrainingData["review"]:
    sentences2 += rev2Sent(i.decode("utf-8"), tokenizer)

print "Total Number of Sentences:",len(sentences2)
# Total Number of Sentences: 795538

print "Print Sentence 0",sentences2[0]
# Print Sentence 0 [u'with', u'all', u'this', u'stuff', u'going', u'down', u'at', u'the', u'moment', u'with', u'mj', u'i', u've', u'started', u'listening', u'to', u'his', u'music', u'watching', u'the', u'odd', u'documentary', u'here', u'and', u'there', u'watched', u'the', u'wiz', u'and', u'watched', u'moonwalker', u'again']


print "Print Sentence 1",sentences2[1]
# Print Sentence 1 [u'maybe', u'i', u'just', u'want', u'to', u'get', u'a', u'certain', u'insight', u'into', u'this', u'guy', u'who', u'i', u'thought', u'was', u'really', u'cool', u'in', u'the', u'eighties', u'just', u'to', u'maybe', u'make', u'up', u'my', u'mind', u'whether', u'he', u'is', u'guilty', u'or', u'innocent']
