import pandas as pd
import numpy as np
import csv
import re
from collections import Counter
from keras.preprocessing import sequence
from keras.utils import np_utils

phrase = []
labels = []
test_phrase = []

with open("train.tsv") as fd:
    rd = csv.reader(fd, delimiter="\t", quotechar='"')
    for row in rd:
        phrase.append(row[2])
        labels.append(row[3])

with open("test.tsv") as testing:
    test = csv.reader(testing, delimiter="\t", quotechar='"')
    for s in test:
        test_phrase.append(s[2])

del phrase[0]
del test_phrase[0]
del labels[0]

def clean_phrase(phrase):
    words = (re.sub("[^a-zA-Z]", " ", phrase)).lower()
    return words

clean_phrases = []
for x in phrase:
    new = clean_phrase(x)
    clean_phrases.append(new)

test_clean_phrases = []
for xw in test_phrase:
    new_test = clean_phrase(xw)
    test_clean_phrases.append(new_test)

all_text=' /n '.join(clean_phrases)
test_all_text=' /n '.join(test_clean_phrases)

reviews = all_text.split(' /n ')
all_text = ' '.join(reviews)

words = all_text.split()

test_reviews = test_all_text.split(' /n ')
test_all_text = ' '.join(test_reviews)
test_words = test_all_text.split()

labels_cleaned = '\n'.join(labels)
labels_cleaned_last = labels_cleaned.split('\n')

labels_sentiment = [int(i) for i in labels_cleaned_last]
labels = np.array(labels_sentiment)

full_words = words + test_words

counts = Counter(full_words)
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: ii for ii, word in enumerate(vocab, 1)}

reviews_ints = []
for each in reviews:
    reviews_ints.append([vocab_to_int[word] for word in each.split( )])
    
test_reviews_ints = []
for eachs in test_reviews:
    test_reviews_ints.append([vocab_to_int[word] for word in eachs.split( )])


review_lens = Counter([len(x) for x in reviews_ints])
print("Zero-length reviews: {}".format(review_lens[0]))
print("Maximum review length: {}".format(max(review_lens)))

non_zero_idx = [ii for ii, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[ii] for ii in non_zero_idx]
labels = np.array([labels[ii] for ii in non_zero_idx])

review_lens = Counter([len(x) for x in reviews_ints])

max_review_length = 12
X_train = sequence.pad_sequences(reviews_ints, maxlen=max_review_length)
x_test = sequence.pad_sequences(test_reviews_ints, maxlen=max_review_length)

y_train = np_utils.to_categorical(labels, 5)





















