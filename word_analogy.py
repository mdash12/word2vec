import os
import pickle
import numpy as np
from numpy import dot
from numpy.linalg import norm
from scipy import spatial

model_path = './models/'
loss_model = 'cross_entropy'
# loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%s.model' % (loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""

examples = []


def calc_max_diff(source, target):
    diff = np.subtract(embeddings[dictionary[source]], embeddings[dictionary[target]])
    return diff


def cos_sim(a, b):
    # return np.sum(a*b)
    return (1 - spatial.distance.cosine(a,b))


file_name = 'word_analogy_' + loss_model + '.txt'
fw = open(file_name, 'w+')

with open('word_analogy_test.txt') as f:
    for line in f:
        line_split = line.split("||")
        examples = line_split[0].split(",")
        choices = line_split[1].split(",")
        choices[-1] = choices[-1].strip('\n')
        x = []
        es = embeddings[dictionary[examples[0].strip('"').split(":")[0]]]
        avg_diff = np.zeros(es.shape[0])
        # print("shape: ", avg_diff.shape)
        for example in examples:
            el = example.strip('"').split(":")
            diff = np.subtract(embeddings[dictionary[el[0]]], embeddings[dictionary[el[1]]])
            avg_diff = np.add(avg_diff, diff)
        avg_diff /= len(examples)

        y = []
        choice_diff = []
        for choice in choices:
            el = choice.strip('"').split(":")
            y.append(el)
            choice_diff.append(cos_sim(avg_diff, calc_max_diff(el[0], el[1])))
        max_sim = max(choice_diff)
        max_pair = y[choice_diff.index(max(choice_diff))]
        # print(" max_sim: ", max_sim)
        # print("max_pair: ", max_pair)

        min_sim = min(choice_diff)
        min_pair = y[choice_diff.index(min(choice_diff))]
        # print(" min_sim: ", min_sim)
        # print("min_pair: ", min_pair)


        max_pair = ':'.join(max_pair)
        min_pair = ':'.join(min_pair)
        c = ' '.join(choices)

        c += ' "' + min_pair + '" "' + max_pair + '"'
        fw.write(c+"\n")

fw.close()
