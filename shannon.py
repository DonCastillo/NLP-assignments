# Using a language model generated from the Brown corpus which is available in NLTK,
# write a Python program shannon.py that generates random sentences
# (i.e., Shannon Visualization Method) using:
# bigrams
# trigrams


import nltk
# download brown corpus
# nltk.download('brown')
import random
from nltk.corpus import brown
from nltk.util import ngrams

STR_TOKEN = "<s>"
END_TOKEN = "</s>"


class Corpus:
    def __init__(self):
        corpus = brown.sents(fileids=["ca16"])
        corpus = self.__augment_corpus(corpus)
        corpus = self.__flatten_corpus(corpus)
        self.corpus = corpus

    def __flatten_corpus(self, corpus):
        temp_corpus = []
        for sen in corpus:
            for word in sen:
                temp_corpus.append(word)
        return temp_corpus

    def __augment_corpus(self, corpus):
        temp_corpus = []
        for sen in corpus:
            sen.append(END_TOKEN)
            sen.insert(0, STR_TOKEN)
            temp_corpus.append(sen)
        return temp_corpus

    def get_corpus(self):
        return list(self.corpus)


class N_GramModel:
    def __init__(self, corpus, n_gram):
        self.corpus = list(corpus)
        self.ngram_list = list(ngrams(corpus, n_gram))
        self.ngram_seq = []
        self.n = n_gram

    def __freq_context(self, prev_words):
        freq = 0
        for tup in self.ngram_list:
            if prev_words == tup[0:self.n - 1]:
                freq += 1
        return freq

    def __get_next_tuple(self, prev_tuple):
        prev_words = prev_tuple[1:]  # (a, b, c, d, e) => (b, c, d, e)
        self.ngram_dict = {}

        for tup in self.ngram_list:
            # (b, c, d, e) == (b, c, d, e, f)
            if prev_words == tup[0:(self.n - 1)]:
                freq_ngram = self.ngram_list.count(tup)
                freq_prev = self.__freq_context(prev_words)
                self.ngram_dict[tup] = freq_ngram / freq_prev

        candidate_tuples = []
        maxim = max(self.ngram_dict.values())
        for k, v in self.ngram_dict.items():
            if v == maxim:
                candidate_tuples.append(k)

        shuffled = random.sample(candidate_tuples, len(candidate_tuples))
        return shuffled[0]

    def __get_first_tuple(self):
        starting_tuples = []
        for tup in self.ngram_list:
            if tup[0] == STR_TOKEN:
                starting_tuples.append(tup)
        r = random.randrange(len(starting_tuples))
        return starting_tuples[r]

    def __make_sentence(self, ngram_seq):
        sentence_list = []
        for i, v in enumerate(ngram_seq):
            if i == 0:
                sentence_list += list(v)
            else:
                sentence_list.append(list(v).pop())
        sentence = " ".join(sentence_list)
        sentence = sentence.replace(STR_TOKEN, "")
        sentence = sentence.replace(END_TOKEN, "")
        return sentence

    def generate_random_sentence(self):
        self.ngram_seq.append(self.__get_first_tuple())
        while(True):
            prev_tuple = self.ngram_seq[len(self.ngram_seq) - 1]

            if END_TOKEN in prev_tuple:
                break
            else:
                next_tuple = self.__get_next_tuple(prev_tuple)
                self.ngram_seq.append(next_tuple)
        # print(self.ngram_seq)
        # print()
        return self.__make_sentence(self.ngram_seq)

    def get_ngram(self):
        return list(self.ngram_list)


def main():
    corpus = Corpus()
    print("Random sentences using")
    print()

    ngram = N_GramModel(corpus.get_corpus(), 2)
    print('Bigram:')
    print(ngram.generate_random_sentence())
    print()

    ngram = N_GramModel(corpus.get_corpus(), 3)
    print('Trigram:')
    print(ngram.generate_random_sentence())
    print()


# 1.4.1 The Shannon Visualization Method
# Choose a random bigram (<s>, w) according to its probability
# Now choose a random bigram (w, x) according to its probability
# And so on until we choose </s>
# Then string the words together
main()
