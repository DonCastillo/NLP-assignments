#  Using the Movie Reviews corpus available in NLTK, 
# write a Python program *classify.py* that classify movies according to their sentiment polarity, 
# and evaluate your program. You should use the Naive Bayes classifier, 
# and split the data into training and testing (e.g., 70\% and 30\%).


import nltk
import random
import string
from nltk import FreqDist, NaiveBayesClassifier
from nltk.corpus import movie_reviews, stopwords
# download movie_reviews corpus
# nltk.download('movie_reviews')
# nltk.download('stopwords')


def clean_document(word_list):
    """removes all unimportant words like punctuations and stopwords

    :param word_list: list of words
    :type word_list: list
    :return: clean list of words
    :rtype: list
    """
    word_list_temp = []
    for word in word_list:
        if (word not in stopwords.words("english")) and (word not in string.punctuation):
            word_list_temp.append(word)
    return word_list_temp


def get_all_documents(corpus):
    """get all documents with associated category

    :param corpus: corpus
    :type corpus: corpus
    :return: list of tuples in a format: ([document's list of words], category)
    :rtype: list
    """
    all_documents_temp = []
    for category in corpus.categories():
        for fileid in corpus.fileids(category):
            all_documents_temp.append((corpus.words(fileid), category))
    random.shuffle(all_documents_temp)
    return all_documents_temp


def get_top_unique_words(corpus, n):
    """list the most frequent unique words

    :param corpus: corpus
    :type corpus: corpus
    :param n: only consider the first n in the list
    :type n: int
    :return: list of words
    :rtype: list
    """
    unique_words = list(FreqDist([word.lower()
                                  for word in corpus.words()]))[:n]
    unique_words = clean_document(unique_words)
    return unique_words


def main():

    all_documents = get_all_documents(movie_reviews)
    top_unique_words = get_top_unique_words(movie_reviews, 2000)

    def document_features(document):
        """checks if the top words in the corpus exists in the document

        :param document: document to be examined
        :type document: list
        :return: set that lists the occurrences of the words in the documents 
        :rtype: set
        """
        document_uniq_word = set(document)
        features = {}
        for word in top_unique_words:
            features[word] = word in document_uniq_word
        return features

    feature_sets = []
    for (document, category) in all_documents:
        feature_sets.append((document_features(document), category))

    training_set = feature_sets[:1400]
    testing_set = feature_sets[1400:]

    classifier = NaiveBayesClassifier.train(training_set)
    accuracy = nltk.classify.accuracy(classifier, testing_set)

    print("Accuracy rate: {}".format(accuracy))
    classifier.show_most_informative_features(10)

    col_1 = "TEST DOCUMENT #"
    col_2 = "ACTUAL CLASS"
    col_3 = "CLASSIFIED BY NAIVE BAYES"
    title = "SOME OF THE TEST SET RESULT"

    print()
    print("-" * (len(col_1) + len(col_2) + len(col_3) + 10))
    print(f"{title:^{len(col_1) + len(col_2) + len(col_3) + 10}}")
    print("-" * (len(col_1) + len(col_2) + len(col_3) + 10))
    print("| {} | {} | {} |".format(col_1, col_2, col_3))

    for (index, (document, category)) in enumerate(testing_set[:20]):
        guess = classifier.classify(document)
        print(
            f"| {index:^{len(col_1)}} | {category:^{len(col_2)}} | {guess:^{len(col_3)}} |")


main()
