# Given 2 text documents, write your own function that computes the similarity between the 2 documents using the
# cosine similarity measure. The documents are represented using the TF-IDF.  Apply your function to the Movie
# Review corpus to compute the average similarity of the positive and the negative reviews.
import string
import math
from nltk.corpus import movie_reviews, stopwords


class CompareModel:
    docs_collection = []
    vocabulary = []
    idf = {}

    def __init__(self, docs_collection):
        for f in docs_collection:
            d = self.__sanitize_doc(f)
            self.docs_collection.append(d)


    def __sanitize_doc(self, doc):
        doc_temp = []
        for word in doc:
            if (word not in stopwords.words("english")) and (word not in string.punctuation):
                doc_temp.append(word)
        return doc_temp


    def __extract_vocabulary(self):
        set_temp = set()
        for doc in self.docs_collection:
            for word in doc:
                set_temp.add(word)
        return sorted(list(set_temp))


    def __set_idf(self):
        N = len(self.docs_collection)
        for word in self.vocabulary:
            df = 0
            for doc in self.docs_collection:
                if word in doc:
                    df += 1
            self.idf[word] = math.log10(N / df)


    def __set_doc_vector(self, doc):
        doc_vector = {}
        for v in self.vocabulary:
            tf = math.log10(doc.count(v) + 1)
            doc_vector[v] = tf * self.idf[v]
        return doc_vector


    def __cos_similarity(self, vct1, vct2):
        vw = 0 
        v2 = 0
        w2 = 0
        for v in self.vocabulary:
            vw += vct1.get(v) * vct2.get(v)
        for k, v in vct1.items():
            v2 += v ** 2
        for k, v in vct2.items():
            w2 += v ** 2
        return (vw / (math.sqrt(v2) * math.sqrt(w2)))
            

    def compare(self, doc1, doc2):
        self.vocabulary = self.__extract_vocabulary()
        self.__set_idf()
        doc1 = self.__sanitize_doc(doc1)
        doc2 = self.__sanitize_doc(doc2)
        v1 = self.__set_doc_vector(doc1)
        v2 = self.__set_doc_vector(doc2)
        cos = self.__cos_similarity(v1, v2)
        return cos



def main():

    def _print(a, b ,c):
        print('cos_similarity({}, {})\t\t= {}'.format(a, b, c))

    # testing
    my_docs = [
        ['food', 'restaurant', 'customer', 'restaurant', 'waitress'],
        ['food', 'store', 'customer', 'cashier'],
        ['appliance', 'store', 'customer', 'store', 'cashier']
    ]
    myModel = CompareModel(my_docs)
    print('Testing:')
    print('my_docs[0] = "food restaurant customer restaurant waitress"')
    print('my_docs[1] = "food store customer cashier"')
    print('my_docs[2] = "appliance store customer store cashier"')
    print()
    _print('my_docs[0]', 'my_docs[0]', myModel.compare(my_docs[0], my_docs[0]))
    _print('my_docs[1]', 'my_docs[2]', myModel.compare(my_docs[1], my_docs[2]))
    _print('my_docs[0]', 'my_docs[1]', myModel.compare(my_docs[0], my_docs[1]))
    _print('my_docs[0]', 'my_docs[2]', myModel.compare(my_docs[0], my_docs[2]))
    print()
    print()


    # movie reviews
    pos_files = movie_reviews.fileids(categories='pos')[:5]
    neg_files = movie_reviews.fileids(categories='neg')[:5]

    pos = []
    neg = []

    for p in pos_files:
        pos.append(movie_reviews.words(p))

    for n in neg_files:
        neg.append(movie_reviews.words(n))

    # docs_collection = pos + neg
    
    pos_model = CompareModel(pos)

    print()
    print('Comparing all possible pairs of positive reviews')
    cos1 = pos_model.compare(pos[0], pos[1])
    cos2 = pos_model.compare(pos[0], pos[2])
    cos3 = pos_model.compare(pos[0], pos[3])
    cos4 = pos_model.compare(pos[0], pos[4])
    cos5 = pos_model.compare(pos[1], pos[2])
    cos6 = pos_model.compare(pos[1], pos[3])
    cos7 = pos_model.compare(pos[1], pos[4])
    cos8 = pos_model.compare(pos[2], pos[3])
    cos9 = pos_model.compare(pos[2], pos[4])
    cos10 = pos_model.compare(pos[3], pos[4])
    cosAve = (cos1 + cos2 + cos3 + cos4 + cos5 + cos6 + cos7 + cos8 + cos9 + cos10) / 10
    _print('pos 0', 'pos 1', cos1)
    _print('pos 0', 'pos 2', cos2)
    _print('pos 0', 'pos 3', cos3)
    _print('pos 0', 'pos 4', cos4)
    _print('pos 1', 'pos 2', cos5)
    _print('pos 1', 'pos 3', cos6)
    _print('pos 1', 'pos 4', cos7)
    _print('pos 2', 'pos 3', cos8)
    _print('pos 2', 'pos 4', cos9)
    _print('pos 3', 'pos 4', cos10)
    print('Average cosine similarity of all\npossible positive review pairs: {}'.format(cosAve))
    print()
    print()

    neg_model = CompareModel(neg)

    print()
    print('Comparing all possible pairs of negative reviews')
    cos1 = neg_model.compare(neg[0], neg[1])
    cos2 = neg_model.compare(neg[0], neg[2])
    cos3 = neg_model.compare(neg[0], neg[3])
    cos4 = neg_model.compare(neg[0], neg[4])
    cos5 = neg_model.compare(neg[1], neg[2])
    cos6 = neg_model.compare(neg[1], neg[3])
    cos7 = neg_model.compare(neg[1], neg[4])
    cos8 = neg_model.compare(neg[2], neg[3])
    cos9 = neg_model.compare(neg[2], neg[4])
    cos10 = neg_model.compare(neg[3], neg[4])
    cosAve = (cos1 + cos2 + cos3 + cos4 + cos5 + cos6 + cos7 + cos8 + cos9 + cos10) / 10
    _print('neg 0', 'neg 1', cos1)
    _print('neg 0', 'neg 2', cos2)
    _print('neg 0', 'neg 3', cos3)
    _print('neg 0', 'neg 4', cos4)
    _print('neg 1', 'neg 2', cos5)
    _print('neg 1', 'neg 3', cos6)
    _print('neg 1', 'neg 4', cos7)
    _print('neg 2', 'neg 3', cos8)
    _print('neg 2', 'neg 4', cos9)
    _print('neg 3', 'neg 4', cos10)
    print('Average cosine similarity of all\npossible negative review pairs: {}'.format(cosAve))
    print()
    print()

main()