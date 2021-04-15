import random
import numpy
import gensim
import string
from sklearn.neural_network import MLPClassifier
from nltk.corpus import movie_reviews as mr, stopwords as sw
from gensim.models.doc2vec import TaggedDocument, Doc2Vec



def sanitize_doc(doc):
    doc_temp = []
    for word in doc:
        if (word not in sw.words("english")) and (word not in string.punctuation):
            doc_temp.append(word)
    return doc_temp


def label_docs(sources):
    labeled_docs = []
    for k, v in sources.items():
        for i, f in enumerate(v):
            words = (mr.words(f))
            tags = [k + '_' + str(i)]
            labeled_docs.append(TaggedDocument(words=words, tags=tags))
    return labeled_docs


def main():
    DOC_SIZE = 500
    TRAIN_SIZE = int(DOC_SIZE * 0.90)

    pos_files = mr.fileids(categories='pos')[:DOC_SIZE]
    neg_files = mr.fileids(categories='neg')[:DOC_SIZE]


    sources = {
        'train_pos': pos_files[:TRAIN_SIZE],
        'train_neg': neg_files[:TRAIN_SIZE],
        'test_pos': pos_files[TRAIN_SIZE:],
        'test_neg': neg_files[TRAIN_SIZE:]
    }

    # Tag documents in format: 
    # TaggedDocument(words=['word_1', 'word_2', 'word_n'], tags=['pos'])
    corpus = label_docs(sources)

    doc2vecModel = Doc2Vec(documents=corpus, 
                           min_count=1, 
                           window=10, 
                           vector_size=100, 
                           workers=7, 
                           sample=1e-4, 
                           negative=5)

    for epoch in range(10):
        random.shuffle(corpus)
        doc2vecModel.train(corpus, total_examples=doc2vecModel.corpus_count, epochs=doc2vecModel.epochs)
        

    # set training dataset
    x_train = numpy.zeros((TRAIN_SIZE * 2, 100))
    y_train = numpy.zeros(TRAIN_SIZE * 2)

    for i in range(TRAIN_SIZE):
        x_train[i] = doc2vecModel['train_pos_' + str(i)]
        x_train[TRAIN_SIZE + i] = doc2vecModel['train_neg_' + str(i)]
        y_train[i] = 1
        y_train[TRAIN_SIZE + i] = 0

    
    # set testing dataset
    TEST_SIZE = DOC_SIZE - TRAIN_SIZE
    x_test = numpy.zeros((TEST_SIZE * 2, 100))
    y_test = numpy.zeros(TEST_SIZE * 2,)

    for i in range(TEST_SIZE):
        x_test[i] = doc2vecModel['test_pos_' + str(i)]
        x_test[TEST_SIZE + i] = doc2vecModel['test_neg_' + str(i)]
        y_test[i] = 1
        y_test[TEST_SIZE + i] = 0

    
    # classification
    classifier = MLPClassifier(hidden_layer_sizes=(5,10))
    classifier.fit(x_train, y_train)

    print()
    print('Corpus Size: {}'.format(len(pos_files + neg_files)))
    print('Training Size:\n\tpositive: {}\tnegative: {}'.format(len(sources['train_pos']), len(sources['train_neg'])))
    print('Testing Size:\n\tpositive: {}\tnegative: {}'.format(len(sources['test_pos']), len(sources['test_neg'])))
    print()


    # mean accuracy of the model using the test dataset
    print("Accuracy Score:", classifier.score(x_test, y_test))

    # sample predictions
    print('Predicting movie reviews using Neural Network classifier:')
    print()

    # print(len(classifier.predict(x_test)))

    # random.shuffle(x_test)
    # test_datasets = x_test
    # test_labels = y_test
    pred_list = classifier.predict(x_test)
    cat = {0: 'NEG', 1: 'POS'}

    # print(len(x_test))

    for i in range(10):
        r = random.randint(0, len(x_test) - 1)
        print('Test doc: {}\t\tactual class: {}\t\tprediction: {}\t\t accurate?: {}'
        .format(r, cat[int(y_test[r])], cat[int(pred_list[r])], y_test[r] == pred_list[r]))


main()
