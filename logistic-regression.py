# Using the Movie Reviews corpus available in NLTK, write a program *classify.py* 
# that classify movies according to their sentiment polarity, and evaluate your program. 
# You should use the logistic regression as a classifier.

import random
import sklearn.utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from nltk.corpus import movie_reviews, stopwords


def create_bunch(col):
    bunch_data = []
    bunch_target = []
    for f in col:
        bunch_data.append(movie_reviews.raw(f))
        if (movie_reviews.categories(f)[0] == 'pos'):
            cat = 1
        else:
            cat = 0
        bunch_target.append(cat)
    return sklearn.utils.Bunch(data=bunch_data, target=bunch_target)



def main():

    DOC_SIZE = 1000
    TRAIN_SIZE = int(DOC_SIZE * 0.75)

    pos_files = movie_reviews.fileids(categories='pos')[:DOC_SIZE]
    neg_files = movie_reviews.fileids(categories='neg')[:DOC_SIZE]

    train_pos_files = pos_files[:TRAIN_SIZE]
    train_neg_files = neg_files[:TRAIN_SIZE]

    test_pos_files = pos_files[TRAIN_SIZE:] 
    test_neg_files = neg_files[TRAIN_SIZE:] 

    print('Corpus Size: {}'.format(len(pos_files + neg_files)))
    print('Training Size:\n\tpositive: {}\tnegative: {}'.format(len(train_pos_files), len(train_neg_files)))
    print('Testing Size:\n\tpositive: {}\tnegative: {}'.format(len(test_pos_files), len(test_neg_files)))
    print()

    # training datasets
    datasets = create_bunch(train_pos_files + train_neg_files)
    text_train, y_train = datasets.data, datasets.target

    # testing datasets
    datasets = create_bunch(test_pos_files + test_neg_files)
    text_test, y_test = datasets.data, datasets.target

    # vectorize training and testing data sets
    vectorizer = CountVectorizer(min_df=5, ngram_range=(2, 2))
    x_train = vectorizer.fit(text_train).transform(text_train)
    x_test = vectorizer.transform(text_test)
    
    # vocabulary
    features = vectorizer.get_feature_names()
    # print(features)
    param_grid = {'C': [0.001, 0.01, 0.1, 1, 10]}
    grid = GridSearchCV(LogisticRegression(solver='lbfgs', max_iter=200), param_grid, cv=5)
    grid.fit(x_train, y_train)

    lr = grid.best_estimator_
    lr.fit(x_train, y_train)
    lr.predict(x_test)

    print("Accuracy score: {:.2f}".format(lr.score(x_test, y_test)))
    print()
    print()

    # predictions
    print('Predicting movie reviews using Logical Regression classifier:')
    print('prediction:')
    print('[0] => negative\n[1] => positive')
    print()

    test_datasets = test_pos_files + test_neg_files
    random.shuffle(test_datasets)

    for i in range(10):
        r = random.randint(0, len(test_datasets) - 1)
        f = test_datasets[r]
        actual = movie_reviews.categories(f)
        raw = [movie_reviews.raw(f)]
        predict = lr.predict(vectorizer.transform(raw))
        print('Test doc: {}\t\tactual class: {}\t\tprediction: {}'.format(r, actual, predict))


main()
