# Assignment 1 -- Programming Part [20 points]

Given you have:
 * a fill-in the gap question file *question.in* that contains a text with some gaps,
 * a student script that contains the student answers *script.in* file with the gaps in the text filled, and
 * the *answer.in* file that contains the list of tokens to be placed in the gaps in the order of their appearance in *question.in*. Those tokens are stemmed using the Porter's stemmer.

Write a Python program *a1.py* that checks the student script. Each answer token is worth 1 point. Your program should extract from the student script the answers and compare them with the list of tokens in the *answer.in* file.


# Assignment 2 [45 points]
# Due on February 26, 2021 by midnight

# shannon.py [15 points]
Using a language model generated from the Brown corpus which is available in NLTK, write a Python program *shannon.py* that generates random sentences (i.e., Shannon Visualization Method) using:
 * bigrams
 * trigrams

# classify.py [30 points]
 Using the Movie Reviews corpus available in NLTK, write a Python program *classify.py* that classify movies according to their sentiment polarity, and evaluate your program. You should use the Naive Bayes classifier, and split the data into training and testing (e.g., 70\% and 30\%).

# Assignment 3  [40 points]
# Due on March 26, 2021 by midnight

# compare.py [20 points]
Given 2 text documents, write your own function that computes the similarity between the 2 documents using the cosine similarity measure. The documents are represented using the TF-IDF.  Apply your function to the Movie Review corpus to compute the average similarity of the positive and the negative reviews.


# classify.py [20 points]
Using the Movie Reviews corpus available in NLTK, write a program *classify.py* that classify movies according to their sentiment polarity, and evaluate your program. You should use the logistic regression as a classifier.

# Assignment 4  [40 points]
# Due on April 9, 2021 by midnight

# classify.py [40 points]
Using the Movie Review corpus, write a program that classify reviews according to their positive/negative polarity, and evaluate the results of your program. You should use Word2Vec to represent the words, and the Multi-layer neural network classifier (about 10 hidden layers with 5 units each). You should split the data into training and testing (e.g., 90\% and 10\%).

