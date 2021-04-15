import nltk
# install the punkt package first
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


def get_array_of_tokens(fileName):
    f = open(fileName, mode='r')
    tempArray = []
    for content in f:
        for word in word_tokenize(content):
            tempArray.append(word)
    return tempArray


# get all tokens from the files
script = get_array_of_tokens('./script.in')
question = get_array_of_tokens('./question.in')
answerKey = get_array_of_tokens('./answer.in')
# print(script)
# print()
# print(question)
# print()
# print(answerKey)
# print()


# get all the student answers from script
studentAnswers = []
for index, value in enumerate(question):
    if value != script[index]:
        studentAnswers.append(script[index])



# check if the student answers match the stem of words in the answerKey
score = 0
ps = PorterStemmer()
for index, value in enumerate(studentAnswers):
    if ps.stem(value) == answerKey[index]:
        score = score + 1

print("Student answers:\n{}".format(", ".join(studentAnswers)))
print()
print("Answer keys:\n{}".format(", ".join(answerKey)))
print()
print("Score:\n{0}/{1}".format(score, len(answerKey)))