#paragraph or corpus

with open("train.txt" , "r"  , encoding= "utf-8") as file:
 lines = file.readlines()

sentences= []
labels = []
for line in lines:
   line =line.strip()
   if ";" in line:
      sentence , label = line.rsplit(";" , 1)#split only at the last ";"
      sentences.append(sentence)
      labels.append(label)


import nltk
from nltk.corpus import stopwords
nltk.download("punkt" )
nltk.download('punkt_tab')


## lemmetization 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#cleaning the data
import re
corpus = []
for sentence in sentences:
    test = re.sub("[^a-zA-Z]", " ", sentence)
    test = test.lower()
    words = nltk.word_tokenize(test)
    words = [lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('english'))]
    cleaned_sentence = " ".join(words)
    corpus.append(cleaned_sentence)


from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(ngram_range=(1,3))
x = cv.fit_transform(corpus)
##print(corpus[0])

x.toarray()


## save this x
import joblib

joblib.dump(x , "tfidf_matrix.pkl")
joblib.dump(labels , "labels.pkl")
joblib.dump(cv , "tfidf_vectorizer.pkl")






