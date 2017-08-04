import os
import string
import matplotlib.pyplot as plt
import nltk
import numpy as np
import sklearn.datasets

from io import StringIO

from snowballstemmer import stemmer
from wordcloud import WordCloud
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter  # process_pdf
from pdfminer.pdfpage import PDFPage
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk import word_tokenize, re
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


# stemmer = SnowballStemmer("english")


def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems


def pdf_to_text(pdf_path):
    # PDFMiner boilerplate
    rsrcmgr = PDFResourceManager()
    sio = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, sio, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)

    # Extract text
    fp = open(pdf_path, 'rb')
    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
    fp.close()

    # Get text from StringIO
    text = sio.getvalue()

    # Cleanup
    device.close()
    sio.close()

    return text


def pdf_to_dataset(path,x):

    data = []
    file_names = np.array([])

    for subdir, dirs, files in os.walk(path):
        for file in files:
            file_path = subdir + os.path.sep + file
            print(file_path)
            text = pdf_to_text(file_path)
            text = ''.join(ch for ch in text if ch not in string.punctuation).lower().replace('\n', '').replace("higher order","higherorder")
            data.append(text)
            file_names = np.append(file_names, file)
            #print("sunu okudum: %s" % (file_names))

    dataset = sklearn.datasets.base.Bunch(data=data, filenames=file_names)

    stop_words = ENGLISH_STOP_WORDS.union(["al","et","set",'±',"document","term","used","based","?","altınel","7","62","2017","5","b","table","10","20","30",
                                           "50","1","2","s","0","international","conference"])

    # term frequency
    tf_vectorizer = CountVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, lowercase=True, ngram_range=(2,2))  # tf
    X_data_tf = tf_vectorizer.fit_transform(dataset.data)

    # term frequency - inverse document frequency
    tfidf_vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), stop_words=stop_words, lowercase=True, ngram_range=(2,2))  # td-idf
    X_data_tfidf = tfidf_vectorizer.fit_transform(dataset.data)

    # document frequency
    #X_data_df = tfidf_vectorizer.inverse_transform(X=dataset.data)

    # x=10
    freqs_tf = [(word, X_data_tf.getcol(idx).sum()) for word, idx in tf_vectorizer.vocabulary_.items()]
    sorted_freqs_tf = sorted(freqs_tf, key = lambda x: -x[-1])[:x]
    print(sorted_freqs_tf) # returns a list

    wordcloud_text_tf = [item[0] for item in sorted_freqs_tf]
    string_tf = ",".join(wordcloud_text_tf).replace(" ","_").replace(","," ")
    wordcloud_tf = WordCloud(background_color="white").generate(string_tf)
    plt.imshow(wordcloud_tf, interpolation='bilinear')
    plt.axis("off")
    plt.title("TF")
    fig_tf = plt.figure(1)
    fig_tf.savefig("tf.png")

    freqs_tfidf = [(word, X_data_tfidf.getcol(idx).sum()) for word, idx in tfidf_vectorizer.vocabulary_.items()]
    sorted_freqs_tfidf =sorted(freqs_tfidf, key = lambda x: -x[-1])[:x]
    print(sorted_freqs_tfidf)

    wordcloud_text_tfidf = [item[0] for item in sorted_freqs_tfidf]
    string_tfidf = ",".join(wordcloud_text_tfidf).replace(" ","_").replace(","," ")
    wordcloud_tfidf = WordCloud(background_color="white").generate(string_tfidf)
    fig_tfidf = plt.figure(2)
    plt.imshow(wordcloud_tfidf, interpolation='bilinear')
    plt.axis("off")
    plt.title("TF-IDF")
    fig_tfidf.savefig("tfidf.png")
    plt.show()

    return


pdf_to_dataset("/media/ayneen/HDD/Users/Talha/PycharmProjects/BIGDATA_Lab_Staj_Talha/PDFs",25)
"""
# JVM başlat
# Aşağıdaki adresleri java sürümünüze ve jar dosyasının bulunduğu klasöre göre değiştirin
jpype.startJVM(jpype.getDefaultJVMPath(),"-Djava.class.path=D:/Users/Talha/PycharmProjects/BIGDATA_Lab_Staj_Talha/zemberek-tum-2.0.jar", "-ea")
# Türkiye Türkçesine göre çözümlemek için gerekli sınıfı hazırla
Tr = jpype.JClass("net.zemberek.tr.yapi.TurkiyeTurkcesi")
# tr nesnesini oluştur
tr = Tr()
# Zemberek sınıfını yükle
Zemberek = jpype.JClass("net.zemberek.erisim.Zemberek")
# zemberek nesnesini oluştur
zemberek = Zemberek(tr)

#Çözümlenecek örnek kelimeleri belirle
kelimeler = ["merhabalaştık","dalgalarının","habercisi","tırmalamışsa"]
for kelime in kelimeler:
    if kelime.strip()>'':
        yanit = zemberek.kelimeCozumle(kelime)
        if yanit:
            print("{}".format(yanit[0]))
        else:
            print("{} ÇÖZÜMLENEMEDİ".format(kelime))
#JVM kapat
jpype.shutdownJVM()
"""