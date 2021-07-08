
import pyspark
import json 
import re
from itertools import combinations
from pyspark import SparkContext, SparkConf, StorageLevel
from operator import add
import sys
import time
import random
import math

##Initiation
start_time = time.time()
inputs = sys.argv
input_file = inputs[1]
output_file = inputs[2]
stop_words = inputs[3]
conf=SparkConf()
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf)

##Read stop words 
stop = open(stop_words, 'r')
stop_words = []
for i in stop: 
    stop_words.append(i.strip("\n"))
print("stop words length: ", len(stop_words))

##Group by businesses
inputRDD = sc.textFile(input_file).map(lambda x: json.loads(x))
inputRDD.persist(StorageLevel.DISK_ONLY)
def removeChars(x):
    text = x['text'].lower()
    text = re.sub(r'[^\w\s]', ' ', text)    
    text = ''.join(w for w in text if not w.isdigit())  
    return (x['business_id'], (x["user_id"], text))
##With user Ids
#############Use Again
businessRDD = inputRDD.map(removeChars).groupByKey().map(lambda x: (x[0], list(set(x[1])))).partitionBy(30)
userRDD = inputRDD.map(lambda x: (x['user_id'], x['business_id'])).groupByKey().map(lambda x: (x[0], list(x[1])))
num_docs = businessRDD.count()
print("Number of businesses: ", num_docs)
num_users = userRDD.count()
print("Number of users: ", num_users)

##clean string
def catString(x, filter_words): 
    returned_string = []
    for item in x[1]:
        returned_string.append(item[1])
    long_string = " ".join(returned_string)
    filtered_sentence = [w for w in long_string.split() if not w in filter_words]
    return (x[0], filtered_sentence)
# ##No user id, with business id

# #############Use Again
business_reviews = businessRDD.map(lambda x: catString(x, stop_words))
# print("one business: ", business_reviews.take(1))

##Calculate IDFs
def wordsBus(x):
    returned = []
    words = x[1]
    for i in words:
        if i != "": 
            returned.append((i, x[0]))
    return returned
##Pure Doc
idfs = business_reviews.map(wordsBus).flatMap(lambda x: x).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
def calIDF(x, num_docs): 
    idf = math.log2(num_docs/len(x))
    return idf
idfs = idfs.mapValues(lambda x: calIDF(x, num_docs))
print("idfs count: ", idfs.count())

##Calculate TFs
def calTf(x):
    w_dict = {}
    for w in x: 
        if w in w_dict: 
            w_dict[w] += 1
        else: 
            w_dict[w] = 1
    max_show = max(w_dict.values())
    tfs = []
    for word, num in w_dict.items():
        if word != "": 
            tfs.append([word, num / max_show])
    return tfs
tfs = business_reviews.mapValues(lambda x: calTf(x)).flatMap(lambda x: [(x[0], y) for y in x[1]]).map(lambda x: (x[1][0], (x[0], x[1][1])))
print("tf counts: ", tfs.count())

#Join tf-idf
joined = tfs.join(idfs)
def word_tfidf(x): 
    business = x[1][0][0]
    word = x[0]
    tf = x[1][0][1]
    idf = x[1][1]
    return (business, (word, tf*idf))
business_words = joined.map(word_tfidf).groupByKey()
def generateProfile(x):
    words = list(x[1])
    sorted_words = sorted(words, key = lambda x: -x[1])
    return (x[0], sorted_words[:200])
business_profile = business_words.map(generateProfile)
# print("one b: ", business_profile.take(1))

# Make a words dictionary and convert business words to dict
words = business_profile.flatMap(lambda x: [y[0] for y in x[1]]).distinct().zipWithIndex().collect()
print("word count: ", len(words))
words_dict = dict(words)
def convertWordstoNum(row):
    words_with_tfifs = row[1]
    returned = []
    for i in words_with_tfifs: 
        returned.append((words_dict[i[0]], i[1]))
    return (row[0], returned)
business_profile = business_profile.map(convertWordstoNum)
# print("business profile: ", business_profile.take(1))
b_dict = dict(business_profile.collect())
print("dictionary done")
final_business_profile = business_profile.map(lambda x: (x[0], [y[0] for y in x[1]])).collect()
# print("final business: ", final_business_profile[0])

def mapBusiness(x): 
    all_words = []
    for i in x[1]:
        businesses = b_dict[i]
        for j in businesses: 
            all_words.append(j)
    no_replication = set(all_words)
    sort_words = sorted(no_replication, key = lambda i: -i[1])[:500]
    returned_words = []
    for i in sort_words:
        returned_words.append(i[0])
    return (x[0], returned_words)
user_profile = userRDD.map(mapBusiness).collect()
# print("final user: ", user_profile[0])

##Write to file
f = open(output_file, "a")
for i in final_business_profile: 
    dic = {}
    dic["business"] = i[0]
    dic["features"] = i[1]
    json.dump(dic, f)
    f.write("\n")
for i in user_profile: 
    dic = {}
    dic["user"] = i[0]
    dic["features"] = i[1]
    json.dump(dic, f)
    f.write("\n")
f.close()

print("Duration: ", "{:.1f}".format(time.time() - start_time), " s")