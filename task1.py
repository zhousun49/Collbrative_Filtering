import pyspark
import json 
from itertools import combinations, groupby
from pyspark import SparkContext, SparkConf, StorageLevel
import sys
import time
import random

##Initiation
start_time = time.time()
inputs = sys.argv
input_file = inputs[1]
output_file = inputs[2]
conf=SparkConf()
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf)

##Get distinct users and businesses
##Column is user and row is business
inputRDD = sc.textFile(input_file).map(lambda x: json.loads(x))
userRDD = inputRDD.map(lambda x: x['user_id']).distinct()
businessRDD = inputRDD.map(lambda x: x['business_id']).distinct()
user_count = userRDD.count()
business_count = businessRDD.count()
print("Number of Unique users: ", user_count)
print("Number of Unique businesses: ", business_count)
##Table of Size business_count*user_count
user_id = userRDD.zipWithIndex().collect()
user_dict = dict(user_id)
reviewRDD = inputRDD.map(lambda x: (user_dict[x['user_id']], x["business_id"]))

##Building Hashtable
n = 25
m = user_count
hashtable = []
for row in range(m):
    one_row = []
    for i in range(n):
        a = random.randint(1, 100000)
        b = random.randint(1, 100000)
        one_row.append((a*i+b)%m)
    hashtable.append(one_row)
# print("Number of rows: ", len(hashtable))
# print("Number of elements in one row: ", len(hashtable[0]))

##Build the signature Matrix
sigMat = reviewRDD.groupByKey().map(lambda x: (x[0], list(set(x[1]))))
sigMat = sigMat.map(lambda x: (x[1], hashtable[x[0]]))
##all businesses that belong to one user    
def getSingleBusiness(row):
    returned_list = []
    for i in row[0]:
        returned_list.append((i,row[1]))
    return returned_list
sigMat = sigMat.map(getSingleBusiness).flatMap(lambda x: x)
# print(sigMat.take(1))
##Flatmapping: for each user, get (business, hash) pair  
def HashMat(row):
    returned_list = []
    for i in list(row[1]):
        returned_list.append(i)
    return (row[0], returned_list)
sigMat = sigMat.groupByKey().map(HashMat)
# print(sigMat.take(1))  
def getMinValue(x):
    minVals = []
    transpose_lst = list(zip(*x[1]))
    for i in transpose_lst:
        minVals.append(min(i))
    return (x[0], minVals)
sigMat = sigMat.map(getMinValue)
print("signature matrix length: ", sigMat.count())

##Divide
##By setting each row to be 1, then don't need to divide into more buckets (100)
n = 25
b = 25
r = n/b
def HashMinValue(row):
    returned = []
    for minval in row[1]: 
        v = (minval, )
        hashval = hash(v)
        returned.append((hashval, row[0]))
    return returned
sigMat = sigMat.map(HashMinValue).flatMap(lambda x: x).groupByKey().map(lambda x: (x[0], list(set(x[1])))).filter(lambda x: len(x[1]) >= 2)
sigMat = sigMat.map(lambda x: x[1])
print("returned group business")
# print(sigMat.take(1))

candidates = set()
for i in sigMat.collect():
    for item in combinations(i, 2):
        candidates.add(item)
print("length of candidates: ", len(candidates))

##Business grouping
groupbyBusinessRdd = reviewRDD.map(lambda x: (x[1], x[0])).groupByKey().map(lambda x: (x[0], set(x[1]))).collect()
business_dict = dict(groupbyBusinessRdd)
#print(groupbyBusinessRdd.take(1))

##Write to file
f = open(output_file,"a")
output = []
for cand in candidates:
    b1 = cand[0]
    b2 = cand[1]
    b1_usr = business_dict[b1]
    b2_usr = business_dict[b2]
    sim = float(len(b1_usr & b2_usr)/len(b1_usr | b2_usr))
    if sim >= 0.05:
        # print("similarity: ", sim)
        dic = {}
        dic["b1"] = cand[0]
        dic["b2"] = cand[1]
        dic["sim"] = sim
        output.append(dic)
        json.dump(dic, f)
        f.write("\n")
f.close()
print("length of output: ", len(output))
print("Duration: ", "{:.1f}".format(time.time() - start_time), " s")