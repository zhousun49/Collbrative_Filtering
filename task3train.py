import json 
from itertools import combinations
from pyspark import SparkContext, SparkConf, StorageLevel
from operator import add
import sys
import time
import math
import random 

##Initiation
start_time = time.time()
inputs = sys.argv
#input_file = "./data/train_review.json"
#output_file = "task3item.model"
#option = "item_based"
input_file = inputs[1]
output_file = inputs[2]
option = inputs[3]
conf=SparkConf()
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf)
sc.setLogLevel("OFF")

if "user" in option: 
    inputRDD = sc.textFile(input_file).map(lambda x: json.loads(x))
    ######Use userRDD again
    userRDD = inputRDD.map(lambda x: (x['user_id'], (x["business_id"], float(x["stars"])))).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    businessRDD = inputRDD.map(lambda x: x['business_id']).distinct()
    # user_count = userRDD.count()
    business_count = businessRDD.count()
    # print("Number of Unique users: ", user_count)
    print("Number of Unique businesses: ", business_count)
    ##Table of Size business_count*user_count
    business_dict = businessRDD.zipWithIndex().collect()
    business_dict = dict(business_dict)
    # print(userRDD.take(1))
    user_business = userRDD.map(lambda x: (x[0],[business_dict[y[0]] for y in x[1]]))
    # print(user_business.take(1))

    ##Hash table
    n = 13
    m = business_count
    hashtable = []
    for row in range(m):
        one_row = []
        for i in range(n):
            a = random.randint(1, 100000)
            b = random.randint(1, 100000)
            one_row.append((a*i+b)%m)
        hashtable.append(one_row)
    print("Number of rows: ", len(hashtable))
    print("Number of elements in one row: ", len(hashtable[0]))
    #hashtable = dict(hashtable)

    ##Signature matrix: hasing and pick minimum value
    sigMat = user_business.map(lambda x: (x[0], list(zip(*[hashtable[y] for y in x[1]])))).map(lambda x: (x[0], [min(y) for y in x[1]]))
    # print(sigMat.take(1))
    def HashMinValue(row):
        returned = []
        # print(row)
        for minval in row[1]: 
            v = (minval, )
            hashval = hash(v)
            returned.append((hashval, row[0]))
        return returned
    sigMat = sigMat.map(HashMinValue).flatMap(lambda x: x).groupByKey().map(lambda x: (x[0], list(set(x[1])))).filter(lambda x: len(x[1]) >= 2)
    # print("Time for Signature Matrix: ", "{:.1f}".format(time.time() - start_time), " s")
    ##Jaccard Similarity
    def findCandidates(x):
        users = x[1]
        returned = []
        for item in combinations(users, 2):
            returned.append(tuple(sorted(item)))
        return returned
    sigMat = sigMat.map(findCandidates).flatMap(lambda x: x).distinct()
    user_dict = dict(user_business.collect())

    def Jaccard(row):
        u1 = set(user_dict[row[0]])
        u2 = set(user_dict[row[1]])
        similarity = len(u1 & u2)
        if similarity >= 3: 
            sim = float(similarity/len(u1 | u2))
            return (row[0], row[1], sim)
        else: 
            return (row[0], row[1], 0)
    sigMat = sigMat.map(Jaccard).filter(lambda x: x[2] >= 0.01)
    # print(sigMat.take(1))
    filtered = sigMat.collect()
    scores_dict = dict(userRDD.collect())
    # print("filtered length: ", len(filtered))
    # print("one filtered: ", filtered[0])
    # print("one user: ", scores_dict['txu_KwZOGYG6O3yYHjztbg'])
    print("Time for Jaccard and 3 filtering: ", "{:.1f}".format(time.time() - start_time), " s")
    ##Calculate Pearson
    def calPearson(user1, user2): 
        u1_dict = dict(scores_dict[user1])
        u2_dict = dict(scores_dict[user2])
        combined = set(set(list(u1_dict.keys())) & set(list(u2_dict.keys())))
        new_u1 = {}
        new_u2 = {}
        for i in combined: 
            new_u1[i] = u1_dict[i]
            new_u2[i] = u2_dict[i]
        avg_u1 = sum(list(new_u1.values()))/len(new_u1)
        avg_u2 = sum(list(new_u2.values()))/len(new_u2)
        top = 0
        for u in new_u1: 
            new_u1[u] = new_u1[u] - avg_u1
            new_u2[u] = new_u2[u] - avg_u2
            top += new_u1[u]*new_u2[u]
        u1_values = list(new_u1.values())
        u2_values = list(new_u2.values())
        bottom = math.sqrt(sum([i*i for i in u1_values]))*math.sqrt(sum([i*i for i in u2_values]))
        if bottom == 0.0 or top == 0.0:
            return 0
        else: 
            returned = top/bottom
            return returned
        
        
    f = open(output_file, "w+")    
    item = 0
    for i in filtered: 
        pearson = calPearson(i[0], i[1])
        if pearson > 0: 
            item += 1
            dic = {"u1": i[0], "u2": i[1], "sim": float(pearson)}
            json.dump(dic, f)
            f.write("\n")
    f.close()
    print(item, " items saved")
    print("Duration: ", "{:.1f}".format(time.time() - start_time), " s")

elif "item" in option: 
    inputRDD = sc.textFile(input_file).map(lambda x: json.loads(x))
    #(bid, (uid, star))
    businessRDD = inputRDD.map(lambda x: (x['business_id'], (x["user_id"], float(x["stars"])))).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    businessRDD = businessRDD.filter(lambda x: len(x[1]) >= 3)
    print("satisfactory business count: ", businessRDD.count())

    ##Business dictionary
    businessRDD = businessRDD.map(lambda x: (x[0], dict(x[1]))).collect()
    business_dict = dict(businessRDD)
    businesses = list(business_dict.keys())

    def calPearson(u1_dict, u2_dict):
        combined = set(set(list(u1_dict.keys())) & set(list(u2_dict.keys())))
        new_u1 = {}
        new_u2 = {}
        for i in combined: 
            new_u1[i] = u1_dict[i]
            new_u2[i] = u2_dict[i]
        avg_u1 = sum(list(new_u1.values()))/len(new_u1)
        avg_u2 = sum(list(new_u2.values()))/len(new_u2)
        top = 0
        for u in new_u1: 
            new_u1[u] = new_u1[u] - avg_u1
            new_u2[u] = new_u2[u] - avg_u2
            top += new_u1[u]*new_u2[u]
        u1_values = list(new_u1.values())
        u2_values = list(new_u2.values())
        bottom = math.sqrt(sum([i*i for i in u1_values]))*math.sqrt(sum([i*i for i in u2_values]))
        if bottom == 0.0 or top == 0.0:
            return 0
        else: 
            returned = top/bottom
            return returned

    f = open(output_file, "w+")
    item = 0
    for i in combinations(businesses, 2): 
        sorted_b = sorted(i)
        b1 = i[0]
        b2 = i[1]
        b1_users_r = business_dict[b1]
        b2_users_r = business_dict[b2]
        b1_users = b1_users_r.keys()
        b2_users = b2_users_r.keys()
        if len(set(set(b1_users) & set(b2_users))) >= 3:
            pearson = calPearson(b1_users_r, b2_users_r)
            if pearson > 0: 
                item += 1
                dic = {"b1" : b1, "b2" : b2, "sim" : pearson}
                json.dump(dic, f)
                f.write("\n")
    print(item, " written to file")
    f.close()
    print("Duration: ", "{:.1f}".format(time.time() - start_time), " s")