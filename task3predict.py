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
# input_file = "./data/train_review.json"
# test_file = "./data/test_review.json"
# model = "./mine/task3user.model"
# output_file = "task3user.predict"
# option = "user_based"
input_file = inputs[1]
test_file = inputs[2]
model = inputs[3]
output_file = inputs[4]
option = inputs[5]
conf=SparkConf()
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf)
sc.setLogLevel("OFF")

if "item" in option: 
    ##Case 1
    ##File Loading
    trainRDD = sc.textFile(input_file).map(lambda x: json.loads(x)).map(lambda x: (x["user_id"], (x["business_id"], x["stars"]))).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    testRDD = sc.textFile(test_file).map(lambda x: json.loads(x)).map(lambda x: (x['user_id'], x["business_id"]))
    modelRDD = sc.textFile(model).map(lambda x: json.loads(x)).map(lambda x: ((x["b1"], x["b2"]), x["sim"])).collect()
    model_dict = dict(modelRDD)
    joined = testRDD.leftOuterJoin(trainRDD)
    print(joined.count())
    #print(joined.take(1))
    #Filter out None Type
    joined = joined.filter(lambda x: x[1][1] != None).map(lambda x: ((x[0], x[1][0]), x[1][1]))
    print(joined.count())
    #print(joined.take(1))

    def pearson_predict(x, model_dict): 
        user = x[0][0]
        target_b = x[0][1]
        potential_b = x[1]
        pairs = []
        for i in potential_b: 
            key = tuple(sorted([target_b, i[0]]))
            rev = tuple(reversed(key))
            if key in model_dict: 
                pairs.append([model_dict[key], i[1]])
            elif rev in model_dict: 
                pairs.append([model_dict[rev], i[1]])
            else: 
                continue
        selected_pairs = sorted(pairs, key = lambda x: -x[0])[:10]
        top = 0
        bottom = 0
        for i in selected_pairs: 
            top += i[0]*i[1]
            bottom += i[0]
        if top == 0.0 or bottom == 0.0: 
            return (user, target_b, 0)
        else: 
            return (user, target_b, top/bottom)
    joined = joined.groupByKey().map(lambda x: (x[0], [i for b in x[1] for i in b]))
    joined = joined.map(lambda x: pearson_predict(x, model_dict)).filter(lambda x: x[2] != 0)
    print(joined.take(1))

    # f = open(output_file, "w+")
    item = 0
    for i in joined.collect():
        dic = {"user_id": i[0], "business_id": i[1], "stars": i[2]}
        # json.dump(dic, f)
        # f.write("\n")
        item += 1
    # f.close()
    print(item, "items saved")
    print("Duration: ", "{:.1f}".format(time.time() - start_time), " s")

elif "user" in option: 
    ##Case 2
    ##File Loading
    trainRDD = sc.textFile(input_file).map(lambda x: json.loads(x)).map(lambda x: (x["business_id"], (x["user_id"], x["stars"]))).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    testRDD = sc.textFile(test_file).map(lambda x: json.loads(x)).map(lambda x: (x['business_id'], x["user_id"]))
    modelRDD = sc.textFile(model).map(lambda x: json.loads(x)).map(lambda x: ((x["u1"], x["u2"]), x["sim"])).collect()
    model_dict = dict(modelRDD)

    ##Replace this
    ua = sc.textFile(input_file).map(lambda x: json.loads(x)).map(lambda x: (x["user_id"], (x["business_id"], x["stars"]))).groupByKey().map(lambda x: (x[0], list(set(x[1]))))
    def calAverage(row):
        total = 0
        for i in row[1]:
            total += i[1]
        return (row[0], total/len(row[1]))
    ua = ua.map(calAverage).collect()
    user_average = dict(ua)
    # print(user_average)

    joined = testRDD.leftOuterJoin(trainRDD)
    print(joined.count())
    #print(joined.take(1))
    #Filter out None Type
    joined = joined.filter(lambda x: x[1][1] != None).map(lambda x: ((x[0], x[1][0]), x[1][1]))
    print(joined.count())
    #print(joined.take(1))

    def pearson_predict(x, model_dict, user_average): 
        business = x[0][0]
        target_u = x[0][1]
        potential_u = x[1]
        pairs = []
        for i in potential_u: 
            key = tuple(sorted([target_u, i[0]]))
            rev = tuple(reversed(key))
            if key in model_dict: 
                pairs.append([model_dict[key], i[0], i[1]]) #(sim, user_id, rating)
            elif rev in model_dict: 
                pairs.append([model_dict[rev], i[0], i[1]])
            else: 
                continue
        selected_pairs = sorted(pairs, key = lambda x: -x[0])[:11]
        top = 0
        bottom = 0
        for i in selected_pairs: 
            if (i[1]) in user_average: 
                one_usr_avg = user_average[i[1]]
                top += i[0]*(i[2] - one_usr_avg)
                bottom += i[0]
            else: 
                continue
        if top == 0.0 or bottom == 0.0: 
            return (target_u, business, 0)
        else: 
            if target_u in user_average: 
                return (target_u, business, user_average[target_u] + top/bottom)
            else: 
                return (target_u, business, 0)
            
    joined = joined.map(lambda x: pearson_predict(x, model_dict, user_average)).filter(lambda x: x[2] != 0)
    print(joined.take(1))

    item = 0
    f = open(output_file, "w+")
    for i in joined.collect():
        dic = {"user_id": i[0], "business_id": i[1], "stars": i[2]}
        json.dump(dic, f)
        f.write("\n")
        item += 1
    f.close()
    print(item, " items saved")
    print("Duration: ", "{:.1f}".format(time.time() - start_time), " s")