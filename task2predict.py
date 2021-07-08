import json 
from itertools import combinations
from pyspark import SparkContext, SparkConf, StorageLevel
from operator import add
import sys
import time
import math

##Initiation
start_time = time.time()
inputs = sys.argv
input_file = inputs[1]
model = inputs[2]
output_file = inputs[3]
conf=SparkConf()
conf.set("spark.executor.memory", "4g")
conf.set("spark.driver.memory", "4g")
sc = SparkContext.getOrCreate(conf)

##File loading
inputRDD = sc.textFile(input_file).map(lambda x: json.loads(x)).map(lambda x: (x["user_id"], x["business_id"]))
model_profile = sc.textFile(model).map(lambda x: json.loads(x))
business_profile = model_profile.filter(lambda x: "business" in x).map(lambda x: (x["business"], x["features"])).collect()
print("b length: ", len(business_profile))
user_profile = model_profile.filter(lambda x: "user" in x).map(lambda x: (x["user"], x["features"])).collect()
print("u length: ", len(user_profile))
b_dict = dict(business_profile)
u_dict = dict(user_profile)

##Calculate Cosine similarity
def cosSim(x): 
    user = x[0]
    business = x[1]
    # print("b: ", business)
    # print("u: ", user)
    if user in u_dict and business in b_dict: 
        business_feature = set(b_dict[business])
        user_feature = set(u_dict[user])
        connection = len(set(business_feature & user_feature))
        # print("connection: ", connection)
        # print(math.sqrt(len(business_feature)) * math.sqrt(len(business_feature)))
        sim = connection/(math.sqrt(len(business_feature)) * math.sqrt(len(business_feature)))
        return (user, business, sim)
    else: 
        return (user, business, 0)
prediction = inputRDD.map(cosSim).filter(lambda x: x[2] >= 0.01).collect()
print("length of prediction: ", len(prediction))

##Write to file
f = open(output_file, "a")
for i in prediction: 
    dic = {'user_id': i[0], 'business_id': i[1], 'sim': i[2]}
    json.dump(dic, f)
    f.write("\n")
f.close()

print("Duration: ", "{:.1f}".format(time.time() - start_time), " s")