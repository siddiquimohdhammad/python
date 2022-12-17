

import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
from apyori import apriori

store_data=np.array([
    ["Cheese","Butter","Bread","Mayo"],
    ["Cheese","Bread","Butter",""],
    ["Mayo","Cheese","",""],
    ["Mayo","Chocoloate","Icecream","",],
    ["Bread","Biscuit","",""],
    ["Biscuit","Butter","",""],
    ["Biscuit","Chocolate","Icecream",""]
],dtype="object")



records = []
for i in range(0, 7):
    records.append([str(store_data[i][j]) for j in range(0, 4)])
    print("records.........",records)

association_rules = apriori(records, min_support=0.0045, min_confidence=0.2, min_lift=3, min_length=2)
association_results = list(association_rules)
print("result.......",association_results)

for item in association_results:
    # first index of the inner list
    # Contains base item and add item
    pair = item[0] 
    print("pair is ...",pair)
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])
    

    #second index of the inner list
    print("Support: " + str(item[1]))

    #third index of the list located at 0th
    #of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
    

'''
Rule: light cream -> chicken
Support: 0.004532728969470737
Confidence: 0.29059829059829057
Lift: 4.84395061728395
=====================================
Rule: escalope -> mushroom cream sauce
Support: 0.005732568990801226
Confidence: 0.3006993006993007
Lift: 3.790832696715049
=====================================
Rule: escalope -> pasta
Support: 0.005865884548726837
Confidence: 0.3728813559322034
Lift: 4.700811850163794
=====================================
Rule: ground beef -> herb & pepper
Support: 0.015997866951073192
Confidence: 0.3234501347708895
Lift: 3.2919938411349285
=====================================
Rule: ground beef -> tomato sauce
Support: 0.005332622317024397
Confidence: 0.3773584905660377
Lift: 3.840659481324083
=====================================
Rule: whole wheat pasta -> olive oil
Support: 0.007998933475536596
Confidence: 0.2714932126696833
Lift: 4.122410097642296
=====================================
Rule: pasta -> shrimp
Support: 0.005065991201173177
Confidence: 0.3220338983050847
Lift: 4.506672147735896
=====================================

'''