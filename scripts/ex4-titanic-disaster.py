# Naive Bayes classifier as a Bayesian network

import numpy as np
import pandas as pd
from pomegranate import *

# Passengers on the Titanic either survive or perish
passenger = DiscreteDistribution( { 'survive': 0.5, 'perish': 0.5 } )

# Smoothen and return survival probability (Laplace)
def p_survived(list_x):
    return (list_x[(list_x.Survived == 1)].count() + 1 ) / (list_x.count() + 2)

train_data = pd.read_csv('../data/titanic/train.csv', header=0)

gen_male = train_data[train_data.Sex == 'male']
gen_female = train_data[train_data.Sex == 'female']

p_gen_male = p_survived(gen_male)['Sex']
p_gen_female = p_survived(gen_female)['Sex']

# Gender, given survival data
gender = ConditionalProbabilityTable(
            [[ 'survive', 'male',   p_gen_male ],
             [ 'survive', 'female', p_gen_female ],
             [ 'perish', 'male',    1 - p_gen_male ],
	         [ 'perish', 'female',  1 - p_gen_female]], [passenger] )


tclass_1 = train_data[train_data.Pclass == 1]
tclass_2 = train_data[train_data.Pclass == 2]
tclass_3 = train_data[train_data.Pclass == 3]

p_tclass_1 = p_survived(tclass_1)['Pclass']
p_tclass_2 = p_survived(tclass_2)['Pclass']
p_tclass_3 = p_survived(tclass_3)['Pclass']

print p_tclass_1, p_tclass_2, p_tclass_3

train_data.Pclass.median() #Median is third class, So use third class for NaN

# Class, given survival data
tclass = ConditionalProbabilityTable(
            [[ 'survive', 'first',  p_tclass_1 ],
             [ 'survive', 'second', p_tclass_2 ],
             [ 'survive', 'third',  p_tclass_3 ],
             [ 'perish', 'first',  1 - p_tclass_1 ],
             [ 'perish', 'second', 1 - p_tclass_2 ],
	         [ 'perish', 'third',  1 - p_tclass_3]], [passenger] )

age_1 = train_data[train_data.Age < 20]
age_2 = train_data[(train_data.Age >= 20) & (train_data.Age < 40)]
age_3 = train_data[(train_data.Age >= 40) & (train_data.Age < 60)]
age_4 = train_data[(train_data.Age >= 60) & (train_data.Age < 80)]
age_5 = train_data[(train_data.Age >= 80)]

p_age_1 = p_survived(age_1)['Age']
p_age_2 = p_survived(age_2)['Age']
p_age_3 = p_survived(age_3)['Age']
p_age_4 = p_survived(age_4)['Age']
p_age_5 = p_survived(age_5)['Age']

train_data.Age.median() #is between 20 and 40, so use age_2 for NaN

# Age, given survival data
age = ConditionalProbabilityTable(
            [[ 'survive', 'age_1',   p_age_1 ],
             [ 'survive', 'age_2', p_age_2 ],
             [ 'survive', 'age_3', p_age_3 ],
             [ 'survive', 'age_4', p_age_4 ],
             [ 'survive', 'age_5', p_age_5 ],             
             [ 'perish', 'age_1',    1 - p_age_1 ],
             [ 'perish', 'age_2',    1 - p_age_2 ],
             [ 'perish', 'age_3',    1 - p_age_3 ],
             [ 'perish', 'age_4',    1 - p_age_4 ],
	         [ 'perish', 'age_5',  1 - p_age_5]], [passenger] )


s1 = State( passenger, name = "passenger" )
s2 = State( gender, name = "gender" )
s3 = State( tclass, name = "class" )
s4 = State( age, name = "age" )

network = BayesianNetwork( "Titanic Disaster" )

network.add_nodes( [ s1, s2, s3, s4 ] )

network.add_edge( s1, s2 )
network.add_edge( s1, s3 )
network.add_edge( s1, s4 )

network.bake()

def find_age(age):
    if age < 20:
        return 'age_1'
    elif age >= 20 and age < 40:
        return 'age_2'
    elif age >= 40 and age < 60:
        return 'age_3'
    elif age >= 60 and age < 80:
        return 'age_4'
    elif age >= 80:
        return 'age_5'
    else:
        return 'age_2'

def find_class(pclass):
    if pclass == 1:
        return 'first'
    elif pclass == 2:
        return 'second'
    elif pclass == 3:
        return 'third'
    else:
        return 'third'

# Check with 30% of the training data
train_test_data = pd.read_csv('../data/titanic/train_verify.csv', header=0)

train_test_res = []
for row in train_test_data.values:
    observations = {'gender': row[4], 'age': find_age(row[5]), 'class': find_class(row[2]) }
    beliefs = network.forward_backward( observations )
    res = beliefs[0].parameters[0]
    if res['survive'] > res['perish']:
        train_test_res.append(1)
    else:
        train_test_res.append(0)

prediction = zip(train_test_data.Survived, train_test_res)

match_count = 0
for row in prediction:
    if row[0] == row[1]:
        match_count += 1

# Should actually do the false positive thing, but right now just checking the raw match. 
print "Ratio of correctly predicted rows to total rows for 30% training data", match_count / float(len(train_test_data.PassengerId))

# Load the test data and generate our awesome predictions!
test_data = pd.read_csv('../data/titanic/test.csv', header=0)

test_res = []
for row in test_data.values:
    observations = {'gender': row[3], 'age': find_age(row[4]), 'class': find_class(row[1]) }
    beliefs = network.forward_backward( observations )
    res = beliefs[0].parameters[0]
    if res['survive'] > res['perish']:
        test_res.append(1)
    else:
        test_res.append(0)

d = {'PassengerId':test_data.PassengerId, 'Survived': test_res}
df = pd.DataFrame(data = d)

df.to_csv('../data/titanic/output_final.csv', index=False)
print 'Final output written to output_final.csv'