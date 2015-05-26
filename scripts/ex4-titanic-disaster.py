# Naive Bayes classifier as a Bayesian network

import numpy as np
from pomegranate import *
import pandas as pd

# Passengers on the Titanic either survive or perish
passenger = DiscreteDistribution( { 'survive': 0.5, 'perish': 0.5 } )

# Smoothen and return survival probability
p_survived = lambda list_x: (list_x[(list_x.Survived == 1)].count() + 1 ) / (list_x.count() + 2)

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
p_tclass_2 = p_survived(tclass_1)['Pclass']
p_tclass_3 = p_survived(tclass_1)['Pclass']

# Class of travel, given survival data
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


# State objects hold both the distribution, and a high level name.
s1 = State( passenger, name = "passenger" )
s2 = State( gender, name = "gender" )
s3 = State( tclass, name = "class" )
s4 = State( age, name = "age" )

# Create the Bayesian network object with a useful name
network = BayesianNetwork( "Titanic Disaster" )

# Add the three nodes to the network
network.add_nodes( [ s1, s2, s3, s4 ] )

# Add transitions which represent conditional depesndencies, where the second
# node is conditionally dependent on the first node (Monty is dependent on both guest and prize)
network.add_edge( s1, s2 )
network.add_edge( s1, s3 )
network.add_edge( s1, s4 )
network.bake()

# The first observation is that the interpreter is not working
#first_observation = { 'interpreter' : 'nw' }

# beliefs will be an array of posterior distributions or clamped values for each state, indexed corresponding to the order
# in self.states.
#beliefs = network.forward_backward( first_observation )

# Convert the beliefs into a more readable format
#beliefs = map( str, beliefs )

# Print out the state name and belief for each state on individual lines
# What is the probability of the code being buggy ?
#print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )


# Repeat above steps for second observation
# Now what is the probability of the code being buggy ?
#second_observation = { 'interpreter' : 'nw', 'cursor' : 'w' }
#beliefs = network.forward_backward( second_observation )
#beliefs = map( str, beliefs )
#print "\n".join( "{}\t{}".format( state.name, belief ) for state, belief in zip( network.states, beliefs ) )
