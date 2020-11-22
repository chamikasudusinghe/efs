import math
import numpy as np
import pandas as pd
from functools import reduce

from sklearn.model_selection import train_test_split
from cela.evolutionary_feature_synthesis import EFSRegressor

df = pd.read_csv('train.csv')
df = df.dropna()
dft = pd.read_csv('test.csv')
dft = dft.dropna()

num_context = 24

train = {}
test = {}
dropping = ['index','Cluster']
for i in range(num_context):
    dff = df.loc[df['context'+str(i)] == 1.0]
    dff.reset_index(inplace=True) 
    train['df'+str(i)] = dff
    dff = dft.loc[dft['context'+str(i)] == 1.0]
    dff.reset_index(inplace=True) 
    test['dft'+str(i)] = dff
    dropping.append('context'+str(i))
    
train_selected = {}
test_selected = {}
for i in range(num_context):
    dff = train['df'+str(i)]
    dff = dff.drop(dropping,axis=1)
    nunique = dff.apply(pd.Series.nunique)
    cols_to_drop = nunique[nunique == 1].index
    dff = dff.drop(cols_to_drop, axis=1)
    train_selected['df'+str(i)] = dff
    dff = test['dft'+str(i)]
    dff = dff.drop(dropping,axis=1)
    dff = dff.drop(cols_to_drop, axis=1)
    test_selected['dft'+str(i)] = dff
    
def training(i,train_selected,train,test_selected,test):
    
    feature_list = []
    train_context = {}
    test_context = {}
    
    for seed in range (1,41):
    
        np.random.seed(seed)
        df = train_selected['df'+str(i)]
        df1 = train['df'+str(i)]
        
        dft = test_selected['dft'+str(i)]
        dft1 = test['dft'+str(i)]

        X_train = df.drop(columns=['Target'])
        y_train = df['Target']
        
        X_test = dft.drop(columns=['Target'])
        y_test = dft['Target']
        
        size = len(X_train.columns)
        
        X_train = X_train.to_numpy()
        X_test = X_test.to_numpy()
        y_train = y_train.to_numpy()
        y_test = y_test.to_numpy()
        

        sr = EFSRegressor(seed=seed,verbose=0,max_gens=200,num_additions=(2*size)-1,max_useless_steps=50)
        features, gen = sr.fit(X_train, y_train)
        
        sr = EFSRegressor(seed=seed,verbose=0,max_gens=gen,num_additions=(2*size)-1,max_useless_steps=50)
        features, gen = sr.fit(X_train, y_train)
        
        train_score = sr.score(X_train, y_train)
        print("Context :",i,"Seed :",seed,'Train MSE Score: {}'.format(train_score))
        
        test_score = sr.score(X_test, y_test)
        print("Context :",i,"Seed :",seed,'Test MSE Score: {}'.format(test_score))
        
        train_ml = sr.pred(X_train,y_train)
        df1 = df1.assign(name=train_ml)
        df1 = df1.rename(columns={'name': "prediction-context-"+str(i)})
        train_context['seed-'+str(seed)] = df1
        
        test_ml = sr.pred(X_test,y_test)
        dft1 = dft1.assign(name=test_ml)
        dft1 = dft1.rename(columns={'name': "prediction-context-"+str(i)})
        test_context['seed-'+str(seed)] = dft1 
                           
        for j in range(len(features)):
            name  = features[j].string
            fitness = str(features[j].fitness)
            feature_list.append([seed,i,gen,name,fitness])
    fitness = pd.DataFrame(feature_list, columns = ['Seed', 'Context', 'Generation', 'Feature Name', 'Feature Fitness'])
    return train_context,test_context,fitness

train_all_context = {}
test_all_context = {}
fitness_context = {}

for i in range(num_context):
    train_context,test_context,fitness = training(i,train_selected,train,test_selected,test)
    train_all_context['context-'+str(i)] = train_context
    test_all_context['context-'+str(i)] = test_context
    fitness_context['context-'+str(i)] = fitness
    
index = df.index
number_of_rows = len(index)

indexarr = []

for i in range(0,number_of_rows):
    indexarr.append(i)

df = df.assign(index=indexarr)

for seed in range (1,2):
        
    training_seed = []
    
    for i in range(num_context):
        dff = train_all_context['context-'+str(i)]['seed-'+str(seed)]
        training_seed.append(dff)
        
    dfs = training_seed
    
    common_cols = list(set.intersection(*(set(df.columns) for df in dfs)))
        
    for i in range(num_context):
        dff = training_seed[i]
        #print(dff)
        training_seed[i] = pd.merge(df,dff,on=common_cols,how='left')    
    
    dfs = training_seed
    
    df_final = reduce(lambda left,right: pd.merge(left,right,on=common_cols), dfs)
    
    df_final.to_csv('df-train-seed-'+str(seed)+'.csv')
    
index = dft.index
number_of_rows = len(index)

indexarr = []

for i in range(0,number_of_rows):
    indexarr.append(i)

dft = dft.assign(index=indexarr)

for seed in range (1,2):
    
    testing_seed = []
    
    for i in range(num_context):
        dff = test_all_context['context-'+str(i)]['seed-'+str(seed)]
        testing_seed.append(dff)
        
    dfs = testing_seed
    
    common_cols = list(set.intersection(*(set(dft.columns) for dft in dfs)))
    
    for i in range(num_context):
        dff = testing_seed[i]
        testing_seed[i] = pd.merge(dft,dff,on=common_cols,how='left')    
    
    dfs = testing_seed
    
    df_final = reduce(lambda left,right: pd.merge(left,right,on=common_cols), dfs)
    
    df_final.to_csv('df-test-seed-'+str(seed)+'.csv')

def amend(i,df):
    feature_list = []
    seed_length = 24
    prev_seed = 1
    iterator = 0
    for index, row in df.iterrows():
        if index>1:
            prev_seed = df.at[index-1,'Seed']
        seed = row["Seed"]
        if seed == prev_seed:
            if iterator<seed_length:
                context = row["Context"]
                name  = row["Feature Name"]
                gen = row["Generation"]
                fitness = row["Feature Fitness"]
                feature_list.append([seed,context,gen,name,fitness])
            else:
                print("Error in Input")
        elif seed != prev_seed:
            while iterator != seed_length:
                context = row["Context"]
                name  = ""
                gen = row["Generation"]
                fitness = ""
                feature_list.append([prev_seed,context,prev_gen,name,fitness])        
                iterator+=1
            iterator = 0
            seed = row["Seed"]
            context = row["Context"]
            name  = row["Feature Name"]
            gen = row["Generation"]
            fitness = row["Feature Fitness"]
            feature_list.append([seed,context,gen,name,fitness])
        prev_gen = row["Generation"]
        iterator+=1
    for j in range (iterator,seed_length):
        name  = ""
        fitness = ""
        feature_list.append([prev_seed,context,prev_gen,name,fitness])
    if i == 0:
        dff = pd.DataFrame(feature_list, columns = ['Seed', 'Context', 'Generation', 'Feature Name', 'Feature Fitness'])
    else:
        for x in feature_list:
            del x[0]
        dff = pd.DataFrame(feature_list, columns = ['Context', 'Generation', 'Feature Name', 'Feature Fitness'])
    return dff

fitness_concat = amend(0,fitness_context['context-'+str(i)])
for i in range(1,num_context):
    dff = amend(i,fitness_context['context-'+str(i)])
    fitness_concat = pd.concat([fitness_concat, dff], axis=1)
fitness_concat.to_csv('df-fitness.csv')