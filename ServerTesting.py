from module.prepare import *
from itertools import product
from sklearn.externals import joblib
# import pygraphviz

def LGBTuning(Xtrain,Ytrain):
    
    clf = lgb.LGBMClassifier(objective='binary',
                             silent=False,
                             verbose=1,
                             random_state=seed,
                             n_jobs=4,
#                              class_weight
                            )
    
    gridParams = {
        # step 1
#     'learning_rate': [0.01,0.05,0.1],
#     'boosting_type':['gbdt','goss'],
#     'n_estimators': [50,200,500],
#     'num_iterations':[200,400,1000],
        # step 1 fixed
    'learning_rate': [0.1], ### 0.1
    'boosting_type':['gbdt'], ### goss>gbdt
    'n_estimators': [300],
    'num_iterations':[2000], ### 2000
        # step 2
    'num_leaves': [680], ### 680
#     'max_bin':[127,255,511],
        # step 2 fixed
#     'num_leaves': [800],
    'max_bin':[256],
        # step 3
#     'max_depth':[7,8,9,10], ### missed
    'colsample_bytree' : [0.8], ### 0.8
    'subsample_freq':[1,2,3],
    'subsample' : [0.6,0.8,1],
    'reg_alpha' : [0,0.1,0.5],
    'reg_lambda' : [0,0.1,0.5],
    }

    print('default params\n',clf.get_params())

    grid = GridSearchCV(clf, gridParams,
                    scoring='roc_auc',
#                     refit=False,
                    verbose=3,
                    cv=5,
                    n_jobs=1)

    grid.fit(Xtrain,Ytrain)
    
    return grid

#### configure

hyper_params = GetConfigure()
num_hyper_params = len(hyper_params)

generalize_ratio = 0.3
test_ratio = 0.3
cv = 1
mi_use = True

tuning_mode = True
if tuning_mode:
    cv = 1

cv_results = []


#### main

[data,T] = ReadData()

# for i in 
for batch in range(cv):
    if mi_use==True:
        arr = ToMatrix(data,'sparse')
        [X_train,X_test,Y_train,Y_test] = MutualInformationFeatureSelection2(arr,data,generalize_ratio)
        [X_train,X_test,Y_train,Y_test] = \
            RandomForestDimensionalityReduction(X_train,X_test,Y_train,Y_test)
    else:
        [X,Y] = ToMatrix(data,'dense')
        [X_train,X_test,Y_train,Y_test] = SplitDataset(X,Y,generalize_ratio)
        [X_train,X_test,Y_train,Y_test] = \
            RandomForestDimensionalityReduction(X_train,X_test,Y_train,Y_test)
    if tuning_mode:
        [Xtrain,Ytrain] = merge_train_test(X_train,X_test,Y_train,Y_test)
        grid = LGBTuning(Xtrain,Ytrain)
        cv_results.append(grid)

joblib.dump(grid,'./result-temp/2-19-lgb-grid-tune1.m')