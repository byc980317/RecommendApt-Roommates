import xgboost as xgb

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=321, num_rounds=4000):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.04
    param['max_depth'] = 5
    param['silent'] = 1
    param['num_class'] = 3
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 1
    param['subsample'] = 0.5
    param['colsample_bytree'] = 0.7
    param["reg_alpha"] = 0.5
    param['seed'] = seed_val

    # param = {}
    # param['booster'] = 'gbtree'
    # param['objective'] = 'multi:softprob'
    # param['bst:eta'] = 0.04
    # param['seed'] = 1
    # param['bst:max_depth'] = 6
    # param['bst:min_child_weight'] = 1.
    # param['silent'] = 1
    # param['nthread'] = 12  # put more if you have
    # param['bst:subsample'] = 0.7
    # param['gamma'] = 1.0
    # param['colsample_bytree'] = 1.0
    # param['num_parallel_tree'] = 3
    # param['colsample_bylevel'] = 0.7
    # param['lambda'] = 5
    # param['num_class'] = 3
    # param['seed'] = seed_val

    num_rounds = num_rounds

    plst = list(param.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)

    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=200)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)

    pred_test_y = model.predict(xgtest)
    return pred_test_y, model