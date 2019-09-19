from operator import itemgetter
import numpy as np
import pandas as pd
import operator
from scipy import sparse
from sklearn.preprocessing import PolynomialFeatures
from sklearn import cross_validation
from sklearn.metrics import log_loss
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer,HashingVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import xgboost as xgb
import math
def get_data():
    a = 40.705628
    b = -74.010278
    all=pd.read_csv("hello.csv").fillna(-1)

    Min_lis_id=all["listing_id"].min()
    Min_time=all["time"].min()
    all["gradient"]=((all["listing_id"])-Min_lis_id)/(all["time"]-Min_time)

    # building feature
    all["building_dif"]=all["price"]-all["building_mean"]
    all["building_rt"]=all["price"]/all["building_mean"]

    # mean feature
    add = pd.DataFrame(all.groupby(["manager_id"]).building_rt.mean()).reset_index()
    add.columns = ["manager_id", "manager_pay"]
    all = all.merge(add, on=["manager_id"], how="left")

    # count of manager
    add = pd.DataFrame(all.groupby(["jwd_class"]).manager_id.nunique()).reset_index()
    add.columns = ["jwd_class", "manager_num_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    # location feature
    add = pd.DataFrame(all.groupby(["manager_id"]).jwd_class.nunique()).reset_index()
    add.columns = ["manager_id", "manager_jwd_class"]
    all = all.merge(add, on=["manager_id"], how="left")

    # mean price
    add = pd.DataFrame(all.groupby(["jwd_class"]).price.median()).reset_index()
    add.columns = ["jwd_class", "price_mean_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    # building count
    add = pd.DataFrame(all.groupby(["jwd_class"]).building_id.nunique()).reset_index()
    add.columns = ["jwd_class", "building_num_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    # photo count
    add = pd.DataFrame(all.groupby(["manager_id"]).photo_num.mean()).reset_index()
    add.columns = ["manager_id", "manager_photo"]
    all = all.merge(add, on=["manager_id"], how="left")

    # description count
    add = pd.DataFrame(all.groupby(["manager_id"]).num_description_words.mean()).reset_index()
    add.columns = ["manager_id", "manager_desc"]
    all = all.merge(add, on=["manager_id"], how="left")

    # feature mean
    add = pd.DataFrame(all.groupby(["manager_id"]).feature_num.mean()).reset_index()
    add.columns = ["manager_id", "manager_feature"]
    all = all.merge(add, on=["manager_id"], how="left")

    # location influence
    add = pd.DataFrame(all.groupby(["bathrooms","bedrooms"]).price.median()).reset_index()
    add.columns = ["bathrooms","bedrooms", "fangxing_mean"]
    all = all.merge(add, on=["bathrooms","bedrooms"], how="left")

    all["fangxing_mean_dif_building"]=all["fangxing_mean"]-all["building_mean"]
    #all["fangxing_mean_rt_building"] = all["fangxing_mean"]/all["building_mean"]

    # mean feature
    price_mean_all=all.price.median()
    all["price_all_dif_jwd"]=price_mean_all-all["price_mean_jwd"]

    # mean in different district
    add = pd.DataFrame(all.groupby(["jwd_class","bathrooms","bedrooms"]).price.median()).reset_index()
    add.columns = ["jwd_class","bathrooms","bedrooms", "type_jwd_price_mean"]
    all = all.merge(add, on=["jwd_class","bathrooms","bedrooms"], how="left")


    all["type_jwd_price_mean_dif"]=all["price"]-all["type_jwd_price_mean"]
    all["type_jwd_price_mean_rt"]=all["price"]/all["type_jwd_price_mean"]

    all["type_jwd_building_mean_dif"]=all["building_mean"]-all["type_jwd_price_mean"]
    all["type_jwd_building_mean_rt"]=all["building_mean"]/all["type_jwd_price_mean"]

    all["fangxing_mean_dif_jwd"] = all["fangxing_mean"] - all["type_jwd_price_mean"]
    all["fangxing_mean_rt_jwd"] = all["fangxing_mean"]/all["type_jwd_price_mean"]


    add = pd.DataFrame(all.groupby(["manager_id"]).type_jwd_price_mean_rt.mean()).reset_index()
    add.columns = ["manager_id", "manager_pay_jwd"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["building_id"]).type_jwd_building_mean_rt.mean()).reset_index()
    add.columns = ["building_id", "building_pay_jwd"]
    all = all.merge(add, on=["building_id"], how="left")

    add = pd.DataFrame(all.groupby(["jwd_class"]).fangxing_mean_rt_jwd.mean()).reset_index()
    add.columns = ["jwd_class", "jwd_pay_all"]
    all = all.merge(add, on=["jwd_class"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).building_pay_jwd.mean()).reset_index()
    add.columns = ["manager_id", "manager_own_ud"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).jwd_pay_all.mean()).reset_index()
    add.columns = ["manager_id", "manager_own_ud_all"]
    all = all.merge(add, on=["manager_id"], how="left")

    all["manager_building_all_rt"]=all["manager_own_ud"]/all["manager_own_ud_all"]

    from sklearn.cluster import KMeans
    clf = KMeans(n_clusters=5,random_state=1)

    all["longitude"]=all["longitude"].apply(lambda x:-73.75 if x>=-73.75 else x)
    all["longitude"]=all["longitude"].apply(lambda x:-74.05 if x<=-74.05 else x)
    all["latitude"]=all["latitude"].apply(lambda x:40.4 if x<=40.4 else x)
    all["latitude"]=all["latitude"].apply(lambda x:40.9 if x>=40.9 else x)

    data=all[["latitude","longitude"]].values
    clf.fit(data)
    all["where"]=pd.Series(clf.labels_)

    all["all_hours"]=all["time"]*24+all["created_hour"]
    add = pd.DataFrame(all.groupby(["manager_id"]).all_hours.nunique()).reset_index()
    add.columns = ["manager_id", "manager_hours"]
    all = all.merge(add, on=["manager_id"], how="left")
    all["manager_hours_rt"]=all["manager_hours"]/all["manager_active"]

    all["manager_price_mean"]=0

    add = pd.DataFrame(all.groupby(["manager_id"]).price.sum()).reset_index()
    add.columns = ["manager_id", "manager_price_sum"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).bedrooms.sum()).reset_index()
    add.columns = ["manager_id", "manager_bedrooms_sum"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).building_dif.sum()).reset_index()
    add.columns = ["manager_id", "earn_all"]
    all = all.merge(add, on=["manager_id"], how="left")

    all["manager_price_mean"]=all["manager_price_sum"]/all["manager_bedrooms_sum"]

    all["earn_everyday"]=all["earn_all"]/all["manager_active"]

    all["earn_all_rt"]=all["earn_all"]/all["manager_price_sum"]

    all["manager_price_"] = all["manager_price_sum"] / all["manager_active"]

    neak=pd.read_csv("timeout.csv")
    aaaa=neak[["jwd_type_low_than_num","jwd_type_all","jwd_type_rt","listing_id"]]
    all=all.merge(aaaa,on="listing_id",how="left")
    #all["jwd_type_low_than_num"]=map(lambda lo,la,ba,be,p:all[(all.latitude>la-0.01)&(all.latitude<la+0.01)&(all.longitude>lo-0.01)&(all.longitude<lo+0.01)&(all.bathrooms==ba)&(all.bedrooms==be)&(all.price<=p)].shape[0],all["longitude"],all["latitude"],all["bathrooms"],all["bedrooms"],all["price"])
    #all["jwd_type_all"]=map(lambda lo,la,ba,be:all[(all.latitude>la-0.01)&(all.latitude<la+0.01)&(all.longitude>lo-0.01)&(all.longitude<lo+0.01)&(all.bathrooms==ba)&(all.bedrooms==be)].shape[0],all["longitude"],all["latitude"],all["bathrooms"],all["bedrooms"])
    #all["jwd_type_rt"]=all["jwd_type_low_than_num"]/all["jwd_type_all"]

    add = pd.DataFrame(all.groupby(["manager_id"]).jwd_type_rt.mean()).reset_index()
    add.columns = ["manager_id", "manager_pay_jwd_type_rt"]
    all = all.merge(add, on=["manager_id"], how="left")

    where_mean={}
    where_list=list(all["where"].value_counts().index)
    for w in where_list:
        where_mean[w]=all[all["where"]==w].price.mean()
    all["where_mean"]=all["where"].apply(lambda x:where_mean[x])
    all["where_mean_rt"]=all["price"]/all["where_mean"]

    add = pd.DataFrame(all.groupby(["manager_id"]).distance.mean()).reset_index()
    add.columns = ["manager_id", "manager_distance"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).created_hour.var()).reset_index()
    add.columns = ["manager_id", "manager_post_hour_var"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).created_hour.mean()).reset_index()
    add.columns = ["manager_id", "manager_post_hour_mean"]
    all = all.merge(add, on=["manager_id"], how="left")

    all["manager_price_distance_rt"]=all["manager_price_mean"]/(all["manager_distance"]+5)
    all["fangxing_mean_distance_rt"]=all["fangxing_mean"]/(all["distance"]+5)
    all["building_mean_distance_rt"]=all["building_mean"]/(all["distance"]+5)
    all["price_mean_jwd_distance_rt"]=all["price_mean_jwd"]/(all["distance"]+5)

    def string_add(input):
        x,y = input[0],input[1]
        return str(x)+str(y)
    all["man_bui_id"]=all[["manager_id","building_id"]].apply(string_add,axis=1)
    all["price_bath_bed"] = all["price"]/(all["bathrooms"]/2.0 + all["bedrooms"]+1)

    #"""
    manager_list = list(all["manager_id"].value_counts().index)
    manager_feature={}
    for man in manager_list:
        content = []
        for i in all[all.manager_id==man]['features']:
            content.extend(i.lower().replace("[","").replace("]","").replace("-","").replace("/","").replace(" ","").split(","))
        abc = pd.Series(content).value_counts()
        new=list(abc.index)[:20]
        try:
            feature=",".join(new)
        except:
            feature=""
        manager_feature[man]=feature+","
    all["manager_features"]=all["manager_id"].apply(lambda x:manager_feature[x])

    all["post_day"] = all["manager_count"] / all["manager_active"]

    all["features"]=all["features"].apply(lambda x:x.lower().replace("[","").replace("]","").replace("-","").replace("/","").replace(" ",""))

    content = []
    for i in all[(all.interest_level=="high")|(all.interest_level=="medium")]["features"]:
        if i != "":
            content.extend(i.split(","))
    good = pd.Series(content).value_counts().to_frame(name="num_good")
    content = []
    for i in all[(all.interest_level=="low")]["features"]:
        if i != "":
            content.extend(i.split(","))
    bad = pd.Series(content).value_counts().to_frame(name="num_bad")
    tongji=good.merge(bad, left_index=True, right_index=True,how="outer").fillna(0)#iloc[0:200]
    abc=tongji["num_good"]/(tongji["num_bad"]+1)
    def score(x):
        score=0
        for i in x.split(","):
            try:
                score+=abc[i]
            except:
                pass
        return score
    all["manager_feature_score"]=all["manager_features"].apply(lambda x:score(x))

    add = pd.DataFrame(all.groupby(["manager_id"]).price_bath_bed.mean()).reset_index()
    add.columns = ["manager_id", "manager_price_bath_bed_mean"]
    all = all.merge(add, on=["manager_id"], how="left")

    manager_building_zero_count={}
    for man in manager_list:
        manager_building_zero_count[man]=all[(all.manager_id==man)&(all.building_id.astype("str")=="0")].shape[0]
    all["manager_building_zero_count"]=all["manager_id"].apply(lambda x:manager_building_zero_count[x])
    all["manager_building_zero_count_rt"]=all["manager_building_zero_count"]/all["manager_count"]

    add = pd.DataFrame(all.groupby(["manager_id"]).longitude.median()).reset_index()
    add.columns = ["manager_id", "manager_longitude_median"]
    all = all.merge(add, on=["manager_id"], how="left")

    add = pd.DataFrame(all.groupby(["manager_id"]).latitude.median()).reset_index()
    add.columns = ["manager_id", "manager_latitude_median"]
    all = all.merge(add, on=["manager_id"], how="left")

    def add_feat(input):
        a,b,c,d,e=input[0],input[1],input[2],input[3],input[4]
        return str(a)+str(b)+str(c)+str(d)+str(e)

    all["same"]=all[["manager_id","bedrooms","bathrooms","building_id","features"]].apply(add_feat,axis=1)
    same_count = all["same"].value_counts()
    all["same_count"] = all["same"].apply(lambda x: same_count[x])

    man_bui_id_count = all["man_bui_id"].value_counts()
    all["man_bui_id_count"] = all["man_bui_id"].apply(lambda x: man_bui_id_count[x])

    all["man_bui_id_count_rt"] = all["man_bui_id_count"]/all["building_count"]

    all["acreage"]=1+all["bedrooms"] + all["bathrooms"]/2.0

    def calc_jq(input):
        x,y = input[0],input[1]
        return (abs(x - b)+abs(y-a))*111
    all["jq_distance"] = all[["longitude","latitude"]].apply(calc_jq,axis=1)

    add = pd.DataFrame(all.groupby(["jwd_class"]).listing_id.count()).reset_index()
    add.columns = ["jwd_class", "listing_num_jwd"]
    all = all.merge(add, on=["jwd_class"], how="left")

    all["building_listing_num_jwd_rt"]=all["building_num_jwd"]/all["listing_num_jwd"]

    all["lo_la"] = (all["longitude"]-b) / (all["latitude"]-a)

    building_zeros_la=list(all[all.building_id.astype("str")=="0"].latitude)
    building_zeros_lo=list(all[all.building_id.astype("str")=="0"].longitude)
    building_zeros=zip(building_zeros_la,building_zeros_lo)
    def building_zero_num(la,lo,n):
        num=0
        for s in building_zeros:
            slo=float(s[1])
            sla=float(s[0])
            dis=math.sqrt((la-sla)**2+(lo-slo)**2)*111
            if dis<=n:
                num+=1
        return num

    #需要优化
    aaaa=neak[["listing_id","building_zero_num"]]
    all=all.merge(aaaa,on="listing_id",how="left")
    #all["building_zero_num"] = map(lambda la, lo: building_zero_num(la, lo,1), all["latitude"], all["longitude"])

    time_stamp=pd.read_csv("listing_image_time.csv")
    all=all.merge(time_stamp,on="listing_id")

    la1, lo1 =40.778772,-73.96684
    la2, lo2=40.849209,-73.888508
    la3, lo3 =40.747844,-73.901731
    la4, lo4 =40.678722,-73.951174
    la5, lo5 =40.688788,-73.870111
    la6, lo6 =40.624861,-73.967846
    def calc1(input):
        x,y = input[0],input[1]
        return abs(x-la1)+abs(y-lo1)
    def calc2(input):
        x,y = input[0],input[1]
        return abs(x-la2)+abs(y-lo2)
    def calc3(input):
        x,y = input[0],input[1]
        return abs(x-la3)+abs(y-lo3)
    def calc4(input):
        x,y = input[0],input[1]
        return abs(x-la4)+abs(y-lo4)
    def calc5(input):
        x,y = input[0],input[1]
        return abs(x-la5)+abs(y-lo5)
    def calc6(input):
        x,y = input[0],input[1]
        return abs(x-la6)+abs(y-lo6)
    all["dis_1"]=all[["latitude","longitude"]].apply(calc1,axis=1)
    all["dis_2"]=all[["latitude","longitude"]].apply(calc2,axis=1)
    all["dis_3"]=all[["latitude","longitude"]].apply(calc3,axis=1)
    all["dis_4"]=all[["latitude","longitude"]].apply(calc4,axis=1)
    all["dis_5"]=all[["latitude","longitude"]].apply(calc5,axis=1)
    all["dis_6"]=all[["latitude","longitude"]].apply(calc6,axis=1)

    all["class_lo_la"]=np.argmin(all[["dis_1","dis_2","dis_3","dis_4","dis_5","dis_6"]].values,axis=1)
    all["class_lo_la_dis"]=np.min(all[["dis_1","dis_2","dis_3","dis_4","dis_5","dis_6"]].values,axis=1)

    import json
    with open("jpgs.json", "r") as f:
        data = f.read()

    data = json.loads(data)
    img_dic = {}
    for i in data.keys():
        img_list = data[i]
        shape_list = []
        for img in img_list:
            shape = img[0] * img[1]
            shape_list.append(shape)
        leng = len(img_list)
        try:
            img_dic[int(i)] = sum(shape_list) / leng
        except:
            img_dic[int(i)] = 0

    all["pic_mean"]=all["listing_id"].apply(lambda x:img_dic.get(x,0))

    train_add=pd.read_csv("train_gdy.csv")
    test_add=pd.read_csv("test_gdy.csv")
    add=train_add.append(test_add)
    all=all.merge(add,on="listing_id",how="left")

    all["feature_price_rt"]=all["price"]/all["feature_num"]
    all["photo_price_rt"]=all["price"]/all["photo_num"]

    price_today=pd.DataFrame(all.groupby(["time"]).price.median()).reset_index()
    price_today.columns=["time","price_today"]
    all=all.merge(price_today,on="time",how="left")

    price_created_month=pd.DataFrame(all.groupby(["created_month"]).price.median()).reset_index()
    price_created_month.columns=["created_month","price_today"]
    all=all.merge(price_created_month,on="created_month",how="left")

    all["price_rt_jwd"] = all["price"] / all["type_jwd_price_mean"]

    all.to_csv("all20.csv",index=None)

    addclass=["man_bui_id",]
    categorical = ["display_address", "manager_id", "building_id", "street_address"]+addclass
    # categorical = ["display_address","manager_id", "building_id"]
    for f in categorical:
        if all[f].dtype == 'object':
            # print(f)
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(all[f].values))
            all[f] = lbl.transform(list(all[f].values))

    all=all.replace({"interest_level":{"high":0,"medium":1,"low":2,"nnnn":3},
                     "description":{0:"o"}
                     })
    train = all[all.interest_level != 3].copy()
    valid = all[all.interest_level == 3].copy()
    y_train=train["interest_level"]

    train_num=train.shape[0]

    tfidf = CountVectorizer(stop_words='english', max_features=100)
    all_sparse=tfidf.fit_transform(all["features"].values.astype('U'))
    tr_sparse = all_sparse[:train_num]
    te_sparse = all_sparse[train_num:]
    #print tfidf.get_feature_names()

    x_train = train.drop(["interest_level","features","description","manager_features","same"],axis=1)
    x_valid = valid.drop(["interest_level","features","description","manager_features","same"],axis=1)

    x_train = sparse.hstack([x_train.astype(float),tr_sparse.astype(float)]).tocsr()
    x_valid = sparse.hstack([x_valid.astype(float),te_sparse.astype(float)]).tocsr()

    return x_train,y_train,x_valid,valid

def run(train_matrix,test_matrix):
    params = {'booster': 'gbtree',
              #'objective': 'multi:softmax',
              'objective': 'multi:softprob',
              'eval_metric': 'mlogloss',
              'gamma': 1,
              'min_child_weight': 1.5,
              'max_depth': 5,
              'lambda': 10,
              'subsample': 0.7,
              'colsample_bytree': 0.7,
              'colsample_bylevel': 0.7,
              'eta': 0.03,
              'tree_method': 'exact',
              'seed': 2017,
              'nthread': 12,
              "num_class":3
              }
    num_round = 10000
    early_stopping_rounds = 50
    watchlist = [(train_matrix, 'train'),
                 (test_matrix, 'eval')
                 ]
    if test_matrix:
        model = xgb.train(params, train_matrix, num_boost_round=num_round, evals=watchlist,
                      early_stopping_rounds=early_stopping_rounds
                      )
        pred_test_y = model.predict(test_matrix,ntree_limit=model.best_iteration)
        return pred_test_y, model
    else:
        model = xgb.train(params, train_matrix, num_boost_round=num_round)
        return model


def XGB():

    X,y,z,v = get_data()

    #V=xgb.DMatrix(v_X,label=v_y)
    z = xgb.DMatrix(z)

    #print X.shape
    #print z.shape
    #train_x=X[:40000]
    #test_x=X[40000:]

    #train_y=y[:40000]
    #test_y=y[40000:]

    #train_matrix = xgb.DMatrix(X, label=y)
    cv_scores = []
    model_list=[]
    preds_list=[]
    kf = cross_validation.KFold(X.shape[0],n_folds=5,shuffle=True,random_state=1)
    for dev_index, val_index in kf:
        train_x, test_x = X[dev_index, :], X[val_index, :]
        train_y, test_y = y[dev_index], y[val_index]
        train_matrix = xgb.DMatrix(train_x, label=train_y,missing=-1)
        test_matrix = xgb.DMatrix(test_x, label=test_y,missing=-1)
        preds, model = run(train_matrix, test_matrix)
        cv_scores.append(log_loss(test_y, preds))
        model_list.append(model)
        preds_list.append(preds)

        with open("result.txt","a") as f:
            f.write(str(cv_scores)+"\n")
        #break

    for i in range(len(preds_list)):
        if i==0:
            pre_v=model_list[i].predict(z,ntree_limit=model.best_iteration)
        else:
            pre_v=(pre_v+model_list[i].predict(z,ntree_limit=model.best_iteration))

    pre_v=pre_v/len(preds_list)

    loss_mean=np.mean(cv_scores)
    print('Mean loss is:',loss_mean)
    with open("result.txt", "a") as f:
        f.write(str(loss_mean) + "\n")

    result=pre_v
    out_df = pd.DataFrame(result)
    out_df.columns = ["high", "medium", "low"]
    out_df["listing_id"] = v.listing_id.values
    out_df.to_csv("xgb_cv10_%s.csv" % str(loss_mean), index=False)

    for model in model_list:
        importance = model.get_fscore()
        importance = sorted(importance.items(), key=operator.itemgetter(1))
        print(importance)
if __name__ == '__main__':
    XGB()