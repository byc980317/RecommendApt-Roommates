#encoding=utf-8
import pandas as pd
import dask.dataframe as dd
import math

train=dd.read_csv("train.csv")
test=dd.read_csv("test.csv")
train = train.compute()
test = test.compute()
print('Finish loading datasets')
test["interest_level"]="nnnn"
df=train.append(test)
df=df.fillna("0")

df["photo_num"]= df["photos"].apply(len)
df["feature_num"]=df["features"].apply(len)
print('Finish calculate length')
a=40.705628
b=-74.010278

#distance to the doc select
def calc_dist(input):
    x,y = input[0],input[1]
    a = 40.705628
    b = -74.010278
    return ((x-b)**2+(y-1)**2)**55.5
df["distance"]=df[['longitude','latitude']].apply(calc_dist,axis=1)
print('Finish distance')
df["num_description_words"] = df["description"].apply(lambda x: len(str(x).split(" ")))

df["created"] = pd.to_datetime(df["created"])
df["created_month"] = df["created"].dt.month
df["created_day"] = df["created"].dt.day
df["created_hour"] = df["created"].dt.hour
print('Time Finish')
def time_long(input):
    x,y = input[0],input[1]
    if x==4:
        return y
    if x==5:
        return 30+y
    if x==6:
        return 30+31+y
df["time"]=df[["created_month","created_day"]].apply(time_long,axis=1)

df["price_bed"] = df["price"]/(df["bedrooms"]+1)
df["price_bath"] = df["price"]/(df["bathrooms"]+1)
df["price_bath_bed"] = df["price"]/(df["bathrooms"] + df["bedrooms"]+1)
df["bed_bath_dif"] = df["bedrooms"]-df["bathrooms"]

df["bed_bath_per"] = df["bedrooms"]/df["bathrooms"]
df["room_sum"] = df["bedrooms"]+df["bathrooms"]
df["bed_all_per"] = df["bedrooms"]/df["room_sum"]

#counts of these
display=df["display_address"].value_counts()
manager_id=df["manager_id"].value_counts()
building_id=df["building_id"].value_counts()
street=df["street_address"].value_counts()
bedrooms=df["bedrooms"].value_counts()
bathrooms=df["bathrooms"].value_counts()
days=df["time"].value_counts()

df["display_count"]=df["display_address"].apply(lambda x:display[x])
df["manager_count"]=df["manager_id"].apply(lambda x:manager_id[x])
df["building_count"]=df["building_id"].apply(lambda x:building_id[x])
df["street_count"]=df["street_address"].apply(lambda x:street[x])
df["bedrooms_count"]=df["bedrooms"].apply(lambda x:bedrooms[x])
df["bathrooms_count"]=df["bathrooms"].apply(lambda x:bathrooms[x])
df["day_count"]=df["time"].apply(lambda x:days[x])

#how many days the manager active
add=pd.DataFrame(df.groupby(["manager_id"]).time.nunique()).reset_index()
add.columns=["manager_id","manager_active"]
df=df.merge(add,on=["manager_id"],how="left")

#how many buildings the manager own
add=pd.DataFrame(df.groupby(["manager_id"]).building_id.nunique()).reset_index()
add.columns=["manager_id","manager_building"]
df=df.merge(add,on=["manager_id"],how="left")

df["manager_building_post_rt"]=df["manager_building"]/df["manager_count"]

df["build_day"]=df["manager_building"]/df["manager_active"]

#the range place manager active
managet_place={}
for man in list(manager_id.index):
    la=df[df["manager_id"] == man]["latitude"].copy()
    lo=df[df["manager_id"] == man]["longitude"].copy()
    managet_place[man]=10000*((la.max()-la.min())*(lo.max()-lo.min()))
df["manager_place"]=df["manager_id"].apply(lambda x:managet_place[x])

df["midu"]=df["manager_building"]/df["manager_place"]

#the building own by how many manager
add=pd.DataFrame(df.groupby(["building_id"]).manager_id.nunique()).reset_index()
add.columns=["building_id","building_manager"]
df=df.merge(add,on=["building_id"],how="left")

#the manager post how many listings that day
add=pd.DataFrame(df.groupby(["time","manager_id"]).listing_id.count()).reset_index()
add.columns=["time","manager_id","day_manager"]
df=df.merge(add,on=["time","manager_id"],how="left")
df["day_manager_rt"]=df["day_manager"]/df["day_count"]

#the building have same bedrooms and bathrooms,
add=pd.DataFrame(df.groupby(["building_id","bedrooms","bathrooms"]).price.median()).reset_index()
add.columns=["building_id","bedrooms","bathrooms","building_mean"]
df=df.merge(add,on=["building_id","bedrooms","bathrooms"],how="left")

def calc_jwd(input):
    x,y = input[0],input[1]
    return (int(x*100)%100)*100+(int(-y*100)%100)
df["jwd_class"]= df[['latitude','longitude']].apply(calc_jwd,axis=1)

df=df.drop(["photos","id","created"],axis=1)
df.to_csv("hello.csv",index=None)
