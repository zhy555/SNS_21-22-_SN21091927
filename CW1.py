import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#import geopandas as gpd
#from geopy.distance import geodesic
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold

# Part1 Load and clean dataset

filePath = './London_property.csv'  # data path
df = pd.read_csv(filePath)  # load dataset
# print(df)
Area_list = df['Area_name'].unique()  # get unique value of Area_name
Area_dic = {}
for i in range(0, len(Area_list)):
    Area_dic[Area_list[i]] = i
Propertytype_list = df['propertytype'].unique()
Propertytype_dic = {}
for i in range(0, len(Propertytype_list)):
    Propertytype_dic[Propertytype_list[i]] = i
duration_list = df['duration'].unique()
duration_dic = {}
for i in range(0, len(duration_list)):
    duration_dic[duration_list[i]] = i
postcode_list = df['postcode'].unique()
postcode_dic = {}
for i in range(0, len(postcode_list)):
    postcode_dic[postcode_list[i]] = i
print(Area_dic)
print(Propertytype_dic)
print(duration_dic)
print(postcode_dic)

# Area_list = ['City of London']
# print(Area_list)
df.drop(columns=['CONSTRUCTION_AGE_BAND', 'year', 'dateoftransfer',
                 'transactionid', 'Code', 'lad21cd', 'classt', 'price'],
        axis=1,
        inplace=True
        )  # remove unnecessary columns
# print(df)
df.drop_duplicates('id', inplace=True)  # remove duplicate data with key 'id'
df.dropna(subset=['numberrooms'], inplace=True)  # remove data with missing values, key = 'numberrooms'
df['numberrooms'] = df['numberrooms'].astype('int64')  # convert numberrooms field from float to int
df['id'] = df['id'].astype('str')  # convert id field from int to str
subdf_list = []  # save 33 subdf

# 0 is count, 1 is mean, 2 is std, 3 is min, 4/5/6 is 25%/50%/75%, 7 is max
for area in Area_list:  # remove outlier with key 'priceper'
    subdf = df.loc[df['Area_name'] == area]  # get subdf with key 'Area_name'
    des_result = subdf['priceper'].describe()  # get describe series from each subdf

# remove outlier range with IQR method
    q1 = des_result[4]  # Q1 25%
    q3 = des_result[6]  # Q3 75%
    iqr = q3 - q1  # iqr
    data_low = q1 - 1.5 * iqr  # low outlier
    data_high = q3 + 1.5 * iqr  # high outlier
    print('for area:{}, q1 is {}, q3 is {}, iqr is {}, data_low is {}, data_high is {} '.format(
        area, q1, q3, iqr, data_low, data_high
    ))
    for index, row in subdf.iterrows():
        # print('The price of {} is {}'.format(index+2, row['priceper']))
        if row['priceper'] > data_high or row['priceper'] < data_low:  # find outlier
            # print(index)
            subdf.drop(index=index, inplace=True)
    subdf_list.append(subdf)  # add this subdf to subdf_list
    # print(subdf)
df_cleaned = pd.concat(subdf_list)  # put 33 subdf into 1 df
# print(df_cleaned)

# Part2 Describe and Explore the dataset (EDA)

loc_filePath = './London_loc.csv'
df_loc = pd.read_csv(loc_filePath)  # load London_loc dataset
df_loc['Eastings'] = df_loc['Eastings'].astype('str')  # convert Eastings field from int to str
df_loc['Northings'] = df_loc['Northings'].astype('str')  # convert Northings field from int to str
df_merged = pd.merge(df_cleaned, df_loc, left_on='postcode', right_on='Postcode')  # merge 2 dataset with postcode
df_merged.drop(columns='Postcode', axis=1, inplace=True)  # delete the duplicate column
df_merged['Area_name'].replace(Area_dic, inplace=True)
df_merged['propertytype'].replace(Propertytype_dic, inplace=True)
df_merged['duration'].replace(duration_dic, inplace=True)
df_merged['postcode'].replace(postcode_dic, inplace=True)

# check if there is null data during merge
print(df_merged.isnull().any(axis=0))  # check the column
print(df_merged.isnull().any(axis=1))  # check the row

df_merged.info()
print(df_merged.describe())

# draw bar plot
df_merged[0:30].plot.bar(x='postcode', y='priceper', figsize=(10, 4), color='black')
plt.savefig('./bar_plt.png')
plt.clf()
# plt.show()

# draw box plot
plt.figure(figsize=(10, 12))
plt.title('2020 London house price per m square')
sns.boxplot(df_merged['priceper'], orient='h')
plt.xticks(rotation=90)
plt.savefig('./box_plt.png')
plt.clf()
# plt.show()

# Compute the correlation matrix and draw heatmap
corr = df_merged.corr()
# print(corr)
plt.figure(figsize=(10, 10))  # Set up the matplotlib figure
cmap = sns.diverging_palette(0, 230, 90, 60, as_cmap=True)  # color map
sns.heatmap(corr, vmax=1, cmap=cmap, vmin=-1, center=0, annot=True,
            square=True, linewidths=3, cbar_kws={"shrink": .8})  # Draw the heatmap
plt.title('Correlation Matrix', loc='left', fontsize=18)
plt.yticks(rotation=0)
plt.savefig('./heatmap.png')
plt.clf()

# visualize the dataset geographically using geopandas
london_filePath = './LSOA_2011_London_gen_MHW.shp'
London = gpd.read_file(london_filePath)  # load london graph
ax = London.plot(color='white', edgecolor='lightgrey', figsize=(20, 20))   # draw the ax
gdf = gpd.GeoDataFrame(df_merged, crs='EPSG:27700',
                       geometry=gpd.points_from_xy(df_merged['Eastings'], df_merged['Northings'])
                       )  # load data with geopandas
gdf.plot(figsize=(20, 20), alpha=0.6, linewidth=0.6, ax=ax, markersize=1,  # draw the scatter with priceper
         cmap='Reds', scheme='BoxPlot', column='priceper')  # the deeper red is, the higher price is
plt.savefig('./priceper_scatter.png')
plt.clf()

# Part3 Feature Engineering

pd_new_feature = gdf.to_crs('EPSG:4326')['geometry']  # change crs to EPSG:4236 for getting (long, lat)
# print(pd_new_feature.shape[0])
# print(pd_new_feature)
# print(pd_new_feature[0].x)
dist_list = []  # to save the distance from each point to hydePark
hydePark_pos = (-0.17278, 51.50639)  # position of hydePark center
for i in range(0, pd_new_feature.shape[0]):  # calculate distance (km)
    pos = (pd_new_feature[i].x, pd_new_feature[i].y)
    dis = geodesic(pos, hydePark_pos).km
    dist_list.append(dis)
# print(dist_list)
df_new_feature = pd.DataFrame({'distance': dist_list})  # transform list into dataframe
df_merged = pd.concat([df_merged, df_new_feature], axis=1)  # merge new feature into df_merged
print(df_merged)
gdf_new_feature = gpd.GeoDataFrame(df_merged, crs='EPSG:27700')  # visualize the new feature
ax = London.plot(color='white', edgecolor='lightgrey', figsize=(20, 20))
gdf_new_feature.plot(figsize=(20, 20), alpha=0.6, linewidth=0.6, ax=ax, markersize=1,
                     cmap='Reds', scheme='BoxPlot', column='distance')  # the deeper red is, the further distance is
plt.savefig('./distance_scatter.png')
plt.clf()

# Part4 Build a linear regressor to predict house price

# transforms the skewed data
df_merged.drop(columns='geometry', axis=1, inplace=True)
X = df_merged[df_merged.columns[df_merged.columns != 'priceper']].drop(columns=['id', 'Eastings', 'Northings'])
Y_withLoc = df_merged[['priceper', 'Eastings', 'Northings']]
numerice_data_list = ['tfarea', 'numberrooms', 'distance',
                      'CURRENT_ENERGY_EFFICIENCY', 'POTENTIAL_ENERGY_EFFICIENCY'
                      ]
numerice_data = pd.DataFrame(X, columns=numerice_data_list)
print(numerice_data)
skewed = X[numerice_data.columns].apply(lambda x: skew(x))
skewed_positive = skewed[(skewed > 0.75)]
skewed_positive = skewed_positive.index

X[skewed_positive].hist(bins=20, figsize=(15, 2), color='lightblue',
                        xlabelsize=0, ylabelsize=0, grid=False, layout=(1, 6)
                        )
plt.savefig('./skewed_positive_hist.png')
plt.clf()

X[skewed_positive] = np.log1p(X[skewed_positive])
X[skewed_positive].hist(bins=20, figsize=(15, 2), color='lightblue',
                        xlabelsize=0, ylabelsize=0, grid=False, layout=(1, 6)
                        )
plt.savefig('./skewed_positive__loglp_hist.png')
plt.clf()

skewed_negative = skewed[(skewed < -0.75)]
skewed_negative = skewed_negative.index

X[skewed_negative].hist(bins=20, figsize=(15, 2), color='lightblue',
                        xlabelsize=3, ylabelsize=3, grid=False, layout=(1, 6)
                        )
plt.savefig('./skewed_negative_hist.png')
plt.clf()

X[skewed_negative] = np.log1p(X[skewed_negative])
plt.xticks(fontsize=5)
X[skewed_negative].hist(bins=20, figsize=(15, 2), color='lightblue',
                        xlabelsize=3, ylabelsize=3, grid=False, layout=(1, 6)
                        )
plt.savefig('./skewed_negative__loglp_hist.png')
plt.clf()

# correlation matrix to understand the association between variables
corr = X.corr()
f, ax = plt.subplots(figsize=(10, 10))
sns.heatmap(corr, center=0, cmap=plt.get_cmap('viridis'),
            square=True, linewidths=.05, annot=True, vmin=-1, vmax=1, ax=ax)
plt.savefig('./X_heatmap.png')
plt.clf()

# hierarchically-clustered heatmap of correlation matrix to understand the association between variables
corr = X.corr()
sns.clustermap(corr, center=0, cmap=plt.get_cmap('viridis'),
               square=True, linewidths=.05, annot=True, vmin=-1, vmax=1)
plt.savefig('./X_clustermap.png')
plt.clf()

# Scale data
scaler = StandardScaler()
X[numerice_data.columns] = scaler.fit_transform(X[numerice_data.columns])
scaled_X = pd.DataFrame(X, columns=X.columns)
scaled_X.boxplot(vert=True, figsize=(10, 10))
plt.savefig('./scaled_X_boxplot.png')
plt.clf()

# drop id column to train model
scaled_X_featured = scaled_X
print(scaled_X_featured.info())
# split the data into a train and test set
(X_train_featured, X_test_featured, Y_train_withLoc, Y_test_withLoc) = train_test_split(scaled_X_featured,
                                                                                        Y_withLoc,
                                                                                        train_size=0.7
                                                                                        )

# train two regression models
# section1 baseline + new feature
model1 = LinearRegression()
model1.fit(X_train_featured, Y_train_withLoc['priceper'])  # fit model on training data


def adj_r2_score(test, pred, ncount, dimension):
    R = r2_score(test, pred)
    n = ncount
    p = dimension
    r2adj = 1-np.divide(np.multiply((1-R), n), (n-p-1))
    return r2adj


# report accuracy score for linear regression model
Y_pred_featured = model1.predict(X_test_featured)
print('MSE for baseline + feature model: ', mean_squared_error(Y_test_withLoc['priceper'], Y_pred_featured))
print('MAE for baseline + feature model: ', mean_absolute_error(Y_test_withLoc['priceper'], Y_pred_featured))
print('adjR2 for baseline + feature model: ',
      adj_r2_score(Y_test_withLoc['priceper'], Y_pred_featured, scaled_X_featured.shape[0], scaled_X_featured.shape[1]))

coef_featured = pd.DataFrame(model1.coef_, index=scaled_X_featured.columns)
coef_featured = coef_featured.reset_index()
coef_featured.columns = ['features', 'importance']
coef_featured['importance'] = np.abs(coef_featured['importance'])
coef_featured = coef_featured.sort_values(by='importance', ascending=False)
plt.figure(figsize=(8, 8))
sns.barplot(x='importance', y='features', data=coef_featured)
plt.savefig('./coef_X_featured.png')
plt.clf()

# section2 baseline
scaled_X_baseline = scaled_X_featured.drop(columns='distance', axis=1)
X_train_baseline = X_train_featured.drop(columns='distance', axis=1)
X_test_baseline = X_test_featured.drop(columns='distance', axis=1)
# (X_train_baseline, X_test_baseline, Y_train, Y_test) = train_test_split(scaled_X_baseline,
#                                                                         Y_test_withID,
#                                                                         train_size=0.7
#                                                                         )
model2 = LinearRegression()
model2.fit(X_train_baseline, Y_train_withLoc['priceper'])  # fit model on training data
# report accuracy score for linear regression model
Y_pred_baseline = model2.predict(X_test_baseline)
print('MSE for baseline model: ', mean_squared_error(Y_test_withLoc['priceper'], Y_pred_baseline))
print('MAE for baseline model: ', mean_absolute_error(Y_test_withLoc['priceper'], Y_pred_baseline))
print('adjR2 for baseline model: ',
      adj_r2_score(Y_test_withLoc['priceper'], Y_pred_baseline, scaled_X_baseline.shape[0], scaled_X_baseline.shape[1])
      )

coef_baseline = pd.DataFrame(model2.coef_, index=scaled_X_baseline.columns)
coef_baseline = coef_baseline.reset_index()
coef_baseline.columns = ['features', 'importance']
coef_baseline['importance'] = np.abs(coef_baseline['importance'])
coef_baseline = coef_baseline.sort_values(by='importance', ascending=False)
plt.figure(figsize=(8, 8))
sns.barplot(x='importance', y='features', data=coef_baseline)
plt.savefig('./coef_X_baseline.png')
plt.clf()

ax = London.plot(color='white', edgecolor='lightgrey', figsize=(20, 20))   # draw the ax
gdf = gpd.GeoDataFrame(Y_test_withLoc, crs='EPSG:27700',
                       geometry=gpd.points_from_xy(Y_test_withLoc['Eastings'], Y_test_withLoc['Northings'])
                       )  # load data with geopandas
gdf.plot(figsize=(20, 20), alpha=0.6, linewidth=0.6, ax=ax, markersize=1,  # draw the scatter with priceper
         cmap='Reds', scheme='BoxPlot', column='priceper')  # the deeper red is, the higher price is
plt.savefig('./test_priceper_scatter.png')
plt.clf()

ax = London.plot(color='white', edgecolor='lightgrey', figsize=(20, 20))   # draw the ax
gdf = gpd.GeoDataFrame(Y_pred_featured, crs='EPSG:27700',
                       geometry=gpd.points_from_xy(Y_test_withLoc['Eastings'], Y_test_withLoc['Northings'])
                       )  # load data with geopandas
gdf.plot(figsize=(20, 20), alpha=0.6, linewidth=0.6, ax=ax, markersize=1,  # draw the scatter with priceper
         cmap='Reds', scheme='BoxPlot')  # the deeper red is, the higher price is
plt.savefig('./pred_featured_priceper_scatter.png')
plt.clf()

ax = London.plot(color='white', edgecolor='lightgrey', figsize=(20, 20))   # draw the ax
gdf = gpd.GeoDataFrame(Y_pred_baseline, crs='EPSG:27700',
                       geometry=gpd.points_from_xy(Y_test_withLoc['Eastings'], Y_test_withLoc['Northings'])
                       )  # load data with geopandas
gdf.plot(figsize=(20, 20), alpha=0.6, linewidth=0.6, ax=ax, markersize=1,  # draw the scatter with priceper
         cmap='Reds', scheme='BoxPlot')  # the deeper red is, the higher price is
plt.savefig('./pred_baseline_priceper_scatter.png')
plt.clf()

# Part5 Additional Regression model

model3 = Lasso(alpha=0.1)
tuned_parameters = [{'alpha': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6]}]
LASSO_GridSearch = GridSearchCV(model3, tuned_parameters, cv=2, scoring='r2')

k = KFold(n_splits=3, shuffle=True, random_state=0)
i = 1
Y = df_merged['priceper']
for train_index, test_index in k.split(scaled_X_featured):
    X_train = scaled_X_featured.iloc[train_index, :]
    X_test = scaled_X_featured.iloc[test_index, :]
    Y_train = Y[train_index]
    Y_test = Y[test_index]
    LASSO_GridSearch.fit(X_train, Y_train)
    model3 = Lasso(alpha=LASSO_GridSearch.best_params_['alpha'])
    model3.fit(X_train, Y_train)
    Y_pred = model3.predict(X_test)
    print('MSE(Lasso) for fold {}: {}'.format(i, mean_squared_error(Y_test, Y_pred)))
    print('MAE(Lasso) for fold {}: {}'.format(i, mean_absolute_error(Y_test, Y_pred)))
    print('adjR2(Lasso) for fold {}: {} '.format(i, adj_r2_score(Y_test, Y_pred, X_train.shape[0], X_train.shape[1])))
    i += 1












