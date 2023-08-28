#k-means clustering

from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
df=pd.read_csv("/content/keywordtrend.csv")
df.head()

df.columns = ['Date', 'Total_Positivecase', 'Total_Sample', 'Total negativecase', 'Totaldeath',
              'People affected with lungs problem']
print(df)

plt.scatter(df.Total_Sample,df['Total_Positivecase'])
plt.xlabel('Total Sample')
plt.ylabel('Total_Postitivecase')

km= KMeans(n_clusters=3)
predicted= km.fit_predict(df[['Total_Sample','Total_Positivecase']])
predicted

df['cluster']=predicted
df.head()
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==3]
plt.scatter(df1.Total_Sample,df1['Total_Positivecase'],color='green')
plt.scatter(df2.Total_Sample,df2['Total_Positivecase'],color='red')
plt.scatter(df3.Total_Sample,df3['Total_Positivecase'],color='blue')
plt.xlabel('Total_Sample')
plt.ylabel('Total_Positivecase')

scale= MinMaxScaler()
scale.fit(df[['Total_Positivecase']])
df['Total_Positivecase']=scale.transform(df[['Total_Positivecase']])
scale.fit(df[['Total_Sample']])
df['Total_Sample']=scale.transform(df[['Total_Sample']])
km=KMeans(n_clusters=3)
predicted=km.fit_predict(df[['Total_Sample','Total_Positivecase']])
predicted

df=df.drop(['cluster'],axis='columns')
df['cluster']=predicted
df.head()
df1=df[df.cluster==0]
df2=df[df.cluster==1]
df3=df[df.cluster==3]
plt.scatter(df1.Total_Sample,df1['Total_Positivecase'],color='green')
plt.scatter(df2.Total_Sample,df2['Total_Positivecase'],color='red')
plt.scatter(df3.Total_Sample,df3['Total_Positivecase'],color='blue')
plt.xlabel('Total_Sample')
plt.ylabel('Total_Positivecase')

plt.scatter(df1.Total_Sample,df1['Total_Positivecase'],color='green')
plt.scatter(df2.Total_Sample,df2['Total_Positivecase'],color='red')
plt.scatter(df3.Total_Sample,df3['Total_Positivecase'],color='blue')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='black',marker='*')
plt.xlabel('Total_Sample')
plt.ylabel('Total_Positivecase')
