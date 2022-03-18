import csv
import os
import glob
import numpy as np

f = open('./London_loc.csv','r')
dict = {}
lst = []
lst_perprice = []

with f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'Postcode':
            continue
        dict[row[0]] = 1
#print(dict)


datapath = '.\data'
filepath = './london_house_price_origin.csv'
csv_files = glob.glob(os.path.join(datapath,'*.csv'))
#print(csv_files)

for csv_file in csv_files:
    f = open(csv_file,'r')
    with f:
        reader = csv.reader(f)
        for row in reader:
            if row[6] == 'postcode':
                continue
            postcode = row[6].replace(' ','')
            if postcode in dict:
                lst.append(row)

with open(filepath,'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(lst)

output_path = './london_house_price.csv'
head = ['priceper', 'year', 'dateoftransfer', 'propertytype', 'duration', 'price', 'postcode', 'lad21cd',
        'transactionid', 'id', 'tfarea', 'numberrooms', 'classt', 'CURRENT_ENERGY_EFFICIENCY',
        'POTENTIAL_ENERGY_EFFICIENCY', 'CONSTRUCTION_AGE_BAND']
head_output = ['priceper', 'year', 'propertytype', 'duration', 'price',
        'tfarea', 'numberrooms', 'classt', 'CURRENT_ENERGY_EFFICIENCY',
        'POTENTIAL_ENERGY_EFFICIENCY']
#initialize
propertytype_dic = {}
duration_dic = {}
classt_dic = {}
i = 0
j = 0
k = 0
for row in lst:
    if 'NA' in row:
        continue
    if row[3] not in propertytype_dic:
        propertytype_dic[row[3]] = i
        i = i + 1
    if row[4] not in duration_dic:
        duration_dic[row[4]] = j
        j = j + 1
    if row[12] not in classt_dic:
        classt_dic[row[12]] = k
        k = k + 1
    # Create perprice list
    lst_perprice.append(row[0])
# Computing IQR
Q1 = np.quantile(lst_perprice,0.25,interpolation='lower') # Q1 = 0.25
Q3 = np.quantile(lst_perprice,0.75,interpolation='lower') # Q3 = 0.75
Q1 = float(Q1)
Q3 = float(Q3)
IQR = Q3 - Q1 # IQR
Q1 = Q1 - 0.5 * IQR # low outlier
Q3 = Q3 + 1.5 * IQR  # high outlier
print(Q1,Q3)

with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(head_output)
    for row in lst:
        if 'NA' in row:
            continue
        if float(row[0]) < Q1 or float(row[0]) > Q3:
            continue
        row_output = [row[0],row[1],propertytype_dic[row[3]],duration_dic[row[4]],row[5],row[10],row[11],classt_dic[row[12]],row[13],row[14]]
        #print(row_output)
        writer.writerow(row_output)

print('csv create done!')

# Calculate the mean of priceper
dict_perprice = {}
for row in lst:
    if 'NA' in row:
        continue
    if int(row[1]) not in dict_perprice:
        dict_perprice[int(row[1])] = []
    else:
        dict_perprice[int(row[1])].append((float(row[0])))
for year in dict_perprice.keys():
    dict_perprice[year] = np.mean(dict_perprice[year])
print(dict_perprice)