
# coding: utf-8

# In[119]:


import csv
import pandas as pd
print(csv.__file__)
import os
import re
from nltk.translate.bleu_score import sentence_bleu

dataCsv = []

reference = []
candidate = []

referencecsv = []
candidatecsv = []
scorecsv = []

# membaca data csv kaliamat prediksi
def csvData(namaFile):
    path_origin = "C:\\Users\\User\\Downloads\\Code Python\\Last Testing\\NgramWord\\Komentar\\"
    with open(path_origin + namaFile, encoding='utf-8-sig', errors='ignore') as f:
        count = 0
        reader = csv.reader(f)
        for row in reader:
            if row:
                teks1 = re.sub('\n',' ',row[0])
                # menghapus tanda 	
                teks2 = re.sub('	','',teks1)
                teks = re.sub('	','',teks2)
                # menghapus angka
                result = ''.join(i for i in teks if not i.isdigit())
                # menghapus semua karakter ga berguna
                clean = re.sub('[-.!@#$%^&*)(_+=}{\|:;"<>,?/"“”…•–]',' ',result)
                spasi = re.sub(' +', ' ',clean)
                # mengecilkan huruf
                lower = spasi.lower()
                #split = lower.split(".")
                # kalau menemukan "." atau "\n" dijadikan satu array
                dataCsv.append(lower)
                count = count+1

# membaca data csv kaliamat benar
def csvDataReference():
    with open("C:\\Users\\User\\Downloads\\Code Python\\Last Testing\\komentar.csv", encoding='utf-8-sig', errors='ignore') as f:
        count = 0
        reader = csv.reader(f)
        for row in reader:
            if row:
                teks1 = re.sub('\n',' ',row[0])
                # menghapus tanda 	
                teks2 = re.sub('	','',teks1)
                teks = re.sub('	','',teks2)
                # menghapus angka
                result = ''.join(i for i in teks if not i.isdigit())
                # menghapus semua karakter ga berguna
                clean = re.sub('[-.!@#$%^&*)(_+=}{\|:;"<>,?/"“”…•–]',' ',result)
                spasi = re.sub(' +', ' ',clean)
                # mengecilkan huruf
                lower = spasi.lower()
                #split = lower.split(".")
                # kalau menemukan "." atau "\n" dijadikan satu array
                reference.append(lower)
                count = count+1
n = 0
akurasi = 1
csvDataReference()
path_origin = "C:\\Users\\User\\Downloads\\Code Python\\Last Testing\\NgramWord\\Komentar\\"
for file in os.listdir(path_origin):
    file = file.split("\\")
    file_name = file[len(file)-1]
    count = 0
    akurasi = 1
    dataCsv.clear()
    csvData(file_name)
    for data in dataCsv:
        sleng_num = 0  
        count_char = 0
        max_char = 0
        input_sen = data
        senInput = input_sen.split()
        candidate.append(senInput)
        senInputreference = reference[count].split()
#       menghitung menggunakan bleu score
        score = sentence_bleu(candidate, senInputreference)
        akurasi = akurasi + score
        if score < 0.0000000001:
            print(candidate)
            print(senInputreference)
            print(score)
            n+=1
            
        referencecsv.append(reference[count])
        candidatecsv.append(input_sen)
        scorecsv.append(score)      
            
        count+=1
        candidate.clear()
        
    test = pd.DataFrame(scorecsv)
    test.to_csv("C:\\Users\\User\\Downloads\\BleuScore\\NGramWord\\Komentar\\BleuScore%sScore.csv"%(file_name),header = False, index = True) 
    test2 = pd.DataFrame(referencecsv)
    test2.to_csv("C:\\Users\\User\\Downloads\\BleuScore\\NGramWord\\Komentar\\BleuScore%sReference.csv"%(file_name),header = False, index = True) 
    test3 = pd.DataFrame(candidatecsv)
    test3.to_csv("C:\\Users\\User\\Downloads\\BleuScore\\NGramWord\\Komentar\\BleuScore%sCandidate.csv"%(file_name),header = False, index = True) 
    referencecsv.clear()
    candidatecsv.clear()
    scorecsv.clear()
    print(akurasi/count)
print(n)
        
    

