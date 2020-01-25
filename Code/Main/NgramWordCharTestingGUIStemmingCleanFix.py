
# coding: utf-8

# In[3]:


from nltk.util import ngrams
from collections import defaultdict
from collections import OrderedDict
from tkinter import *

import string
import time
import gc
from math import log10
import sqlite3
from sqlite3 import Error
start_time = time.time()
import csv
import pandas as pd
print(csv.__file__)

import glob, os

import requests
import os
import re
from bs4 import BeautifulSoup
import sys, time
from itertools import cycle

global keyword


# # Buat Koneksi

# In[4]:


# membuat koneksi
def create_connection(db_file):
    try:
        conn = sqlite3.connect(db_file)
        return conn
    except Error as e:
        print(e)
    return None


# # Cek Kata

# In[5]:


# mengecek kalimat yang tidak ada di korpus kata
def cekWrong(conn,inputWrong):
    x = 0

#   memecah inputan menjadi per kata
    temp_l = inputWrong.split()
    i = 0
    j = 0

#   Membersihkan inputan
    for word in temp_l :
        j = 0
        for l in word :
            if l in string.punctuation:
                if l == "'":
                    if j+1<len(word) and word[j+1] == 's':
                        j = j + 1
                        continue
                word = word.replace(l," ")
#                     print(j,word[j])
            j += 1
        temp_l[i] = word.lower()
        i=i+1   

    content = " ".join(temp_l)
    token = content.split()
    
#   cek perkata ada di korpus kata atau tidak
    for teks in token:
        kata.append(teks)
        teksStemming = fetch(teks)
        cur = conn.cursor()
#       query untuk cek yang tidak memiliki tag
        cur.execute('Select kata from korpus_kata where kata = (?) and tag != "Entri tidak ditemukan" and tag != "tidak ditemukan kata yang dicari" and tag != "entri tidak ditemukan" ',(teksStemming,))
        a = cur.fetchone()
        conn.commit()
        if a==None:
#           kaliamt yang tidak ada di korpus kata disimpan di wrong
            wrong.append(x)
        x= x + 1


# # Memotong Kalimat

# In[6]:


# fungsi untuk mengambil 3 kata sebelum kalimat singkatan
def kalimatTeks(i):
    n = wrong[i]
    if n >= 3 :
        temp = kata[n-3] + ' ' + kata[n-2]  + ' ' + kata[n-1]
        input_teks.append(temp)
    elif n == 2 :
        temp = 's' + ' ' + kata[n-2] + ' ' + kata[n-1]
        input_teks.append(temp)
    elif n == 1 :
        temp = 's' + ' ' + 's' + ' ' + kata[n-1]
        input_teks.append(temp)
    elif n == 0 :
        temp = 's' + ' ' + 's' + ' ' + 's'
        input_teks.append(temp)
    print(n)


# # Tokenize dan Load Korpus

# In[7]:


# fungsi membersihkan kalimat
def removePunctuations(sen):
    temp_l = sen.split()
    i = 0
    j = 0
    
#   menghilangkan tanda baca dan membuat jadi huruf kecil
    for word in temp_l :
        j = 0
        for l in word :
            if l in string.punctuation:
                if l == "'":
                    if j+1<len(word) and word[j+1] == 's':
                        j = j + 1
                        continue
                word = word.replace(l," ")
            j += 1

        temp_l[i] = word.lower()
        i=i+1   
        
    content = " ".join(temp_l)
    return content


# In[8]:


# memuat korpus kalimat untuk menjadi dataset dan menghitung jumlah dari quadgram, trigram dan bigram 
def loadCorpus(file_path, bi_dict, tri_dict, quad_dict, vocab_dict):

    w1 = ''    #variabel untuk menyimpan 3 kalimat terakhir untuk token set
    w2 = ''    #variabel untuk menyimpan 2 kalimat terakhir untuk token set
    w3 = ''    #variabel untuk menyimpan kalimat terakhir untuk token set
    token = []

    word_len = 0

#   memuat korpus kalimat dan di baca line per line
    with open(file_path,'r', encoding="utf-8") as file:
        print(file)
        for line in file:
            print(line)
#           memecah kalimat menjadi kata
            temp_l = line.split()
            i = 0
            j = 0
            
#           menghilangkan tanda baca dan mebuat kata menjadi huruf kecil
            for word in temp_l :
                j = 0
                for l in word :
                    if l in string.punctuation:
                        if l == "'":
                            if j+1<len(word) and word[j+1] == 's':
                                j = j + 1
                                continue
                        word = word.replace(l," ")
                    j += 1

                temp_l[i] = word.lower()
                i=i+1   

            content = " ".join(temp_l)

            token = content.split()
#           menghitung jumlah kata
            word_len = word_len + len(token)  

            if not token:
                continue

#           menambahkan kalimat terakhir ke variabel
            if w3!= '':
                token.insert(0,w3)
            
#           token untuk bigrams
            temp0 = list(ngrams(token,2))

#           menambahkan kalimat dua terakhir ke variabel
            if w2!= '':
                token.insert(0,w2)

#           token untuk trigrams
            temp1 = list(ngrams(token,3))

#           menambahkan kalimat tiga terakhir ke variabel
            if w1!= '':
                token.insert(0,w1)

#           menambahkan kata unik ke vocaulary
            for word in token:
                if word not in vocab_dict:
                    vocab_dict[word] = 1
                else:
                    vocab_dict[word]+= 1
                    
#           token untuk quadgrams
            temp2 = list(ngrams(token,4))

    
#           menghitung frekuensi dari bigram
            for t in temp0:
                sen = ' '.join(t)
                bi_dict[sen] += 1

#           menghitung frekuensi dari trigram
            for t in temp1:
                sen = ' '.join(t)
                tri_dict[sen] += 1

#           menghitung frekuensi dari quadgrams
            for t in temp2:
                sen = ' '.join(t)
                quad_dict[sen] += 1

#           menghitung panjang token
            n = len(token)

#           menambahkan kalimat ke variabel
            if (n -3) >= 0:
                w1 = token[n -3]
            if (n -2) >= 0:
                w2 = token[n -2]
            if (n -1) >= 0:
                w3 = token[n -1]
    return word_len


# In[9]:


# memuat korpus kalimat untuk menjadi dataset character dan menghitung jumlah dari quadgram, trigram dan bigram 
def loadCorpusChar(file_path, bi_dict_char, tri_dict_char, quad_dict_char, vocab_dict_char):

    w1_char = ''    #variabel untuk menyimpan 3 huruf terakhir untuk token set
    w2_char = ''    #variabel untuk menyimpan 2 huruf terakhir untuk token set
    w3_char = ''    #variabel untuk menyimpan huruf terakhir untuk token set
    token = []
    
    word_len_char = 0

#   memuat korpus kalimat dan di baca line per line
    with open(file_path,'r', encoding="utf-8") as file:
        for line in file:

#           memecah kalimat menjadi karakter
            temp_l_char = list(line)
            i = 0
            j = 0
            
#           menghilangkan tanda baca dan mebuat huruf kecil
            for char in temp_l_char :
                j = 0
                for l in char :
                    if l in string.punctuation:
                        if l == "'":
                            if j+1<len(char) and char[j+1] == 's':
                                j = j + 1
                                continue
                        char = char.replace(l," ")
                        #print(j,word[j])
                    j += 1

                temp_l_char[i] = char.lower()
                i=i+1   


            content_char = " ".join(temp_l_char)

            token_char = content_char.split()
#           menghitung jumlah karakter
            word_len_char = word_len_char + len(token_char)  

            if not token_char:
                continue

#           menambahkan huruf terakhir ke variabel
            if w3_char!= '':
                token_char.insert(0,w3_char)

#           token untuk bigrams
            temp0_char = list(ngrams(token_char,2))

#           menambahkan 2 huruf terakhir ke variabel
            if w2_char!= '':
                token_char.insert(0,w2_char)

#           token untuk trigrams
            temp1_char = list(ngrams(token_char,3))

#           menambahkan 3 huruf terakhir ke variabel
            if w1_char!= '':
                token_char.insert(0,w1_char)

#           menambahkan huruf unik ke vocaulary
            for char in token_char:
                if char not in vocab_dict_char:
                    vocab_dict_char[char] = 1
                else:
                    vocab_dict_char[char]+= 1
                  
                
#           token untuk quadgrams
            temp2_char = list(ngrams(token_char,4))

#           menghitung frekuensi dari bigram
            for t in temp0_char:
                sen_char = ''.join(t)
                bi_dict_char[sen_char] += 1

#           menghitung frekuensi dari trigram
            for t in temp1_char:
                sen_char = ''.join(t)
                tri_dict_char[sen_char] += 1

#           menghitung frekuensi dari quadgram
            for t in temp2_char:
                sen_char = ''.join(t)
                quad_dict_char[sen_char] += 1


#           menghitung panjang token
            n_char = len(token_char)

#           menambahkan kalimat ke variabel
            if (n_char -3) >= 0:
                w1_char = token_char[n_char -3]
            if (n_char -2) >= 0:
                w2_char = token_char[n_char -2]
            if (n_char -1) >= 0:
                w3_char = token_char[n_char -1]
    return word_len_char


# In[10]:


# membuat dict untuk menyimpan probabilitas kalimat quadgrams
def findQuadgramProbGT(vocab_dict, bi_dict, tri_dict, quad_dict, quad_prob_dict, nc_dict, k):
    
    i = 0
    V = len(vocab_dict)

    for quad_sen in quad_dict:
#       membagi kaliamat quadgrams menjadi perkata
        quad_token = quad_sen.split()
        
#       membuat kalimat trigram dari kalimat quadgrams
        tri_sen = ' '.join(quad_token[:3])

        quad_count = quad_dict[quad_sen]
        tri_count = tri_dict[tri_sen]
        
#       menggunakan goot turing smoothing
        if quad_dict[quad_sen] <= k  or (quad_sen not in quad_dict):
            quad_count = findGoodTuringAdjustCount( quad_dict[quad_sen], k, nc_dict)
        if tri_dict[tri_sen] <= k  or (tri_sen not in tri_dict):
            tri_count = findGoodTuringAdjustCount( tri_dict[tri_sen], k, nc_dict)

#       menghitung probabilitas quadgram dengan membagi kalimat quadgrams dengan trigram
        prob = quad_count / tri_count
        
#       menyimpan probabilitas quadgrams ke dict
        if tri_sen not in quad_prob_dict:
            quad_prob_dict[tri_sen] = []
            quad_prob_dict[tri_sen].append([prob,quad_token[-1]])
        else:
            quad_prob_dict[tri_sen].append([prob,quad_token[-1]])
  
    prob = None
    quad_token = None
    tri_sen = None


# In[11]:


# membuat dict untuk menyimpan probabilitas kalimat trigrams
def findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, nc_dict, k):
    V = len(vocab_dict)

    for tri in tri_dict:
#       membagi kaliamat trigrams menjadi perkata
        tri_token = tri.split()

#       membuat kalimat bigram dari kalimat trigrams
        bi_sen = ' '.join(tri_token[:2])
        
        tri_count = tri_dict[tri]
        bi_count = bi_dict[bi_sen]
        
#       menggunakan goot turing smoothing
        if tri_dict[tri] <= k or (tri not in tri_dict):
            tri_count = findGoodTuringAdjustCount( tri_dict[tri], k, nc_dict)
        if bi_dict[bi_sen] <= k or (bi_sen not in bi_dict):
            bi_count = findGoodTuringAdjustCount( bi_dict[bi_sen], k, nc_dict)

#       menghitung probabilitas trigram dengan membagi kalimat trigram dengan bigram
        prob = tri_count / bi_count
        
#       menyimpan probabilitas trigrams ke dict
        if bi_sen not in tri_prob_dict:
            tri_prob_dict[bi_sen] = []
            tri_prob_dict[bi_sen].append([prob,tri_token[-1]])
        else:
            tri_prob_dict[bi_sen].append([prob,tri_token[-1]])
    
    prob = None
    tri_token = None
    bi_sen = None


# In[12]:


# membuat dict untuk menyimpan probabilitas kalimat bigrams
def findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, nc_dict, k):
    V = len(vocab_dict)
    bigram = []
    bigram_prob = []
    
    for bi in bi_dict:
#       membagi kaliamat bigrams menjadi perkata
        bi_token = bi.split()
        
        bigram.append(bi)
        
#       membuat kalimat unigrams dari kalimat bigrams
        unigram = bi_token[0]

        bi_count = bi_dict[bi]
        uni_count = vocab_dict[unigram]
        
#       menggunakan goot turing smoothing
        if bi_dict[bi] <= k or (bi not in bi_dict):
            bi_count = findGoodTuringAdjustCount( bi_dict[bi], k, nc_dict)
        if vocab_dict[unigram] <= k or (unigram not in vocab_dict):
            uni_count = findGoodTuringAdjustCount( vocab_dict[unigram], k, nc_dict)
        
#       menghitung probabilitas bigram dengan membagi kalimat bigram dengan unigram
        prob = bi_count / uni_count
        bigram_prob.append(prob)
        
#       menyimpan probabilitas bigrams ke dict
        if unigram not in bi_prob_dict:
            bi_prob_dict[unigram] = []
            bi_prob_dict[unigram].append([prob,bi_token[-1]])
        else:
            bi_prob_dict[unigram].append([prob,bi_token[-1]])

    prob = None
    bi_token = None
    unigram = None


# In[13]:


# membuat dict untuk menyimpan probabilitas karakter trigrams
def findCharTrigramProbGT(vocab_dict_char, bi_dict_char, tri_dict_char, tri_prob_dict_char, nc_dict_char, k):
    V = len(vocab_dict_char)
    trigram = []
    trigram_prob = []    
    
    for tri_char in tri_dict_char:
#       membagi kaliamat trigrams menjadi perhuruf
        tri_token_char = tri_char.split()
        char = ' '.join(tri_token_char[0])
        tri_tokenc = char.split()
        
#       membuat huruf bigram dari huruf trigrams
        bi_sen_char = ''.join(tri_tokenc[:2])
        
        tri_count_char = tri_dict_char[tri_char]
        bi_count_char = bi_dict_char[bi_sen_char]
        
#       menggunakan goot turing smoothing
        if tri_dict_char[tri_char] <= k or (tri_char not in tri_dict_char):
            tri_count_char = findGoodTuringAdjustCount( tri_dict_char[tri_char], k, nc_dict_char)
        if bi_dict_char[bi_sen_char] <= k or (bi_sen_char not in bi_dict_char):
            bi_count_char = findGoodTuringAdjustCount( bi_dict_char[bi_sen_char], k, nc_dict_char)
        
#       menghitung probabilitas trigram dengan membagi huruf trigram dengan bigram
        prob_char = tri_count_char / bi_count_char
        trigram_prob.append(prob_char)
        tri_prob_dict_char[tri_char] = prob_char        
        
#       menyimpan probabilitas trigrams ke dict
        if bi_sen_char not in tri_prob_dict_char:
            tri_prob_dict_char[bi_sen_char] = []
            tri_prob_dict_char[bi_sen_char].append([prob_char,tri_token_char[-1]])
        else:
            tri_prob_dict_char[bi_sen_char].append([prob_char,tri_token_char[-1]])
    
    prob_char = None
    tri_token_char = None
    bi_sen_char = None


# In[14]:


# membuat dict untuk menyimpan probabilitas karakter bigrams
def findCharBigramProbGT(vocab_dict_char, bi_dict_char, bi_prob_dict_char, nc_dict_char, k):
           
    #vocabulary size
    V = len(vocab_dict_char)
    bigram = []
    bigram_prob = []
    for bi_char in bi_dict_char:
#       membagi kaliamat bigrams menjadi perhuruf
        bi_token_char = bi_char.split()
        
        bigram.append(bi_char)
        
#       membuat huruf unigram dari huruf bigrams
        unigram_char = bi_token_char[0][0]
        test.append(unigram_char)


        bi_count_char = bi_dict_char[bi_char]
        uni_count_char = vocab_dict_char[unigram_char]

#       menggunakan goot turing smoothing
        if bi_dict_char[bi_char] <= k or (bi_char not in bi_dict_char):
            bi_count_char = findGoodTuringAdjustCount( bi_dict_char[bi_char], k, nc_dict_char)
        if vocab_dict_char[unigram_char] <= k or (unigram_char not in vocab_dict_char):
            uni_count_char = findGoodTuringAdjustCount( vocab_dict_char[unigram_char], k, nc_dict_char)
        
#       menghitung probabilitas bigram dengan membagi huruf bigram dengan unigram
        prob_char = bi_count_char / uni_count_char
        bigram_prob.append(prob_char)
        bi_prob_dict_char[bi_char] = prob_char

#       menyimpan probabilitas bigrams ke dict
        if unigram_char not in bi_prob_dict_char:
            bi_prob_dict_char[unigram_char] = []
            bi_prob_dict_char[unigram_char].append([prob_char,bi_token_char[-1]])
        else:
            bi_prob_dict_char[unigram_char].append([prob_char,bi_token_char[-1]])
    
    prob_char = None
    bi_token_char = None
    unigram_char = None


# In[15]:


# untuk mengurutkan probabilitas dari yang terbesar
def sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict):
    for key in bi_prob_dict:
        if len(bi_prob_dict[key])>1:
            bi_prob_dict[key] = sorted(bi_prob_dict[key],reverse = True)
    
    for key in tri_prob_dict:
        if len(tri_prob_dict[key])>1:
            tri_prob_dict[key] = sorted(tri_prob_dict[key],reverse = True)
    
    for key in quad_prob_dict:
        if len(quad_prob_dict[key])>1:
            quad_prob_dict[key] = sorted(quad_prob_dict[key],reverse = True)[:2]


# In[16]:


# untuk mengambil inputan
def takeInput():
    cond = False
    while(cond == False):
        sen = input('Enter the string\n')
        temp = sen.split()
        if len(temp) < 0:
            print("Please enter atleast 1 words !")
        else:
            cond = True
            temp = temp[:]
    sen = " ".join(temp)
    return sen


# # Regression related stuff

# In[17]:



# menghitung fit line terbaik dari simple regression
from statistics import mean
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style

# menghitung slope dari fit line terbaik
def findBestFitSlope(x,y):
    m = (( mean(x)*mean(y) - mean(x*y) ) / 
          ( mean(x)** 2 - mean(x**2)))

    return m
      
# menghitung intercept dari fit line terbaik
def findBestFitIntercept(x,y,m):
    c = mean(y) - m*mean(x)
    return c


# # Menghitung Nc untuk quadgrams dan trigrams ketika c > k , k = 5

# In[18]:



# menghitung nc untuk quadgrams dan trigrams dimana c > 5
def findFrequencyOfFrequencyCount(ngram_dict, k, n, V, token_len):
    #for keeping count of 'c' value i.e Nc
#   untuk tetap menghitung dari nilai c
    nc_dict = {}
    
#   menghitung nilai dari Nc,c = 0  dengan v^n - (total ngram tokens)
    nc_dict[0] = V**n - token_len
#   menghitung nc dimana c = k, disini menggunakan k = 5

#   menghitung dari ngram
    for key in ngram_dict:
        if ngram_dict[key] <= k + 1:
            if ngram_dict[key] not in nc_dict:
                nc_dict[ ngram_dict[key]] = 1
            else:
                nc_dict[ ngram_dict[key] ] += 1
    
    
#   check jika semua value dari nc berada di nc_dict atau tidak
    val_present = True
    for i in range(1,7):
        if i not in nc_dict:
            val_present = False
            break
    if val_present == True:
        return nc_dict
    
#   mengisi nilai dari nc dimana menggunakan regressi di atas c = 6

    data_pts = {}
    i = 0
#   untuk quadgrams
    for key in ngram_dict:
        if ngram_dict[key] not in data_pts:
                data_pts[ ngram_dict[key] ] = 1
                i += 1
        if i >5:
            break
            
#   mendapat nilai nc 
    for key in ngram_dict:
        if ngram_dict[key] in data_pts:
            data_pts[ ngram_dict[key] ] += 1
    
#   membuat coordinat x, y dari regresi
    x_coor = [ np.log(item) for item in data_pts ]
    y_coor = [ np.log( data_pts[item] ) for item in data_pts ]
    x = np.array(x_coor, dtype = np.float64)
    y = np.array(y_coor , dtype = np.float64)
  
    slope_m = findBestFitSlope(x,y)
    intercept_c = findBestFitIntercept(x,y,slope_m)

#   mencari nc yang hilang dan memberikan nilai menggunakan regressi
    for i in range(1,(k+2)):
        if i not in nc_dict:
            nc_dict[i] = (slope_m*i) + intercept_c
    
    return nc_dict


# # Mencari Good Turing Probability

# In[19]:


# mencari adjusted count c* di good turing smoothing
def findGoodTuringAdjustCount(c, k, nc_dict):
   
    adjust_count = ( ( (( c + 1)*( nc_dict[c + 1] / nc_dict[c])) - ( c * (k+1) * nc_dict[k+1] / nc_dict[1]) ) /
                     ( 1 - (( k + 1)*nc_dict[k + 1] / nc_dict[1]) )
                   )
    return adjust_count


# # Driver function untuk melakukan prediksi

# In[20]:


# mencari prediksi kalimat menggunakan backoff
def doPredictionBackoffGT(input_sen, bi_dict, tri_dict, quad_dict, bi_prob_dict, tri_prob_dict, quad_prob_dict):
    token = input_sen.split()
    
    if input_sen in quad_prob_dict and quad_prob_dict[ input_sen ][0][0]>0:
        print("quad")
        pred = quad_prob_dict[input_sen][0]
    elif ' '.join(token[1:]) in tri_prob_dict and tri_prob_dict[' '.join(token[1:])][0][0]>0:
        print("tri")
        pred = tri_prob_dict[ ' '.join(token[1:]) ][0]
    elif ' '.join(token[2:]) in bi_prob_dict and bi_prob_dict[ ' '.join(token[2:]) ][0][0]>0:
        print("bi")
        pred = bi_prob_dict[' '.join(token[2:])][0]
    else:
        print("kosong")
        pred = []
    return pred


# # Script Mencari Kemungkinan Kata

# In[21]:


# fungsi mencari kemungkiann kata dari kata singkatan
def contained(pred,kata):
    jumlahs = len(pred)
    jumlaht = len(kata)
    jumlah = 0
    value = 0
    i=0
    j=0
    n = ""
    veri = 0
#   untuk kata yang berulang seperti yaaaaaannnnggg menjadi yang
    if len(kata) > 2: 
        for k in range(0,len(kata)):
            if len(kata) > 0:
                if k < len(kata)-1:
                    if kata[k] != kata[k+1]:
                        if value == 1:
                            n = n + kata[k] + kata[k]
                            veri = 1
                            value = 0
                        else: 
                            n = n + kata[k]
                            veri = 1
                            value = 0
                    elif kata[k] == kata[k+1]:
                        value+=1
                        if k == len(kata)-2:
                            if value == 1:
                                n = n + kata[k]

        if veri != 0:
            n = n + kata[len(kata)-1]
            kata = n
            jumlaht = len(kata)


#   menghitung berapa kalimat singkatan yang mengandung kaliamt prediksi
    while i < jumlaht:
        while j < jumlahs:
            if kata[i]==pred[j]:
                jumlah+=1
                j+=1
                break
            j+=1
        i+=1    
        
#   mengembalikan nilai true apa bila semua mengandung kata tersebut
    if jumlah >= jumlaht:
        return True
    else :
        return False        


# # Data Testing

# In[22]:


# fungsi untuk memanggil data dari csv
def csvData(namaFile):
    path_origin = "C:\\Users\\User\\Downloads\\testcase\\"
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
                clean = re.sub('[-!@#$%^&*)(_+=}{\|:;"<>,?/"“”…•–]',' ',result)
                spasi = re.sub(' +', ' ',clean)
                # mengecilkan huruf
                lower = spasi.lower()
                #split = lower.split(".")
                # kalau menemukan "." atau "\n" dijadikan satu array
                dataCsv.append(lower)
                count = count+1


# # GUI

# In[23]:


# membuat gui

# membuat form
def makeform(root, field):
    print(field)
    row = Frame(root)
    lab = Label(row, width=22, text=field+": ", anchor='w')
    ent = Text(row, height=5)
    row.pack(side=TOP, 
             fill=X, 
             padx=5, 
             pady=5)
    lab.pack(side=LEFT)
    ent.pack(side=RIGHT, 
             expand=YES, 
             fill=X)
    entries[field] = ent
    return entries

# membuat tabel
def maketable(root,kataslang,prediksi):
    b = Frame(root)
    b.pack(side=TOP, 
             fill=X, 
             padx=5, 
             pady=5)
    n = len(kataslang)
    count = 0
    height = 8
    width = 2
    for i in range(n): #Rows
        for j in range(width): #Columns
            tab = Entry(b, text="")
            if j == 0:
                tab.insert(0, kataslang[count])
            else:
                tab.insert(0, prediksi[count])
            tab.grid(row=i, column=j)
        count+=1
    return tab
            
def clicked(entries,senInput):
    entries['Output'].insert('0.0', senInput[:] )


# # Stemming

# In[24]:


# Fungsi Stemming bahasa
def fetch(keyword):
    rootWord = keyword
    pure = keyword
    if contains(rootWord) == True:
        rootWord = keyword

    
    elif (rootWord.endswith("kan") or rootWord.endswith("kanlah") or rootWord.endswith("kankah") or rootWord.endswith("kanku") 
    or rootWord.endswith("kanmu") or rootWord.endswith("kannya") or rootWord.endswith("kanpun") or rootWord.endswith("i") 
    or rootWord.endswith("ilah") or rootWord.endswith("ikah") or rootWord.endswith("iku") or rootWord.endswith("imu") 
    or rootWord.endswith("inya") or rootWord.endswith("ipun")):
        rootWord = removeSuffix(rootWord)
        rootWord = removePrefix(rootWord)
        print(rootWord)
        if contains(rootWord) == None:
            rootWord = keyword;
            rootWord = removeSuffiks(rootWord)
            rootWord = removePrefix(rootWord)
            print(rootWord)
            if contains(rootWord) == None:
                rootWord = keyword;
                rootWord = removePrefix(rootWord)
            
        
    else:
        rootWord = removeSuffix(rootWord)
        rootWord = removePrefix(rootWord)
        if contains(rootWord) == None:
            rootWord = keyword;
            rootWord = removePrefix(rootWord)
        
    if contains(rootWord) == None:
        return pure;
    else:
        return rootWord;

def removeSuffiks(keyword):
    if contains(keyword) == None:
        keyword = removePossesiveSuffiks(keyword)
        
    if contains(keyword) == None:
        keyword = removeDerivationSuffiks(keyword)
        
    return keyword;    

def removePossesiveSuffiks(keyword):
    if keyword.endswith("lah"):
        keyword = keyword[:-3]
    elif keyword.endswith("kah"):
        keyword = keyword[:-3]
    elif keyword.endswith("pun"):
        keyword = keyword[:-3]
        
    if keyword.endswith("ku"):
        keyword = keyword[:-2]
    elif keyword.endswith("mu"):
        keyword = keyword[:-2]
    elif keyword.endswith("nya"):
        keyword = keyword[:-3]
        
    return keyword;   

def removeDerivationSuffiks(keyword):
    if keyword.endswith("kan"):
        keyword = keyword[:-3]
        
    return keyword;     

def removeSuffix(keyword):
    return removeSuffixes(keyword);

def removeSuffixes(keyword):
    if contains(keyword) == None:
        keyword = removePossesive(keyword)
        
    if contains(keyword) == None:
        keyword = removeDerivationSuffix(keyword)
        
    return keyword;     

def removePossesive(keyword):
    if keyword.endswith("lah"):
        keyword = keyword[:-3]
    elif keyword.endswith("kah"):
        keyword = keyword[:-3]
    elif keyword.endswith("pun"):
        keyword = keyword[:-3]
        
    if keyword.endswith("ku"):
        keyword = keyword[:-2]
    elif keyword.endswith("mu"):
        keyword = keyword[:-2]
    elif keyword.endswith("nya"):
        keyword = keyword[:-3]
        
    return keyword;      

def removeDerivationSuffix(keyword):
    if keyword.endswith("i"):
        keyword = keyword[:-1]
    elif keyword.endswith("an"):
        keyword = keyword[:-2]
        
    return keyword
    
def removePrefix(keyword):
    return removeDerivation(keyword)

def removeDerivation(keyword):
    if contains(keyword) == None:
        if keyword.startswith("di"):
            keyword = keyword[2:]
            if contains(keyword) == None and (keyword.startswith("ke") or keyword.startswith("se")):
                keyword = keyword[2:]
                
        elif keyword.startswith("ke"):
            keyword = keyword[2:]
            if contains(keyword) == None and (keyword.startswith("se") or keyword.startswith("di")):
                keyword = keyword[2:]
            
        elif keyword.startswith("se"):
            keyword = keyword[2:]
            if contains(keyword) == None and (keyword.startswith("ke") or keyword.startswith("di")):
                keyword = keyword[2:]
            
    if contains(keyword) == None:
        if keyword.startswith("me"):
            if re.search("[lrwy]",keyword[2:3]) and re.search("[aiueo]",keyword[3:4]):
                keyword = keyword[2:]
            
            elif keyword.startswith("meter") and len(keyword)>4:
                if re.search("[aiueo]",keyword[5:6]) and contains("r"+keyword[5]):
                    keyword = "r" + keyword[5:]
                    
                elif re.search("[aiueo]",keyword[5:6]):
                    keyword = keyword[5:]
                    
                elif re.search("[aiueor]",keyword[5:6]) == None and re.search("er",keyword[5:6]) == None:
                    keyword = keyword[5:]
                    
                elif len(keyword)>7:
                    if re.search("[aiueor]",keyword[5:6]) == None and re.search("er",keyword[6:8]):
                        keyword = keyword[5:]
                
        
            elif keyword.startswith("mete"):
                if (len(keyword)>7):
                    if re.search("[aiueor]",keyword[4:5]) == None and re.search("er",keyword[5:7]) and re.search("[aiueo]",keyword[7:8]) == None:
                        keyword = keyword.substring(4);            
            
            elif keyword.startswith("meng"):
                if re.search("[ghq]",keyword[4:5]):
                    keyword = keyword[4:]
                elif re.search("[aiueo]",keyword[4:5]) and contains(keyword[4:]):
                    keyword = keyword[4:]
                elif re.search("[aiueo]",keyword[4:5]) and contains("k"+keyword[4:]):
                    keyword = "k"+keyword[4:]
                    
            elif keyword.startswith("menter") and len(keyword)>4:
                    
                if keyword.substring(6, 7).matches("[aiueo]") and contains("r"+keyword[6:]):
                    keyword = "r"+keyword[6:]
                elif keyword.substring(6, 7).matches("[aiueo]"):
                    keyword = keyword[6:]
                elif keyword.substring(6, 7).matches("[aiueor]") == None and keyword.substring(6, 7).matches("er") == None:
                    keyword = keyword[6:]
                elif (len(keyword)>8):
                    if keyword.substring(6, 7).matches("[aiueor]") == None and keyword.substring(7, 9).matches("er"):
                        keyword = keyword[6:]
                
            elif keyword.startswith("mente"):
                if (len(keyword)>8):
                    if keyword.substring(5, 6).matches("[aiueor]") == None and keyword.substring(6, 8).matches("er") and keyword.substring(8, 9).matches("[aiueo]") == None:
                        keyword = keyword[5:]
    
            elif keyword.startswith("meny"):
                if contains("s" + keyword[4:]):
                    keyword = "s" + keyword[4:]
            
            elif keyword.startswith("mempe") and contains(keyword[4:]):
                keyword = keyword[3:]
        
            elif keyword.startswith("mempe") and len(keyword)>9:
                if keyword == "mempelajari":
                    keyword = "ajar"
                    
                elif keyword.substring(5, 6).matches("[l]") and keyword.substring(6, 7).matches("[aiueo]"):
                    keyword = keyword[5:]   
                    
                elif keyword.substring(5, 6).matches("[wy]"):
                    keyword = keyword[5:]            
                    
                elif keyword.substring(5, 6).matches("[rwylmn]") == None and keyword.substring(6, 8).matches("er") and keyword.substring(8, 9).matches("[aiueo]"):
                    keyword = keyword[5:]             

                elif keyword.substring(5, 6).matches("[rwylmn]") == None and keyword.substring(6, 8).matches("er") == None:
                    keyword = keyword[5:]             

                elif keyword.startswith("memper"):
                    if keyword.substring(6, 7).matches("[aiueo]") and contains("r" + keyword[6:]):
                        keyword = "r" + keyword[3:]
                        
                    elif keyword.substring(6, 7).matches("[aiueo]"):
                        keyword = keyword[6:]
                        
                    elif keyword.substring(6, 7).matches("[aiueor]") == None and keyword.substring(8, 10).matches("er") and keyword.substring(10, 11).matches("[aiueo]"):
                        keyword = keyword[3:]                  
             
                    elif keyword.substring(6, 7).matches("[aiueor]") == None and keyword.substring(8, 10).matches("er") == None:
                        keyword = keyword[6:]                  
             
                elif keyword.startswith("mempem"):
                    if keyword.substring(6, 7).matches("[bfv]") and contains(keyword[6:]):
                        keyword = keyword[6:]
                        
                    elif keyword.substring(6, 7).matches("[r]") and keyword.substring(7, 8).matches("[aiueo]") and contains("m" + keyword[6]):
                        keyword = "m" + keyword[6:]
                        
                    elif keyword.substring(6, 7).matches("[r]") and keyword.substring(7, 8).matches("[aiueo]") and contains("p" + keyword[6:]):
                        keyword = "p" + keyword[6:]

                    elif keyword.substring(6, 7).matches("[aiueo]") and contains("m" + keyword[6:]):
                        keyword = "m" + keyword[6:]
                        
                    elif keyword.substring(6, 7).matches("[aiueo]") and contains("p" + keyword[6:]):
                        keyword = "p" + keyword[6:]
                        
                elif keyword.startswith("mempeny"):
                    if contains("s" + keyword[7:]):
                        keyword = "s" + keyword[7:]

                elif keyword.startswith("mempeng"):
                    if keyword.substring(7, 8).matches("[ghq]"):
                        keyword = keyword[7:]
                    elif keyword.substring(7, 8).matches("[aiueo]"):
                        keyword = keyword[7:]
                    elif keyword.substring(7, 8).matches("[aiueo]") and contains("k" + keyword[7:]):
                        keyword = "k" + keyword[7:]
                        
                elif keyword.startswith("mempen"):
                    if keyword.substring(6, 7).matches("[aiueo]") and contains("t" + keyword[6:]):
                        keyword = "t" + keyword[6:]                  
                    elif keyword.substring(6, 7).matches("[aiueo]") and contains("n" + keyword[6:]):
                        keyword = "n" + keyword[6:]
                    elif keyword.substring(6, 7).matches("[jdcz]"):
                        keyword = keyword[6:]
                
            elif keyword.startswith("memp") and re.search("[aiueo]",keyword[5:6]) and contains("p"+keyword[4:]):
                keyword = "p"+keyword[4:]
            elif keyword.startswith("member") and len(keyword)>6:
                if re.search("[aiueor]",keyword[6:7]) == None and re.search("er",keyword[8:10]) and re.search("[aiueo]",keyword[10:11]):
                    keyword = keyword[6:] 
                
                elif re.search("[aiueor]",keyword[6:7]) == None and re.search("er",keyword[8:10] == None):
                    keyword = keyword[6:] 
                
                elif re.search("[aiueo]",keyword[6:7]) and contains("r"+keyword[6:]):
                    keyword = "r"+keyword[6:]
                
                elif re.search("[aiueo]",keyword[6:7]):
                    keyword = keyword[6:]
               
            elif keyword.startswith("mem"):
                
                if re.search("[bvf]",keyword[3:4]):
                    keyword = keyword[3:]
                
                elif re.search("[r]",keyword[3:4]) and re.search("[aiueo]",keyword[4:5]) and contains("m"+keyword[3:]): 
                    keyword = "m"+keyword[4:]
                
                elif re.search("[r]",keyword[3:4]) and re.search("[aiueo]",keyword[4:5]) and contains("p"+keyword[3:]): 
                    keyword = "p"+keyword[4:]
                
                elif re.search("[aiueo]",keyword[3:4]) and contains("m"+keyword[3:]): 
                    keyword = "m"+keyword[3:]
                
                elif re.search("[aiueo]",keyword[3:4]) and contains("p"+keyword[3:]): 
                    keyword = "p"+keyword[3:]
                
            elif keyword.startswith("menter"):
                
                if re.search("[aiueo]",keyword[6:7]) and contains("r"+keyword[5:]):
                    keyword = "r"+keyword[5:]
                
                elif re.search("[aiueo]",keyword[6:7]):
                    keyword = keyword[6:]
                
                elif re.search("[aiueor]",keyword[6:7]) == None and re.search("er",keyword[7:9]) and re.search("[aiueo]",keyword[9:10]):
                    keyword = keyword[6:]
                
                elif re.search("[aiueor]",keyword[6:7]) == None and re.search("er",keyword[6:8]) == None:
                    keyword = keyword[6:]
                
            elif keyword.startswith("men"):
                
                if re.search("[cdjz]",keyword[3:4]):
                    keyword = keyword[3:]
                
                elif re.search("[aiueo]",keyword[3:4]) and contains("n"+keyword[3:]):
                    keyword = "n"+keyword[3:]
                
                elif re.search("[aiueo]",keyword[3:4]) and contains("t"+keyword[3:]):
                    keyword = "t"+keyword[3:]
                
        elif keyword.startswith("te"):
            print("masuk 01")
            if keyword.startswith("ter"):    
                if re.search("[aiueo]",keyword[3:4]) and contains("r"+keyword[3:]):
                    keyword = "r"+keyword[3:]

                elif re.search("[aiueo]",keyword[3:4]):
                    keyword = keyword[3:]

                elif re.search("[aiueor]",keyword[3:4]) == None and re.search("er",keyword[3:5]) == None:
                    keyword = keyword[3:]

                elif len(keyword)>5:
                    if re.search("[aiueor]",keyword[3:4]) == None and re.search("er",keyword[4:6]):
                        keyword = keyword[3:]
                
            elif keyword.startswith("te"):
                if len(keyword)>2:
                    if re.search("[aiueor]",keyword[2:3]) == None and re.search("er",keyword[3:5]) and re.search("[aiueo]",keyword[5:6]) == None:
                            keyword = keyword[2:]
            
        elif keyword.startswith("be"):
            if keyword == "belajar":
                keyword = "ajar"
                
            elif re.search("[aiueolr]",keyword[2:3]) == None and re.search("er",keyword[3:5]) and  re.search("[aiueor]",keyword[5:6]) == None:
                keyword = keyword[2:]
                
            elif keyword.startswith("berke") and contains(keyword[5:]):
                keyword = keyword[5:]
                
            elif keyword.startswith("berke") and contains(keyword[3:]):
                keyword = keyword[3:]
                
            elif keyword.startswith("ber"):
                
                if re.search("[aiueo]",keyword[3:4]) and contains("r"+keyword[3:]):
                    keyword = "r"+keyword[3:]
                
                elif re.search("[aiueo]",keyword[3:4]):
                    keyword = keyword[3:]
                
                elif len(keyword)>6:
                    if re.search("[aiueor]",keyword[3:4]) == None and re.search("er",keyword[5:7]) == None:
                        keyword = keyword[3:]
                
                    elif re.search("[aiueor]",keyword[3:4]) == None and re.search("er",keyword[5:7]) and re.search("[aiueo]",keyword[7:8]):
                        keyword = keyword[3:]
                   
        elif keyword.startswith("pe"):
            
            if keyword == "pelajar":
                keyword = "ajar"
                
            elif re.search("[l]",keyword[2:3]) and re.search("[aiueo]",keyword[3:4]):
                keyword = keyword[2:]
                 
            elif re.search("[wy]",keyword[2:3]):
                keyword = keyword[2:]
                
            elif re.search("[rwylmn]",keyword[2:3]) == None and re.search("er",keyword[3:5]) and re.search("[aiueo]",keywordyword[5:6]):
                keyword = keyword[2:]
             
            elif re.search("[rwylmn]",keyword[2:3]) == None and re.search("er",keyword[3:5])== None:
                keyword = keyword[2:]
             
            elif keyword.startswith("per"):
                print("per")
                if re.search("[aiueo]",keyword[3:4]) and contains("r" + keyword[3:]):
                    keyword = "r" + keyword[3:]
                
                elif re.search("[aiueo]",keyword[3:4]):
                    keyword = keyword[3:]
                
                elif re.search("[aiueor]",keyword[3:4]) == None and re.search("er",keyword[5:7]) and re.search("[aiueo]",keyword[7:8]):
                    keyword = keyword[3:]
                    
                elif re.search("[aiueor]",keyword[3:4]) == None and re.search("er",keyword[5:7]) == None:
                    keyword = keyword[3:]
                
            elif keyword.startswith("pem"):
                
                if re.search("[bfv]",keyword[3:4]) and contains(keyword[3:]):
                    keyword = keyword[3:]
                    
                elif re.search("[r]",keyword[3:4]) and re.search("[aiueo]",keyword[4:5]) and contains("m" + keyword[3:]):
                    keyword = "m" + keyword[3:]
                                                                                                 
                elif re.search("[r]",keyword[3:4]) and re.search("[aiueo]",keyword[4:5]) and contains("p" + keyword[3:]):     
                    keyword = "p" + keyword[3:]
                                                                                                           
                elif re.search("[aiueo]",keyword[3:4]) and contains("m" + keyword[3:]):
                    keyword = "m" + keyword[3:]
                    
                elif re.search("[aiueo]",keyword[3:4]) and contains("p" + keyword[3:]):
                    keyword = "p" + keyword[3:]
                                                                                                      
                elif re.search("[aiueo]",keyword[6:7]) and contains("r" + keyword[6:]):
                    keyword = "r" + keyword[6:]
                
            elif keyword.startswith("peny"):
                if contains("s" + keyword[4:]):
                    keyword = "s" + keyword[4:]
                            
            elif keyword.startswith("peng"):
                            
                if re.search("[ghq]",keyword[4:5]):
                    keyword = keyword[4:]
                
                elif re.search("[aiueo]",keyword[4:5]):
                    keyword = keyword[4:]
                            
                elif re.search("[aiueo]",keyword[4:5]) and contains("k" + keyword[4:]):
                    keyword = "k" + keyword[4:]
                    
            elif keyword.startswith("penter") and len(keyword)>4:
                if re.search("[aiueo]",keyword[6:7]) and contains("r" + keyword[6:]):
                    keyword = "r" + keyword[6:]
                elif re.search("[aiueo]",keyword[6:7]):
                    keyword = keyword[6:]
                elif re.search("[aiueor]",keyword[6:7]) == None and re.search("er",keyword[6:7]) == None :
                    keyword = keyword[6:]
                elif len(keyword) > 8:
                    if re.search("[aiueor]",keyword[6:7]) == None and re.search("er",keyword[7:9]):
                        keyword = keyword[6:]
                        
            elif keyword.startswith("pente"):
                if len(keyword)>8:
                     if re.search("[aiueor]",keyword[5:6]) == None and re.search("er",keyword[6:8]) and re.search("[aiueo]",keyword[8:9]) == None:
                        keyword = keyword[5:] 
                        
            elif keyword.startswith("pen"):
                if re.search("[aiueo]",keyword[3:4]) and contains("t" + keyword[3:]):
                    keyword = "t" + keyword[3:]

                elif re.search("[aiueo]",keyword[3:4]) and contains("n" + keyword[3:]):
                    keyword = "n" + keyword[3:]
                    
                elif re.search("[jdcz]",keyword[3:4]):
                    keyword = keyword[3:]
                    
            elif keyword.startswith("pen"):
                
                if re.search("[aiueo]",keyword[3:4]) and contains("t" + keyword[3:]):
                    keyword = "t" + keyword[3:]
                
                elif re.search("[aiueo]",keyword[3:4]) and contains("n" + keyword[3:]):
                    keyword = "n" + keyword[3:]
                
                elif re.search("[jdcz]",keyword[3:4]):
                    keyword = keyword[3:]
            
    return keyword


# mengecek kata ada di korpus kata atau tidak
def contains(teks):
    database = "C:\\db\\korpus.db"
    conn = create_connection(database)
    cur = conn.cursor()
    cur.execute('Select kata from korpus_kata where kata = (?) and tag != "Entri tidak ditemukan" and tag != "tidak ditemukan kata yang dicari" and tag != "entri tidak ditemukan" ',(teks,))
    a = cur.fetchone()
    conn.commit()
    if a==None:
        return None
    else:
        return True


# In[25]:


def main(entries,root):
    sleng_num = 0  
    count_char = 0
    max_char = 0
#   gui input
    input_sen = entries['Input'].get('1.0', END)
    print(input_sen) 
#   memanggil fungsi cekWrong
    with conn:
        cekWrong(conn,input_sen)  
    senInput = input_sen.split()
    print(senInput)

    true = []
    false = []

    for sleng in wrong:
        max_char = 0
        print(wrong)
        kalimatTeks(sleng_num)
        input_sen = input_teks[0]

        token = input_sen.split()
        print(input_teks)
        print(token)
        kataslang.append(kata[wrong[sleng_num]])
        error = ' '.join(token[2:])
        if error not in bi_prob_dict:
            n=0
        else:
            n = len(bi_prob_dict[' '.join(token[2:])])
        print(n)
#       menampilkan semua hasil prediksi
        for i in range(n):
            prob_bigram_char = 1
#           memanggil bi_prob_dict
            pred = bi_prob_dict[' '.join(token[2:])][i]

#           jika kalimat prediksi mengandung kalimat singkatan
            if(contained(pred[1],kata[wrong[sleng_num]])==True):
                b = pred[1]
#               memecah menjadi karakter
                bigramchar = [b[j:j+2] for j in range(len(b)-1)]
                for x in range(len(bigramchar)):
#                   menghitung probabilitas
                    prob_bigram_char = prob_bigram_char * bi_prob_dict_char[bigramchar[x]]

                print(pred[1] ,"%.8f" % prob_bigram_char)
#               menghitung probabilitas terbesar
                if prob_bigram_char > max_char:
                    max_char = prob_bigram_char
                    true.clear()
                    true.append(pred[1])

                if(count_char > 5):
                    break
                count_char = count_char + 1
            list_pred.append(pred[1])

#       jika hasil prediksi tidak ada maka mengembalikan kata awal
        if not true:
            true.append(kata[wrong[sleng_num]])
            prediksi.append(kata[wrong[sleng_num]])
            f = kata[wrong[sleng_num]]


        senInput.pop(sleng)
#       mengganti kalimat awal menjadi kalimat prediksi
        senInput.insert(sleng,true[0])
        print(" ".join(senInput[:]))

        input_teks.clear()
        kata.pop(sleng)
        kata.insert(sleng,true[0])
        prediksi.append(true[0])

        sleng_num = sleng_num + 1
        true.clear()

    print(senInput[:])
    entries['Output'].delete('0.0', END)
    entries['Output'].insert('0.0', senInput[:])
    print(len(prediksi))
    print(prediksi)
    print(len(kata))
    print(kata)
    print(len(wrong))
    
    maketable(root,kataslang,prediksi)
    print("asu")
    kata.clear()
    wrong.clear()
    
    kataslang.clear()
    prediksi.clear()


# In[27]:


token = []
kata = []
index = []
wrong = []
input_teks = []
test = []
true = []
prediksi = []
kataslang = []
dataCsv = []
senInput = []
entries = {}
from tkinter import *
colomn_wrong = []
colomn_pred = []
colomn_prob = []
if __name__ == '__main__':
   
    vocab_dict = defaultdict(int)          #untuk membuat kalimat yang unik menjadi ada frekuensi   
    bi_dict = defaultdict(int)             #untuk menyimpan 2 kalimat 
    tri_dict = defaultdict(int)            #untuk menyimpan 3 kalimat 
    quad_dict = defaultdict(int)           #untuk menyimpan 4 kalimat 
    
    vocab_dict_char = defaultdict(int)          
    bi_dict_char = defaultdict(int)             
    tri_dict_char = defaultdict(int)           
    quad_dict_char = defaultdict(int)           
    
    quad_prob_dict = OrderedDict()              
    tri_prob_dict = OrderedDict()
    bi_prob_dict = OrderedDict()
    
    quad_prob_dict_char = OrderedDict()              
    tri_prob_dict_char = OrderedDict()
    bi_prob_dict_char = OrderedDict()
    
    
    list_pred = []
#   koneksi ke db
    database = "C:\\db\\korpus.db"
    conn = create_connection(database)
    
#   membuka korpus kecil
    train_file = 'Kecilkorpus.txt'

#   menjalankan fungsi loadcorpus dan loadcorpuschar
    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    token_len_char = loadCorpusChar(train_file, bi_dict_char, tri_dict_char, quad_dict_char, vocab_dict_char)


    k = 5
    V = len(vocab_dict)
    
    tri_nc_dict = findFrequencyOfFrequencyCount(tri_dict, k, 3, V, len(tri_dict))
    bi_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 2, V, len(bi_dict))
    uni_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 1, V, len(vocab_dict))
    
    tri_nc_dict_char = findFrequencyOfFrequencyCount(tri_dict_char, k, 3, V, len(tri_dict_char))
    bi_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 2, V, len(bi_dict_char))
    uni_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 1, V, len(vocab_dict_char))
    

#   membuat dic probabilitas bigram, trigram, charbigram, chartrigram
    findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, bi_nc_dict, k)
    findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, tri_nc_dict, k)
    
    findCharBigramProbGT(vocab_dict_char, bi_dict_char, bi_prob_dict_char, bi_nc_dict_char, k)
    findCharTrigramProbGT(vocab_dict_char, bi_dict_char, tri_dict_char, tri_prob_dict_char, tri_nc_dict_char, k)
    
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)

    
    entries = {}
    root = Tk()
    
    ents = makeform(root, 'Input')
    
    
    b1 = Button(root, text='Normalisasi', command=(lambda e=ents: main(e,root)))
    b1.pack(side=TOP, fill=X, padx=5, pady=5)
    
    ents = makeform(root, 'Output') 
    

    root.mainloop()
    
    


# # Bigram Testing

# In[254]:


def main():
    #variable declaration
    
    vocab_dict = defaultdict(int)          #for storing the different words with their frequencies    
    bi_dict = defaultdict(int)             #for keeping count of sentences of two words
    tri_dict = defaultdict(int)            #for keeping count of sentences of three words
    quad_dict = defaultdict(int)           #for keeping count of sentences of four words
    
    vocab_dict_char = defaultdict(int)          
    bi_dict_char = defaultdict(int)             
    tri_dict_char = defaultdict(int)           
    quad_dict_char = defaultdict(int)           
    
    quad_prob_dict = OrderedDict()              
    tri_prob_dict = OrderedDict()
    bi_prob_dict = OrderedDict()
    
    quad_prob_dict_char = OrderedDict()              
    tri_prob_dict_char = OrderedDict()
    bi_prob_dict_char = OrderedDict()
    
    
    list_pred = []
    database = "C:\\db\\korpus.db"
    conn = create_connection(database)
#     with conn:
#         cek(conn)
#     kalimat()
    
    #load the corpus for the dataset
    train_file = 'BesarKorpus.txt'
    #load corpus

    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    token_len_char = loadCorpusChar(train_file, bi_dict_char, tri_dict_char, quad_dict_char, vocab_dict_char)

    #create the different Nc dictionaries for ngrams
    #threshold value
    k = 5
    V = len(vocab_dict)
    
    tri_nc_dict = findFrequencyOfFrequencyCount(tri_dict, k, 3, V, len(tri_dict))
    bi_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 2, V, len(bi_dict))
    uni_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 1, V, len(vocab_dict))
    
    tri_nc_dict_char = findFrequencyOfFrequencyCount(tri_dict_char, k, 3, V, len(tri_dict_char))
    bi_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 2, V, len(bi_dict_char))
    uni_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 1, V, len(vocab_dict_char))
    

    #create bigram probability dictionary
    findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, bi_nc_dict, k)
    findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, tri_nc_dict, k)
    
    findCharBigramProbGT(vocab_dict_char, bi_dict_char, bi_prob_dict_char, bi_nc_dict_char, k)
    findCharTrigramProbGT(vocab_dict_char, bi_dict_char, tri_dict_char, tri_prob_dict_char, tri_nc_dict_char, k)
    
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)
    
    path_origin = "C:\\Users\\User\\Downloads\\testcase\\"
    for file in os.listdir(path_origin):
        file = file.split("\\")
        file_name = file[len(file)-1]

        dataCsv.clear()
        csvData(file_name)
        for data in dataCsv:
            sleng_num = 0  
            count_char = 0
            max_char = 0
            
            input_sen = data
            print(input_sen)  
        #         input_sen = data
            with conn:
                cekWrong(conn,input_sen)  
            senInput = input_sen.split()
            print(senInput)

            true = []
            false = []
            for sleng in wrong:
                max_char = 0
                print(wrong)
                kalimatTeks(sleng_num)
                input_sen = input_teks[0]

                token = input_sen.split()
                print(input_teks)
                print(token)
                print(kata[wrong[sleng_num]])
                error = ' '.join(token[2:])
                if error not in bi_prob_dict:
                    n=0
                else:
                    n = len(bi_prob_dict[' '.join(token[2:])])
                print(n)
                for i in range(n):
                    prob_bigram_char = 1
                    pred = bi_prob_dict[' '.join(token[2:])][i]
            #         print(pred[1])
                    if(contained(pred[1],kata[wrong[sleng_num]])==True):
                        b = pred[1]
                        bigramchar = [b[j:j+2] for j in range(len(b)-1)]
                        for x in range(len(bigramchar)):
                            prob_bigram_char = prob_bigram_char * bi_prob_dict_char[bigramchar[x]]

                        print(pred[1] ,"%.8f" % prob_bigram_char)
                        print(max_char)
                        if prob_bigram_char > max_char:
                            max_char = prob_bigram_char
                            true.clear()
                            true.append(pred[1])
                            benar = pred[1]
                        
                        if(count_char > 5):
                            break
                        count_char = count_char + 1
        #                 else :
        #                     print(pred[1] , "False")
                    list_pred.append(pred[1])

                if not true:
                    true.append(kata[wrong[sleng_num]])
                    f = kata[wrong[sleng_num]]
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(" ")
                    colomn_prob.append(" ")
                    
                else :
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(benar)
                    colomn_prob.append(format(max_char,'.9f'))

        #             bisa dijadikan fungsi    




                senInput.pop(sleng)
                senInput.insert(sleng,true[0])
        #         senInput[sleng] = true[0]
                print(" ".join(senInput[:]))

                input_teks.clear()
                kata.pop(sleng)
                kata.insert(sleng,true[0])

        #         input_teks.append(" ".join(senInput[:-1]))
                sleng_num = sleng_num + 1
                true.clear()

            print(senInput[:])
            kata.clear()
            wrong.clear()
            kalimatpred.append(" ".join(senInput[:]))
            
            


#         test = pd.DataFrame(colomn_wrong)
#         test.to_csv("C:\\Users\\User\\Downloads\\Test Singkatan\\KorpusKecilBigram%sWrong.csv"%(file_name), index = False, header = False)  
#         test1 = pd.DataFrame(colomn_prob)
#         test1.to_csv("C:\\Users\\User\\Downloads\\Test Singkatan\\KorpusKecilBigram%sProb.csv"%(file_name), index = False, header = False)  
#         test2 = pd.DataFrame(colomn_pred)
#         test2.to_csv("C:\\Users\\User\\Downloads\\Test Singkatan\\KorpusKecilBigram%sPred.csv"%(file_name), index = False, header = False)
        test2 = pd.DataFrame(kalimatpred)
        test2.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarKalimatBigram%sPred.csv"%(file_name), index = False, header = False)
        kalimatpred.clear()
#         print(colomn_wrong)
#         print(colomn_pred)
#         print(colomn_prob)
#         colomn_wrong.clear()
#         colomn_pred.clear()
#         colomn_prob.clear()


# In[301]:


token = []
kata = []
index = []
wrong = []
input_teks = []
test = []
true = []
dataCsv = []

kalimatpred = []

colomn_wrong = []
colomn_pred = []
colomn_prob = []
if __name__ == '__main__':
    main()


# # Trigram Testing

# In[256]:


def main():
    #variable declaration
    
    vocab_dict = defaultdict(int)          #for storing the different words with their frequencies    
    bi_dict = defaultdict(int)             #for keeping count of sentences of two words
    tri_dict = defaultdict(int)            #for keeping count of sentences of three words
    quad_dict = defaultdict(int)           #for keeping count of sentences of four words
    
    vocab_dict_char = defaultdict(int)          
    bi_dict_char = defaultdict(int)             
    tri_dict_char = defaultdict(int)           
    quad_dict_char = defaultdict(int)           
    
    quad_prob_dict = OrderedDict()              
    tri_prob_dict = OrderedDict()
    bi_prob_dict = OrderedDict()
    
    quad_prob_dict_char = OrderedDict()              
    tri_prob_dict_char = OrderedDict()
    bi_prob_dict_char = OrderedDict()
    
    
    list_pred = []
    database = "C:\\db\\korpus.db"
    conn = create_connection(database)
#     with conn:
#         cek(conn)
#     kalimat()
    
    #load the corpus for the dataset
    train_file = 'BesarKorpus.txt'
    #load corpus

    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    token_len_char = loadCorpusChar(train_file, bi_dict_char, tri_dict_char, quad_dict_char, vocab_dict_char)

    #create the different Nc dictionaries for ngrams
    #threshold value
    k = 5
    V = len(vocab_dict)
    
    tri_nc_dict = findFrequencyOfFrequencyCount(tri_dict, k, 3, V, len(tri_dict))
    bi_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 2, V, len(bi_dict))
    uni_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 1, V, len(vocab_dict))
    
    tri_nc_dict_char = findFrequencyOfFrequencyCount(tri_dict_char, k, 3, V, len(tri_dict_char))
    bi_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 2, V, len(bi_dict_char))
    uni_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 1, V, len(vocab_dict_char))
    

    #create bigram probability dictionary
    findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, bi_nc_dict, k)
    findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, tri_nc_dict, k)
    
    findCharBigramProbGT(vocab_dict_char, bi_dict_char, bi_prob_dict_char, bi_nc_dict_char, k)
    findCharTrigramProbGT(vocab_dict_char, bi_dict_char, tri_dict_char, tri_prob_dict_char, tri_nc_dict_char, k)
    
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)
    path_origin = "C:\\Users\\User\\Downloads\\testcase\\"
    for file in os.listdir(path_origin):
        file = file.split("\\")
        file_name = file[len(file)-1]

        dataCsv.clear()
        csvData(file_name)
        for data in dataCsv:
            sleng_num = 0  
            count_char = 0
            max_char = 0
            input_sen = data
            print(input_sen) 
            print("input_sen")
        #         input_sen = data
            with conn:
                cekWrong(conn,input_sen)  
            senInput = input_sen.split()
            print(senInput)
            print("senInput")

            true = []
            false = []
            for sleng in wrong:
                max_char = 0
                print(wrong)
                print("wrong")
                kalimatTeks(sleng_num)
                input_sen = input_teks[0]

                token = input_sen.split()
                print(input_teks)
                print("input_teks")
                print(token)
                print("token")
                print(kata[wrong[sleng_num]])
                print("kata wrong")
#                 Bigram
#                 error = ' '.join(token[2:])

#                 Tigram
                error = ' '.join(token[1:])
                print(error)
                print("error")
                if error not in tri_prob_dict:
                    n=0
                else:
                    n = len(tri_prob_dict[' '.join(token[1:])])
                print(n)
                print("panjang n")
                for i in range(n):
                    prob_trigram_char = 1
                    pred = tri_prob_dict[' '.join(token[1:])][i]
            #         print(pred[1])
                    if(contained(pred[1],kata[wrong[sleng_num]])==True):
                        b = pred[1]
                        print(b)
                        trigramchar = [b[j:j+3] for j in range(len(b)-2)]
                        for x in range(len(trigramchar)):
                            if trigramchar[x] not in tri_prob_dict_char:
                                prob_trigram_char = prob_trigram_char * tri_prob_dict_char[trigramchar[x]]
                            else:
                                prob_trigram_char = prob_trigram_char * tri_prob_dict_char[trigramchar[x]]

                        print(pred[1] ,"%.8f" % prob_trigram_char)
                        print("pred dan prob")
                        if prob_trigram_char > max_char:
                            max_char = prob_trigram_char
                            true.clear()
                            true.append(pred[1])
                            benar = pred[1]

                        if(count_char > 5):
                            break
                        count_char = count_char + 1
        #                 else :
        #                     print(pred[1] , "False")
                    list_pred.append(pred[1])

                if not true:
                    true.append(kata[wrong[sleng_num]])
                    f = kata[wrong[sleng_num]]
                    colomn_wrong.append(f)
                    colomn_pred.append(" ")
                    colomn_prob.append(" ")
                else :
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(benar)
                    colomn_prob.append(format(max_char,'.9f'))                  
                

        #             bisa dijadikan fungsi    




                senInput.pop(sleng)
                senInput.insert(sleng,true[0])
        #         senInput[sleng] = true[0]
                print(" ".join(senInput[:]))
                print("Gabungan")
            

                input_teks.clear()
                kata.pop(sleng)
                kata.insert(sleng,true[0])

        #         input_teks.append(" ".join(senInput[:-1]))
                sleng_num = sleng_num + 1
                true.clear()

            print(senInput[:])
            print("senInput")
            kata.clear()
            wrong.clear()
            kalimatpred.append(" ".join(senInput[:]))

        test2 = pd.DataFrame(kalimatpred)
        test2.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarKalimatTrigram%sPred.csv"%(file_name), index = False, header = False)
        kalimatpred.clear()
#         test = pd.DataFrame(colomn_wrong)
#         test.to_csv("C:\\Users\\User\\Downloads\\Test Salah\\KorpusBesarTrigram%sWrong.csv"%(file_name), index = False, header = False)  
#         test1 = pd.DataFrame(colomn_prob)
#         test1.to_csv("C:\\Users\\User\\Downloads\\Test salah\\KorpusBesarTrigram%sProb.csv"%(file_name), index = False, header = False)  
#         test2 = pd.DataFrame(colomn_pred)
#         test2.to_csv("C:\\Users\\User\\Downloads\\Test salah\\KorpusBesarTrigram%sPred.csv"%(file_name), index = False, header = False)  
#         print(colomn_wrong)
#         print(colomn_pred)
#         print(colomn_prob)
#         colomn_wrong.clear()
#         colomn_pred.clear()
#         colomn_prob.clear()


# # Ngram word Testing

# In[261]:


def main():
    #variable declaration
    
    vocab_dict = defaultdict(int)          #for storing the different words with their frequencies    
    bi_dict = defaultdict(int)             #for keeping count of sentences of two words
    tri_dict = defaultdict(int)            #for keeping count of sentences of three words
    quad_dict = defaultdict(int)           #for keeping count of sentences of four words
    
    vocab_dict_char = defaultdict(int)          
    bi_dict_char = defaultdict(int)             
    tri_dict_char = defaultdict(int)           
    quad_dict_char = defaultdict(int)           
    
    quad_prob_dict = OrderedDict()              
    tri_prob_dict = OrderedDict()
    bi_prob_dict = OrderedDict()
    
    quad_prob_dict_char = OrderedDict()              
    tri_prob_dict_char = OrderedDict()
    bi_prob_dict_char = OrderedDict()
    
    
    list_pred = []
    database = "C:\\db\\korpus.db"
    conn = create_connection(database)
#     with conn:
#         cek(conn)
#     kalimat()
    
    #load the corpus for the dataset
    train_file = 'BesarKorpus.txt'
    #load corpus

    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    token_len_char = loadCorpusChar(train_file, bi_dict_char, tri_dict_char, quad_dict_char, vocab_dict_char)

    #create the different Nc dictionaries for ngrams
    #threshold value
    k = 5
    V = len(vocab_dict)
    
    tri_nc_dict = findFrequencyOfFrequencyCount(tri_dict, k, 3, V, len(tri_dict))
    bi_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 2, V, len(bi_dict))
    uni_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 1, V, len(vocab_dict))
    
    tri_nc_dict_char = findFrequencyOfFrequencyCount(tri_dict_char, k, 3, V, len(tri_dict_char))
    bi_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 2, V, len(bi_dict_char))
    uni_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 1, V, len(vocab_dict_char))
    

    #create bigram probability dictionary
    findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, bi_nc_dict, k)
    findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, tri_nc_dict, k)
    
    findCharBigramProbGT(vocab_dict_char, bi_dict_char, bi_prob_dict_char, bi_nc_dict_char, k)
    findCharTrigramProbGT(vocab_dict_char, bi_dict_char, tri_dict_char, tri_prob_dict_char, tri_nc_dict_char, k)
    
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)
    
    path_origin = "C:\\Users\\User\\Downloads\\testcase\\"
    for file in os.listdir(path_origin):
        file = file.split("\\")
        file_name = file[len(file)-1]

        dataCsv.clear()
        csvData(file_name)
        for data in dataCsv:
            sleng_num = 0  
            count_char = 0
            max_char = 0
            
            input_sen = data
            print(input_sen)  
        #         input_sen = data
            with conn:
                cekWrong(conn,input_sen)  
            senInput = input_sen.split()
            print(senInput)

            true = []
            false = []
            for sleng in wrong:
                max_char = 0
                print(wrong)
                kalimatTeks(sleng_num)
                input_sen = input_teks[0]

                token = input_sen.split()
                print(input_teks)
                print(token)
                print(kata[wrong[sleng_num]])
                error = ' '.join(token[2:])
                if error not in bi_prob_dict:
                    n=0
                else:
                    n = len(bi_prob_dict[' '.join(token[2:])])
                print(n)
                for i in range(n):
                    prob_bigram_char = 1
                    pred = bi_prob_dict[' '.join(token[2:])][i]
            #         print(pred[1])
                    if(contained(pred[1],kata[wrong[sleng_num]])==True):
                        b = pred[1]

                        print(pred[1] ,"%.8f" % pred[0])
                        print(max_char)
                        if prob_bigram_char > max_char:
                            max_char = prob_bigram_char
                            true.clear()
                            true.append(pred[1])
                            benar = pred[1]
                        
                        if(count_char > 5):
                            break
                        count_char = count_char + 1
        #                 else :
        #                     print(pred[1] , "False")
                    list_pred.append(pred[1])

                if not true:
                    true.append(kata[wrong[sleng_num]])
                    f = kata[wrong[sleng_num]]
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(" ")
                    colomn_prob.append(" ")
                    
                else :
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(benar)
                    colomn_prob.append(format(max_char,'.9f'))

        #             bisa dijadikan fungsi    

                senInput.pop(sleng)
                senInput.insert(sleng,true[0])
        #         senInput[sleng] = true[0]
                print(" ".join(senInput[:]))

                input_teks.clear()
                kata.pop(sleng)
                kata.insert(sleng,true[0])

        #         input_teks.append(" ".join(senInput[:-1]))
                sleng_num = sleng_num + 1
                true.clear()

            print(senInput[:])
            kata.clear()
            wrong.clear()
            kalimatpred.append(" ".join(senInput[:]))
            
            


        test = pd.DataFrame(colomn_wrong)
        test.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarNgramWord%sWrong.csv"%(file_name), index = False, header = False)  
        test1 = pd.DataFrame(colomn_prob)
        test1.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarNgramWord%sProb.csv"%(file_name), index = False, header = False)  
        test2 = pd.DataFrame(colomn_pred)
        test2.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarNgramWord%sPred.csv"%(file_name), index = False, header = False)

        test3 = pd.DataFrame(kalimatpred)
        test3.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarKalimatNgramWord%sPred.csv"%(file_name), index = False, header = False)
        
        kalimatpred.clear()

#         print(colomn_wrong)
#         print(colomn_pred)
#         print(colomn_prob)
        colomn_wrong.clear()
        colomn_pred.clear()
        colomn_prob.clear()


# # Trigram Word Testing

# In[268]:


def main():
    #variable declaration
    
    vocab_dict = defaultdict(int)          #for storing the different words with their frequencies    
    bi_dict = defaultdict(int)             #for keeping count of sentences of two words
    tri_dict = defaultdict(int)            #for keeping count of sentences of three words
    quad_dict = defaultdict(int)           #for keeping count of sentences of four words
    
    vocab_dict_char = defaultdict(int)          
    bi_dict_char = defaultdict(int)             
    tri_dict_char = defaultdict(int)           
    quad_dict_char = defaultdict(int)           
    
    quad_prob_dict = OrderedDict()              
    tri_prob_dict = OrderedDict()
    bi_prob_dict = OrderedDict()
    
    quad_prob_dict_char = OrderedDict()              
    tri_prob_dict_char = OrderedDict()
    bi_prob_dict_char = OrderedDict()
    
    
    list_pred = []
    database = "C:\\db\\korpus.db"
    conn = create_connection(database)
#     with conn:
#         cek(conn)
#     kalimat()
    
    #load the corpus for the dataset
    train_file = 'BesarKorpus.txt'
    #load corpus

    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    token_len_char = loadCorpusChar(train_file, bi_dict_char, tri_dict_char, quad_dict_char, vocab_dict_char)

    #create the different Nc dictionaries for ngrams
    #threshold value
    k = 5
    V = len(vocab_dict)
    
    tri_nc_dict = findFrequencyOfFrequencyCount(tri_dict, k, 3, V, len(tri_dict))
    bi_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 2, V, len(bi_dict))
    uni_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 1, V, len(vocab_dict))
    
    tri_nc_dict_char = findFrequencyOfFrequencyCount(tri_dict_char, k, 3, V, len(tri_dict_char))
    bi_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 2, V, len(bi_dict_char))
    uni_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 1, V, len(vocab_dict_char))
    

    #create bigram probability dictionary
    findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, bi_nc_dict, k)
    findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, tri_nc_dict, k)
    
    findCharBigramProbGT(vocab_dict_char, bi_dict_char, bi_prob_dict_char, bi_nc_dict_char, k)
    findCharTrigramProbGT(vocab_dict_char, bi_dict_char, tri_dict_char, tri_prob_dict_char, tri_nc_dict_char, k)
    
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)
    path_origin = "C:\\Users\\User\\Downloads\\testcase\\"
    for file in os.listdir(path_origin):
        file = file.split("\\")
        file_name = file[len(file)-1]

        dataCsv.clear()
        csvData(file_name)
        for data in dataCsv:
            sleng_num = 0  
            count_char = 0
            max_char = 0
            input_sen = data
            print(input_sen) 
            print("input_sen")
        #         input_sen = data
            with conn:
                cekWrong(conn,input_sen)  
            senInput = input_sen.split()
            print(senInput)
            print("senInput")

            true = []
            false = []
            for sleng in wrong:
                max_char = 0
                print(wrong)
                print("wrong")
                kalimatTeks(sleng_num)
                input_sen = input_teks[0]

                token = input_sen.split()
                print(input_teks)
                print("input_teks")
                print(token)
                print("token")
                print(kata[wrong[sleng_num]])
                print("kata wrong")
#                 Bigram
#                 error = ' '.join(token[2:])

#                 Tigram
                error = ' '.join(token[1:])
                print(error)
                print("error")
                if error not in tri_prob_dict:
                    n=0
                else:
                    n = len(tri_prob_dict[' '.join(token[1:])])
                print(n)
                print("panjang n")
                for i in range(n):
                    pred = tri_prob_dict[' '.join(token[1:])][i]
                    prob_trigram_char = pred[0]
            #         print(pred[1])
                    if(contained(pred[1],kata[wrong[sleng_num]])==True):
                        b = pred[1]
                        print(b)

                        print(pred[1] ,"%.8f" % pred[0])
                        print("pred dan prob")
                        if prob_trigram_char > max_char:
                            max_char = prob_trigram_char
                            true.clear()
                            true.append(pred[1])
                            benar = pred[1]

                        if(count_char > 5):
                            break
                        count_char = count_char + 1
        #                 else :
        #                     print(pred[1] , "False")
                    list_pred.append(pred[1])

                if not true:
                    true.append(kata[wrong[sleng_num]])
                    f = kata[wrong[sleng_num]]
                    colomn_wrong.append(f)
                    colomn_pred.append(" ")
                    colomn_prob.append(" ")
                else :
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(benar)
                    colomn_prob.append(format(max_char,'.9f'))                  
                

        #             bisa dijadikan fungsi    




                senInput.pop(sleng)
                senInput.insert(sleng,true[0])
        #         senInput[sleng] = true[0]
                print(" ".join(senInput[:]))
                print("Gabungan")
            

                input_teks.clear()
                kata.pop(sleng)
                kata.insert(sleng,true[0])

        #         input_teks.append(" ".join(senInput[:-1]))
                sleng_num = sleng_num + 1
                true.clear()

            print(senInput[:])
            print("senInput")
            kata.clear()
            wrong.clear()
            kalimatpred.append(" ".join(senInput[:]))



        test1 = pd.DataFrame(colomn_prob)
        test1.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarTrigram%sProb.csv"%(file_name), index = False, header = False)  
        test2 = pd.DataFrame(colomn_pred)
        test2.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarTrigram%sPred.csv"%(file_name), index = False, header = False)  
        test3 = pd.DataFrame(kalimatpred)
        test3.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarKalimatTrigramNgramWord%sPred.csv"%(file_name), index = False, header = False)
        kalimatpred.clear()
#         print(colomn_wrong)
#         print(colomn_pred)
#         print(colomn_prob)
        colomn_wrong.clear()
        colomn_pred.clear()
        colomn_prob.clear()


# # N gram no Script Testing

# In[300]:


def main():
    #variable declaration
    
    vocab_dict = defaultdict(int)          #for storing the different words with their frequencies    
    bi_dict = defaultdict(int)             #for keeping count of sentences of two words
    tri_dict = defaultdict(int)            #for keeping count of sentences of three words
    quad_dict = defaultdict(int)           #for keeping count of sentences of four words
    
    vocab_dict_char = defaultdict(int)          
    bi_dict_char = defaultdict(int)             
    tri_dict_char = defaultdict(int)           
    quad_dict_char = defaultdict(int)           
    
    quad_prob_dict = OrderedDict()              
    tri_prob_dict = OrderedDict()
    bi_prob_dict = OrderedDict()
    
    quad_prob_dict_char = OrderedDict()              
    tri_prob_dict_char = OrderedDict()
    bi_prob_dict_char = OrderedDict()
    
    
    list_pred = []
    database = "C:\\db\\korpus.db"
    conn = create_connection(database)
#     with conn:
#         cek(conn)
#     kalimat()
    
    #load the corpus for the dataset
    train_file = 'BesarKorpus.txt'
    #load corpus

    token_len = loadCorpus(train_file, bi_dict, tri_dict, quad_dict, vocab_dict)
    token_len_char = loadCorpusChar(train_file, bi_dict_char, tri_dict_char, quad_dict_char, vocab_dict_char)

    #create the different Nc dictionaries for ngrams
    #threshold value
    k = 5
    V = len(vocab_dict)
    
    tri_nc_dict = findFrequencyOfFrequencyCount(tri_dict, k, 3, V, len(tri_dict))
    bi_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 2, V, len(bi_dict))
    uni_nc_dict = findFrequencyOfFrequencyCount(bi_dict, k, 1, V, len(vocab_dict))
    
    tri_nc_dict_char = findFrequencyOfFrequencyCount(tri_dict_char, k, 3, V, len(tri_dict_char))
    bi_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 2, V, len(bi_dict_char))
    uni_nc_dict_char = findFrequencyOfFrequencyCount(bi_dict_char, k, 1, V, len(vocab_dict_char))
    

    #create bigram probability dictionary
    findBigramProbGT(vocab_dict, bi_dict, bi_prob_dict, bi_nc_dict, k)
    findTrigramProbGT(vocab_dict, bi_dict, tri_dict, tri_prob_dict, tri_nc_dict, k)
    
    findCharBigramProbGT(vocab_dict_char, bi_dict_char, bi_prob_dict_char, bi_nc_dict_char, k)
    findCharTrigramProbGT(vocab_dict_char, bi_dict_char, tri_dict_char, tri_prob_dict_char, tri_nc_dict_char, k)
    
    sortProbWordDict(bi_prob_dict, tri_prob_dict, quad_prob_dict)
    
    path_origin = "C:\\Users\\User\\Downloads\\testcase\\"
    for file in os.listdir(path_origin):
        file = file.split("\\")
        file_name = file[len(file)-1]

        dataCsv.clear()
        csvData(file_name)
        for data in dataCsv:
            sleng_num = 0  
            count_char = 0
            max_char = 0
            
            input_sen = data
            print(input_sen)  
        #         input_sen = data
            with conn:
                cekWrong(conn,input_sen)  
            senInput = input_sen.split()
            print(senInput)

            true = []
            false = []
            for sleng in wrong:
                max_char = 0
                print(wrong)
                kalimatTeks(sleng_num)
                input_sen = input_teks[0]

                token = input_sen.split()
                print(input_teks)
                print(token)
                print(kata[wrong[sleng_num]])
                error = ' '.join(token[2:])
                if error not in bi_prob_dict:
                    n=0
                else:
                    n = len(bi_prob_dict[' '.join(token[2:])])
                print(n)
                for i in range(n):
                    prob_bigram_char = 1
                    pred = bi_prob_dict[' '.join(token[2:])][i]
            #         print(pred[1])
                    if(i < 10):
                        b = pred[1]
                        prob_bigram_char = pred[0]
                        print(pred[1] ,"%.8f" % pred[0])
                        print(max_char)
                        if prob_bigram_char > max_char:
                            max_char = prob_bigram_char
                            true.clear()
                            true.append(pred[1])
                            benar = pred[1]
                        
                        if(count_char > 5):
                            break
                        count_char = count_char + 1
        #                 else :
        #                     print(pred[1] , "False")
                    list_pred.append(pred[1])

                if not true:
                    true.append(kata[wrong[sleng_num]])
                    f = kata[wrong[sleng_num]]
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(" ")
                    colomn_prob.append(" ")
                    
                else :
                    colomn_wrong.append(kata[wrong[sleng_num]])
                    colomn_pred.append(benar)
                    colomn_prob.append(format(max_char,'.9f'))

        #             bisa dijadikan fungsi    

                senInput.pop(sleng)
                senInput.insert(sleng,true[0])
        #         senInput[sleng] = true[0]
                print(" ".join(senInput[:]))

                input_teks.clear()
                kata.pop(sleng)
                kata.insert(sleng,true[0])

        #         input_teks.append(" ".join(senInput[:-1]))
                sleng_num = sleng_num + 1
                true.clear()

            print(senInput[:])
            kata.clear()
            wrong.clear()
            kalimatpred.append(" ".join(senInput[:]))
            
            


        test = pd.DataFrame(colomn_wrong)
        test.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarBigramNOSCRIPTNgramWord%sWrong.csv"%(file_name), index = False, header = False)  
        test1 = pd.DataFrame(colomn_prob)
        test1.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarBigramNOSCRIPTNgramWord%sProb.csv"%(file_name), index = False, header = False)  
        test2 = pd.DataFrame(colomn_pred)
        test2.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarBigramNOSCRIPTNgramWord%sPred.csv"%(file_name), index = False, header = False)

        test3 = pd.DataFrame(kalimatpred)
        test3.to_csv("C:\\Users\\User\\Downloads\\KorpusBesarBigramNOSCRIPTKalimatNgramWord%sPred.csv"%(file_name), index = False, header = False)
        
        kalimatpred.clear()

#         print(colomn_wrong)
#         print(colomn_pred)
#         print(colomn_prob)
        colomn_wrong.clear()
        colomn_pred.clear()
        colomn_prob.clear()

