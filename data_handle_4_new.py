import pickle
import sys
import re
import torch
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import numpy as np
def preprocess(path):
    passwd = []
    exp = re.compile(r'[^\x00-\x7f]')
    num = 0
    try:
        with open(path, 'r', encoding='ISO-8859-1') as wordlist:
            for line in wordlist:  ###弱口令添加
                wl = line.strip()

                passwd.extend([wl])
                if len(passwd) == 1800000:
                    break
    except FileNotFoundError:
        print("The password file does not exist", file=sys.stderr)
    return passwd
dict1={
  "!": 1,
  "\"": 2,
  "#": 3,
  "$": 4,
  "%": 5,
  "&": 6,
  "'": 7,
  "(": 8,
  ")": 9,
  "*": 10,
  "+": 11,
  ",": 12,
  "-": 13,
  ".": 14,
  "/": 15,
  "0": 16,
  "1": 17,
  "2": 18,
  "3": 19,
  "4": 20,
  "5": 21,
  "6": 22,
  "7": 23,
  "8": 24,
  "9": 25,
  ":": 26,
  ";": 27,
  "<": 28,
  "=": 29,
  ">": 30,
  "?": 31,
  "@": 32,
  "A": 33,
  "B": 34,
  "C": 35,
  "D": 36,
  "E": 37,
  "F": 38,
  "G": 39,
  "H": 40,
  "I": 41,
  "J": 42,
  "K": 43,
  "L": 44,
  "M": 45,
  "N": 46,
  "O": 47,
  "P": 48,
  "Q": 49,
  "R": 50,
  "S": 51,
  "T": 52,
  "U": 53,
  "V": 54,
  "W": 55,
  "X": 56,
  "Y": 57,
  "Z": 58,
  "[": 59,
  "\\": 60,
  "]": 61,
  "^": 62,
  "_": 63,
  "`": 64,
  "a": 65,
  "b": 66,
  "c": 67,
  "d": 68,
  "e": 69,
  "f": 70,
  "g": 71,
  "h": 72,
  "i": 73,
  "j": 74,
  "k": 75,
  "l": 76,
  "m": 77,
  "n": 78,
  "o": 79,
  "p": 80,
  "q": 81,
  "r": 82,
  "s": 83,
  "t": 84,
  "u": 85,
  "v": 86,
  "w": 87,
  "x": 88,
  "y": 89,
  "z": 90,
  "{": 91,
  "|": 92,
  "}": 93,
  "~": 94,
}
dict_printable={
  "START":0,
  "!": 1,
  "\"": 2,
  "#": 3,
  "$": 4,
  "%": 5,
  "&": 6,
  "'": 7,
  "(": 8,
  ")": 9,
  "*": 10,
  "+": 11,
  ",": 12,
  "-": 13,
  ".": 14,
  "/": 15,
  "0": 16,
  "1": 17,
  "2": 18,
  "3": 19,
  "4": 20,
  "5": 21,
  "6": 22,
  "7": 23,
  "8": 24,
  "9": 25,
  ":": 26,
  ";": 27,
  "<": 28,
  "=": 29,
  ">": 30,
  "?": 31,
  "@": 32,
  "A": 33,
  "B": 34,
  "C": 35,
  "D": 36,
  "E": 37,
  "F": 38,
  "G": 39,
  "H": 40,
  "I": 41,
  "J": 42,
  "K": 43,
  "L": 44,
  "M": 45,
  "N": 46,
  "O": 47,
  "P": 48,
  "Q": 49,
  "R": 50,
  "S": 51,
  "T": 52,
  "U": 53,
  "V": 54,
  "W": 55,
  "X": 56,
  "Y": 57,
  "Z": 58,
  "[": 59,
  "\\": 60,
  "]": 61,
  "^": 62,
  "_": 63,
  "`": 64,
  "a": 65,
  "b": 66,
  "c": 67,
  "d": 68,
  "e": 69,
  "f": 70,
  "g": 71,
  "h": 72,
  "i": 73,
  "j": 74,
  "k": 75,
  "l": 76,
  "m": 77,
  "n": 78,
  "o": 79,
  "p": 80,
  "q": 81,
  "r": 82,
  "s": 83,
  "t": 84,
  "u": 85,
  "v": 86,
  "w": 87,
  "x": 88,
  "y": 89,
  "z": 90,
  "{": 91,
  "|": 92,
  "}": 93,
  "~": 94,
  "END": 95
}
def encode(password):
    dict2 = set()
    dict3 = {}
    for i in password:
        for j in i:
            dict2.add(j)
    num = 0
    for i in dict2:
        dict3[i] = num
        num += 1
    dict3['END'] = num
    vacabulary = set()
    for i in dict1.keys():
        vacabulary.add(i)
    vacabulary.add("START")
    # vacabulary.add("END")
    # print(len(vacabulary))
    vacabulary = list(vacabulary)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(vacabulary)
    encoded = to_categorical(integer_encoded)
    dict_onehot = dict(zip(vacabulary, encoded))
    return dict3,dict_onehot


special_string = r"""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
password = preprocess('data/trainword_craftrise.txt')
# print(password)
# exit()
dict3,dict_onehot=encode(password)
file=open('dict3_craftrise_new_4_new_180w.txt','w')
file.write(str(dict3))
graph_list=[]
for i in password:
    for j in range(1, len(i) + 2):
        dict1={}
        list1 = []
        list2 = []
        list3 = []
        for m in range(0, j):
            list1.append(m)
            list2.append(m + 1)
        edge_start, edge_end = torch.tensor(list1), torch.tensor(list2)
        dict1['edge_start']=edge_start
        dict1['edge_end']=edge_end
        if j==1:
            dict1['edge_weight']=[1]
        elif j==2:
            dict1['edge_weight'] = [1,1]
        else:
            list1=[]
            list1.append(1)
            list1.append(1)
            for index in range(2,len(edge_start)):
                if i[edge_start[index-2]].isalpha() and i[edge_end[index-2]].isalpha():
                    list1.append()
                elif i[edge_start[index-2]].isdigit() and i[edge_end[index-2]].isdigit():
                    list1.append()
                elif i[edge_start[index-2]] in special_string and i[edge_end[index-2]] in special_string:
                    list1.append()
                elif (i[edge_start[index-2]].isalpha() and i[edge_end[index-2]].isdigit()) or (i[edge_start[index-2]].isdigit() and i[edge_end[index-2]].isalpha()):
                    list1.append()
                elif (i[edge_start[index-2]].isalpha() and i[edge_end[index-2]] in special_string) or (i[edge_start[index-2]] in special_string and i[edge_end[index-2]].isalpha()):
                    list1.append()
                elif (i[edge_start[index-2]].isdigit() and i[edge_end[index-2]] in special_string) or (i[edge_start[index-2]] in special_string and i[edge_end[index-2]].isdigit()):
                    list1.append()
            dict1['edge_weight']=list1
        if j == 1:
            list3.append("START")
            list3.append("START")
        else:
            list3.append("START")
            list3.append("START")
            for m1 in range(0, j - 1):
                list3.append(i[m1])
        dict1['node_label']=list3
        if j == len(i) + 1:
            label = torch.tensor(dict3["END"], dtype=torch.int32)
        else:
            label = torch.tensor(dict3[i[j - 1]], dtype=torch.int32)
        dict1['graph_label']=label
        dict1['number_nodes']=j+1
        dict1['number_edge']=len(edge_start)
        graph_list.append(dict1)

with open('data/craftrise_graph_new_4_new_180w.pkl','wb') as fo:
    pickle.dump(graph_list,fo)
    fo.close()
