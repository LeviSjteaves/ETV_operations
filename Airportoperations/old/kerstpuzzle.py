# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 09:43:24 2023

@author: levi1
"""
import string
import gurobipy as grp
from gurobipy import GRB

l= list(string.ascii_lowercase)
l.insert(0, 'NOT')

l[0]='0'
l[1]='q'
l[2]='b'
l[3]='c'
l[4]='s'#
l[5]='y'
l[6]='i'
l[7]='g'
l[8]='p'
l[9]='z'
l[10]='a'#
l[11]='h'
l[12]='n'
l[13]='k'#
l[14]='j'
l[15]='l'#
l[16]='e'#
l[17]='v'
l[18]='f'
l[19]='d'
l[20]='m'#
l[21]='o'#
l[22]='t'#
l[23]='u'
l[24]='w'#
l[25]='x'
l[26]='r'#


model = grp.Model("Kerstpuzzle20")

PEN = {}  # State of charge before every task

for a in range(len(l)):
    PEN[a] = model.addVar(lb=0, vtype=GRB.INTEGER, name=f"PEN_{a}")

model.setObjective(grp.quicksum(PEN), sense=GRB.MAXIMIZE)

model.addConstr()   


#pink
p1 = (f'{l[7]}{l[10]}{l[10]}{l[22]}{l[26]}{l[21]}')
p2 = (f'{l[20]}{l[16]}{l[22]}')
#OR
p3 =(f'{l[7]}{l[10]}{l[10]}{l[22]}')
p4 =(f'{l[20]}{l[16]}{l[22]}{l[26]}{l[21]}')

#orange
o1 = (f'{l[4]}{l[22]}{l[10]}{l[10]}{l[18]}')#sure
o2 = (f'{l[24]}{l[16]}{l[15]}{l[13]}{l[16]}')#sure

#green
g1 = (f'{l[17]}{l[16]}{l[16]}')#sure

g2 = (f'{l[26]}{l[21]}{l[10]}{l[5]}')
g3 = (f'{l[4]}{l[10]}{l[15]}{l[21]}')
#OR
g4 = (f'{l[4]}{l[10]}{l[5]}')
g5 = (f'{l[26]}{l[21]}{l[10]}{l[15]}{l[21]}')


#Lblue
lb1 = (f'{l[23]}{l[14]}{l[17]}{l[5]}{l[15]}{l[12]}{l[7]}{l[19]}{l[21]}')
lb2 = (f'{l[25]}{l[15]}{l[16]}{l[6]}{l[16]}')
#OR
lb3 = (f'{l[23]}{l[14]}{l[17]}{l[5]}{l[15]}{l[16]}{l[6]}{l[16]}')
lb4 = (f'{l[25]}{l[15]}{l[12]}{l[7]}{l[19]}{l[21]}')


#Dblue
db1 = (f'{l[17]}{l[16]}{l[21]}{l[12]}{l[20]}{l[16]}{l[9]}{l[16]}')

#red
r1 = (f'{l[17]}{l[14]}{l[16]}{l[12]}{l[7]}{l[13]}{l[11]}{l[16]}{l[12]}{l[15]}{l[12]}{l[20]}')
r2 = (f'{l[17]}{l[14]}{l[16]}{l[12]}{l[15]}{l[12]}{l[7]}{l[13]}{l[11]}{l[16]}{l[12]}{l[20]}')

#yellow
y1 = (f'{l[2]}{l[3]}{l[10]}{l[10]}{l[17]}')
y2 = (f'{l[16]}{l[16]}{l[8]}{l[9]}{l[10]}{l[10]}{l[17]}')
y3 = (f'{l[15]}{l[16]}{l[21]}{l[21]}{l[16]}{l[22]}')