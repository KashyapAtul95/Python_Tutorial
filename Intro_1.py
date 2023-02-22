# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from os import chdir, getcwd
wd=getcwd()
chdir(wd)

players = ("Sachin", "Sehwag", "Gambhir", "Dravid", "Raina")
scores = (100, 15, 17, 28, 43)
type(players)
zip(players,scores)
x = zip(scores, players)
print(tuple(x))

#Reading the example code

import numpy as np

a = np.array([1, 2, 3, 4, 5, 6])
print(a)
type(a)

a1 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
print(a1)
