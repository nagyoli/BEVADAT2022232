#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
from typing import Tuple


# In[ ]:


#FONTOS!!!

# CSAK OTT LEHET HASZNÁLNI FOR LOOP-OT AHOL A FELADAT KÜLÖN KÉRI!
# [1,2,3,4] --> ezek az értékek np.array-ek. Ahol listát kérek paraméterként ott külön ki fogom emelni!
# Ha végeztél a feladatokkal, akkor notebook-ot alakítsd át .py.
# A FÁJLBAN CSAK A FÜGGVÉNYEK LEGYENEK! (KOMMENTEK MARADHATNAK)


# In[ ]:


#Készíts egy függvényt ami létre hoz egy nullákkal teli numpy array-t.
#Paraméterei: mérete (tuple-ként), default mérete pedig legyen egy (2,2)
#Be: (2,2)
#Ki: [[0,0],[0,0]]
#create_array()


# In[ ]:


def create_array(size: Tuple)-> np.ndarray:
    return np.zeros(size)


# In[ ]:


#Készíts egy függvényt ami a paraméterként kapott array-t főátlóját feltölti egyesekkel
#Be: [[1,2],[3,4]]
#Ki: [[1,2],[3,1]]
#set_one()


# In[ ]:


def set_one(input_array: np.ndarray) -> np.ndarray:
    np.fill_diagonal(input_array, 1)
    return input_array


# In[ ]:


# Készíts egy függvényt ami transzponálja a paraméterül kapott mártix-ot:
# Be: [[1, 2], [3, 4]]
# Ki: [[1, 3], [2, 4]]
# do_transpose()


# In[ ]:


def do_transpose(input: np.ndarray) -> np.ndarray:
    return np.transpose(input)


# In[ ]:


# Készíts egy olyan függvényt ami az array-ben lévő értékeket N tizenedjegyik kerekíti, ha nincs megadva ez a paraméter, akkor legyen az alapértelmezett a kettő 
# Be: [0.1223, 0.1675], 2
# Ki: [0.12, 0.17]
# round_array()


# In[ ]:


def round_array(input: np.ndarray, decimal: int) -> np.ndarray:
    return np.round(input, decimal)


# In[ ]:


# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 0 - False-ra, az 1 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ True False False], [ True  True  True], [False False False]]
# bool_array()


# In[ ]:


def bool_array(input: np.ndarray) -> np.ndarray:
    return input.astype(bool)


# In[ ]:


# Készíts egy olyan függvényt, ami a bementként kapott 0 és 1 ből álló tömben a 1 - False-ra az 0 True-ra cserélni
# Be: [[1, 0, 0], [1, 1, 1],[0, 0, 0]]
# Ki: [[ False True True], [ False  False  False], [True True True]]
# invert_bool_array()


# In[ ]:


def invert_bool_array(input: np.ndarray) -> np.ndarray:
    return np.invert(input.astype(bool))


# In[ ]:


# Készíts egy olyan függvényt ami a paraméterként kapott array-t kilapítja
# Be: [[1,2], [3,4]]
# Ki: [1,2,3,4]
# flatten()


# In[ ]:


def flatten(input: np.ndarray)-> np.ndarray:
    return input.reshape(-1)

