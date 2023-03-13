#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import datetime
import numpy as np
from typing import List, Any


# In[ ]:


#FONTOS!!!

# CSAK OTT LEHET HASZNÁLNI FOR LOOP-OT AHOL A FELADAT KÜLÖN KÉRI!
# [1,2,3,4] --> ezek az értékek np.array-ek. Ahol listát kérek paraméterként ott külön ki fogom emelni!
# Ha végeztél a feladatokkal, akkor notebook-ot alakítsd át .py.
# A FÁJLBAN CSAK A FÜGGVÉNYEK LEGYENEK! (KOMMENTEK MARADHATNAK)


# In[ ]:


# Írj egy olyan fügvényt, ami megfordítja egy 2d array oszlopait. Bemenetként egy array-t vár.
# Be: [[1,2],[3,4]]
# Ki: [[2,1],[4,3]]
# column_swap()


# In[ ]:


def column_swap(input_array: np.ndarray) -> np.ndarray:
    return np.flip(input_array, axis=1)


# In[ ]:


# Készíts egy olyan függvényt ami összehasonlít két array-t és adjon vissza egy array-ben, hogy hol egyenlőek 
# Pl Be: [7,8,9], [9,8,7] 
# Ki: [1]
# compare_two_array()
# egyenlő elemszámúakra kell csak hogy működjön


# In[ ]:


def compare_two_array(input_array_1: np.ndarray, input_array_2: np.ndarray) -> np.ndarray:
    compared_array: np.ndarray = input_array_1 == input_array_2
    return np.where(compared_array)


# In[ ]:


# Készíts egy olyan függvényt, ami vissza adja string-ként a megadott array dimenzióit:
# Be: [[1,2,3], [4,5,6]]
# Ki: "sor: 2, oszlop: 3, melyseg: 1"
# get_array_shape()
# 3D-vel még műküdnie kell!,


# In[ ]:


def get_array_shape(input_array: np.ndarray) -> str:
    result_string: str
    try:
        result_string = f"sor: {input_array.shape[0]}, oszlop: {input_array.shape[1]}, melyseg: {input_array.shape[2]}"
        pass
    except:
        try:
            result_string = f"sor: {input_array.shape[0]}, oszlop: {input_array.shape[1]}, melyseg: 1"
            pass
        except:
            result_string = f"sor: {input_array.shape[0]}, oszlop: 1, melyseg: 1"

    return result_string


# In[ ]:


# Készíts egy olyan függvényt, aminek segítségével elő tudod állítani egy neurális hálózat tanításához szükséges pred-et egy numpy array-ből.
# Bementként add meg az array-t, illetve hogy mennyi class-od van. Kimenetként pedig adjon vissza egy 2d array-t, ahol a sorok az egyes elemek. Minden nullákkal teli legyen és csak ott álljon egyes, ahol a bementi tömb megjelöli.
# Pl. ha 1 van a bemeneten és 4 classod van, akkor az adott sorban az array-ban a [1] helyen álljon egy 1-es, a többi helyen pedig 0.
# Be: [1, 2, 0, 3], 4
# Ki: [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# encode_Y()


# In[ ]:


def encode_Y(input_array: np.ndarray, number_of_classes: int) -> np.ndarray:
    result_array: np.ndarray = np.zeros((number_of_classes, number_of_classes))
    for i in range(0, number_of_classes):
        result_array[i, input_array[i]] = 1
    return result_array


# In[ ]:


# A fenti feladatnak valósítsd meg a kiértékelését. Adj meg a 2d array-t és adj vissza a decodolt változatát
# Be:  [[0,1,0,0], [0, 0, 1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]
# Ki:  [1, 2, 0, 3]
# decode_Y()
def decode_Y(input_array: np.ndarray) -> np.ndarray:
    result_array = np.where(input_array)[1]
    return result_array


# In[ ]:


# Készíts egy olyan függvényt, ami képes kiértékelni egy neurális háló eredményét! Bemenetként egy listát és egy array-t és adja vissza azt az elemet, aminek a legnagyobb a valószínüsége(értéke) a listából.
# Be: ['alma', 'körte', 'szilva'], [0.2, 0.2, 0.6]. # Az ['alma', 'körte', 'szilva'] egy lista!
# Ki: 'szilva'
# eval_classification()


# In[ ]:


def eval_classification(input_list: List, input_array: np.ndarray) -> Any:
    max_probabiility: int = int(np.argmax(input_array))
    return input_list[max_probabiility]


# In[ ]:


# Készíts egy olyan függvényt, ahol az 1D array-ben a páratlan számokat -1-re cseréli
# Be: [1,2,3,4,5,6]
# Ki: [-1,2,-1,4,-1,6]
# repalce_odd_numbers()


# In[ ]:


def replace_odd_numbers(input_array: np.ndarray) -> np.ndarray:
    odd_condition: np.ndarray = input_array % 2 == 1
    input_array[odd_condition] = -1
    return input_array


# In[ ]:


# Készíts egy olyan függvényt, ami egy array értékeit -1 és 1-re változtatja, attól függően, hogy az adott elem nagyobb vagy kisebb a paraméterként megadott számnál.
# Ha a szám kisebb mint a megadott érték, akkor -1, ha nagyobb vagy egyenlő, akkor pedig 1.
# Be: [1, 2, 5, 0], 2
# Ki: [-1, 1, 1, -1]
# replace_by_value()


# In[ ]:


def replace_by_value(input_array: np.ndarray, threshold: float) -> np.ndarray:
    condition: np.ndarray = input_array < threshold
    input_array[condition] = -1
    input_array[np.invert(condition)] = 1
    return input_array


# In[ ]:


# Készíts egy olyan függvényt, ami egy array értékeit összeszorozza és az eredményt visszaadja
# Be: [1,2,3,4]
# Ki: 24
# array_multi()
# Ha több dimenziós a tömb, akkor az egész tömb elemeinek szorzatával térjen vissza


# In[ ]:


def array_multi(input_array: np.ndarray) -> float:
    result: float = float(np.prod(input_array))
    return result


# In[ ]:


# Készíts egy olyan függvényt, ami egy 2D array értékeit összeszorozza és egy olyan array-el tér vissza, aminek az elemei a soroknak a szorzata
# Be: [[1, 2], [3, 4]]
# Ki: [2, 12]
# array_multi_2d()


# In[ ]:


def array_multi_2d(input_array: np.ndarray) -> np.ndarray:
    result_array: np.ndarray = np.prod(input_array, axis=1)
    return result_array


# In[ ]:


# Készíts egy olyan függvényt, amit egy meglévő numpy array-hez készít egy bordert nullásokkal. Bementként egy array-t várjon és kimenetként egy array jelenjen meg aminek van border-je
# Be: [[1,2],[3,4]]
# Ki: [[0,0,0,0],[0,1,2,0],[0,3,4,0],[0,0,0,0]]
# add_border()


# In[ ]:


def add_border(input_array: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.pad(input_array, 1, mode="constant")
    return result


# In[ ]:


# A KÖTVETKEZŐ FELADATOKHOZ NÉZZÉTEK MEG A NUMPY DATA TYPE-JÁT!


# In[ ]:


# Készíts egy olyan függvényt ami két dátum között felsorolja az összes napot és ezt adja vissza egy numpy array-ben. A fgv ként str vár paraméterként 'YYYY-MM' formában.
# Be: '2023-03', '2023-04'  # mind a kettő paraméter str.
# Ki: ['2023-03-01', '2023-03-02', .. , '2023-03-31',]
# list_days()


# In[ ]:


def list_day(start_date:str, end_date:str) -> np.ndarray:
    result: np.ndarray = np.zeros(5)
    return result


# In[ ]:


# Írj egy fügvényt ami vissza adja az aktuális dátumot az alábbi formában: YYYY-MM-DD. Térjen vissza egy 'numpy.datetime64' típussal.
# Be:
# Ki: 2017-03-24


# In[ ]:


def get_act_date() -> np.datetime64:
    return np.datetime64(datetime.date.today())


# In[ ]:


# Írj egy olyan függvényt ami visszadja, hogy mennyi másodperc telt el 1970 január 01. 00:02:00 óta. Int-el térjen vissza
# Be: 
# Ki: másodpercben az idó, int-é kasztolva
# sec_from_1970()


# In[ ]:


def sec_from_1970() -> int:
    start = datetime.datetime(1970,1,1,0,2,0)
    result = (datetime.datetime.utcnow()- start).total_seconds()
    return int(result)

