#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List
from itertools import chain

# In[ ]:


'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''


# In[ ]:


'''
Készíts egy függvényt, ami egy string útvonalat vár paraméterként, és egy DataFrame ad visszatérési értékként.

Egy példa a bemenetre: 'test_data.csv'
Egy példa a kimenetre: df_data
return type: pandas.core.frame.DataFrame
függvény neve: csv_to_df
'''


# In[ ]:


def csv_to_df(path_to_csv: str) -> pd.DataFrame:
    return pd.read_csv(path_to_csv)

# In[ ]:


'''
Készíts egy függvényt, ami egy DataFrame-et vár paraméterként, 
és átalakítja azoknak az oszlopoknak a nevét nagybetűsre amelyiknek neve nem tartalmaz 'e' betüt.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_capitalized
return type: pandas.core.frame.DataFrame
függvény neve: capitalize_columns
'''


# In[ ]:


def capitalize_columns(input_df: pd.DataFrame) -> pd.DataFrame:
    result_df: pd.DataFrame = input_df.copy()
    result_df.columns= [column if 'e' in column else column.upper() for column in result_df.columns]
    return result_df

# In[ ]:


'''
Készíts egy függvényt, ahol egy szám formájában vissza adjuk, hogy hány darab diáknak sikerült teljesíteni a matek vizsgát.
(legyen az átmenő ponthatár 50).

Egy példa a bemenetre: df_data
Egy példa a kimenetre: 5
return type: int
függvény neve: math_passed_count
'''


# In[ ]:


def math_passed_count(input_df: pd.DataFrame) -> int:
    df: pd.DataFrame = input_df.copy()
    result: int = df['math score'][df['math score'] > 50].count()
    return result


# In[ ]:


'''
Készíts egy függvényt, ahol Dataframe ként vissza adjuk azoknak a diákoknak az adatait (sorokat), akik végeztek előzetes gyakorló kurzust.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_did_pre_course
return type: pandas.core.frame.DataFrame
függvény neve: did_pre_course
'''


# In[ ]:


def did_pre_course(input_df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = input_df.copy()
    return df[df['test preparation course'] == "completed"]


# In[ ]:


'''
Készíts egy függvényt, ahol a bemeneti Dataframet a diákok szülei végzettségi szintjei alapján csoportosításra kerül,
majd aggregációként vegyük, hogy átlagosan milyen pontszámot értek el a diákok a vizsgákon.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_average_scores
return type: pandas.core.frame.DataFrame
függvény neve: average_scores
'''


# In[ ]:


def average_scores(input_df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = input_df.copy()
    return df.groupby(by='parental level of education').mean()


# In[ ]:


'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'age' oszloppal, töltsük fel random 18-66 év közötti értékekkel.
A random.randint() függvényt használd, a random sorsolás legyen seedleve, ennek értéke legyen 42.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_age
return type: pandas.core.frame.DataFrame
függvény neve: add_age
'''


# In[ ]:


def add_age(input_df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = input_df.copy()
    random_seed:int = 42
    np.random.seed(random_seed)
    df['age'] = np.random.randint(18,67, size=len(df))
    return df


# In[ ]:


'''
Készíts egy függvényt, ami vissza adja a legjobb teljesítményt elérő női diák pontszámait.

Egy példa a bemenetre: df_data
Egy példa a kimenetre: (99,99,99) #math score, reading score, writing score
return type: tuple
függvény neve: female_top_score
'''


# In[ ]:


def female_top_score(input_df: pd.DataFrame) -> Tuple:
    df: pd.DataFrame = input_df.copy()
    columns_to_use: List = ['math score', 'reading score', 'writing score']
    df['total score'] = df[columns_to_use].sum(axis=1)
    res: pd.DataFrame = df[df['gender'] == 'female'].nlargest(1, 'total score', keep='first')
    tuple_to_return: Tuple = tuple(chain.from_iterable(res[columns_to_use].values))
    return tuple_to_return


# In[ ]:


'''
Készíts egy függvényt, ami a bementeti Dataframet kiegészíti egy 'grade' oszloppal. 
Számoljuk ki hogy a diákok hány százalékot ((math+reading+writing)/300) értek el a vizsgán, és osztályozzuk őket az alábbi szempontok szerint:

90-100%: A
80-90%: B
70-80%: C
60-70%: D
<60%: F

Egy példa a bemenetre: df_data
Egy példa a kimenetre: df_data_with_grade
return type: pandas.core.frame.DataFrame
függvény neve: add_grade
'''


# In[ ]:


def add_grade(input_df: pd.DataFrame) -> pd.DataFrame:
    df: pd.DataFrame = input_df.copy()
    columns_to_use: List = ['math score', 'reading score', 'writing score']
    df['point'] = df[columns_to_use].sum(axis=1) / 3
    df['grade'] = df['grade'][df['point'] < 100 ] = 'A'
    df['grade'][df['point'] < 90] = 'B'
    df['grade'][df['point'] < 80] = 'C'
    df['grade'][df['point'] < 70] = 'D'
    df['grade'][df['point'] < 60] = 'F'
    return df


# In[ ]:


'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlop diagrammot,
ami vizualizálja a nemek által elért átlagos matek pontszámot.

Oszlopdiagram címe legyen: 'Average Math Score by Gender'
Az x tengely címe legyen: 'Gender'
Az y tengely címe legyen: 'Math Score'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: math_bar_plot
'''


# In[ ]:


def math_bar_plot(input_data: pd.DataFrame) -> plt.Figure:
    df: pd.DataFrame = input_data.copy()
    columns_to_use: List = ['gender','math score']
    fig, ax = plt.subplots()
    df[columns_to_use].groupby(by=columns_to_use[0]).mean().plot(kind='bar',
                                                                 ax=ax,
                                                                 title='Average Math Score by Gender',
                                                                 xlabel = 'Gender',
                                                                 ylabel = 'Math Score',
                                                                 legend = False)
    return fig

# In[ ]:


''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan histogramot,
ami vizualizálja az elért írásbeli pontszámokat.

A histogram címe legyen: 'Distribution of Writing Scores'
Az x tengely címe legyen: 'Writing Score'
Az y tengely címe legyen: 'Number of Students'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: writing_hist
'''


# In[ ]:


def writing_hist(input_data: pd.DataFrame) -> plt.Figure:
    df: pd.DataFrame = input_data.copy()
    columns_to_use: List = ['writing score']
    fig, ax = plt.subplots()
    ax.set_title('Distribution of Writing Scores')
    ax.set_xlabel('Writing Score')
    ax.set_ylabel('Number of Students')
    ax.hist(df[columns_to_use])

    return fig


# In[ ]:


''' 
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja a diákok etnikum csoportok szerinti eloszlását százalékosan.

Érdemes megszámolni a diákok számát, etnikum csoportonként,majd a százalékos kirajzolást az autopct='%1.1f%%' paraméterrel megadható.
Mindegyik kör szelethez tartozzon egy címke, ami a csoport nevét tartalmazza.
A diagram címe legyen: 'Proportion of Students by Race/Ethnicity'

Egy példa a bemenetre: df_data
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: ethnicity_pie_chart
'''


# In[ ]:


def ethnicity_pie_chart(input_data: pd.DataFrame) -> plt.Figure:
    df: pd.DataFrame = input_data.copy()
    df['counter'] = 1
    columns_to_use: List = ['race/ethnicity', 'counter']
    df_to_use: pd.DataFrame = df[columns_to_use].groupby(by=columns_to_use[0]).sum().reset_index()
    fig, ax = plt.subplots()
    ax.pie(df_to_use[columns_to_use[1]],labels= df_to_use[columns_to_use[0]], autopct='%1.1f%%')
    ax.set_title('Proportion of Students by Race/Ethnicity')


    return fig
