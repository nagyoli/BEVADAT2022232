#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict


# In[ ]:


'''
FONTOS: Az első feladatáltal visszaadott DataFrame-et kell használni a további feladatokhoz. 
A függvényeken belül mindig készíts egy másolatot a bemenő df-ről, (new_df = df.copy() és ezzel dolgozz tovább.)
'''


# In[ ]:


'''
Készíts egy függvényt ami a bemeneti dictionary-ből egy DataFrame-et ad vissza.

Egy példa a bemenetre: test_dict
Egy példa a kimenetre: test_df
return type: pandas.core.frame.DataFrame
függvény neve: dict_to_dataframe
'''


# In[ ]:


stats = {"country": ["Brazil", "Russia", "India", "China", "South Africa"],
       "capital": ["Brasilia", "Moscow", "New Dehli", "Beijing", "Pretoria"],
       "area": [8.516, 17.10, 3.286, 9.597, 1.221],
       "population": [200.4, 143.5, 1252, 1357, 52.98] }


# In[ ]:


def dict_to_dataframe(test_dict: Dict) -> pd.DataFrame:
       result: pd.DataFrame = pd.DataFrame.from_dict(test_dict)
       return result


# In[ ]:


'''
Készíts egy függvényt ami a bemeneti DataFrame-ből vissza adja csak azt az oszlopot amelynek a neve a bemeneti string-el megegyező.

Egy példa a bemenetre: test_df, 'area'
Egy példa a kimenetre: test_df
return type: pandas.core.series.Series
függvény neve: get_column
'''


# In[ ]:


def get_column(input_df: pd.DataFrame, column_name: str) -> pd.Series:
       df: pd.DataFrame = input_df.copy()
       result: pd.Series = df[column_name]
       return result


# In[ ]:


'''
Készíts egy függvényt ami a bemeneti DataFrame-ből vissza adja a két legnagyobb területű országhoz tartozó sorokat.

Egy példa a bemenetre: test_df
Egy példa a kimenetre: test_df
return type: pandas.core.frame.DataFrame
függvény neve: get_top_two
'''


# In[ ]:


def get_top_two(input_df: pd.DataFrame) -> pd.DataFrame:
       df: pd.DataFrame = input_df.copy()
       return df.nlargest(2, 'area')


# In[ ]:


'''
Készíts egy függvényt ami a bemeneti DataFrame-ből kiszámolja az országok népsűrűségét és eltárolja az eredményt egy új oszlopba ('density').
(density = population / area)

Egy példa a bemenetre: test_df
Egy példa a kimenetre: test_df
return type: pandas.core.frame.DataFrame
függvény neve: population_density
'''


# In[ ]:


def population_density(input_df: pd.DataFrame) -> pd.DataFrame:
       df: pd.DataFrame = input_df.copy()
       density: pd.Series = df['population']/df['area']
       df['density'] = density
       return df


# In[ ]:



'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan oszlopdiagramot (bar plot),
ami vizualizálja az országok népességét.

Az oszlopdiagram címe legyen: 'Population of Countries'
Az x tengely címe legyen: 'Country'
Az y tengely címe legyen: 'Population (millions)'

Egy példa a bemenetre: test_df
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: plot_population
'''


# In[ ]:


def plot_population(input_df: pd.DataFrame) -> plt.Figure:
       df = input_df.copy()
       fig, ax = plt.subplots()
       ax.bar(df['country'], df['population'])
       ax.set_ylabel('Population (millions)')
       ax.set_xlabel('Country')
       ax.set_title('Population of Countries')
       return fig


# In[ ]:


'''
Készíts egy függvényt, ami a bemeneti Dataframe adatai alapján elkészít egy olyan kördiagramot,
ami vizualizálja az országok területét. Minden körcikknek legyen egy címe, ami az ország neve.

Az kördiagram címe legyen: 'Area of Countries'

Egy példa a bemenetre: test_df
Egy példa a kimenetre: fig
return type: matplotlib.figure.Figure
függvény neve: plot_area
'''


# In[ ]:


def plot_area(input_df: pd.DataFrame) -> plt.Figure:
       df: pd.DataFrame = input_df.copy()
       areas: pd.Series = df['area']
       countries: pd.Series =  df['country']
       fig, ax = plt.subplots()
       ax.set_title('Area of Countries')
       ax.pie(areas, labels = countries)
       return fig

