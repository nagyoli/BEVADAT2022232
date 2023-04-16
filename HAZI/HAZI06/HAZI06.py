"""
1. Értelmezd az adatokat!!!

2. Írj egy osztályt a következő feladatokra:  
     - Neve legyen NJCleaner és mentsd el a NJCleaner.py-ba. Ebben a fájlban csak ez az osztály legyen.
     - Konsturktorban kapja meg a csv elérési útvonalát és olvassa be pandas segítségével és mentsük el a data (self.data) osztályszintű változóba 
     - Írj egy függvényt ami sorbarendezi a dataframe-et 'scheduled_time' szerint növekvőbe és visszatér a sorbarendezett df-el, a függvény neve legyen 'order_by_scheduled_time' és térjen vissza a df-el
     - Dobjuk el a from és a to oszlopokat, illetve a nan-okat és adjuk vissza a df-et. A függvény neve legyen 'drop_columns_and_nan' és térjen vissza a df-el
     - A date-et alakítsd át napokra, pl.: 2018-03-01 --> Thursday, ennek az oszlopnak legyen neve a 'day'. Ezután dobd el a 'date' oszlopot és térjen vissza a df-el. A függvény neve legyen 'convert_date_to_day' és térjen vissza a df-el   
     - Hozz létre egy új oszlopot 'part_of_the_day' névvel. A 'scheduled_time' oszlopból számítsd ki az alábbi értékeit. A 'scheduled_time'-ot dobd el. A függvény neve legyen 'convert_scheduled_time_to_part_of_the_day' és térjen vissza a df-el  
         4:00-7:59 -- early_morning  
         8:00-11:59 -- morning  
         12:00-15:59 -- afternoon  
         16:00-19:59 -- evening  
         20:00-23:59 -- night  
         0:00-3:59 -- late_night  
    - A késéeket jelöld az alábbiak szerint. Az új osztlop neve legyen 'delay'. A függvény neve legyen pedig 'convert_delay' és térjen vissza a df-el
         0 <= x 5  --> 0  
         5 <= x    --> 1  
    - Dobd el a felesleges oszlopokat 'train_id' 'scheduled_time' 'actual_time' 'delay_minutes'. A függvény neve legyen 'drop_unnecessary_columns' és térjen vissza a df-el
    - Írj egy olyan metódust, ami elmenti a dataframe első 60 000 sorát. A függvénynek egy string paramétere legyen, az pedig az, hogy hova mentse el a csv-t (pl.: 'data/NJ.csv'). A függvény neve legyen 'save_first_60k'. 
    - Írj egy függvényt ami a fenti függvényeket összefogja és megvalósítja (sorbarendezés --> drop_columns_and_nan --> ... --> save_first_60k), a függvény neve legyen 'prep_df'. Egy paramnétert várjon, az pedig a csv-nek a mentési útvonala legyen. Ha default value-ja legyen 'data/NJ.csv'

3.  A feladatot a HAZI06.py-ban old meg.
    Az órán megírt DecisionTreeClassifier-t fit-eld fel az első feladatban lementett csv-re.

    A feladat célja az, hogy határozzuk meg azt, hogy a vonatok késnek-e vagy sem. 0p <= x < 5p --> nem késik, ha 5 < x --> késik.
    Az adatoknak a 20% legyen test és a splitelés random_state-je pedig 41 (mint órán)
    A testset-en 80% kell elérni. Ha megvan a minimum százalék, akkor azzal paraméterezd fel a decisiontree-t és azt kell leadni.

    A leadásnál csak egy fit kell, ezt azzal a paraméterre paraméterezd fel, amivel a legjobb accuracy-t elérted.

    A helyes paraméter megtalálásához használhatsz grid_search-öt.
    https://www.w3schools.com/python/python_ml_grid_search.asp 

4.  A tanításodat foglald össze 4-5 mondatban a HAZI06.py-ban a fájl legalján kommentben. Írd le a nehézségeket, mivel próbálkoztál, mi vált be és mi nem. Ezen kívül írd le 10 fitelésed eredményét is, hogy milyen paraméterekkel probáltad és milyen accuracy-t értél el. 
Ha ezt feladatot hiányzik, akkor nem fogadjuk el a házit!

HAZI06-
    -NJCleaner.py
    -HAZI06.py

##################################################################
##                                                              ##
## A feladatok közül csak a NJCleaner javítom unit test-el      ##
## A decision tree-t majd manuálisan fogom lefuttatni           ##
## NJCleaner - 10p, Tanítás - acc-nál 10%-ként egy pont         ##
## Ha a 4. feladat hiányzik, akkor nem tudjuk elfogadni a házit ##
##                                                              ##
##################################################################
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from DecisionTreeClassifier import DecisionTreeClassifier

data_to_fit = pd.read_csv('data/NJ_60k.csv', sep=',', header=0)
X: pd.DataFrame = data_to_fit.iloc[:, :-1].values
Y: pd.DataFrame  = data_to_fit.iloc[:, -1].values.reshape(-1,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2, random_state=41)
classifier = DecisionTreeClassifier(min_samples_split=95, max_depth=11)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
acc = accuracy_score(Y_test, Y_pred)

"""
Kiindulásnál problémát jelentett, hogy a minimum sample split nem lehet 1, így a Grid Search elég sokáig hibára futott már az elején. 
Ameddig a minimum sample split-et 20 alatti értékre állítottam, a max depth 8-as érték mellett ''info gain'' szakaszon hibára futott.
A modell illesztés onnantól működött rendesen, ha legalább 25-ös minimum split-et állítottam be.
A Grig Search nevetségesen lassú volt. A minimum sample split-et [20, 95] zárt intervallumon futtattam, 5-ös lépésközzel, míg a max depth-et [7, 17]-es zárt intervallumon 2-es lépésközzel. Ez a futtatás 2 óráig tartott.
Teljesen biztos, hogy ilyen paraméterezés mellett a döntési fa túltanult.
Mivel a max depth során 1 érték lett a top 10-ben a legjobb, ezért a top 20-at riportálom lent.

Grid Search értékek:
    [accuracy, minimum sample split, max depth]
    [0.8035833333333333, 95, 11],
    [0.8034166666666667, 60, 11],
    [0.8033333333333333, 75, 11],
    [0.8033333333333333, 70, 11],
    [0.8033333333333333, 65, 11],
    [0.80325, 55, 11],
    [0.8030833333333334, 80, 11],
    [0.803, 90, 11],
    [0.8028333333333333, 45, 11],
    [0.8026666666666666, 85, 11],
    [0.8026666666666666, 50, 11],
    [0.8021666666666667, 40, 11],
    [0.8021666666666667, 35, 11],
    [0.8018333333333333, 30, 11],
    [0.8014166666666667, 25, 11],
    [0.80075, 95, 13],
    [0.7999166666666667, 80, 13],
    [0.79975, 90, 13],
    [0.7995833333333333, 75, 13],
    [0.7994166666666667, 85, 13]
"""