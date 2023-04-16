import pandas as pd
from typing import List
import os.path
import datetime


class NJCleaner:
    def __init__(self, csv_path: str):
        self.data = self.csv_to_dataframe(csv_path)

    @staticmethod
    def csv_to_dataframe(csv_path: str) -> pd.DataFrame:
        dataframe = pd.read_csv(csv_path, header=0)
        assert dataframe.shape[1] == 13
        return dataframe

    def order_by_scheduled_time(self) -> pd.DataFrame:
        self.data.sort_values(by=['scheduled_time'])
        return self.data

    def drop_columns_and_nan(self) -> pd.DataFrame:
        to_drop: List = ['from', 'to']
        self.data = self.data.drop(columns=to_drop)
        self.data = self.data.dropna()
        return self.data

    def convert_date_to_day(self) -> pd.DataFrame:
        self.data.date = pd.to_datetime(self.data.date)
        self.data['day'] = self.data.date.dt.day_name()
        self.data = self.data.drop(columns=['date'])
        return self.data

    def convert_scheduled_time_to_part_of_the_day(self) -> pd.DataFrame:
        self.data['part_of_the_day'] = 0
        self.data['scheduled_time'] = pd.to_datetime(self.data['scheduled_time'])
        self.data['time'] = self.data['scheduled_time'].dt.time

        def part(hour) -> str:
            if (hour >= 4) and (hour < 8):
                return 'early_morning'
            elif (hour >= 8) and (hour < 12):
                return 'morning'
            elif (hour >= 12) and (hour < 16):
                return 'afternoon'
            elif (hour >= 16) and (hour < 20):
                return 'evening'
            elif (hour >= 20) and (hour < 24):
                return 'night'
            elif (hour < 4):
                return 'late_night'

        def determine_time(time) -> str:
            hour = time.hour
            name = part(hour)
            return name

        self.data['part_of_the_day'] = self.data['time'].apply(determine_time)
        self.data = self.data.drop(columns=['time', 'scheduled_time'])
        return self.data

    def convert_delay(self) -> pd.DataFrame:
        def create_delay(row):
            if row >= 5:
                val = 1
            else:
                val = 0
            return val

        self.data['delay'] = self.data.delay_minutes.apply(create_delay)
        return self.data

    def drop_unnecessary_columns(self) -> pd.DataFrame:
        columns_to_drop: List = ['train_id', 'actual_time', 'delay_minutes']
        self.data = self.data.drop(columns=columns_to_drop)
        return self.data

    def save_first_60k(self, path: str) -> None:
        self.data.iloc[:60000].to_csv(path, index=False, sep=',')


    def prep_df(self, path: str = 'data/NJ.csv') -> None:
        self.order_by_scheduled_time()
        self.drop_columns_and_nan()
        self.convert_date_to_day()
        self.convert_scheduled_time_to_part_of_the_day()
        self.convert_delay()
        self.drop_unnecessary_columns()
        self.save_first_60k(path)