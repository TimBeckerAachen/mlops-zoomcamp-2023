#!/usr/bin/env python
# coding: utf-8

import sys
import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


categorical = ['PULocationID', 'DOLocationID']


def read_data(filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')
    return df


def write_results(df: pd.DataFrame, output_file_name: str) -> None:
    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predictions'] = y_pred
    print(f'mean predicted duration {y_pred.mean()}')

    df_result.to_parquet(
        output_file_name,
        engine='pyarrow',
        compression=None,
        index=False
    )
    print(f'wrote predictions to {output_file_name}')


def run(year: int = 2021, month: int = 2):
    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    output_file = f'result_yellow_tripdata_{year}-{month:02d}.parquet'
    data = read_data(input_file)
    write_results(data, output_file)


if __name__ == '__main__':
    year = int(sys.argv[1])
    month = int(sys.argv[2])
    run(year, month)
