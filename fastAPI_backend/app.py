# -*- coding: utf-8 -*-
"""
Created on Fri Nov 19 12:38:06 2021

@author: RamziAbdelhafidh
"""

# 1. Librairies
import io
import uvicorn
from fastapi import FastAPI
from starlette.responses import Response
from BankCredit import BankCredit
import pickle
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns


# Creer l'objet app
app = FastAPI()


# charger quelques datas utiles dans la suite
pickle_clf = open("/storage/final_model.pkl", "rb")
final_model = pickle.load(pickle_clf)

pickle_pipe = open("/storage/preprocessing_pipe.pkl", "rb")
pipe_shap = pickle.load(pickle_pipe)

pickle_exp = open("/storage/explainer.pkl", "rb")
explainer = pickle.load(pickle_exp)

most_important_features = joblib.load('/storage/most_important_features.pkl')

features = joblib.load('/storage/features.pkl')


@app.get('/')
def index():
    return{"text": "L'API est lancée"}


@app.post('/predict')
def predict_bank_credit(data: BankCredit):
    """
    Fonction predict qui prend les infos du client sous forme json et retourne
    la décision et la probabilité

    Parameters
    ----------
    data : BankCredit

    Returns
    -------
    dict
        prediction
    """
    data = data.dict()
    data_df = pd.DataFrame.from_dict([data])
    data_df = data_df[features]
    prediction = final_model.predict(data_df)[0]
    # if(prediction == 0):
    #     prediction = "Crédit Accordé"
    # else:
    #     prediction = "Credit Refusé"
    return {'prediction': str(prediction)}


@app.post('/get_waterfall_graph')
def get_waterfall(data: BankCredit):
    """
    Fonction qui prend les infos du client sous forme json et retourne
    le waterfall graph

    Parameters
    ----------
    data : BankCredit

    Returns
    -------
    Response
        waterfall plot
    """
    data = data.dict()
    data_df = pd.DataFrame.from_dict([data])
    data_shap = pipe_shap.transform(data_df)
    shap_values = explainer(data_shap)
    waterfall_plot = plt.figure()
    shap.plots.waterfall(shap_values[0], show=False)
    bytes_io = io.BytesIO()
    waterfall_plot.savefig(bytes_io, format="PNG", bbox_inches='tight')
    plt.close()
    return Response(bytes_io.getvalue(), media_type="image/png")


@app.post('/get_bar_plot_1')
def get_bar_1(data: BankCredit):
    """
    Fonction qui prend les infos du client sous forme json et retourne
    le graph en barre

    Parameters
    ----------
    data : BankCredit

    Returns
    -------
    Response
        bar plot for EXT_SOURCE_1
    """
    barplot_df = pd.read_csv("/storage/barplot_df.csv")
    barplot_df = barplot_df[barplot_df['indicator'] == 'EXT_SOURCE_1']
    data = data.dict()
    data_df = pd.DataFrame.from_dict([data])
    df_user = pd.DataFrame(columns=['TARGET', 'value', 'indicator'])
    df_user = df_user.append({'TARGET': 'user value',
                              'indicator': 'EXT_SOURCE_1',
                              'value': float(data_df['EXT_SOURCE_1'])},
                             ignore_index=True)
    barplot_df = barplot_df.append(df_user)
    bar_plot = plt.figure()
    sns.barplot(data=barplot_df, x='indicator', y='value', hue='TARGET')
    bytes_io = io.BytesIO()
    bar_plot.savefig(bytes_io, format="PNG")
    plt.close()
    return Response(bytes_io.getvalue(), media_type="image/png")


@app.post('/get_bar_plot_2')
def get_bar_2(data: BankCredit):
    """
    Fonction qui prend les infos du client sous forme json et retourne
    le graph en barre

    Parameters
    ----------
    data : BankCredit

    Returns
    -------
    Response
        bar plot for EXT_SOURCE_2
    """
    barplot_df = pd.read_csv("/storage/barplot_df.csv")
    barplot_df = barplot_df[barplot_df['indicator'] == 'EXT_SOURCE_2']
    data = data.dict()
    data_df = pd.DataFrame.from_dict([data])
    df_user = pd.DataFrame(columns=['TARGET', 'value', 'indicator'])
    df_user = df_user.append({'TARGET': 'user value',
                              'indicator': 'EXT_SOURCE_2',
                              'value': float(data_df['EXT_SOURCE_2'])},
                             ignore_index=True)
    barplot_df = barplot_df.append(df_user)
    bar_plot = plt.figure()
    sns.barplot(data=barplot_df, x='indicator', y='value', hue='TARGET')
    bytes_io = io.BytesIO()
    bar_plot.savefig(bytes_io, format="PNG")
    plt.close()
    return Response(bytes_io.getvalue(), media_type="image/png")


@app.post('/get_bar_plot_3')
def get_bar_3(data: BankCredit):
    """
    Fonction qui prend les infos du client sous forme json et retourne
    le graph en barre

    Parameters
    ----------
    data : BankCredit

    Returns
    -------
    Response
        bar plot for EXT_SOURCE_3
    """
    barplot_df = pd.read_csv("/storage/barplot_df.csv")
    barplot_df = barplot_df[barplot_df['indicator'] == 'EXT_SOURCE_3']
    data = data.dict()
    data_df = pd.DataFrame.from_dict([data])
    df_user = pd.DataFrame(columns=['TARGET', 'value', 'indicator'])
    df_user = df_user.append({'TARGET': 'user value',
                              'indicator': 'EXT_SOURCE_3',
                              'value': float(data_df['EXT_SOURCE_3'])},
                             ignore_index=True)
    barplot_df = barplot_df.append(df_user)
    bar_plot = plt.figure()
    sns.barplot(data=barplot_df, x='indicator', y='value', hue='TARGET')
    bytes_io = io.BytesIO()
    bar_plot.savefig(bytes_io, format="PNG")
    plt.close()
    return Response(bytes_io.getvalue(), media_type="image/png")
