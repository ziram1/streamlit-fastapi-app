# -*- coding: utf-8 -*-
"""
@author: RamziAbdelhafidh
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, \
    RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from sklearn import metrics
from sklearn.pipeline import make_pipeline
import pickle
import shap
import matplotlib.pyplot as plt
import utils


def plot_f3_score(thresholds, fbeta_scores):
    plt.figure(figsize=(10, 7))
    plt.grid()
    plt.xlabel("Threshold")
    plt.ylabel('F3-score')
    plt.plot(thresholds, fbeta_scores)
    annot_max(np.array(thresholds), np.array(fbeta_scores))
    plt.ylim(0, 0.55)
    plt.show()


def annot_max(x, y, ax=None):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = "threshold={:.3f}, F3-score={:.3f}".format(xmax, ymax)
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    arrowprops = dict(
        arrowstyle="->",
        connectionstyle="angle,angleA=0,angleB=60",
        color='black')
    kw = dict(xycoords='data', textcoords="axes fraction",
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)


def plot_roc_curve(trained_estimator, X_test, y_test):
    y_proba = trained_estimator.predict_proba(X_test)[:, 1]
    fper, tper, threshold = metrics.roc_curve(y_test, y_proba)
    auc = metrics.auc(fper, tper)
    plt.figure(figsize=(10, 7))
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.text(0.8, 0, "AUC = {:.3f}".format(auc), fontsize=20)
    plt.legend()
    plt.grid()
    plt.show()


def csv_clean_data_import(filename, cols_to_drop):
    data = pd.read_csv(filename)
    data.drop(columns=cols_to_drop, axis=1, inplace=True)
    return data


def take_a_stratified_subset(data, percentage, stratify):
    data_to_keep, data_to_reject = train_test_split(data,
                                                    test_size=percentage,
                                                    stratify=stratify)
    return data_to_keep


def train_test_construction(data, **kwargs):
    train_df, test_df = train_test_split(data, **kwargs)
    x_train = train_df.drop('TARGET', axis=1)
    y_train = train_df['TARGET']
    x_test = test_df.drop('TARGET', axis=1)
    y_test = test_df['TARGET']
    return x_train, y_train, x_test, y_test


def create_imblearn_pipe():
    pipe = Pipeline([('binary_label_encoding', utils.BinaryLabelEncoder()),
                     ('multiple_label_encoding', utils.MultipleLabelEncoding()),
                     ('imputer', utils.Imputer()),
                     ('polynomial_features', utils.PolyFeatures()),
                     ('normalization', utils.Normalization()),
                     ('oversampling', SMOTE()),
                     ('classifier', RandomForestClassifier())
                     ]
                    )
    return pipe


def create_param_grid():
    param_grid = [{'classifier': (SVC(probability=True),),
                   'classifier__C': [1, 2, 3],
                   'classifier__kernel': ['linear', 'rbf'],
                   'classifier__class_weight': [None, 'balanced'],
                   'oversampling': ['passthrough', BorderlineSMOTE(), SMOTE()],
                   'polynomial_features__degree': [3, 4]},

                  {'classifier': (RandomForestClassifier(),),
                   'classifier__n_estimators': [10, 20, 30],
                   'classifier__class_weight': [None, 'balanced'],
                   'oversampling': ['passthrough', BorderlineSMOTE(), SMOTE()],
                   'polynomial_features__degree': [3, 4]},

                  {'classifier': (GradientBoostingClassifier(),),
                   'classifier__n_estimators': [10, 20, 30],
                   'classifier__max_depth': [3, 4, 5],
                   'oversampling': ['passthrough', SMOTE(), BorderlineSMOTE()],
                   'polynomial_features__degree': [3, 4]},
                  ]
    return param_grid


def create_search():
    # Pipeline creation
    pipe = create_imblearn_pipe()

    # RandomizedSearchCV creation
    stratified_kfold = StratifiedKFold(n_splits=5,
                                       shuffle=True,
                                       random_state=11)
    param_grid = create_param_grid()
    search = RandomizedSearchCV(pipe,
                                param_grid,
                                n_jobs=-1,
                                scoring='roc_auc',
                                cv=stratified_kfold,
                                verbose=3,
                                n_iter=10)
    return search


def get_f3_scores_list(trained_estimator, X_test, y_test, resolution=101):
    y_proba = trained_estimator.predict_proba(X_test)[:, 1]
    fbeta_scores = []
    thresholds = np.linspace(0, 1, resolution)
    for threshold in thresholds:
        y_predicted = [1 if proba > threshold else 0 for proba in y_proba]
        fbeta_scores.append(metrics.fbeta_score(y_test, y_predicted, beta=3))
    return thresholds, fbeta_scores


def get_final_model(search_full_pipe, clf_th):
    preprocessing_pipe = Pipeline(search_full_pipe.steps[:-2])
    clf = utils.Classifier(base_classifier=search_full_pipe.steps[-1][1],
                           threshold=clf_th)
    final_model = make_pipeline(preprocessing_pipe, clf)
    return final_model


def explainer_construction(search_full_pipe, clf_th, X_train, y_train):
    preprocessing_pipe = Pipeline(search_full_pipe.steps[:-2])
    clf = utils.Classifier(base_classifier=search_full_pipe.steps[-1][1],
                           threshold=clf_th)
    X_train_preprocessed = preprocessing_pipe.fit_transform(X_train, y_train)
    explainer = shap.Explainer(clf.predict,
                               X_train_preprocessed,
                               feature_names=list(X_train.columns))
    return explainer


def save_objects_as_pkl(dict_of_objects):
    objects = list(dict_of_objects.keys())
    pathes = list(dict_of_objects.values())
    for i in range(len(objects)):
        pickle_out = open(pathes[i], 'wb')
        pickle.dump(objects[i], pickle_out)
        pickle_out.close()


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == '-subset':
        GRAPHS = False
        try:
            percentage_to_drop = 1 - float(args[1])
        except ValueError:
            print('Please enter a valid percentage of the subset to drop. Hint: it should be a float !! ')
    elif len(args) == 4 and args[2] == '-graphs' and args[3] == 'allow':
        GRAPHS = True
        try:
            percentage_to_drop = 1 - float(args[1])
        except ValueError:
            print('Please enter a valid percentage of the subset to drop. Hint: it should be a float !! ')
    elif len(args) == 0:
        GRAPHS = True
        percentage_to_drop = 0

    # Import data
    app_train = csv_clean_data_import(filename='../storage/application_train.csv',
                                      cols_to_drop=['SK_ID_CURR'])

    # Use only a subset as the original dataset is too large
    app_train = take_a_stratified_subset(data=app_train,
                                         percentage=percentage_to_drop,
                                         stratify=list(app_train.TARGET))

    # Train test split
    X_train, y_train, X_test, y_test = train_test_construction(data=app_train,
                                                               test_size=0.2,
                                                               stratify=list(app_train.TARGET))

    # Creating a RandomizedSearchCV and finding best parameters
    search = create_search()
    search.fit(X_train, y_train)

    # Visualizing ROC AUC performances
    if GRAPHS:
        plot_roc_curve(trained_estimator=search.best_estimator_,
                       X_test=X_test,
                       y_test=y_test)

    # Finding probability threshold to get F-3 best score
    thresholds, fbeta_scores = get_f3_scores_list(trained_estimator=search.best_estimator_,
                                                  X_test=X_test,
                                                  y_test=y_test,
                                                  resolution=101)
    threshold = thresholds[np.argmax(fbeta_scores)]
    if GRAPHS:
        plot_f3_score(thresholds, fbeta_scores)

    # The final model with RSCV parameters and the new probability threshold
    final_model = get_final_model(search_full_pipe=search.best_estimator_, clf_th=threshold)

    # Constructing the SHAP explainer that will be used in the dashboard
    explainer = explainer_construction(search_full_pipe=search.best_estimator_,
                                       clf_th=threshold,
                                       X_train=X_train,
                                       y_train=y_train)

    # Saving the final-complete-model and the explainer
    save_objects_as_pkl({Pipeline(search.best_estimator_.steps[:-2]): '../storage/preprocessing_pipe.pkl',
                         final_model:                                 '../storage/final_model.pkl',
                         explainer:                                   '../storage/explainer.pkl',
                         })



if __name__ == '__main__':
    main()
