from bottle import *
import pandas as pd
import numpy as np
from model import *

@route('/')
def index():
    return template('index')


@post('/dataset1')
def index():
    age = float(request.POST.getunicode('age'))
    sex = float(request.POST.getunicode('sex'))
    cp = float(request.POST.getunicode('cp'))
    restbp = float(request.POST.getunicode('restbp'))
    chol = float(request.POST.getunicode('chol'))
    fbs = float(request.POST.getunicode('fbs'))
    restecg = float(request.POST.getunicode('restecg'))
    thalach = float(request.POST.getunicode('thalach'))
    exang = float(request.POST.getunicode('exang'))
    oldpeak = float(request.POST.getunicode('oldpeak'))
    slope = float(request.POST.getunicode('slope'))
    ca = float(request.POST.getunicode('ca'))
    thal = float(request.POST.getunicode('thal'))
    # num = request.POST.getunicode('num')
    model = request.POST.getunicode('model')
    data = [age, sex, cp, restbp, chol, fbs, restecg, thalach, exang,
            oldpeak, slope, ca, thal]
    if model == "logistic":
        return predict(logistic_regression(data))
    elif model == "naive":
        return predict(naive_bayes(data))
    elif model == "svm":
        return predict(SVM(data))
    else:
        return predict(True)
    # return "hhh"


def predict(result):
    if result is True:
        return "It's probably a heart disease"
    else:
        return "probably no heart disease"


def svm(input):
    return True


def xgboost(input):
    return True


run(host='localhost', port=8080, debug=True)


