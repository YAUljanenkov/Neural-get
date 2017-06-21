
# coding: utf-8

# In[703]:

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
import xgboost
from sklearn.metrics import roc_auc_score


# In[704]:

X1 = pd.read_csv('X_1.csv')
X2 = pd.read_csv('X_2.csv')
X3 = pd.read_csv('X_3.csv')
y1 = pd.read_csv('y_1.csv')
y2 = pd.read_csv('y_2.csv')
y3 = pd.read_csv('y_3.csv')


# In[714]:

X2 = X2.dropna()
X3 = X3.dropna()
X1 = X1.dropna()


# In[715]:

def get_y(names, lz):
    Y = [0 for i in range(len(names))]
    lz = list(map(lambda x: x[0], np.array(lz)))
    for i in range(len(lz)):
        lz[i] = lz[i].split()
        lz[i] = lz[i][:-1] if len(lz[i]) == 3 else lz[i]
        lz[i] = set(lz[i])
    for i in range(len(names)):
        name = set(names[i].split()[:-1])
        if name in lz:
            Y[i] = 1
    return Y


# In[716]:

names1 = np.array(X1['Как вас зовут?'])


# In[717]:

Y1 = get_y(names1, y1)


# In[718]:

names2 = np.array(X2['ФИО'])


# In[719]:

Y2 = get_y(names2, y2)


# In[720]:

names3 = np.array(X3['ФИО'])


# In[721]:

y3 = list(map(lambda x: [x[0][:-1]] if x[0][-1] == '+' else x, np.array(y3)))
y3 = list(map(lambda x: ' '.join(x[0].split()), y3))


# In[722]:

yt3 = pd.DataFrame()
yt3['zach'] = y3
y3 = yt3


# In[723]:

Y3 = get_y(names3, y3)


# In[724]:

Y = Y2 + Y3 + Y1


# In[725]:

X = pd.DataFrame()


# In[726]:

X['fio'] = list(X2['ФИО']) + list(X3['ФИО']) + list(X1['Как вас зовут?'])


# In[727]:

X['lang'] = list(X2['Какие языки программирования вы знаете? На каком уровне? Если у вас уже есть опыт в ML, опишите его, пожалуйста.']) + list(X3['Какие языки программирования вы знаете? На каком уровне? Если у вас уже есть опыт в ML & IoT, опишите его, пожалуйста.']) + list(X1['Какие языки программирования вы знаете? Какие проекты в области программирования? Знаете ли вы Python?'])


# In[728]:

X['projects'] = list(X2['Какие проекты вы реализовали? Над какими работаете сейчас? Зачем вам участие в хакатоне?']) + list(X3['Какие проекты вы реализовали? Над какими работаете сейчас? Зачем вам эта школа?']) + list(X1['Какие языки программирования вы знаете? Какие проекты в области программирования и смежных вы реализовали? Знаете ли вы Python?']) 


# In[729]:

X['competetions'] = list(X2['Участвовали ли вы в олимпиадах, конкурсах, хакатонах в области ИТ? В Каких?']) + list(X3['Участвовали ли вы в олимпиадах, конкурсах, хакатонах в области ИТ? В Каких?']) + list(X1['Участвовали ли вы раньше в школах/олимпиадах/конкурсах и хакатонах? В каких?'])


# In[730]:

def norm(x):
    x = x.split('\n')
    x = ' '.join(x)
    x = x.split('\t')
    x = ' '.join(x)
    x = x.split('(')
    x = ' '.join(x)
    x = x.split(')')
    x = ' '.join(x)
    x = x.split('/')
    x = ' '.join(x)
    x = x.split('\\')
    x = ' '.join(x)
    return x


# In[731]:

X.lang = list(map(lambda x: norm(x), X.lang))


# In[732]:

X.competetions = list(map(lambda x: norm(x), X.competetions))


# In[733]:

X.projects = list(map(lambda x: norm(x), X.projects))


# In[734]:

X = list(np.array(X))


# In[735]:

X = pd.DataFrame(X)


# In[736]:

from sklearn.feature_extraction.text import TfidfVectorizer


# In[737]:

vec = TfidfVectorizer()
_ = vec.fit(list(X[1]) + list(X[2]) + list(X[3]))


# In[738]:

X_projects = vec.transform(X[1])
X_compet = vec.transform(X[2])
X_lang = vec.transform(X[3])


# In[739]:

xptr, xpt, ytr, yt = train_test_split(X_projects, Y, test_size=0.3, random_state=0)
xctr, xct, ytr, yt = train_test_split(X_compet, Y, test_size=0.3, random_state=0)
xltr, xlt, ytr, yt = train_test_split(X_lang, Y, test_size=0.3, random_state=0)


# In[740]:

rf1 = RandomForestClassifier(class_weight='balanced')
rf1.fit(xptr, ytr)


# In[741]:

roc_auc_score(yt, list(map(lambda x: x[1], rf1.predict_proba(xpt))))


# In[742]:

rf2 = RandomForestClassifier(n_estimators=11, max_depth=22, random_state=200, class_weight='balanced')
rf2.fit(xctr, ytr)


# In[743]:

roc_auc_score(yt, list(map(lambda x: x[1], rf2.predict_proba(xct))))


# In[744]:

rf3 = RandomForestClassifier(n_estimators=11, max_depth=22, random_state=200, class_weight='balanced')
rf3.fit(xltr, ytr)


# In[745]:

roc_auc_score(yt, list(map(lambda x: x[1], rf3.predict_proba(xlt))))


# In[746]:

xatr, xat, ytr, yt = train_test_split(list(map(lambda x: x if type(x) != 'string' else x.split()[0], list(X2['Возраст']))) + list(map(lambda x: int(x.split()[0]) if x != 'Егор' and x != '12.5' else 12, list(X3['Возраст']))), Y2+Y3, test_size=0.3)


# In[747]:

list(map(lambda x: x if type(x) != 'string' else x.split()[0], list(X2['Возраст']))) + list(map(lambda x: x.split()[0], list(X3['Возраст'])))


# In[748]:

from sklearn.linear_model import LogisticRegression


# In[749]:

xatr = [xatr]


# In[754]:

xatr.append(list(map(lambda x: len(x), list(np.array(X[1]) + np.array(X[2]) + np.array(X[3])))))


# In[768]:

len(xatr[0])


# In[764]:

df = pd.DataFrame()


# In[769]:

df['age'] = xatr[0]
df['lens'] = xatr[1][:484]


# In[805]:

df.shape, len(Y)


# In[806]:

X_train, X_test, Y_train, Y_test = train_test_split(df, Y[:484], test_size=0.3)


# In[808]:

rf4 = LogisticRegression(class_weight='balanced')
rf4.fit(X_train, np.array(Y_train))


# In[811]:

roc_auc_score(Y_test, np.array(list(map(lambda x: x[1], rf4.predict_proba(X_test)))))


# In[434]:

pred1 = np.array(list(map(lambda x: x[1], rf1.predict_proba(xpt))))
pred2 = np.array(list(map(lambda x: x[1], rf2.predict_proba(xct))))
pred3 = np.array(list(map(lambda x: x[1], rf3.predict_proba(xlt))))


# In[435]:

(pred1+pred2+pred3)/3


# In[812]:

def pred(projects, compet, lang, age):
    global vec
    global rf1
    global rf2
    global rf3
    global rf4
    df = pd.DataFrame()
    df['age'] = [age]
    df['lens'] = [len(projects) + len(compet) + len(lang)]
    projects = vec.transform([projects])
    compet = vec.transform([compet])
    lang = vec.transform([lang])
    pred1 = rf1.predict_proba(projects)[0][1]
    pred2 = rf2.predict_proba(compet)[0][1]
    pred3 = rf3.predict_proba(lang)[0][1]
    pred4 = rf4.predict_proba(df)[0][1]
    if df['lens'][0] < 20:
        return 0.0001
    return (pred1+pred2+pred3+pred4*2)/5


# In[820]:

pred('Автоматизирование школьной библиотеки', 'Школа IT рещений, goto camp 2016', 'Python, C++, C#, ', 14)


# In[121]:

grc = GridSearchCV(RandomForestClassifier(), {'max_depth':list(range(1, 50, 2)),
                                             'n_estimators':list(range(1, 40, 2))})
_ = grc.fit(xptr, ytr)


# In[122]:

grc.best_params_


# In[ ]:

grc = GridSearchCV(RandomForestClassifier(), {'random_state':list(range(1, 50, 2))}
_ = grc.fit(X_train, Y_train)


# In[124]:

pd.DataFrame(grc.cv_results_)

