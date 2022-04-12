import pandas as pd 
from xgboost import XGBClassifier
from imblearn.under_sampling import TomekLinks
import joblib as jb
from sklearn.preprocessing import LabelEncoder
import numpy as np

def encoder_var(df,flag):
  if flag == 1:
    ind = df.columns[df.dtypes == 'object']
  if flag == 0:
     ind = df.columns[df.dtypes == 'category']

  for i in range(len(ind)):
    enc = LabelEncoder()
    inteiros = enc.fit_transform(df[ind[i]])
    df[ind[i]] = inteiros

  return df


def tratamento(df, flag):

  df['func_idade_log'] = np.log(df['func_idade'])
  df['contrato_horastrabalho_log'] = np.log(df['contrato_horastrabalho'] + 1)
  df['contrato_salario_log'] = np.log10(df['contrato_salario'])
  if flag == 1:
    df['turnover_apos_1_ano'] = df['turnover_apos_1_ano'].map({'NAO': 0, 'SIM': 1})
  
  df['func_deficiencia'] = df['func_deficiencia'].map({'NAO': 0, 'SIM': 1})
  df['func_sexo'] = df['func_sexo'].map({'FEMININO': 0, 'MASCULINO': 1})

  df = df.drop(columns = ['contrato_horastrabalho_log'])

 # df = encoder_var(df,1)

  col2 = ['func_idade' ,'contrato_salario_log']
  df = df.drop(columns = col2)

  df['contrato_horastrabalho'].loc[df['contrato_horastrabalho'] == 0] =df['contrato_horastrabalho'].loc[df['contrato_horastrabalho'] > 0].mean()

  df['contrato_horastrabalho']= pd.cut(df["contrato_horastrabalho"], bins= np.linspace(0.957,44.0,13))


  #df['contrato_salario_sqrt']= df['contrato_salario']**(1/3)

  #df = encoder_var(df,0)

  return df

df = pd.read_csv('~/Documentos/Churn/Dados/treino.csv',encoding = "UTF-8", sep = ";",decimal = ',',dayfirst = True)

df = tratamento(df, 1)

ind1 = df.columns[df.dtypes == 'object']
ind2 = df.columns[df.dtypes == 'category']

enc = LabelEncoder()

enc0 = LabelEncoder()
enc0.fit(df[ind1[0]]) #func_racacor
jb.dump(enc0, "enc0.pkl")

enc1 = LabelEncoder()
enc1.fit(df[ind1[1]]) #func_racacor
jb.dump(enc1, "enc1.pkl")

enc2 = LabelEncoder()
enc2.fit(df[ind1[2]]) #func_racacor
jb.dump(enc2, "enc2.pkl")

enc3 = LabelEncoder()
enc3.fit(df[ind1[3]]) #func_racacor
jb.dump(enc3, "enc3.pkl")

enc4 = LabelEncoder()
enc4.fit(df[ind1[4]]) #func_racacor
jb.dump(enc4, "enc4.pkl")


enc_o = LabelEncoder()
enc_o.fit(df[ind2[0]]) #func_racacor
jb.dump(enc_o, "enc_o.pkl")

df[ind1[0]] =enc0.fit_transform(df[ind1[0]]) #func_racacor
df[ind1[1]] =enc1.fit_transform(df[ind1[1]]) #func_escolaridade
df[ind1[2]] =enc2.fit_transform(df[ind1[2]]) #func_uf
df[ind1[3]] =enc3.fit_transform(df[ind1[3]]) #empresa_porte
df[ind1[4]] =enc4.fit_transform(df[ind1[4]]) #empresa_setor

df[ind2[0]] =enc_o.fit_transform(df[ind2[0]]) #empresa_setor


X_train  = df.drop(columns='turnover_apos_1_ano')
y_train = df['turnover_apos_1_ano']

lr = 0.027840476080879074
max_depth = 6
n_estimators = 669
colsample_bytree = 1
lamb             = 0.0009938152575176688
alpha            = 0.0
model = XGBClassifier(colsample_bytree = colsample_bytree, learning_rate = lr,
                       max_depth = max_depth, n_estimators = n_estimators,reg_lambda = lamb, reg_alpha = alpha,random_state=40)

# Balacendo as classes
smote            = TomekLinks()
X_smote, y_smote = smote.fit_resample(X_train, y_train)

prf = model.fit(X_smote, y_smote)

jb.dump(prf, "prf.pkl")








