import joblib as jb
import numpy as np
import pandas as pd 

class  Churn ( object ) :
    
    def __init__(self):
        self.enc0 = jb.load('/home/antoniojunior/Documentos/flow_employed/model/enc0.pkl')
        self.enc1 = jb.load('/home/antoniojunior/Documentos/flow_employed/model/enc1.pkl')
        self.enc2 = jb.load('/home/antoniojunior/Documentos/flow_employed/model/enc2.pkl')
        self.enc3 = jb.load('/home/antoniojunior/Documentos/flow_employed/model/enc3.pkl')
        self.enc4 = jb.load('/home/antoniojunior/Documentos/flow_employed/model/enc4.pkl')
        
        self.enc_o = jb.load('/home/antoniojunior/Documentos/flow_employed/model/enc_o.pkl')
        
    
    def tratamento(self,df):

        df['func_idade_log'] = np.log(df['func_idade'])
        df['contrato_horastrabalho_log'] = np.log(df['contrato_horastrabalho'] + 1)
        df['contrato_salario_log'] = np.log10(df['contrato_salario'])
        
        df['func_deficiencia'] = df['func_deficiencia'].map({'NAO': 0, 'SIM': 1})
        df['func_sexo'] = df['func_sexo'].map({'FEMININO': 0, 'MASCULINO': 1})

        df = df.drop(columns = ['contrato_horastrabalho_log'])

        
        df = df.drop(columns = ['func_idade' ,'contrato_salario_log'])
        

        df['contrato_horastrabalho'].loc[df['contrato_horastrabalho'] == 0] =df['contrato_horastrabalho'].loc[df['contrato_horastrabalho'] > 0].mean()

        df['contrato_horastrabalho'] = pd.cut(df["contrato_horastrabalho"], bins= np.linspace(0.957,44.0,13))
     
        #df = df.drop(columns='turnover_apos_1_ano')

        return df

    def Data_labelEncoder(self,df):
        ind1 = df.columns[df.dtypes == 'object']
        ind2 = df.columns[df.dtypes == 'category']
        
        
        df[ind1[0]] =self.enc0.fit_transform(df[ind1[0]].values) #func_racacor
        df[ind1[1]] =self.enc1.fit_transform(df[ind1[1]].values) #func_escolaridade
        df[ind1[2]] =self.enc2.fit_transform(df[ind1[2]].values) #func_uf
        df[ind1[3]] =self.enc3.fit_transform(df[ind1[3]].values) #empresa_porte
        df[ind1[4]] =self.enc4.fit_transform(df[ind1[4]].values) #empresa_setor
        
        df[ind2[0]] =self.enc_o.fit_transform(df[ind2[0]]) #empresa_setor
        
        return df

