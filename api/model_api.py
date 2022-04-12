from flask import Flask,request
import joblib as jb
import pandas as pd
from churn.Churn import Churn



# Inicializando o Flask
app = Flask(__name__)

 
# Carregando o modelo
model = jb.load('/home/antoniojunior/Documentos/flow_employed/model/prf.pkl')


@app.route('/Churn/predtic',methods=['POST'])
def churn_predtic():
    Dados_api = request.get_json() #Dados de entrada do API
    

# Coletando os dados e enviando para o API
    if Dados_api:
        if isinstance(Dados_api,dict): # Quando a entrada for apenas uma linha
            df_raw = pd.DataFrame(Dados_api,index=[0])
        else : #Quando os dados forem diversas linhas
            df_raw = pd.DataFrame(Dados_api,columns =Dados_api[0].keys())
            
# Tratamento dos dados

    c = Churn()

    df = c.tratamento(df_raw)
    df = c.Data_labelEncoder(df)


# Predições 
    pred = model.predict_proba(df)[:,1]
    
    
# Retornando a predição para o cliente
    df_raw['Pred'] = pred

    return df_raw.to_json(orient = 'records')



if __name__=='__main__':
    # start api
    app.run(host='0.0.0.0',port='5000')
   