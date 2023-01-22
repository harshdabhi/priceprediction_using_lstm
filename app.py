from flask import Flask,render_template,request
import yfinance as yf
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.layers import LSTM,Dense
from keras.models import Sequential
from PIL import Image
import base64
import io



app=Flask(__name__)


@app.route('/',methods=['GET','POST'])
def basepage():
    try:
        os.remove('/images/image_file1.png')
        os.remove('/images/image_file2.png')
        os.remove('/images/image_file3.png')
        os.remove('/images/image_file4.png')
        os.remove('/images/image_file5.png')
       

    except:
        pass

    return render_template("index_priceprediction.html")


@app.route('/projection', methods=['GET','POST'])
def results_page():
    if request.method=='POST':


        # retriving data from html page

        assest=request.form['assest']
        interval=request.form['interval']       
        password=request.form['password']


        if password=='1234':
             
            #defining interval caveats 

            if interval in ['15m','30m']:
                period=60
            else:
                period=360


            data=yf.download(tickers=assest,start=pd.date_range(end=datetime.date.today(),periods=period)[0],interval=interval)

            df=data.filter(['Close'])
            df=df.values

            #get the number of closing data and retrieve 80% of data for training model

            training_data_len=int(df.shape[0]*0.8)


            #scaling data using MinMax scaler 
            
            scaler=MinMaxScaler(feature_range=(0,1))
            scaler_data=scaler.fit_transform(df)
           

            #Create the train scaled dataset

            train_data=scaler_data[0:training_data_len,]

            #split data into training and test data for model
            X_train=[]
            y_train=[]

            for i in range(60,len(train_data)):
                X_train.append(train_data[i-60:i,0])
                y_train.append(train_data[i,0])

            #convert the X_train and y_train to numpy arrays
            X_train,y_train=np.array(X_train),np.array(y_train)

            #Reshape the data 
            X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))
            
            #build the model
            model=Sequential()
            model.add(LSTM(50,return_sequences=True,input_shape=(X_train.shape[1],1)))
            model.add(LSTM(50,return_sequences=False))
            model.add(Dense(25))
            model.add(Dense(1))

            #complining model

            model.compile(optimizer='adam',loss='mean_squared_error')

            #train the model
            model.fit(X_train,y_train,batch_size=1,epochs=1)


            df1=yf.download(tickers=assest,start=pd.date_range(end=datetime.date.today(),periods=period)[0],interval=interval)
            #number of previous days data input 
            data1=df1.filter(['Close'])[-61:]
            df2=scaler.transform(data1.values)
            df2=df2.reshape(1,-1)

            temp_input=list(df2)
            temp_input=temp_input[0].tolist()

            #prediction for next 30 days

            list_output=[]
            n_steps=60

            j=0
            while(j<30):

                if(len(temp_input)>60):

                    X1_input=np.array(temp_input[1:])

                    X1_input=X1_input.reshape(1,-1)

                    X1_input=X1_input.reshape((1,n_steps,1))

                    y1_pred=model.predict(X1_input,verbose=0)
                    temp_input.extend(y1_pred[0].tolist())
                    temp_input=temp_input[1:]
                    list_output.extend(y1_pred.tolist())
                    j=j+1


                else:
                    X1_input=X1_input.reshape((1,n_steps,1))
                    y1_pred=model.predict(X1_input,verbose=0)
                    temp_input.extend(y1_pred[0].tolist())
                    temp_input=temp_input[1:]
                    list_output.extend(y1_pred.tolist())
                    j=j+1



            day_new=np.arange(1,61)
            day_pred=np.arange(61,91)

            df3=df2.tolist()[0]
            df3.extend(list_output)

            fig=plt.figure(figsize=(30,20))
            plt.plot(day_new,data1.Close[-60:])
            plt.plot(day_pred,scaler.inverse_transform(list_output))
            fg = str(np.random.randint(1, 5))

            fig.savefig(f'images/image_file{fg}.png')



            im=Image.open(f'images/image_file{fg}.png')
            data=io.BytesIO()
            im.save(data,'png')
            encoded_img_data=base64.b64encode(data.getvalue())    

    
        else:
            return render_template('index_priceprediction')
      


    return render_template("result_prediction.html",img_data=encoded_img_data.decode('utf-8'),assest=assest,interval=interval)





if __name__=="__main__":
    app.run()