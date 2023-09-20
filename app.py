from flask import Flask, render_template, request, jsonify
app = Flask(__name__)

initial = "Hey, this is crop recommendation chatbot. I can help you pick best crop so that you can yield more efficiently and can earn more."

@app.route("/")
def index():
    return render_template("chat.html")
k = 1
i = 0
flag = 0
user_input = []
new_user_input = []
@app.route("/get", methods=["POST"])
def chat():
    global user_input
    global new_user_input
    global i, flag, k
    msg = request.form["msg"]   
    if(k == 1 and (msg == "1" or msg== "2")):
        flag = int(msg)
        k = 0
    if(flag == 1):
        question_list = ["Enter your loaction:", "Enter the Nitrogen(N) content of soil: ", "Enter the Phosphorus(P) content of soil", "Enter the Potassium(K) content of soil", "Enter the temperature:", "Enter the humidity:","Enter the ph value:", "Enter the rainfall:" ]
        if(i < len(question_list)):
            user_input.append(msg)
            # print(user_input)
            msg = question_list[i]
            i+=1
            response = {'result': msg}
            return jsonify(response)
        else:
            user_input.append(msg)
            print(user_input[2:])
            #83, 45, 60, 28, 70.3, 7.0, 150.9
            user_input_array = np.array(user_input[2:])
            user_input_array = np.array([user_input_array])
            predicted_value = RF.predict(user_input_array.reshape(1, -1))
            predicted_value = {
                "result": "According to my prediction you must grow <b>{"+ predicted_value[0] + "}</b> this year.<br>Is there anything else I can help you with."
            }
            k = 1
            i = 0
            flag = 0
            return jsonify(predicted_value)
    elif(flag == 2):
        question_list = ["Enter the state: ", "Enter the district: ", "Enter the market: ", "Enter the commodity: ", "Enter the variety: ", "Enter the grade: ", "Enter the Weekday: ", "Enter the min price: ", "Enter the max price: "]
        if i <len(question_list):
            new_user_input.append(msg)
            msg = question_list[i]
            i+=1
            response = {'result': msg}
            return jsonify(response)
        else:
            new_user_input.append(msg)
            print(new_user_input)
            # State,District,Market,Commodity,Variety,Grade,Weekday,Min Price,Max Price,Modal Price
            # Gujarat,Amreli,Damnagar,Bhindi(Ladies Finger),Bhindi,FAQ,Thursday,4100,4500,4350
            user_input = pd.DataFrame({
                'State': [new_user_input[1]],
                'District': [new_user_input[2]],
                'Market': [new_user_input[3]],
                'Commodity': [new_user_input[4]],
                'Variety': [new_user_input[5]],
                'Grade': [new_user_input[6]],
                'Weekday': [new_user_input[7]],
                'Min Price': [new_user_input[8]],
                'Max Price': [new_user_input[9]]
            })
            # Preprocess user input and scale it
            user_input_preprocessed = preprocessor.transform(user_input)
            user_input_preprocessed = sc.transform(user_input_preprocessed)

            # Predict the crop price for user input
            predicted_price = regressor.predict(user_input_preprocessed)
            msg = f"Predicted Crop Price: { predicted_price[0]}"
            response = {'result': msg }
            new_user_input.clear()
            k = 1
            i = 0 
            flag = 0
            return jsonify(response)
        
    else:
        msg = '🌾 Welcome to the Crop Recommendation Chatbot! 🌾<br>I am here to help you make informed decisions about the best crops to plant based on atmospheric conditions and price predictions. Whether you are a seasoned farmer or just getting started, I have got you covered. <br><br>You can ask me questions like: <br>- Press 1: "What should I plant this season?" <br>- Press 2: "Can you recommend crops with good price predictions?" <br><br>Feel free to explore and ask anything related to crop recommendations and atmospheric conditions. Let us work together to maximize your agricultural yield! <br>'
        response = {'result' : msg}
        flag = 0
        return jsonify(response)

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv('Crop_recommendation.csv')
    features = df[['N', 'P','K','temperature', 'humidity', 'ph', 'rainfall']]
    target = df['label']
    labels = df['label']
    acc = []
    model = []
    from sklearn.model_selection import train_test_split
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(features,target,test_size = 0.2,random_state =2)
    from sklearn.ensemble import RandomForestClassifier
    global RF
    RF = RandomForestClassifier(n_estimators=20, random_state=0)
    RF.fit(Xtrain,Ytrain)


    nf = pd.read_csv('dataset.csv')
    # Split data into features (X) and target variable (y)
    X = nf.drop(columns=['Modal Price'])  # All columns except 'Sales' are features
    y = nf['Modal Price']

    print(X.shape)

    # Define the columns to be one-hot encoded
    categorical_cols = ['State','District','Market','Commodity','Variety','Grade','Weekday']

    # Create a column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_cols)
        ],
        remainder='passthrough'  # Pass through any other columns
    )

    # Apply the preprocessor to the feature columns (X)
    X_encoded = preprocessor.fit_transform(X)

    print(X_encoded.shape)

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.2, random_state = 0)

    sc = StandardScaler(with_mean=False)
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.ensemble import RandomForestRegressor
    regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)
    regressor.fit(X_train, y_train)






    try:
        app.run(debug=True)
    except Exception as e:
        print(f"An error occurred: {e}")