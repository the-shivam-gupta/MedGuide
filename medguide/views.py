from flask import Blueprint,render_template, request
import dill
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor

views = Blueprint('views',__name__)


@views.route('/', methods=['GET','POST'])
def index():
        if request.method == 'POST':
                age = request.form.get('age')
                sex = request.form.get('sex')
                symptom = request.form.get('symptom')
                 

                df = pd.read_csv('notebook/data/disease_with_132_features.csv')
                df_drop_target = df.drop(['prognosis'],axis =1)
                numerical_columns = np.array(df_drop_target.columns)
                            
                input_values = {
                        "itching" : [0],
                        "skin_rash" : [0],
                        "nodal_skin_eruptions" : [0],
                        "inflammatory_nails" : [0],
                        "chills" : [0],
                        "yellow_crust_ooze": [0],
                        "blister": [0]
                }

                default_input_values = {feature :[0] for feature in numerical_columns}

                # Updating the default input values with the symptoms that are given by the patients.
                default_input_values.update(input_values)
                
                # converting from 2d to 1d
                all_values = [value[0] for value in default_input_values.values()]

                if 1 not in all_values:
                        print("Enter atleast one symptom:")
                # else:
                input_df = pd.DataFrame(default_input_values)

                model = dill.load(open('artifacts/preprocessor_v1.pkl', 'rb'))
                final_model = dill.load(open('artifacts/best_model_v1.pkl', 'rb'))
                
                # Transform the input data
                data_scaled = model.transform(input_df)

                # Evaluate the performance of the model
                # print(data_scaled)

                # Make predictions
                pred = final_model.predict(data_scaled)

                diseases = get_diseases()

                index = np.where(pred[0] > 0.7)[0][0]
                
                # print(type(symptom))
                return f"{symptom} type : {type(symptom)}"
                # return render_template(
                #         'possible_disease.html', 
                #         title="Possible disease",
                #         possible_disease = diseases[index]
                #         )
                

        
        return render_template('tell_symptoms.html', title="Possible disease")


def get_diseases():

        train_df = pd.read_csv('artifacts/train.csv')

        target_column_name = "prognosis"
        
        # Separate input and target features for training and testing datasets
        input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
        target_feature_train_df = train_df[[target_column_name]]

        # Apply one-hot encoding to the target feature
        one = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')

        output_feature_train_df = one.fit_transform(target_feature_train_df)

        # Decoding the disease column
        diseases = np.array([column for column in  output_feature_train_df.columns])

        return diseases
    



