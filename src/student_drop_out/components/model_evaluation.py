import os
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from student_drop_out.utils.common import save_json
from urllib.parse import urlparse
import numpy as np
import joblib
from student_drop_out.entity.config_entity import ModelEvaluationConfig
from pathlib import Path


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)
  
        test_data =test_data.drop(['StudentID','FirstName','FamilyName'], axis=1)            
        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]  

        numeric_transformer = StandardScaler()
        oh_transformer = OneHotEncoder()
        
        num_features= ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
        oh_columns =  ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']
        
        preprocessor = ColumnTransformer(
            [
                ("OneHotEncoder", oh_transformer, oh_columns),                     
                ("StandardScaler", numeric_transformer, num_features)
            ]
        )
 
        processed = preprocessor.fit(test_x) 
        test_x = pd.DataFrame(processed.transform(test_x)) 

        predicted_qualities = model.predict(test_x)

        (rmse, mae, r2) = self.eval_metrics(test_y, predicted_qualities)
        
        # Saving metrics as local
        scores = {"rmse": rmse, "mae": mae, "r2": r2}
        save_json(path=Path(self.config.metric_file_name), data=scores)
