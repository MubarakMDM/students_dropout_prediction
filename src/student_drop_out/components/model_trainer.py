import pandas as pd
import os
from student_drop_out import logger
from student_drop_out import exception
import sys
from sklearn.linear_model import ElasticNet
import joblib
from student_drop_out.entity.config_entity import ModelTrainerConfig

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, PowerTransformer
from sklearn.compose import ColumnTransformer

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)
 
        train_data =train_data.drop(['StudentID','FirstName','FamilyName'], axis=1)     
        test_data =test_data.drop(['StudentID','FirstName','FamilyName'], axis=1)               
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)

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

        processed = preprocessor.fit(train_x)
        processed = preprocessor.fit(test_x)
        train_x = pd.DataFrame(processed.transform(train_x))
        test_x = pd.DataFrame(processed.transform(test_x))

        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]

        lr = ElasticNet(alpha=self.config.alpha, l1_ratio=self.config.l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        joblib.dump(lr, os.path.join(self.config.root_dir, self.config.model_name))




    def get_data_transformer_object(self) -> Pipeline:
        """
        Method Name :   get_data_transformer_object
        Description :   This method creates and returns a data transformer object for the data
        
        Output      :   data transformer object is created and returned 
        On Failure  :   Write an exception log and then raise an exception
        """
        logger.info(
            "Entered get_data_transformer_object method of DataTransformation class"
        )

        try:
            logger.info("Got numerical cols from schema config")

            numeric_transformer = StandardScaler()
            oh_transformer = OneHotEncoder()
            #ordinal_encoder = OrdinalEncoder()

            logger.info("Initialized StandardScaler, OneHotEncoder, OrdinalEncoder")

            oh_columns = ['age', 'Medu', 'Fedu', 'traveltime', 'studytime', 'failures', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']
            num_features =  ['sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher', 'internet', 'romantic']

            #logger.info("Initialize PowerTransformer")

            # transform_pipe = Pipeline(steps=[
            #     ('transformer', PowerTransformer(method='yeo-johnson'))
            # ])
            preprocessor = ColumnTransformer(
                [
                    ("OneHotEncoder", oh_transformer, oh_columns),
                    #("Ordinal_Encoder", ordinal_encoder, or_columns),
                    #("Transformer", transform_pipe, transform_columns),
                    ("StandardScaler", numeric_transformer, num_features)
                ]
            )

            logger.info("Created preprocessor object from ColumnTransformer")

            logger.info(
                "Exited get_data_transformer_object method of DataTransformation class"
            )
            return preprocessor

        except Exception as e:
            raise e