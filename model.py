import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

def predict():
    TRAIN_PATH = "internship_train.csv"
    TEST_PATH = "internship_hidden_test.csv"
    PREDICTIONS_OUTPUT_PATH = 'model_predictions.csv'
    
    ''' read train and test data files '''
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)


    ''' train data set mean normalization '''
    X = train.iloc[:,:-1]
    X_norm = (X - X.mean()) / X.std()
    X_train = X_norm.to_numpy()
    y_train = train.iloc[:,-1].to_numpy()


    ''' test dataset mean normalization '''
    X_test = test.to_numpy()
    X_test = (X_test - X_test.mean()) / X_test.std()


    ''' best model - Decision Tree Regressor '''
    dtr = DecisionTreeRegressor().fit(X_train, y_train)
    predictions = dtr.predict(X_test)
    print("Model prediction completed")

    ''' save model predictions to .csv file'''
    predictions = pd.DataFrame(predictions)
    predictions.to_csv(PREDICTIONS_OUTPUT_PATH, index=False)
    print("File with model predictions saved")


if __name__ == "__main__":
    predict()