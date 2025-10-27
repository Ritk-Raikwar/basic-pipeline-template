import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from hyperopt import fmin, tpe, hp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# reading data from csv

def read_csv(file_path):
	return pd.read_csv(file_path)

def create_feature(data):
    #no new features for iris dataset
    return data

# training a classifier model
def train_classifier(data):
	X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42)
	model = RandomForestClassifier()
	model.fit(X_train,y_train)
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test,y_pred)

	return model,accuracy

# hyperparameter tuning with hyperopt
def objective(params):
	model = RandomForestClassifier(**params)
	score = cross_val_score(model,X,y,cv=5).mean()
	return -score

def evaluate_model(model,X_test,y_test):
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test,y_pred)
	return accuracy


if __name__ == '__main__':
	#file_path = "/data/iris/iris.csv"
    file_path = "data/iris_dataset.csv"
    data = read_csv(file_path)
    data = create_feature(data)
	
    #split features and target
    X = data.drop('species',axis=1)
    y = data['species']
	
    #split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)
    
    # define pipeline
    pipeline = Pipeline(
		[
			('preprocessor', ColumnTransformer(
                transformers=[
					('num', StandardScaler(), X.columns)
	            ],
                remainder='passthrough'
            )),
			('classifier', RandomForestClassifier())
        ]
    )
    # train model
    pipeline.fit(X_train, y_train)
    
    # evaluate model
    accuracy = evaluate_model(pipeline, X_test, y_test)
    print(f"Model accuracy: {accuracy}")
	
    # hyperparameter tuning (commented out for simplicity)
    space = {
        'n_estimators': hp.choice('n_estimators', [50, 100, 200]),
        'max_depth': hp.choice('max_depth', [None, 10, 20, 30])
    }
	
    best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)
    print(f"Best hyperparameters: {best}")