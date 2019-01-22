# load modules
import sys
import pandas as pd
import numpy as np
# model validation
from sklearn.model_selection import ShuffleSplit
# transform target variable
from sklearn.compose import TransformedTargetRegressor
# model: Lasso-regularized linear regression
from sklearn.linear_model import LassoCV
# performance metrics
from sklearn.metrics import mean_absolute_error as mae


def load_data(file_path: str) -> pd.DataFrame:
    '''Returns loaded dataset from CSV file'''

    dataset = pd.read_csv(file_path, sep=',')
    return dataset


def check_missing(dataset: pd.DataFrame) -> pd.DataFrame:
    '''Returns data frame with missing values (can be empty)'''

    missing_data = dataset[dataset.isna().any(axis='columns')]
    return missing_data


def preprocess_data(dataset: pd.DataFrame,
                    random_state: int) -> (pd.DataFrame, pd.Series):
    '''Returns preprocessed predictors (X) and target variable (y)'''

    # remove rows with missing values
    dataset.dropna(axis='rows', inplace=True)

    # drop useless 'instant' feature, drop casual and registered counts
    dataset.drop(['instant', 'casual', 'registered'], axis='columns', inplace=True)

    # strip day, add it to frame
    dataset['dteday'] = dataset['dteday'].apply(lambda x: int(x[-2:]))

    # replace feeling temperature with temperature difference
    location = len(dataset.columns) - 1
    value = dataset['temp'] - dataset['atemp']
    dataset.insert(loc=location, column='temp_diff', value=value)
    dataset.drop('atemp', axis='columns', inplace=True)

    # fix categorical variables data type
    cats = ['dteday', 'season', 'yr', 'mnth', 'hr',
            'holiday', 'weekday', 'workingday', 'weathersit']
    nums = ['temp', 'temp_diff', 'windspeed', 'hum']
    X_cats = dataset[cats].astype('category')
    X = pd.concat([X_cats, dataset[nums]], axis='columns')

    # encode categorical predictors
    X = pd.get_dummies(X, drop_first=True)
    # target variable
    y = dataset['cnt']

    return (X, y)


def build_model(X: pd.DataFrame, y: pd.Series, random_state: int,
                y_transformer=np.log1p, y_inverse_transformer=np.expm1,
                cv_splits=5, n_jobs=-1):
    '''Build linear regression model with Lasso regularization.
       Uses all available processors (n_jobs=-1).
       Returns sklearn fitted model object.'''

    # intialize cross-validation generator to 5-fold CV with data shuffling
    cv_generator = ShuffleSplit(n_splits=cv_splits, test_size=0.25,
                                random_state=random_state)

    # instantiate regressor
    regr = LassoCV(fit_intercept=True, normalize=True,
                   cv=cv_generator, random_state=random_state)

    # apply logarithmic transformation to target variable, y -> ln(y + 1)
    model = TransformedTargetRegressor(regressor=regr,
                                       func=y_transformer,
                                       inverse_func=y_inverse_transformer)

    # fit model
    model.fit(X, y)
    # get parameters
    params = model.regressor_.get_params()

    # mean absolute error on whole dataset
    y_pred = model.predict(X)
    mae_model = mae(y_pred, y)

    return (params, model, mae_model)


def main(file_path, rand):
    # load dataset
    counts = load_data(file_path)

    # check missing data
    if check_missing(dataset=counts).empty is False:
        print('Missing {} rows in dataset.'.format(check_missing.count()))
        print('Missing rows will be removed from the dataset.\n')

    # get preprocessed predictors and target
    X, y = preprocess_data(dataset=counts, random_state=rand)

    # build model
    params, model, mae_model = build_model(X, y, random_state=rand)

    return (params, model, mae_model)


# ==== run main script ====
if __name__ == "__main__":
    # two inputs must be supplied:
    # data file path and random number for reproducibility
    # file_path = path to hour.csv file
    file_path = sys.argv[1]
    rand = int(sys.argv[2])

    params, model, mae_model = main(file_path, rand)

    # print model parameters
    print(('**** Lasso Regression Model fitted with parameters: ****'))
    for p, value in params.items():
        print(p, ': ', value)

    print('**** Mean Absolute Error on dataset: {:.0f} counts'.format(mae_model))
