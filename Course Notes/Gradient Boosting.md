# Gradient boosting
We begin by loading the training and validation data in X_train, X_valid, y_train, and y_valid.

    import pandas as pd
    from sklearn.model_selection import train_test_split

    # Read the data
    data = pd.read_csv('../input/melbourne-housing-snapshot/melb_data.csv')

    # Select subset of predictors
    cols_to_use = ['Rooms', 'Distance', 'Landsize', 'BuildingArea', 'YearBuilt']
    X = data[cols_to_use]

    # Select target
    y = data.Price

    # Separate data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y)


In this example, you'll work with the XGBoost library. XGBoost stands for extreme gradient boosting, which is an implementation of gradient boosting with several additional features
focused on performance and speed (scikit-learn has another version of gradient boosting, but XGBoost has some technical advantages).

In the next code cell, we import the scikit-learn API for XGBoost (xgboost.XGBRegressor). This allows us to build and fit a model just as we would in scikit-learn. As you'll see in the
output, the XGBRegressor class has many tunable parameters -- you'll learn about those soon!

    from xgboost import XGBRegressor

    my_model = XGBRegressor()
    my_model.fit(X_train, y_train)

    XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
                 colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                 early_stopping_rounds=None, enable_categorical=False,
                 eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                 importance_type=None, interaction_constraints='',
                 learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                 max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                 missing=nan, monotone_constraints='()', n_estimators=100, n_jobs=0,
                 num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
                 reg_lambda=1, ...)


We also make predictions and evaluate the model.

    from sklearn.metrics import mean_absolute_error

    predictions = my_model.predict(X_valid)
    print("Mean Absolute Error: " + str(mean_absolute_error(predictions, y_valid)))

    Mean Absolute Error: 241041.5160392121

