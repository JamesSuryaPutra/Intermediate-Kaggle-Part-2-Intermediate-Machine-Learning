# 1st example: n_estimators
Typical values range from 100-1000, though this depends a lot on the learning_rate parameter discussed below.
Here is the code to set the number of models in the ensemble.

    my_model = XGBRegressor(n_estimators=500)
    my_model.fit(X_train, y_train)

    XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
                 colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                 early_stopping_rounds=None, enable_categorical=False,
                 eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                 importance_type=None, interaction_constraints='',
                 learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                 max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                 missing=nan, monotone_constraints='()', n_estimators=500, n_jobs=0,
                 num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
                 reg_lambda=1, ...)


# 2nd example: early_stopping_rounds
When using early_stopping_rounds, you also need to set aside some data for calculating the validation scores - this is done by setting the eval_set parameter.
We can modify the example above to include early stopping.

    my_model = XGBRegressor(n_estimators=500)
    my_model.fit(X_train, y_train, 
                 early_stopping_rounds=5, 
                 eval_set=[(X_valid, y_valid)],
                 verbose=False)

    /opt/conda/lib/python3.7/site-packages/xgboost/sklearn.py:797: UserWarning:
    `early_stopping_rounds` in `fit` method is deprecated for better compatibility with scikit-learn, use `early_stopping_rounds` in constructor or`set_params` instead.
    UserWarning,

    XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
                 colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                 early_stopping_rounds=None, enable_categorical=False,
                 eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                 importance_type=None, interaction_constraints='',
                 learning_rate=0.300000012, max_bin=256, max_cat_to_onehot=4,
                 max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                 missing=nan, monotone_constraints='()', n_estimators=500, n_jobs=0,
                 num_parallel_tree=1, predictor='auto', random_state=0, reg_alpha=0,
                 reg_lambda=1, ...)


# 3rd example: learning_rate
In general, a small learning rate and large number of estimators will yield more accurate XGBoost models, though it will also take the model longer to train since it does more iterations
through the cycle. As default, XGBoost sets learning_rate=0.1.
Modifying the example above to change the learning rate yields the following code.

    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
    my_model.fit(X_train, y_train, 
             early_stopping_rounds=5, 
             eval_set=[(X_valid, y_valid)], 
             verbose=False)

    XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
                 colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                 early_stopping_rounds=None, enable_categorical=False,
                 eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                 importance_type=None, interaction_constraints='',
                 learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
                 max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                 missing=nan, monotone_constraints='()', n_estimators=1000,
                 n_jobs=0, num_parallel_tree=1, predictor='auto', random_state=0,
                 reg_alpha=0, reg_lambda=1, ...)


# 4th example: n_jobs
Here's the modified example.

    my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4)
    my_model.fit(X_train, y_train, 
                 early_stopping_rounds=5, 
                 eval_set=[(X_valid, y_valid)], 
                 verbose=False)

    XGBRegressor(base_score=0.5, booster='gbtree', callbacks=None,
                 colsample_bylevel=1, colsample_bynode=1, colsample_bytree=1,
                 early_stopping_rounds=None, enable_categorical=False,
                 eval_metric=None, gamma=0, gpu_id=-1, grow_policy='depthwise',
                 importance_type=None, interaction_constraints='',
                 learning_rate=0.05, max_bin=256, max_cat_to_onehot=4,
                 max_delta_step=0, max_depth=6, max_leaves=0, min_child_weight=1,
                 missing=nan, monotone_constraints='()', n_estimators=1000,
                 n_jobs=4, num_parallel_tree=1, predictor='auto', random_state=0,
                 reg_alpha=0, reg_lambda=1, ...)

