# Task: General template for ML/NN/AI projects in Python
____________________________________________________________________________________________________
Step 0: imports
    import pandas as pd
    import numpy as np 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import datetime
    import lightgbm as lgb
    import category_encoders as ce
    import itertools

    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_absolute_error
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.model_selection import cross_val_score 
    from slearn import metrics
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_selection import SelectFromModel



    from xgboost import XGBRegressor

    from scipy import stats

    from mlxtend.preprocessing import minimax_scaling
____________________________________________________________________________________________________
Step 1: load data, analyze, process
    # filepath
        data_path='path/file'

    # load data
        data = pd_read_csv(data_path)
        data = pd_read_csv(data_path, index_col='Id')
        data = pd_read_xlsx(data_path)

    # review loaded data
        print(data.columns())
        print(data.describe())
        print(data.head())
        print(data.tail())
        print(data.shape)
        print(data['column'].dtype)

    # get number of missing data points per columns
        missing_val_count = data.isnull().sum()
        print(missing_val_count)

        # missing values as percentage
            total_cells = np. product(data.shape)
            total_missing = missing_val_count.sum()
            percent_missing = (total_missing/total_cells)*100
            print(percent_missing)

        # drop missing values (think of na as "not available")
            data = data.dropna(axis=0)
            data = data.dropna(axis=0, subset=['Column1'], inplace=True)

        # replace all NA's with 0
            data.fillna(0)
        
        # replace all NA's with value that comes next
            data.fillna(method='bfill', axis=0).fillna(0)

    # scaling - change range of data
        # mix-max scale data between 0 and 1
            scaled_data = minmax_scaling(original_data, columns=[0])

        # normalization - change distribution of data (bell curve - Gaussian/Normal distribution)
            # normalize data with boxcox
                normalized_data = stats.boxcox(original_data)
________________________________________________________________________________________________
Step 2: Select prediction target (y) and features (X)
    # obtain prediction target (y)
        y = data.column_1
        y = data['column_1']

        # review y
            print(data.describe())
            print(data.head())
            print(data.tail()))

    # obtain features (X)
        data_features = ['column_2','column_3','column_4']
        X = data.[data_features]

    # use only one type of data (numerical)
        X = X.select_dtypes(exclude=['object'])
        X = X.select_dtypes(include=['number'])

        # review X
            print(data.describe())
            print(data.head())
            print(data.tail())

    # drop columns based on specified condition (i.e. column to be predicted by the model)
        data = data.query('column != "string_target"')

        # generate new column with 1/0 assigned based on string
            data = data.assign(outcome=(data['column'] == 'string_target').astype(int))



    # Missing values
        # number of missing values in each column of training data
            missing_val_by_column = (X_train.isnull().sum())
            print(missing_val_by_column[missing_val_by_column > 0])

        # drop columns with missing values
            # get columns with missing values
                    cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
                
            # drop columns in training and validation data
                    reduced_X_train = X_train.drop(cols_with_missing, axis=1)
                    reduced_X_valid = X_valid.drop(cols_with_missing, axis=1)
            
            # print MAE after dropping missing columns
                    print('MAE (drop columns with missing values: ')
                    print(score_dataset(reduced_X_train, reduced_X_valid, y_train, y_valid))

        # impute missing values
            # imputation
                my_imputer = SimpleImputer()
                my_imputer = SimpleImputer(strategy='median')
                imputed_X_train = pd.DataFrame(my_imputer.fit_transform(X_train))
                imputed_X_valid = pd.DataFrame(my_imputer.fit(X_valid))
        
            # putting back coluns names removed by imputation
                imputed_X_train = X_train.columns
                imputed_X_valid = X_valid.columns
            
            # print MAE after imputing missing columns
                print('MAE (impute columns with missing values: ')
                print(score_dataset(imputed_X_train, imputed_X_valid, y_train, y_valid))

        # impute missing values and keep track of imputed values (Extension to Imputation)
            # make copy to avoid changing original data
                X_train_plus = X_train.copy()
                X_valid_plus = X_valid.copy()

            # make new columnd indicated imputed values
                for col in cols_with_missing:
                    X_train_plus[col = '_was_missing'] = X_train_plus[col].isnull()
                    X_valid_plus[col = '_was_missing'] = X_valid_plus[col].isnull()
            # imputation
                my_imputer = SimpleImputer()
                imputed_X_train_plus = pd.DataFrame(my_imputer.fit_transform(X_train_plus))
                imputed_X_valid_plus = pd.DataFrame(my_imputer.fit_transform(X_valid_plus))
            
            # putting back coluns names removed by imputation
                imputed_X_train_plus.columns = X_train.columns
                imputed_X_valid_plus.columns = X_valid.columns
            
            # print MAE after imputing missing columns (Extension to Imputation)
                print('MAE (Extension to Imputation: ')
                print(score_dataset(imputed_X_train_plus, imputed_X_valid_plus, y_train, y_valid))

    # Prepare categorical data
        # get list of categorical variables
            s = (X_train.dtypes == 'object')
            object_cols = list(s[s].index)
            print('Categorical variables:')
            print(object_cols)

        # print unique entries in each column (cardinality)
            object_nunique = list(map(lambda col: X_train[col].nunique(), object_cols))
            d = dict(zip(object_cols, object_nunique))

            sorted(d.items(), key=lambda x: x[1])

        # drop categorical variables - if columns don't contain useful information
            drop_X_train - X_train.select_dtypes(exclude=['object'])
            drop_X_valid - X_valid.select_dtypes(exclude=['object'])

            # print MAE
                print('MAE (Drop categorical variables):')
                print(score_dataset(drop_X_train, drop_X_valid, y_train, y_valid))
                
        # label encoding - assign unique values to different integer
            # apply label encoder to each column with categorical data (random unique value to a different integer)
                label_encoder = LabelEncoder()
               
                for col in object_cols:
                    label_X_train[col] = label_encoder.fit_transform(X_train[col])
                    label_X_valid[col] = label_encoder.transform(X_valid[col])
                
                # print MAE
                    print('MAE (Label Encoding):')
                    print(score_dataset(label_X_train, label_X_valid, y_train, y_valid))
                
            # verification which columns can ba safely encoded (values present in both training and validation data)
                # all categorical columns
                    object_cols = [col for col in X_train.columns if X_train[col].dtype=='object']

                # column that can be safely encoded
                    good_label_cols = [col for col in object_cols if set(X_train[col]==set(X-valid[col])]
                    print('Categorica columns that will be encoded', good_label_cols)
                
                # problematic columns to be dropped from the dataset
                    bad_label_cols = list(set(objcet_cols)-set(good_labels_cols))
                    print('Categorical columns that will be dropped', bad_label_cols)

                # drop problematic columns
                    label_X_train = X_train.drop(bad_label_cols, axis=1)
                    label_X_valid = X_valid.drop(bad_label_cols, axis=1)

        # one-hot encoding - columns indicating the presence (or absence) of each possible value in the original data
            # apply one-hot encoder to each column with categorical data
                OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
                
                OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[object_cols]))
                OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[object_cols]))
            
            # placing index after one-hot encoding removed it
                OH_cols_train.index = X_train.index
                OH_cols_valid.index = X_valid.index
            
            # remove categorical columns
                num_X_train = X_train.drop(object_cols, axis=1)
                num_X_valid = X_valid.drop(object_cols, axis=1)

            # add one-hot encoded columns to numerical features
                OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
                OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)

                # print MAE
                    print('MAE (One-Hot Encoding):')
                    print(score_dataset(OH_X_train, OH_X_valid, y_train, y_valid))    

        # count encoding - replace categorial value with the number it appears
            cat_features = ['col1', 'col2', 'col3']

            # create encoder
                count_enc = ce.CountEncoder()
            
            # transform features
                count_encoded = count_enc.transform(data[cat_features])
            
            # rename columns with suffix and join with dataframe
                data = data.join(count_encoded.add_suffix('_count'))

            # train model with encoded features
                train, valid , test = get_data_splits(data)
                train_model(train, valid)

        # target encoding - replace categorical value with average value of the target for that value of the feature
            # learn target encodings ONLY from training set - DATA LEAKAGE
            # create encoder
                target_enc = ce.TargetEncoder(cols=cat_features)
                target_enc.fit(train[cat_features], train['col1'])
            
            # transform features
                train_target_enc = target_enc.transform(data[cat_features]).add_suffix('_target')
                valid_target_enc = valid.join(target_enc,transform(valid[cat_features]).add_suffix('_target'))

            # train model with encoded features
                train_model(train_target_enc, valid_target_enc)  

        # CatBoost encoding - target probability calculated for each row calculated from the rows before it
            # create encoder
                target_CB_enc = ce.CatBoostEncoder(cols=cat_features)
                target_enc.fit(train[cat_features], train['outcome'])

            # transform features
                train_CBE = train.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))
                valid_CBE = valid.join(target_enc.transform(train[cat_features]).add_suffix('_cb'))
 
            # train model
                train_model(train_CBE, valid_CBE)

    # Feature Generation
        # interactions
            interactions =  data['feature1'] + '_' + data['feature2']

            # label encode interaction feature
                label_enc = LabelEncoder()
                data_interaction = data.assign(feature1_feauture2=label_enc.fit_transform(interactions))

            # create with selected column as index value with rolling
                new_series = pd.Series(data.index, index=data.feature1, name='count_new').sort_index()
                count_time_period = new_series.rolling('window').count()-1

                # adjust the index to enable joining generated feature with existing training data
                    count_time_period.index = new_series.values
                    count_time_period = count_time_period.reindex(data.index)
                
                # join new feature with other data
                    data.join(count_time_period)
            
            # create features with timedeltas .groupby / .transform
                def time_delta(series):
                    return series.diff().dt.total_seconds()/3600
                
                df = data[['feature1','feature2']].sort_values('feature1')
                
                timedeltas = df.groupby('category').transform(time_delta)

                # fill NaNs with median and reset the index
                    timedeltas = timedeltas.fillna(timedeltas.median()).reindex(data.index)
                
            # transform numerical features - change distribution
                # plot original feature histogram
                    plt.hist(data.feature1, range=(0, 100000), bins=50);
                    plt.title('no transformation')

                # square-root transformation
                    data.feature1 = np.sqrt(data.feature1), range=(0, 400), bins=50);
                    plt.title('SQRT transformation')
                
                # logarithimic transformation
                    data.feature1 = np.log(data.feature1), range=(0, 25), bins=50);
                    plt.title('LOG transformation')
                       

    # Feature Selection - keep the most informative
        # univariate feature selection - how strongly target depends on the feature
            feature_cols = data.columns.drop('target')
            train, valid = get_data_splits(baseline_data)

            selector = SelectKBest(f_classif, k=5)

            X_new = selector.fit_transform(train[feature_cols], train['target'])

            # get selected features, returns dropped columns filled with zeros
                selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                index=train.index,
                columns=feature_cols)
            
            # update valid dataset, dropped columns have zero variance
                selected_columns = selected_features.columns[selected_features.var() != 0]
                valid[selected_columns].head()

            # choosing the best K value - fit multiple models with different values of K and loop over to note validation scores
        
        #  L1 regularization (Lasso) - penalizes the absolute magnitude of the coefficients
            #  as the strength of linearization is increased the less important features for prediction are set to 0
            # prepare sets
                train, valid, _ = get_data_splits(data)
                X, y = train[train.columns.drop('target')], train['target']

            # set regularization parameter C
                logistic = LogisticRegression(C=1,
                penalty='l1',
                solver='liblinear',
                random_state=7).fit(X, y)

                model = SelectFromModel(logistic, prefit=True)

                X_new = model.transform(X)

            # get selected features, returns dropped columns filled with zeros
                selected_features = pd.DataFrame(selector.inverse_transform(X_new),
                index=X.index,
                columns=X.columns)
            
            # update valid dataset, dropped columns have zero variance
                selected_columns = selected_features.columns[selected_features.var() != 0]


    # Search for values with condition
        searched = np.where([column == 1])[1]
        searched_data = data.loc[searched]
____________________________________________________________________________________________________
Step 3: Build model
    # generate training and validation data
        train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

        # for lightgbm(lgb)
            dtrain = lgb.Dataset(train[feature_cols], label=train['result'])
            dvalid = lgb.Dataset(valid[feature_cols], label=valid['result'])

    # simple training, validation data and test splits (80%-10%-10%)
        valid_fraction = 0.1
        valid_size = int(len(data)*valid_fraction)

        train = data[:-2*valid_size]
        valid = data[-2*valid_size:-valid_size]
        test = data[-valid_size:]

    # review data to be used
        print(train_X.describe())
        print(train_X.head())
        print(train_X.tail())

    # define type & import libraries
        data_model = DecisionTreeRegressor(random_state=1)
        data_model = RandomForestRegressor(random_state=1)
        
        # lightgbm(lgb)
        param = {'num_leaves':64, 'objective': 'binary'}
        param['metric'] = 'auc'
        num_round = 1000
        data_model = lgb.train(param, dtrain, num_round, valid_sets=[dvalid], early_stopping_rounds=10, verbose_eval=False)


    # generate different versions for scoring
        data_model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
        data_model_2 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
        data_model_3 = RandomForestRegressor(n_estimators=150, min_samples_split=20, random_state=0)
        data_model_4 = RandomForestRegressor(n_estimators=200, max_depth=7, random_state=0)

        models = [data_model_1, data_model_2, data_model_3, data_model_4]

    # Function to help compare MAE scores for different models
        def score_model(model, X_t=X_train, X_v=X_vaild, y_t=y_train, y_v=y_valid):
            model.fit(X_t, y_t)
            preds = model.predict(X_v)
            return mean_absolute_error(y_v, preds)

        for i in range(0. len(models)):
            mae = score_model(models[i])
            print('model %d MAE: %d' %(i+1, mae))

    # Fit
        data_model.fit(X,y)
        data_model.fit(train_X, train_y)

    # Make predictions
        predictions = data_model.predict(X)
        val_predictions = data_model.predict(val_X)

    # Evaluation using AUC
        score_AUC = metrics.roc_auc_score(test['result'], predictions)
        print('model AUC: %d' %(score_AUC))

    # Evaluate/Validation
        predictions_MAE = mean_absolute_error(y, predictions)
        predictions_MAE = mean_absolute_error(val_y, val_predictions)

    # Function to help compare MAE scores from different values of max_leaf_nodes in DecisionTreeRegressor mdoels
        def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
            model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
            model.fit(train_X, train_y)
            preds_val = model.predict(val_X) 
            mae = mean_absolute_error(val_y, preds_val)
            return(mae)

    # help function usage
        for max_leaf_nodes in [5, 50, 500, 5000]:
            my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
            print('MAX leaf nodes: %d \t\t MAE: %d' %(max_leaf_nodes, my_mae))

        scores = {leaf_size: get_mae(leaf_size, train_X, val_X, train_y, val_y) for leaf_size in candidate_max_leaf_nodes}
        best_tree_size = min(scores, key=scores.get)
____________________________________________________________________________________________________
Step 4:Optimize / Improve / Other ?? tunning(over-under fitting)?
    # cross-validation (use in case of small data sets)
        # define help-pipeline to fill in missing values and use RandomForestRegressor
            help_pipeline = Pipeline(steps=[('preprocessor', SimpleImputer()),
            ('model', RandomForestRegressor(n_estimators=50, random_state=0))])
        
        #  obtain cross-validation scores
            scores = -1*cross_val_score(my_pipeline, X, y, cv=5, scoring='neg_mean_absolute_error')
            print('MAE scores: \n', scores)
            print('average MAE score: \n', scores.mean())
        
        # in a function form
            def get_score(n_estimators):
                my_pipeline = Pipeline(steps=[
                    ('preprocessor', SimpleImputer()),
                    ('model', RandomForestRegressor(n_estimators, random_state=0))
                ])
                scores = -1 * cross_val_score(my_pipeline, X, y,
                                  cv=3,
                                  scoring='neg_mean_absolute_error')
                return scores.mean()

    # APPLY XGBoost - extreme gradient boosting
        XGBR_model = XGBRegressor(
            n_estimators = , #number of XGB cycles (typ.100-1000, low=underfitting, hihg=overfitting)
            learning_rate = # default=01
            n_jobs = #for big data sets, makes fitting faster, typically equal to number of machine cores
        )

        XGBR_model.fit(X_train, y_train,
            early_stopping_rounds = , #stop iterating when model when validation score stop improving   
            eval_set = [(X_valid, y_valid)],
            verbose = False
        )
    
        # XGBR predictions
            predictions = XGBR_model.predict(X_valid)

        # XGBR evaluation
            print('MAE (XGBR): \n', mean_absolute_error(predictions, y_valid))

    # data leakage prevention
        
    # random data generation
        data = np.random.exponential(size=1000)

    # joining generated dataframe
        new_data = data1[['column1', 'column2']].join(data2)

    # datetime
        # convert date columns to datetime dtype
            data['date_conv'] = pd.to_dataframe(data['date'], format='%d/%m/%Y')
        
        # get the day of the month
            day_of_month_data = data['date_conv'].dt.day
    
    # timestamps conversion
        data = data.assign(
            hour=data.feature.dt.hour,
            day=data.feature.dt.day,
            month=data.feature.dt.month,
            yeat=data.feature.dt.year)

    # string encoding utf-8, utf-32, ascii, 
        encoded_string = org_string.encode('utf-8', errors='replace')
        decoded_string = encoded_string.decode('utf-8')
____________________________________________________________________________________________________
Step 5:Generate results in desired output form
    
    # files 
    output = pd.DataFrame({'Id': X_test.index, 'Column1':preds_test})
    output_results.to_csv('path/output.csv', index=False)
    output_results.to_xlsx('path/output.xlsc', index=False)
    
    # plots
