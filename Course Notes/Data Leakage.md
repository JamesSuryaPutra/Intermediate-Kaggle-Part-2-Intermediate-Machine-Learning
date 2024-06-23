# Data leakage
In this example, you will learn one way to detect and remove target leakage.

We will use a dataset about credit card applications and skip the basic data set-up code. The end result is that information about each credit card application is stored in a DataFrame X.
We'll use it to predict which applications were accepted in a Series y.

    import pandas as pd

    # Read the data
    data = pd.read_csv('../input/aer-credit-card-data/AER_credit_card_data.csv', 
                       true_values = ['yes'], false_values = ['no'])

    # Select target
    y = data.card

    # Select predictors
    X = data.drop(['card'], axis=1)

    print("Number of rows in the dataset:", X.shape[0])
    X.head()

    Number of rows in the dataset: 1319

    reports	age	income	share	expenditure	owner	selfemp	dependents	months	majorcards	active
    0	0	37.66667	4.5200	0.033270	124.983300	True	False	3	54	1	12
    1	0	33.25000	2.4200	0.005217	9.854167	False	False	3	34	1	13
    2	0	33.66667	4.5000	0.004156	15.000000	True	False	4	58	1	5
    3	0	30.50000	2.5400	0.065214	137.869200	False	False	0	25	1	7
    4	0	32.16667	9.7867	0.067051	546.503300	True	False	2	64	1	5


Since this is a small dataset, we will use cross-validation to ensure accurate measures of model quality.

    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    # Since there is no preprocessing, we don't need a pipeline (used anyway as best practice!)
    my_pipeline = make_pipeline(RandomForestClassifier(n_estimators=100))
    cv_scores = cross_val_score(my_pipeline, X, y, 
                                cv=5,
                                scoring='accuracy')

    print("Cross-validation accuracy: %f" % cv_scores.mean())

    Cross-validation accuracy: 0.981052


With experience, you'll find that it's very rare to find models that are accurate 98% of the time. It happens, but it's uncommon enough that we should inspect the data more closely for
target leakage. Here is a summary of the data, which you can also find under the data tab:
- card: 1 if credit card application accepted, 0 if not
- reports: Number of major derogatory reports
- age: Age n years plus twelfths of a year
- income: Yearly income (divided by 10,000)
- share: Ratio of monthly credit card expenditure to yearly income
- expenditure: Average monthly credit card expenditure
- owner: 1 if owns home, 0 if rents
- selfempl: 1 if self-employed, 0 if not
- dependents: 1 + number of dependents
- months: Months living at current address
- majorcards: Number of major credit cards held
- active: Number of active credit accounts

A few variables look suspicious. For example, does expenditure mean expenditure on this card or on cards used before applying?
At this point, basic data comparisons can be very helpful.

    expenditures_cardholders = X.expenditure[y]
    expenditures_noncardholders = X.expenditure[~y]

    print('Fraction of those who did not receive a card and had no expenditures: %.2f' \
          %((expenditures_noncardholders == 0).mean()))
    print('Fraction of those who received a card and had no expenditures: %.2f' \
          %(( expenditures_cardholders == 0).mean()))

    Fraction of those who did not receive a card and had no expenditures: 1.00
    Fraction of those who received a card and had no expenditures: 0.02


As shown above, everyone who did not receive a card had no expenditures, while only 2% of those who received a card had no expenditures. It's not surprising that our model appeared to
have a high accuracy. But this also seems to be a case of target leakage, where expenditures probably means expenditures on the card they applied for.

Since share is partially determined by expenditure, it should be excluded too. The variables active and majorcards are a little less clear, but from the description, they sound
concerning. In most situations, it's better to be safe than sorry if you can't track down the people who created the data to find out more.

We would run a model without target leakage as follows.

    # Drop leaky predictors from dataset
    potential_leaks = ['expenditure', 'share', 'active', 'majorcards']
    X2 = X.drop(potential_leaks, axis=1)

    # Evaluate the model with leaky predictors removed
    cv_scores = cross_val_score(my_pipeline, X2, y, 
                                cv=5,
                                scoring='accuracy')

    print("Cross-val accuracy: %f" % cv_scores.mean())

    Cross-val accuracy: 0.830919


This accuracy is quite a bit lower, which might be disappointing. However, we can expect it to be right about 80% of the time when used on new applications, whereas the leaky model
would likely do much worse than that (in spite of its higher apparent score in cross-validation).
