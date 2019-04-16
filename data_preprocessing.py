from sklearn.preprocessing import MinMaxScaler, LabelBinarizer


def np_normalize_data(x):
    scaler = MinMaxScaler()
    return scaler.fit_transform(x)


def df_normalize_data(df):
    scaler = MinMaxScaler()
    return scaler.fit_transform(df.values)


def binarize_target(y):
    lb = LabelBinarizer()
    lb.fit(y)
    return lb.transform(y)
