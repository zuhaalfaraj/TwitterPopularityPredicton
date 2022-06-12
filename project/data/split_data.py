from sklearn.model_selection import train_test_split


class SplitData:
    def __init__(self, test_size, random_state):
        self.test_size = test_size
        self.random_state = random_state

    def __call__(self, df, val=True):
        df_train, df_test = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
        if not val:
            return df_train, df_test

        df_val, df_test = train_test_split(df, test_size=0.5, random_state=self.random_state)
        return df_train, df_test, df_val