from sklearn.model_selection import train_test_split

class HandlerModels(object):

    def __init__(
            self,
            dataset=None, 
            response=None, 
            test_size=None, 
            cv=None):
        
        self.dataset = dataset
        self.response = response
        self.test_size = test_size
        self.cv = cv
    
    def prepare_dataset(
            self, 
            random_state=42):
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.dataset, 
            self.response, 
            test_size=0.30, 
            random_state=random_state)
        
        return X_train, X_test, y_train, y_test
    
    def training_exploring(
            self, 
            random_state=42):
        pass

    def training_individual(
            self, 
            name_algorithm="knn",
            random_state=42,
            name_export=None):
        
        pass