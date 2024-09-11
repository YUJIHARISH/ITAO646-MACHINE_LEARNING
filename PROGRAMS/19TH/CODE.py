import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = []
        for c in self.classes:
            X_c = X[y == c]
            self.parameters.append({
                'mean': np.mean(X_c, axis=0),
                'var': np.var(X_c, axis=0),
                'prior': len(X_c) / len(X)
            })

    def _pdf(self, x, mean, var):
        return np.exp(-(x-mean)**2 / (2*var)) / np.sqrt(2*np.pi*var)

    def predict(self, X):
        return [self._predict(x) for x in X]

    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(self.parameters[i]['prior'])
            posterior = np.sum(np.log(self._pdf(x, self.parameters[i]['mean'], self.parameters[i]['var'])))
            posterior += prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

iris = load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

nb = NaiveBayes()
nb.fit(X_train, y_train)
predictions = nb.predict(X_test)
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f"Accuracy: {accuracy}")
