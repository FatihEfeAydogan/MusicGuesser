from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

class TempModel:
    def __init__(self):
        self.models = [
            MLPClassifier(activation='relu', learning_rate='adaptive', solver='adam', max_iter=1000, hidden_layer_sizes=(100,)),
            LogisticRegression(penalty='l1', solver='liblinear', max_iter=5000),
            RandomForestClassifier(n_estimators=1000, criterion='entropy'),
            LinearDiscriminantAnalysis(solver='lsqr', shrinkage=None),
            KNeighborsClassifier(n_neighbors=10, algorithm='ball_tree', weights='distance', leaf_size=50, p=1, n_jobs=-1),
            SVC(kernel='rbf', C=10, gamma='scale'),
            GaussianNB(),
            AdaBoostClassifier(algorithm='SAMME.R', n_estimators=1000),
            LinearSVC(C=1, max_iter=1000, penalty='l2'),
            GradientBoostingClassifier(loss='log_loss', n_estimators=1000)
        ]

    # En iyi modeli eğitme
    def train_best_model(self, X_train, X_val, y_train, y_val):
        best_model = None
        best_score = 0

        for model in self.models:
            print(f"Training {model.__class__.__name__}...")
            model.fit(X_train, y_train)
            score = model.score(X_val, y_val)
            print(f"Validation score for {model.__class__.__name__}: {score}")
            if score > best_score:
                best_score = score
                best_model = model

        print(f"Best model: {best_model.__class__.__name__} with score: {best_score}")
        return best_model

    # Test seti üzerinde en iyi modelin performansını değerlendirme
    def evaluate_best_model(self, best_model, X_test, y_test):
        test_score = best_model.score(X_test, y_test)
        print("Test score of the best model:", test_score)

# Örnek kullanım
if __name__ == "__main__":
    # Varsayılan bazı veri setleri oluşturun
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Veri seti oluşturma
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Model eğitimi ve değerlendirme
    temp_model = TempModel()
    best_model = temp_model.train_best_model(X_train, X_val, y_train, y_val)
    temp_model.evaluate_best_model(best_model, X_test, y_test)
