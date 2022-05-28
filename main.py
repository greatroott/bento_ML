from sklearn import svm 
from sklearn import datasets

# Load training data 
iris = datasets.load_iris()
X,y = iris.data, iris.target


# Model training
clf = svm.SVC(gamma='scale')
clf.fit(X,y)

from iris_classifier import IrisClassifier

iris_classifier_service = IrisClassifier()

# Pack the newly trained model artifact
iris_classifier_service.pack('model', clf)
# Save the prediction service to disk for model serving
saved_path = iris_classifier_service.save()