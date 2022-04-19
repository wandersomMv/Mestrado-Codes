
#%%
import matplotlib.pyplot as plt # Import matplotlib library for show images
 
# Import scikit-learn library
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#13.233 imagens de 5749 classes de formas 125 * 94
# Dabase for aplication 
from sklearn.datasets import fetch_lfw_people 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from time import time
 
import numpy as np

# Function to plot images in 3 * 4 

def plot_gallery(images, h, w, titles=None ,n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=0.01, right=0.99, top=0.90, hspace=0.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        if titles:
            plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

# Generate true labels above the images
def true_title(Y, target_names, i):
    true_name = target_names[Y[i]].rsplit(' ', 1)[-1]
    return 'true label:   % s' % (true_name)


# breakpoint()
# this command will download the LFW_people's dataset to hard disk.
lfw_people = fetch_lfw_people(min_faces_per_person = 70, resize = 0.4)
 
# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape
 
# Instead of providing 2D data, X has data already in the form  of a vector that
# is required in this approach.
X = lfw_people.data
n_features = X.shape[1]
 
# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]
 
# Print Details about dataset
print(f"Number of Data Samples: {n_samples}")
print(f"Size of a data sample: {n_features}")
print(f"Number of Class Labels: {n_classes}")


 

 
true_titles = [true_title(y, target_names, i)
                     for i in range(y.shape[0])]
plot_gallery(X, h, w, true_titles)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)
print(f"size of training Data is {y_train.shape[0]} and Testing Data is {y_test.shape[0]}")


n_components = 150
 
t0 = time()
pca = PCA(n_components = n_components, svd_solver ='randomized',whiten = True).fit(X_train)
print(f"done in {((time() - t0))} s")
 
eigenfaces = pca.components_.reshape((n_components, h, w)) #autovalores of the faces
breakpoint()
plot_gallery(eigenfaces, h, w, true_titles)
 
print("Projecting the input data on the eigenfaces orthonormal basis")
t0 = time()
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
print(f"done in {((time() - t0))} s" )
print("Sample Data point after applying PCA\n", X_train_pca[0])
print("-----------------------------------------------------")
print(f"Dimesnsions of training set =  {X_train.shape} and Test Set = {X_test.shape}")

print("Fitting the classifier to the training set")
t0 = time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(kernel ='rbf', class_weight ='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)
print("done in % 0.3fs" % (time() - t0))
print("Best estimator found by grid search:")
print(clf.best_estimator_)
 
print("Predicting people's names on the test set")
t0 = time()
y_pred = clf.predict(X_test_pca)
print("done in % 0.3fs" % (time() - t0))
# print classification results
print(classification_report(y_test, y_pred, target_names = target_names))
# print confusion matrix
print("Confusion Matrix is:")
print(confusion_matrix(y_test, y_pred, labels = range(n_classes)))
# %%
