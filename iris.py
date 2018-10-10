from sklearn.datasets import load_iris;
from sklearn import tree;
import numpy as np;


iris = load_iris();
test_idx = [0, 50, 100]

#training data
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)


#testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print(clf.predict(test_data))


#visualization

from sklearn.externals.six import StringIO
import graphviz as gp;
dot_data = StringIO()
tree.export_graphviz(clf,
                        out_file = dot_data,
                        feature_names = iris.feature_names,
                        class_names = iris.target_names,
                        filled = True,
                        rounded = True,
                        impurity=False)
graph = gp.Source(dot_data.getvalue())
graph.render("iris.pdf", view=True)
