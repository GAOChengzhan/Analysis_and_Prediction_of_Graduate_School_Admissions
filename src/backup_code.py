'''
Results = list()
# Field number
f=0
# min number of samples
min_samples = 10
records=[]
for i, field in enumerate(Z_label[f]):
    idx = np.where(Z[:,f] == i)[0]
    if len(idx)>min_samples:
        records.append()
        result = dict()
        accuracy = list()
        y_ = list()
        # for i, (train_index, test_index) in enumerate(kf.split(idx)):
        clf = GaussianNB()
        X_train = X[idx][:,:-2]

        for k in range(len(Z_label)):
            if k != f:
                X_train = np.hstack((X_train, get_ohv(Z[idx],Z, k)))

        clf.fit(X_train, Y[idx])
        accuracy.append(clf.score(X_train, Y[idx]))
        print(np.mean(accuracy)*100)
'''
