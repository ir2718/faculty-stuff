from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from data import *

class KSVMWrap():
    """
        Metode:
        __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
            Konstruira omotač i uči RBF SVM klasifikator
            X, Y_:           podatci i točni indeksi razreda
            param_svm_c:     relativni značaj podatkovne cijene
            param_svm_gamma: širina RBF jezgre

        predict(self, X)
            Predviđa i vraća indekse razreda podataka X

        get_scores(self, X):
            Vraća klasifikacijske mjere
            (engl. classification scores) podataka X;
            ovo će vam trebati za računanje prosječne preciznosti.

        support
            Indeksi podataka koji su odabrani za potporne vektore
    """
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        model = SVC(C=param_svm_c, gamma=param_svm_gamma)
        model.fit(X, Y_)
        self.model = model
        self.support = np.argwhere(X == model.support_vectors_)
    
    def predict(self, X):
        return self.model.predict(X)

    def get_scores(self, X):
        return self.model.predict_proba(X)

if __name__ == "__main__":
    # inicijaliziraj generatore slučajnih brojeva
    np.random.seed(100)

    # instanciraj podatke X i labele Yoh_
    X,Yoh_ = sample_gmm_2d(K=6, C=2, N=10)
    # X,Yoh_ = sample_gmm_2d(K=3, C=3, N=100)
    
    # definiraj model:
    svm = KSVMWrap(X, Yoh_, )

    preds = svm.predict(X)
    print(eval_perf_multi(preds, Yoh_))

    # iscrtaj rezultate, decizijsku plohu
    decfun = svm.predict
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(decfun, bbox, offset=0.5)

    graph_data(X, Yoh_, preds, special = svm.support)
    plt.show()