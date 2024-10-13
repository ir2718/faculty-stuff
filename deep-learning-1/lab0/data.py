import matplotlib.pyplot as plt
import numpy as np

class Random2DGaussian():
    def __init__(self):
        self.minx, self.maxx = 0, 10
        self.miny, self.maxy = 0, 10

        min_, max_ = np.array([self.minx, self.miny]), np.array([self.maxx, self.maxy])
        self.mu =  min_ + np.multiply(np.random.random_sample(size=2), max_)
        
        eigvals = (np.random.random_sample(2)*(max_ - min_)/5)**2

        theta = np.random.random_sample()*np.pi*2
        rot = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        self.cov = rot.T.dot(np.diag(eigvals)).dot(rot)

    def get_sample(self, n=100):
        return np.random.multivariate_normal(self.mu, self.cov, n)

def sample_gauss_2d(C, N):
    data_tmp, labels_tmp = [], []
    for i in range(C):
        data = Random2DGaussian().get_sample(N)
        data_tmp.append(data)
        labels_tmp.append(np.array([i] * data.shape[0]))
    data = np.concatenate(tuple(data_tmp), axis=0)
    labels = np.concatenate(tuple(labels_tmp), axis=0)
    return data, labels

def eval_perf_binary(Y, Y_):
    tp = len([y == y_ for y, y_ in zip(Y, Y_) if y == 1 and y_ == 1])
    tn = len([y == y_ for y, y_ in zip(Y, Y_) if y == 0 and y_ == 0])
    fp = len([y == y_ for y, y_ in zip(Y, Y_) if y == 0 and y_ == 1])
    fn = len([y == y_ for y, y_ in zip(Y, Y_) if y == 1 and y_ == 0])
    acc    = (tp + tn) / (tp + tn + fn + fp)
    prec   = tp / (tp + fp)
    recall = tp / (tp + fn) 
    return acc, prec, recall

def eval_AP(Yr):
    ap = 0
    for i in range(len(Yr)):
        c0, c1 = Yr[:i], Yr[i:]
        tp = np.sum(np.equal(c1, 1))
        fp = np.sum(np.equal(c1, 0))
        ap += (0 if tp == 0 else tp/(tp + fp))*Yr[i]
    return ap/np.sum(Yr)

# def graph_data(X, Y_, Y):
#     '''
#         X  ... podatci (np.array dimenzija Nx2)
#         Y_ ... točni indeksi razreda podataka (Nx1)
#         Y  ... predviđeni indeksi razreda podataka (Nx1)
#     '''
#     X, Y, Y_ = np.array(X), np.array(Y_), np.array(Y)
#     colors = np.array(['white' if y == y_ else 'dimgray' for y, y_ in zip(Y, Y_)])
#     indices_true, indices_false = np.array(Y_ == Y), np.array(Y_ != Y)
#     plt.scatter(X[indices_true, 0], X[indices_true, 1], marker='o', c=colors[indices_true], edgecolors='black')
#     plt.scatter(X[indices_false, 0], X[indices_false, 1], marker='s', c=colors[indices_false])

def graph_data(X,Y_, Y, special=[]):
    """Creates a scatter plot (visualize with plt.show)

    Arguments:
        X:       datapoints
        Y_:      groundtruth classification indices
        Y:       predicted class indices
        special: use this to emphasize some points

    Returns:
        None
    """
    # colors of the datapoint markers
    palette=([0.5,0.5,0.5], [1,1,1], [0.2,0.2,0.2])
    colors = np.tile([0.0,0.0,0.0], (Y_.shape[0],1))
    for i in range(len(palette)):
        colors[Y_==i] = palette[i]

    # sizes of the datapoint markers
    sizes = np.repeat(20, len(Y_))
    sizes[special] = 40

    # draw the correctly classified datapoints
    good = (Y_==Y)
    plt.scatter(X[good,0],X[good,1], c=colors[good], 
                s=sizes[good], marker='o', edgecolors='black')

    # draw the incorrectly classified datapoints
    bad = (Y_!=Y)
    plt.scatter(X[bad,0],X[bad,1], c=colors[bad], 
                s=sizes[bad], marker='s', edgecolors='black')

def graph_surface(fun, rect, offset=0.5, width=256, height=256):
    """Creates a surface plot (visualize with plt.show)

    Arguments:
    fun: surface to be plotted
    rect:     function domain provided as:
                ([x_min,y_min], [x_max,y_max])
    offset:   the level plotted as a contour plot

    Returns:
    None
    """

    lsw = np.linspace(rect[0][1], rect[1][1], width) 
    lsh = np.linspace(rect[0][0], rect[1][0], height)
    xx0,xx1 = np.meshgrid(lsh, lsw)
    grid = np.stack((xx0.flatten(),xx1.flatten()), axis=1)

    #get the values and reshape them
    values = fun(grid).reshape((width,height))

    # fix the range and offset
    delta = offset if offset else 0
    maxval = max(np.max(values)-delta, - (np.min(values)-delta))

    # draw the surface and the offset
    plt.pcolormesh(xx0, xx1, values, vmin=delta-maxval, vmax=delta+maxval, shading='auto')

    if offset != None:
        plt.contour(xx0, xx1, values, colors='black', levels=[offset])

def myDummyDecision(X):
    scores = X[:,0] + X[:,1] - 5
    return scores

def eval_perf_multi(Y, Y_):
    pr = []
    n = max(Y_)+1
    M = np.bincount(n * Y_ + Y, minlength=n*n).reshape(n, n)
    for i in range(n):
        tp_i = M[i,i]
        fn_i = np.sum(M[i,:]) - tp_i
        fp_i = np.sum(M[:,i]) - tp_i
        tn_i = np.sum(M) - fp_i - fn_i - tp_i
        recall_i = tp_i / (tp_i + fn_i)
        precision_i = tp_i / (tp_i + fp_i)
        pr.append( (recall_i, precision_i) )
    accuracy = np.trace(M)/np.sum(M)
    return accuracy, pr, M


if __name__=="__main__":
  
    # get the training dataset
    X,Y_ = sample_gauss_2d(2, 100)
  
    # get the class predictions
    Y = myDummyDecision(X)>0.5
  
    # graph the data points
    graph_data(X, Y_, Y) 
  
    # show the results
    plt.show()



# if __name__=="__main__":
#     np.random.seed(100)
#     G=Random2DGaussian()
#     X=G.get_sample(100)
#     plt.scatter(X[:,0], X[:,1])
#     plt.show()