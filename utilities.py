import numpy

def colFromRow(v): #this function turns a row into a column
    return v.reshape((v.size, 1))

def rowFromCol(v): #this function turns a column into a row
    return v.reshape((1, v.size))


def load_dataset(filename):
    features=[]
    labels=[]
    with open(filename, 'r') as f:
        for line in f:
            feats= [float(fe) for fe in line.split(',')[0:6]]
            
            features.append(colFromRow(numpy.array(feats)))
            
            label=line.split(',')[6].strip()
            labels.append(label)
                
    L=numpy.array(labels, dtype=numpy.int32) #1-d array from a list
    D=numpy.hstack(features) 
    #each column of D is a sample, each sample has 6 rows and each row represents a feature
    #print(D.shape) #6x2371
    #print(L.shape) #2371
    return D,L

def logpdf_GAU_ND(x, mu, C):
    M = x.shape[0]
    _, det = numpy.linalg.slogdet(C)
    det = numpy.log(numpy.linalg.det(C))
    inv = numpy.linalg.inv(C)    
    res = []
    x_centered = x - mu
    for x_col in x_centered.T:
        res.append(numpy.dot(x_col.T, numpy.dot(inv, x_col)))

    return -M/2*numpy.log(2*numpy.pi) - 1/2*det - 1/2*numpy.hstack(res).flatten()