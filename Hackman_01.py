__author__ = 'Dave'

import scipy.io
import scipy.optimize
import numpy

mat = scipy.io.loadmat('D:\\Dave\\Trading\\Models\\Gene\\futsdata.mat')
tdatesMat = mat['tdates']
tickersMat = mat['tickers']
typesMat = mat['types']
ccyMat = mat['ccy']
priceMat = mat['price']
returnsMat = mat['returns']
annVolsMat = mat['annvols']
adjReturnsMat = mat['adjreturns']

tdates = []
tickers = []
types = []
ccys = []

# initialise the 2 numerical matrices as empty mats then set to NAN
prices = numpy.empty(priceMat.shape)
returns = numpy.empty(priceMat.shape)
annvols = numpy.empty(priceMat.shape)
adjreturns = numpy.empty(priceMat.shape)
prices[:] = numpy.NAN
returns[:] = numpy.NAN
annvols[:] = numpy.NAN
adjreturns[:] = numpy.NAN

for i in range(len(tdatesMat)):
    tdates.append(tdatesMat[i].real[0])
    
for j in range(tickersMat.size):
    tickers.append(tickersMat[:, j].real[0].real[0])
    types.append(typesMat[:, j].real[0].real[0])
    ccys.append(ccyMat[:, j].real[0].real[0])
    
    for i in range(len(tdatesMat)):
        prices[i, j] = priceMat[i, j]
        returns[i, j] = returnsMat[i, j]
        annvols[i, j] = annVolsMat[i, j]
        adjreturns[i, j] = adjReturnsMat[i, j]



