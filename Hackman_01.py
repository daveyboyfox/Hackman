__author__ = 'Dave'

import scipy.io
import scipy.optimize
import numpy
import matplotlib.pyplot as plt
import random
import multiprocessing as mp
import os
from functools import partial

mat = scipy.io.loadmat('D:\\Dave\\Trading\\Models\\Gene\\futsdata_small.mat')
#mat = scipy.io.loadmat('C:\\Users\\Dave\\PycharmProjects\\Hackman\\futsdata_small.mat')
tdatesMat = mat['tdates']
tickersMat = mat['tickers']
typesMat = mat['types']
firstIndexMat = mat['firstindex']
ccyMat = mat['ccy']
priceMat = mat['price']
returnsMat = mat['returns']
annVolsMat = mat['annvols']
adjReturnsMat = mat['adjreturns']

tdates = numpy.empty(tdatesMat.shape)
tdates[:] = numpy.NAN
tickers = []
types = []
firstindex = numpy.empty(firstIndexMat.shape)
firstindex[:] = numpy.NAN
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
    tdates[i] = tdatesMat[i].real[0]
        
for j in range(tickersMat.size):
    tickers.append(tickersMat[:, j].real[0].real[0])
    types.append(typesMat[:, j].real[0].real[0])
    ccys.append(ccyMat[:, j].real[0].real[0])
    firstindex[:, j] = (firstIndexMat[:, j].real[0])
    
    for i in range(len(tdatesMat)):
        prices[i, j] = priceMat[i, j]
        returns[i, j] = returnsMat[i, j]
        annvols[i, j] = annVolsMat[i, j]
        adjreturns[i, j] = adjReturnsMat[i, j]


# Define downside volatility calcualtion and sortino
def downsidevol(returnSeries,annualMAR,periodsPerYear):
     periodMAR=numpy.log(annualMAR+1)/periodsPerYear
     logreturns=numpy.log(numpy.array(returnSeries)+1)
     underperformance=[(lr-periodMAR) for lr in logreturns if (lr-periodMAR)<0]
     downdeviation=(sum([x**2 for x in underperformance])/len(returnSeries))**0.5
     return downdeviation*(periodsPerYear**0.5) 


def sortino(returnSeries,annualMAR,periodsPerYear):
     """Return the Annualized Sortino Ratio of a list of returns of arbitrary periodicity.
     Inputs:
          returns is a sequence of period returns in percent (P2-P1)/P1
          annualMAR is the minimum acceptable return, as percent per annum.
          periodsPerYear is the periodicity of the returns (i.e. 2 years of monthly data will
               have len(returns)==24, periodsPerYear=12
     Returns:
          float: annualized sortino ratio"""
     periodMAR=numpy.log(annualMAR+1)/periodsPerYear
     logreturns=numpy.log(numpy.array(returnSeries)+1)
     underperformance=[(lr-periodMAR) for lr in logreturns if (lr-periodMAR)<0]
     downdeviation=(sum([x**2 for x in underperformance])/len(returnSeries))**0.5
     retperperiod=sum(logreturns)/len(logreturns)
     periodsortino=(retperperiod-periodMAR)/downdeviation
     return periodsortino*(periodsPerYear**0.5) 
     
     
# Define the model functions in here - may need to set weights
def dsvarcalc(wghts, *retMatrix):
    # Returns the downside variance of the weighted returns
    # wghts must be a 1*x numpy.array as retMatrix is n*x
    wghtRets = numpy.multiply(wghts,retMatrix)
    retSeries = numpy.nansum(wghtRets, axis=1)
    retSeriesNN = numpy.nan_to_num(retSeries)
    retSeriesNN = numpy.cumsum(retSeriesNN)
    return downsidevol(retSeriesNN,0,260)


def sortinocalc(wghts, *retMatrix):
    # Returns the sortino of the weighted returns
    # wghts must be a 1*x numpy.array as retMatrix is n*x
    wghtRets = numpy.multiply(wghts,retMatrix)
    retSeries = numpy.nansum(wghtRets, axis=1)
    retSeriesNN = numpy.nan_to_num(retSeries)
    retSeriesNN = numpy.cumsum(retSeriesNN)
    srtRes = sortino(retSeriesNN,0,260)
    if srtRes <= 0:
        return 1000
    elif ~numpy.isfinite(srtRes):
        return 1000
    else:
        return 1/srtRes
    


def checkbounds(tWghts):
    posWghts = sum([x for x in tWghts if x > 0])
    negWghts = sum([x for x in tWghts if x < 0])
    if posWghts < 10 and negWghts >-10:
        return 0
    else:
        return 1
        

# Backtest should start here
def runmultibacktest(adjretmatrix, firstindexarray, x):
    optWghts = numpy.empty(adjretmatrix.shape)
    optWghts[:] = 0
    cons = {'type': 'eq', 'fun': checkbounds}
    startGuess = tuple(random.uniform(-1, 1) for i in range(firstindexarray.size))


    for d in range(521, len(tdates)-1):
        fIndex = firstindexarray < d
        vIndex = numpy.intp(fIndex)

        vWghts = tuple((max(min(vIndex[:, i].real[0]*(optWghts[d, i]-1),9), -10),
                        min(max(vIndex[:, i].real[0]*(optWghts[d, i]+1),-9), 10))
                        for i in range(vIndex.size))

        if d == 521:
            bWghts = startGuess
        else:
            bWghts=optWghts[d,:].tolist()

        subReturnSeries = adjretmatrix[d-260:d, :]

        minimizer_kwargs = {"args": subReturnSeries, "bounds": vWghts,
                            "constraints": cons}  # set bounds

        # this is causing the pool to raise an exception so simplest to carry forward
        resultMat = scipy.optimize.basinhopping(sortinocalc, bWghts,
                                                minimizer_kwargs=minimizer_kwargs,
                                                niter=10, stepsize=0.05)
        optWghts[d+1, :] = resultMat.x

        if numpy.mod(d, 100) == 0:
            print('process id: ', os.getpid(), d)

    return optWghts
    #output.put(optWghts)

def pooledbacktest(adjretmatrix, firstindexarray, num_iterations):
    args = [adjreturns, firstindex]
    partialBacktest = partial(runmultibacktest, *args)
 
    pool =mp.Pool() #creates a pool of process, controls worksers
    result_set = pool.map(partialBacktest, range(num_iterations)) #make our results with a map call
    pool.close() #we are not adding any more processes
    pool.join() #tell it to wait until all threads are done before going on
 
    return result_set

if __name__ == '__main__':
    #result_queue = mp.Queue()
    #procruns = [runmultibacktest(adjreturns, firstindex, result_queue) for x in range(20)]
    #jobs = [mp.Process(pr) for pr in procruns]
    #for job in jobs: job.start()
    #for job in jobs: job.join()
    #results = [result_queue.get() for pr in procruns]

    results = pooledbacktest(adjreturns, firstindex, 20)

    allWghts = results[0]
    for i in range(1, len(results)):
        allWghts = allWghts + results[i]

    allWghts = allWghts/len(results)

    allReturns = numpy.multiply(allWghts,adjreturns)
    retSeries = numpy.nansum(allReturns, axis=1)
    retSeriesNN = numpy.nan_to_num(retSeries)
    retSeriesCum = numpy.cumsum(retSeriesNN)

    plt.plot(retSeriesCum)
    plt.show()

    plt.plot(allWghts)
    plt.show()

    