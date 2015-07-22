__author__ = 'Dave'

import scipy.io
import scipy.optimize
import numpy
import pylab as pl

#mat = scipy.io.loadmat('D:\\Dave\\Trading\\Models\\Gene\\futsdata.mat')
mat = scipy.io.loadmat('C:\\Users\\Dave\\PycharmProjects\\Hackman\\futsdata.mat')
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
     
     
# Define the model functions in here
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
    return sortino(retSeriesNN,0,260)


# Backtest should start here
optWghts = numpy.empty(priceMat.shape)
optWghts[:] = 0
lookback = 260


for d in range(521, len(tdates)-1):
    fIndex = firstindex > d
    vIndex = numpy.intp(fIndex)
    vWghts = []
    
    for i in range(vIndex.size):
        vWghts.append((vIndex[:,i].real[0]*-10, vIndex[:,i].real[0]*10))
    
    subReturnSeries = adjreturns[d-260:d,:]
    
    resultMat = scipy.optimize.differential_evolution(dsvarcalc, vWghts, subReturnSeries)
    
    optWghts[d+1,:] = resultMat.x
    
    if numpy.mod(d, 10)==0:
        print(d)

allReturns = numpy.multiply(optWghts,adjreturns)
retSeries = numpy.nansum(allReturns, axis=1)
retSeriesNN = numpy.nan_to_num(retSeries)
retSeriesCum = numpy.cumsum(retSeriesNN)

pl.plot(retSeriesCum)
pl.show()






