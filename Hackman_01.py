__author__ = 'Dave'

import scipy.io
import scipy.optimize
import numpy

mat = scipy.io.loadmat('D:\\Dave\\Trading\\Models\\Gene\\futsdata.mat')
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
    firstindex[:,j] = (firstIndexMat[:,j].real[0])    
    
    for i in range(len(tdatesMat)):
        prices[i, j] = priceMat[i, j]
        returns[i, j] = returnsMat[i, j]
        annvols[i, j] = annVolsMat[i, j]
        adjreturns[i, j] = adjReturnsMat[i, j]

# Define downside volatility calcualtion and sortino
def downsideVol(returns,annualMAR,periodsPerYear):
     periodMAR=numpy.log(annualMAR+1)/periodsPerYear
     logreturns=numpy.log(numpy.array(returns)+1)
     underperformance=[(lr-periodMAR) for lr in logreturns if (lr-periodMAR)<0]
     downdeviation=(sum([x**2 for x in underperformance])/len(returns))**0.5
     return downdeviation*(periodsPerYear**0.5) 

def sortino(returns,annualMAR,periodsPerYear):
     """Return the Annualized Sortino Ratio of a list of returns of arbitrary periodicity.
     Inputs:
          returns is a sequence of period returns in percent (P2-P1)/P1
          annualMAR is the minimum acceptable return, as percent per annum.
          periodsPerYear is the periodicity of the returns (i.e. 2 years of monthly data will
               have len(returns)==24, periodsPerYear=12
     Returns:
          float: annualized sortino ratio"""
     periodMAR=numpy.log(annualMAR+1)/periodsPerYear
     logreturns=numpy.log(numpy.array(returns)+1)
     underperformance=[(lr-periodMAR) for lr in logreturns if (lr-periodMAR)<0]
     downdeviation=(sum([x**2 for x in underperformance])/len(returns))**0.5
     retperperiod=sum(logreturns)/len(logreturns)
     periodsortino=(retperperiod-periodMAR)/downdeviation
     return periodsortino*(periodsPerYear**0.5) 
     
     
# Define the model functions in here
def dsVarCalc(wghts, retMatrix):
    # Returns the downside variance of the weighted returns
    # wghts must be a 1*x numpy.array as retMatrix is n*x
    wghtRets = numpy.multiply(wghts,retMatrix)
    retSeries = numpy.nansum(wghtRets, axis=1)
    retSeriesNN = numpy.nan_to_num(retSeries)
    return downsideVol(retSeriesNN,0,260)

def sortinoCalc(wghts, retMatrix):
    # Returns the sortino of the weighted returns
    # wghts must be a 1*x numpy.array as retMatrix is n*x
    wghtRets = numpy.multiply(wghts,retMatrix)
    retSeries = numpy.nansum(wghtRets, axis=1)
    retSeriesNN = numpy.nan_to_num(retSeries)
    return sortino(retSeriesNN,0,260)


# Backtest should start here
optWghts = numpy.empty(priceMat.shape)
optWghts[:] = 0
lookback = 260


for d in range(261, len(tdates)):
    fIndex=firstindex>d
    vIndex = numpy.intp(fIndex)
    vWghtsUp = [[vIndex*10], [vIndex*-10]]
    
    
    
    










