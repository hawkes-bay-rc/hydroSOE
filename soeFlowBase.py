import numpy as np
import pandas as pd
import scipy.stats as sp
from scipy import interpolate

import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, date

import math
import random

import goodness_of_fit as gof
import pymannkendall as pmk

from htmlTables import Table


#plots
from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, FactorRange, DatetimeTickFormatter
from bokeh.transform import factor_cmap
from bokeh.palettes import d3#viridis
output_notebook(hide_banner=True)

#interactive sessions
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

#Hilltop connector
"""Using emar api instead"""
apiRoot = "https://data.hbrc.govt.nz/Envirodata/EMAR.hts?Service=Hilltop"
##https://data.hbrc.govt.nz/EnviroData/Telemetry.hts?service=Hilltop&request=GetData&Site=HAWQi&Measurement=Temperature%205m&From=1/12/2017&To=1/1/2020
#apiRoot="https://data.hbrc.govt.nz/EnviroData/Telemetry.hts?service=Hilltop"

#variables
mySite = None
myMeasurement = None
myStartDate = None
myEndDate = None
climStart = None #30 year climatology

def getSiteList(measurement):
    global myMeasurement
    myMeasurement = measurement
    #get all the sites for requisite measurement type
    requestType = "SiteList"
    myWebRequest =  apiRoot + '&Request='+requestType + '&Measurement='+myMeasurement
    #print(myWebRequest)
    r = requests.get(myWebRequest,headers={'origin': 'https://dev.hbrc.govt.nz'})
    #print(r.text,'reply')
    sites = []
    root = ET.fromstring(r.content)
    for child in root.iter('*'):
        #print(child.tag,child.attrib)
        if child.tag == 'Site':
            sites.append(child.attrib['Name'])
    return sites

def fetchRiverPCal():
    global climStart
    
    try:
        temp = datetime.strptime(myStartDate,'%Y-%m-%d %H:%M:%S') - timedelta(days=31*365.25) #31 years?
    except Exception as e:
        try:
            temp = datetime.strptime(myStartDate,'%Y-%m-%d') - timedelta(days=31*365.25)
        except Exception as er:
            print(err)
    #this has to be updated, timedelta doesn't have functionality for years 
    #and days can alter in case of leap years
    try:
        climStart = datetime.strptime(str(temp.timetuple().tm_year)+myStartDate[4:],'%Y-%m-%d %H:%M:%S')
    except Exception as e:
        try:
            climStart = datetime.strptime(str(temp.timetuple().tm_year)+myStartDate[4:],'%Y-%m-%d')
        except Exception as er:
            print(err)
    startYear = int(myStartDate[:4])
    
    myWebRequest =  apiRoot + "&Request=Hydro"
    #print(myWebRequest)
    #get the range of values to be applied
    myHqObj = {
        "Command" : "PCal",
        "Site" : str(mySite),
        "Measurement" : str(myMeasurement),
        "From" : str(climStart), #30 year climatology
        "To" : str(myEndDate),
        "statistic": "Mean",
        "DAY": "00:00:00",
        "MONTH": "July"
    }
    
    headers = {"Content-Type": "application/json"}
    params = {"Service":"Hilltop", "Request":"Hydro"}
    r = requests.post(myWebRequest, json = myHqObj, headers=headers, params=params)
    #print(r.text)
    
    currentStats = {}
    pastStats = {}
    longTermStats = {}
    gapYears = []
    root = ET.fromstring(r.content)
    #print(root)
    if len(root.findall("Error")) > 0:
        print('Could not fetch the range, please specify yourself')
        return -1,-1
    else :
        for node in root:
            if node.tag == 'Results':
                for child in node:
                    if child.tag == 'Row':
                        #print(child.attrib["Year"])
                        #temp = child.text.split(',')
                        if '?' in child.text:
                            gapYears.append(child.attrib["Year"])
                        if int(child.attrib["Year"]) >= startYear:
                            currentStats[child.attrib["Year"]] = child.text.split(',')
                        else:
                            pastStats[child.attrib["Year"]] = child.text.split(',')
            if node.tag == 'Summary':
                for child in node:
                    if child.tag == 'Min':
                        longTermStats["Min"] = child.text.split(',')
                    if child.tag == 'Max':
                        longTermStats["Max"] = child.text.split(',')
                    if child.tag == 'Mean':
                        longTermStats["Mean"] = child.text.split(',')
        
        print('Data for this site is available from ', list(pastStats.keys())[0])
        print('The following years have data gaps')
        print(gapYears)
        
        ##pragma code
        currentStats = clearBlanksConvert(currentStats)
        pastStats = clearBlanksConvert(pastStats)
        #print(pastStats)
        longTermStats = clearBlanksConvert(longTermStats)
        #print(currentStats,longTermStats)
        
        doThePlots(currentStats,longTermStats)
        
        print("NSE Threshold",round(gof.nse(0.75*np.array(longTermStats['Mean'][:12]),longTermStats['Mean'][:12]),3))
        doTheTable(currentStats,longTermStats)
        
        print("Trends in annual mean flow")
        trendClim = doTheAnnMeanFlowTrends(pastStats) #return two results for original and modified tests, use index 0
        print(list(pastStats.keys())[0],'-',list(pastStats.keys())[-1],':', trendClim[0].trend)
        trendAll = doTheAnnMeanFlowTrends(pastStats,currentStats)
        print(list(pastStats.keys())[0],'-',list(currentStats.keys())[-1],':',trendAll[0].trend)
        if trendClim[0].h is True and trendAll[0].h is True :
            print(list(pastStats.keys())[0],'-',list(pastStats.keys())[-1],':', trendClim[0].slope)
            print(list(pastStats.keys())[0],'-',list(currentStats.keys())[-1],':',trendAll[0].slope)
        
def clearBlanksConvert(jsonVar):
    for thsKeys in jsonVar:
            verifiedSets = []
            for vals in jsonVar[thsKeys]:
                try:
                    verifiedSets.append(float(vals)/1000) #lps to cu.m/s
                except Exception as e:
                    verifiedSets.append(float('nan'))
            jsonVar[thsKeys] = verifiedSets
    return jsonVar

def doTheAnnMeanFlowTrends(pastStats, currentStats = None):
    tSeries = []
    for key in pastStats:
        #tSeries.extend(pastStats[key][:12])
        tSeries.append(pastStats[key][12])
        #print(pastStats[key])
    
    if currentStats is not None:
        for key in currentStats:
            tSeries.append(currentStats[key][12])
    
    #print(len(tSeries), sum(tSeries)/len(tSeries))
    #print(tSeries)
    return [pmk.original_test(tSeries), pmk.hamed_rao_modification_test(tSeries)]

def doTheTable(currentStats,longTermStats):
    monthlyStatsTable = Table(('Analysis period',
                               'Jul', 'Aug','Sep','Oct','Nov','Dec',
                               'Jan','Feb','Mar','Apr','May','Jun','NSE'))
    
    #print(currentStats)
    for thsYear in currentStats:
        #print(currentStats[thsYear][:12])
        temp = [thsYear+'-'+str(int(thsYear)+1)]
        temp.extend(list(currentStats[thsYear][:12]))
        
        dftemp = pd.DataFrame({'yr':currentStats[thsYear][:12],'mn':longTermStats['Mean'][:12]})
        dftemp = dftemp.dropna()
        #print(dftemp.head())
        temp.extend([round(gof.nse(dftemp['yr'], dftemp['mn']),3)])
        #print(gof.rd(dftemp['yr'], dftemp['mn']))
        
        monthlyStatsTable.add(temp)
    
    for stat in longTermStats:
        #print(currentStats[thsYear][:12])
        temp = ['long term '+stat]
        temp.extend(list(longTermStats[stat][:12]))
        temp.extend([''])
        monthlyStatsTable.add(temp)
    
    monthlyStatsTable.display_notebook()
    print('The Min Mean and Max of Annual values are for complete years only.')
        
def doThePlots(currentStats,longTermStats):        
        p1 = figure(x_axis_type="datetime", y_axis_type="log", title='Monthly mean flow' + ' at ' + mySite)
        #p1 = figure(x_axis_type="datetime", title='Monthly mean flow' + ' at ' + mySite)
        p1.xaxis.formatter=DatetimeTickFormatter(days="%b", months="%b", hours="%b", minutes="%b")
        p1.grid.grid_line_alpha=0.3
        p1.xaxis.axis_label = 'Month'
        p1.yaxis.axis_label = 'Flow cu.m/s'
    
        time = []
        for i in range(1,13):
            time.append(datetime(date.today().year, 6, 1) + timedelta(days=30*i))
        #print(time)
        
        #extremes band
        p1.varea(x=time,
                 y1=longTermStats["Min"][:12],
                 y2=longTermStats["Max"][:12],
                 legend_label='Extreme', color='grey', alpha=0.5)
        
        #normal band
        p1.varea(x=time,
                 y1=[1.25*y for y in longTermStats["Mean"][:12]],
                 y2=[0.75*y for y in longTermStats["Mean"][:12]],
                 legend_label='Normal', color='yellow', alpha=0.5)
        
        #print(len(currentStats))
        dashDeck = list(range(4, 4+2*len(currentStats)+1))
        random.shuffle(dashDeck)
        
        colorDeck = list(range(0, len(currentStats)+1))
        random.shuffle(colorDeck)
        
        lnWidthDeck = list(range(0,len(currentStats)+2))
        random.shuffle(lnWidthDeck)
        
        for thsYear in currentStats:
            p1.line(time, currentStats[thsYear][:12],  legend_label=thsYear,
                    line_dash=[dashDeck.pop(), dashDeck.pop()],
                    color=d3['Category10'][len(currentStats)+1][colorDeck.pop()], line_width=lnWidthDeck.pop())
        
        show(gridplot([[p1]], ))

############################################################################
##Alt method, might not be coherent with the stats generated by Hilltop
import logging
myFormat = "%(asctime)s: %(message)s"
logging.basicConfig(format=myFormat, level=logging.INFO,
                        datefmt="%H:%M:%S")
#data = None
def fetchData(daily=False):
    #print('fetching data: ',mySite,myMeasurement,myStartDate,myEndDate)
    #global data
    logging.info("start fetching")
    
    timeList = []
    obsList = []
    #get the observation data for each site
    requestType = "GetData"
    myWebRequest =  apiRoot + '&Request='+requestType+'&Site='+mySite+'&Measurement='+myMeasurement+'&From='+str(climStart)+'&To='+myEndDate
    if daily:
        myWebRequest += "&Interval=1 day&method=Average"
    #print(myWebRequest)
    
    r = requests.get(myWebRequest)
    #print(r.text)
    root = ET.fromstring(r.content)
    for child in root.iter('E'):
        #print(child.tag,child.attrib)
        for miter in child.iter('*'):
            #print(miter.tag,miter.text)
            if miter.tag == 'T':
                timeList.append(np.datetime64(datetime.strptime(miter.text,'%Y-%m-%dT%H:%M:%S')))#.timestamp()))
                #timeList.append(datetime.strptime(miter.text,'%Y-%m-%dT%H:%M:%S').timestamp())
            if miter.tag == 'I1':
                obsList.append(miter.text)
    
    df={'timestamp':np.array(timeList), myMeasurement:np.true_divide(np.array(obsList).astype('float'),1000)}
    data = pd.DataFrame (df, columns = ['timestamp',myMeasurement])
    #print(data.head())
    #print(hasattr(df['timestamp'], 'dtype'))
    #print(np.issubdtype(df['timestamp'].dtype, np.datetime64))
    
    logging.info("end fetching")
    return data

def genHydroStats():
    print('')
    print("Long term statistics for",mySite)
    
    dailyData = fetchData(daily=True)
    print("climatology data spans between",climStart,"and",myEndDate)    
    
    df_7dmin = dailyData[myMeasurement].rolling(7).mean()
    if df_7dmin.min() < 0:
        print("Time series data set has issues with minimum values : Negative values observed");
        df_7dmin[df_7dmin<0] = float('nan')
    dailyData['7daymean'] = df_7dmin
    #print(dailyData.head(10))
    
    print(".")
    InstData = fetchData(daily=False)
    print("..")
    #print(InstData.head())
    annualStatsTable = Table(('Analysis period',
                               '7-day min', 'Max','Mean','Median',
                              #'Q5','Q25','Q75','Q95',
                              'Q 0.5Median %t', 'Q FRE3 %t',
                              #'7-day MALF','MAMF','7-day MALF/Mean',
                              'Mean/Median'))
    
    temp = genStatsSubFn(dailyData,InstData,int(climStart.year),int(myEndDate[:4]))
    ltMedian = temp[4] #see the headings in line before the above for indices
    sevenDayMALF = temp[8]
    MAMF = temp[9]
    
    temp1 = [temp[0]]
    temp1.extend(np.around(np.asarray(temp[1:], dtype='float64'), decimals=3))
    annualStatsTable.add(temp1[:8])

    print('Long term values for this site')
    ltStatsTable = Table(('Parameter','Value'))
    ltStatsTable.add(['Climatology period',temp1[0]])
    ltStatsTable.add(['7-day MALF','{:.3f}'.format(temp1[8])])
    ltStatsTable.add(['MAMF','{:.3f}'.format(temp1[9])])
    #ltStatsTable.add(['MAMF',temp1[9]])
    ltStatsTable.add(['MALF/Mean %','{:.3f}'.format(temp1[8]/temp1[3]*100)])
    ltStatsTable.display_notebook()
    
    for yr in range(int(myStartDate[:4]),int(myEndDate[:4])):
        temp = genStatsSubFn(dailyData,InstData,yr,yr+1,ltMedian)
        temp1 = [temp[0]]
        temp1.extend(np.around(np.asarray(temp[1:], dtype='float64'), decimals=3))
        annualStatsTable.add(temp1[:8])
    
    print("Exceedence probabilities and Observations on Key parameters")
    annualStatsTable.display_notebook()
    
    #trends on low flow
    print("Trends in 7-day annual low flow")
    trendClim = pmk.original_test(Ann7daymin(dailyData,int(climStart.year),int(myStartDate[:4])))
    print(trendClim.trend,'for',int(climStart.year),'-',int(myStartDate[:4]))
    trendAll = pmk.original_test(Ann7daymin(dailyData,int(climStart.year),int(myEndDate[:4])))
    print(trendAll.trend,'for',int(climStart.year),'-',int(myEndDate[:4]))
    #print(type(trendClim.h), trendAll)
    if trendClim.h and trendAll.h :
            print(int(climStart.year),'-',int(myStartDate[:4]),':', trendClim.slope)
            print(int(climStart.year),'-',int(myEndDate[:4]),':',trendAll.slope)
    
def flowExceedanceProbab(myData,reqProbab,reqProbabInv):
    #print(reqProbab,reqProbabInv)
    myData = np.sort(myData)
    ranks = sp.rankdata(myData, method='average')
    ranks = ranks[::-1]
    prob = [100*(ranks[i]/(len(myData)+1)) for i in range(len(myData)) ]
    """
    graph = figure(title = '')
    graph.scatter(prob,myData) 
    show(graph) 
    """
    f = interpolate.interp1d(prob, myData,fill_value="extrapolate")
    y_interp = []
    for x_interp in reqProbab:
        #y_interp.append(np.interp(x_interp, prob, myData))
        y_interp.append(f(x_interp))
        
    fI = interpolate.interp1d(myData,prob,fill_value="extrapolate")
    y_interpI = []
    for x_interpI in reqProbabInv:
        #y_interp.append(np.interp(x_interp, prob, myData))
        y_interpI.append(fI(x_interpI))
        
    return y_interp, y_interpI

def Ann7daymin(dailyData,stDate,eDate):
    Annual7daymin = []
    for yr in range(stDate,eDate+1): #range end is not inclusive so add 1
        thsYrDf = dailyData[dailyData["timestamp"].between(np.datetime64(datetime.strptime(str(yr)+'-07-01 00:00:00','%Y-%m-%d %H:%M:%S')), np.datetime64(datetime.strptime(str(yr+1)+'-07-01 00:00:00','%Y-%m-%d %H:%M:%S')))]
        #print(thsYrDf.head())
        Annual7daymin.append(thsYrDf['7daymean'].min())#[myMeasurement].min()) #check this part again
    return Annual7daymin

def genStatsSubFn(dailyData,InstData,stDate,eDate,ltMedian=False):
    #temp = np.datetime64(datetime.strptime(eDate,'%Y-%m-%d %H:%M:%S'))
    #print(temp.year)
    strDtObj = np.datetime64(datetime.strptime(str(stDate)+'-07-01 00:00:00','%Y-%m-%d %H:%M:%S'))
    endDtObj = np.datetime64(datetime.strptime(str(eDate)+'-07-01 00:00:00','%Y-%m-%d %H:%M:%S'))

    Annualmax = []
    if (eDate-stDate) > 1:
        Annual7daymin = Ann7daymin(dailyData,stDate,eDate)
        #print(Annual7daymin)
        sevendayMALF = np.nanmean(np.array(Annual7daymin), dtype=np.float64)
    
        for yr in range(stDate,eDate+1): #range end is not inclusive so add 1
            thsYrDf = dailyData[dailyData["timestamp"].between(np.datetime64(datetime.strptime(str(yr)+'-07-01 00:00:00',
                                                                                               '%Y-%m-%d %H:%M:%S')),
                                                          np.datetime64(datetime.strptime(str(yr+1)+'-07-01 00:00:00',
                                                                                          '%Y-%m-%d %H:%M:%S')))]
            Annualmax.append(thsYrDf[myMeasurement].max())
        MAMF = np.nanmean(np.array(Annualmax), dtype=np.float64)
    else :
        sevendayMALF = float('nan')
        MAMF = float('nan')
    
    sevendayMin = dailyData['7daymean'][dailyData['timestamp'].between(strDtObj,endDtObj)].min()
    thsMax = InstData[myMeasurement][InstData['timestamp'].between(strDtObj,endDtObj)].max()
    thsMean = InstData[myMeasurement][InstData['timestamp'].between(strDtObj,endDtObj)].mean()
    thsMedian = InstData[myMeasurement][InstData['timestamp'].between(strDtObj,endDtObj)].median()
    
    
    redData = InstData[InstData['timestamp'].between(strDtObj,endDtObj)]
    #print(redData.head())
    if ltMedian:
        temp, temp1 = flowExceedanceProbab(redData[myMeasurement].dropna().values,[5,25,75,95],[0.5*ltMedian,3*ltMedian])
    else:
        temp, temp1 = flowExceedanceProbab(redData[myMeasurement].dropna().values,[5,25,75,95],[0.5*thsMedian,3*thsMedian])
    pc5 = temp[0]
    pc25 = temp[1]
    pc75 = temp[2]
    pc95 = temp[3]
    #print(temp1)
    
    QhalfMedian = 100 - temp1[0]
    QFRE3 = temp1[1]
    Qmean = temp[2]
    
    #return str(stDate)+'-'+str(eDate),sevendayMin,thsMax,thsMean,thsMedian,pc5,pc25,pc75,pc95,sevendayMALF,MAMF,(sevendayMALF/thsMean*100),(thsMean/thsMedian)
    return str(stDate)+'-'+str(eDate),sevendayMin,thsMax,thsMean,thsMedian,QhalfMedian,QFRE3,(thsMean/thsMedian),sevendayMALF,MAMF
    

    """
    p2 = figure(x_axis_type="datetime", y_axis_type="log", title='Daily flow' + ' at ' + mySite)
    #p2.xaxis.formatter=DatetimeTickFormatter(days="%b", months="%b", hours="%b", minutes="%b")
    p2.grid.grid_line_alpha=0.3
    p2.xaxis.axis_label = 'day'
    p2.yaxis.axis_label = 'Flow cu.m/s'
    p2.line(data["timestamp"], data[myMeasurement],  legend_label="Raw",color='yellow')
    p2.line(data["timestamp"], df_7dmin,  legend_label="mv avg",color='purple')
    show(gridplot([[p2]], ))
    
    #hist, edges = np.histogram(redData[myMeasurement], density=True, bins=100)
    #p3 = figure()
    ##p3.quad(top=np.cumsum(hist), bottom=0, left=edges[:-1], right=edges[1:],fill_color="#036564", line_color="#033649")
    #p3.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],fill_color="#036564", line_color="#033649")
    #show(gridplot([[p3]], ))
    """
    
############################################################################
def siteSelector(Site,StartDate,EndDate):
    global mySite
    mySite = Site
    print('please wait for the right options to load')
    global myStartDate
    global myEndDate
    myStartDate = str(StartDate)
    myEndDate = str(EndDate)
    if myStartDate and myEndDate :
        fetchRiverPCal()
        #sevenDayMin()
        #otherAnnStats()
        genHydroStats()
        

        
###########################################################################
"""
def sevenDayMin():
    myWebRequest =  apiRoot + "&Request=Hydro"
    #print(myWebRequest)
    #get the range of values to be applied
    myHqObj = {
        "Command" : "PExt",
        "Site" : str(mySite),
        "Measurement" : str(SFB.myMeasurement),
        "From" : str(climStart), #30 year climatology
        "To" : str(myEndDate),
        "MONTH": "July",
        "Yearly": "True",
        "Moving": "True",
        "Interval": "7 Day"
    }
    
    headers = {"Content-Type": "application/json"}
    params = {"Service":"Hilltop", "Request":"Hydro"}
    r = requests.post(myWebRequest, json = myHqObj, headers=headers, params=params)
    print(r.text)

    
def otherAnnStats():
    myWebRequest =  apiRoot + "&Request=Hydro"
    #print(myWebRequest)
    #get the range of values to be applied
    myHqObj = {
        "Command" : "P3",
        "Site" : str(mySite),
        "Measurement" : str(myMeasurement),
        "From" : str(climStart), #30 year climatology
        "To" : str(myEndDate),
        "STAT" : "Annual Stats",
        "Lower" : "5",
        "Upper" : "95",
        "MONTH": "July"
    }
    
    headers = {"Content-Type": "application/json"}
    params = {"Service":"Hilltop", "Request":"Hydro"}
    r = requests.post(myWebRequest, json = myHqObj, headers=headers, params=params)
    print(r.text)
"""