#!/usr/bin/env python
# coding: utf-8

# In[5]:




#importing libraries
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import h5py
from numpy import random
from scipy.stats import norm
def CemalGuven_Adal_21703986_hw3(question):
    if question == '1' :
        #---------------------------------------------------------------------------Question1--------------------------------------------------------------
        #loading .mat file to python
        data1 = h5py.File('hw3_data2.mat','r')
        Y=np.array(data1['Yn']).T
        X=np.array(data1['Xn']).T
        np.size(Y)




        def Ridge(landa,x,y):
            xt=np.transpose(x)
            w=np.linalg.inv((np.dot(xt,x)+landa*np.identity(100)))@(xt)@(y)
            return w






        def foldvalidation(x,y):
            c=np.logspace(0,12,num=1000,base=10)
            Rvalidarray=np.zeros(1000)
            Rtestarray=np.zeros(1000)
            rv=0
            for i in range(0,1000):
                landa=c[i]
                Rvalidtoplam=0
                Rtesttoplam=0
                warray=np.zeros((100,1))


                for a in range(0,10):

                    if(a<9):
                        xtest=x[100*a:(a+1)*100,:]
                        ytest=y[100*a:(a+1)*100,:]
                        xvalid=x[100*(a+1):(a+2)*100,:]
                        yvalid=y[100*(a+1):(a+2)*100,:]
                        xtrain=np.concatenate((x[100*(a+2):,:],x[0:(a)*100,:]),axis=0)
                        ytrain=np.concatenate((y[100*(a+2):,:],y[0:a*100,:]),axis=0)
                        w=Ridge(landa,xtrain,ytrain)
                        yp=xtest@w
                        ypvalid=xvalid@w
                        Rvalid=np.sum((yvalid-ypvalid)*(yvalid-ypvalid))/100
                        Rtest=np.sum((ytest-yp)*(ytest-yp))/100
                        R1valid=1-(Rvalid/np.var(ytrain))
                        R1test=1-(Rtest/np.var(ytrain))
                        Rvalidtoplam=Rvalidtoplam+R1valid
                        Rtesttoplam=Rtesttoplam+R1test
                    else:
                        xtest=x[900:,:]
                        ytest=y[900:,:]
                        xvalid=x[0:100,:]
                        yvalid=y[0:100,:]
                        xtrain=x[100:900,:]
                        ytrain=y[100:900,:]
                        w=Ridge(landa,xtrain,ytrain)
                        yp=xtest@w
                        ypvalid=xvalid@w
                        Rvalid=np.sum((yvalid-ypvalid)*(yvalid-ypvalid))/100
                        Rtest=np.sum((ytest-yp)*(ytest-yp))/100
                        R1valid=1-(Rvalid/np.var(ytrain))
                        R1test=1-(Rtest/np.var(ytest))
                        Rvalidtoplam=Rvalidtoplam+R1valid
                        Rtesttoplam=Rtesttoplam+R1test

                if(rv<Rvalidtoplam):
                    w1=w
                    i0=i
                    rv=Rvalidtoplam

                Rvalidarray[i]=Rvalidtoplam/10
                Rtestarray[i]=Rtesttoplam/10
            return Rtestarray,Rvalidarray,w1,i0
        c=np.logspace(0,12,num=1000,base=10)
        plt.figure()
        Rtest,Rvalid,w,i=foldvalidation(X,Y)                           
        plt.plot(c,Rtest)
        plt.plot(c,Rvalid)
        plt.plot()
        plt.title("R^2 vs λ")
        plt.ylabel("R^2")
        plt.xlabel("λ")
        plt.xscale("log")
        plt.grid()
        plt.show()
        display(w)


        optlanda=np.argmax(Rtest)
        print("lamda optimal =")
        c[optlanda]


        def bootstrap(x,y):
            bootx=np.zeros((1000,100))
            booty=np.zeros((1000,1))
            for j in range(1000):
                i = random.randint(1000)
                bootx[j,:]=x[i,:]
                booty[j,:]=y[i,:]
            return bootx,booty
        #for ols
        warrayols=np.zeros((500,100))
        for i in range(500):
            bx,by=bootstrap(X,Y)
            w=Ridge(0,bx,by).reshape(1,100)
            warrayols[i,:]=w
        meanols=np.mean(warrayols,axis=0,keepdims=True).reshape(100,1)
        varols=np.var(warrayols,axis=0,keepdims=True).reshape(100,1)
        boundols=(2.62*np.sqrt(varols)).reshape(100)
        pols=meanols/((np.sqrt(varols)))
        pols1=[]
        for i in range(100):
            if (1.96<pols[i] or pols[i]<-1.96):
                pols1=np.append(pols1,i+1)
        display("indexes of ols")
        display(pols1)
        #FOR OPTimum w
        warrayopt=np.zeros((500,100))
        for i in range(500):
            kx,ky=bootstrap(X,Y)
            w=Ridge(c[optlanda],kx,ky).reshape(1,100)
            warrayopt[i,:]=w
        meanopt=np.mean(warrayopt,axis=0,keepdims=True).reshape(100,1)
        varopt=np.var(warrayopt,axis=0,keepdims=True).reshape(100,1)
        boundopt=(2.62*np.sqrt(varopt)).reshape(100)
        x4=np.arange(1,101,1).reshape(100,1)

        display(meanols)
        display(meanols.shape)
        plt.figure(figsize=(10,8))
        plt.title("Ridge Regression with λ=0 ols and and %95 confidence interval")
        plt.errorbar(x4,meanols,yerr=boundols,ecolor='r',elinewidth=1,fmt='k',capsize=2)
        plt.ylabel("Weight values")
        plt.xlabel("Weight indices")
        plt.grid()
        plt.show()
        #part c
        plt.figure(figsize=(10,8))
        plt.title("Ridge Regression with λ=213 optimum and and %95 confidence interval")
        plt.errorbar(x4,meanopt,yerr=boundopt,ecolor='r',elinewidth=1,fmt='k',capsize=2)
        plt.ylabel("Weight values")
        plt.xlabel("Weight indices")
        plt.grid()
        plt.show()
        polopt=meanopt/((np.sqrt(varopt)))
        pols2=[]
        for i in range(100):
            if (1.96<polopt[i] or polopt[i]<-1.96):
                pols2=np.append(pols2,i+1)
        display("indexes of opt")
        display(pols2)
    elif question == '2' :
        
        #-------------------QUESION2------------


        #Question 2
        #part a
        #loading .mat file to python
        data2 = h5py.File('hw3_data3.mat','r')
        pop1=np.array(data2['pop1'])
        pop2=np.array(data2['pop2'])
        np.size(pop1)


        def bootstrap2(x,y):
            bootx=np.zeros(7)
            booty=np.zeros(5)
            for j in range(7):
                i = random.randint(7)
                bootx[j]=x[i,:]
            for j in range(5):
                i = random.randint(5)
                booty[j]=y[i,:]
            return bootx,booty
        differencemean=[]
        for j in range(10000):
            popboot1,popboot2=bootstrap2(pop1,pop2)
            meanpop1=np.mean(popboot1)
            meanpop2=np.mean(popboot2)

            differencemean=np.append(differencemean,meanpop1-meanpop2)
        plt.figure()
        plt.title("Population of the Difference Mean")
        plt.ylabel("p(x)")
        plt.xlabel("Difference of Means")
        plt.hist(differencemean,bins=100,edgecolor='black')
        plt.show()

        vardifmean=np.std(differencemean)
        meandifmean=np.mean(differencemean)
        zvalue=abs((meandifmean/vardifmean))
        print("z value")
        print(zvalue)
        p=2*(1-norm.cdf(zvalue))
        print("two sided p value")
        print(p)



        #Part b
        def bootstrap4(x,y):
            bootx=np.zeros((50,1))
            booty=np.zeros((50,1))
            for j in range(50):
                i = random.randint(50)
                bootx[j,:]=x[i,:]
                booty[j,:]=y[i,:]
            return bootx,booty
        data3 = h5py.File('hw3_data3.mat','r')
        vox1=np.array(data3['vox1'])
        vox2=np.array(data3['vox2'])
        np.size(vox2)
        corr=[]
        c1=np.zeros(10000)
        for j in range(10000):
            voxboot1,voxboot2=bootstrap4(vox1,vox2)
            meanvox1=np.mean(voxboot1)
            meanvox2=np.mean(voxboot2)


            toplamvox=np.sum(voxboot1*voxboot2)
            c1[j]=((1/50)*toplamvox-meanvox1*meanvox2)/(np.std(voxboot1)*np.std(voxboot2))


        plt.figure()
        plt.title("Population of the Difference Correlation")
        plt.ylabel("p(x)")
        plt.xlabel("Correlation")
        plt.hist(c1,bins=100,edgecolor='black')
        plt.show()
        meancor=np.mean(c1)
        varcor=np.var(c1)
        print("mean of correlation")
        print(meancor)
        print("std of correlation")
        print(np.sqrt(varcor))
        upconfidence=meancor+1.96*np.sqrt(varcor) 
        lowconfidence=meancor-1.96*np.sqrt(varcor) 
        print("upper interval")
        print(upconfidence)
        print("lower interval")
        print(lowconfidence)
        #Percentile of bootstrap
        #Percentile is 0 because there is no pdf at 0 as it can be seen in histogram


        #Part c
        def bootstrap3(x,y):
            bootx=np.zeros((50,1))
            booty=np.zeros((50,1))
            for j in range(50):
                i = random.randint(50)
                bootx[j,:]=x[i,:]
                booty[j,:]=y[i,:]
            for j in range(50):
                i = random.randint(50)
                booty[j,:]=y[i,:]
            return bootx,booty
        data3 = h5py.File('hw3_data3.mat','r')
        vox1=np.array(data3['vox1'])
        vox2=np.array(data3['vox2'])
        np.size(vox2)
        corr=[]
        c1=np.zeros(10000)
        for j in range(10000):
            voxboot1,voxboot2=bootstrap3(vox1,vox2)
            meanvox1=np.mean(voxboot1)
            meanvox2=np.mean(voxboot2)


            toplamvox=np.sum(voxboot1*voxboot2)
            c1[j]=((1/50)*toplamvox-meanvox1*meanvox2)/(np.std(voxboot1)*np.std(voxboot2))


        plt.figure()
        plt.title("Population of the Difference Correlation")
        plt.ylabel("p(x)")
        plt.xlabel("Correlation")
        plt.hist(c1,bins=100,edgecolor='black')
        plt.show()
        meancor=np.mean(c1)
        varcor=np.var(c1)
        print("mean of correlation")
        print(meancor)
        print("std of correlation")
        print(np.sqrt(varcor))
        upconfidence=meancor+1.96*np.sqrt(varcor) 
        lowconfidence=meancor-1.96*np.sqrt(varcor) 
        print("upper interval")
        print(upconfidence)
        print("lower interval")
        print(lowconfidence)
        zc=np.abs(meancor/varcor)
        print("z value for c")
        print(zc)

        pvaluec=(norm.cdf(zc))
        print("one sided p value for c")
        print(pvaluec)



        #part d
        #taking data
        data4 = h5py.File('hw3_data3.mat','r')
        face=np.array(data4['face'])
        build=np.array(data4['building'])
        np.size(face)

        def bootstrap5(x,y):
            bootx=np.zeros((20,1))
            booty=np.zeros((20,1))
            for j in range(20):
                i = random.randint(20)
                bootx[j,:]=x[i,:]
                booty[j,:]=y[i,:]
            return bootx,booty
        differencemean=[]
        for j in range(10000):
            popboot1,popboot2=bootstrap5(face,build)
            meanpop1=np.mean(popboot1)
            meanpop2=np.mean(popboot2)

            differencemean=np.append(differencemean,meanpop1-meanpop2)
        plt.figure()
        plt.title("Population of the Difference Mean Subject Same")
        plt.ylabel("p(x)")
        plt.xlabel("Difference of Means")
        plt.hist(differencemean,bins=100,edgecolor='black')
        plt.show()

        vardifmean=np.std(differencemean)
        meandifmean=np.mean(differencemean)
        zvalue=abs((meandifmean/vardifmean))
        print("z value")
        print(zvalue)
        p=2*(1-norm.cdf(zvalue))
        print("two sided p value")
        print(p)


        #part e
        def bootstrap6(x,y):
            bootx=np.zeros((20,1))
            booty=np.zeros((20,1))
            for j in range(20):
                i = random.randint(20)
                bootx[j,:]=x[i,:]
                booty[j,:]=y[i,:]
            for j in range(20):
                i = random.randint(20)
                booty[j,:]=y[i,:]
            return bootx,booty
        differencemean=[]
        for j in range(10000):
            popboot1,popboot2=bootstrap6(face,build)
            meanpop1=np.mean(popboot1)
            meanpop2=np.mean(popboot2)

            differencemean=np.append(differencemean,meanpop1-meanpop2)
        plt.figure()
        plt.title("Population of the Difference Mean Subject Different")
        plt.ylabel("p(x)")
        plt.xlabel("Difference of Means")
        plt.hist(differencemean,bins=100,edgecolor='black')
        plt.show()

        vardifmean=np.std(differencemean)
        meandifmean=np.mean(differencemean)
        zvalue=abs((meandifmean/vardifmean))
        print("z value")
        print(zvalue)
        p=2*(1-norm.cdf(zvalue))
        print("two sided p value")
        print(p)

question=input("enter question number")
CemalGuven_Adal_21703986_hw3(question)


# In[ ]:




