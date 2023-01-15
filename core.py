import csv
import matplotlib.pyplot as plt
import itertools
from scipy.signal import find_peaks
import numpy as np
import scipy
import os
import glob
from sklearn import linear_model
import re
import pickle

class calibration:
    def __init__(self,list,weighted,split=True,split_point=500,intercept=True):

        # list.sort(key= lambda x:x.MC_area ,reverse=False)
        self.comp = list[0].comp
        self.ISTD = list[0].ISTD
        self.concs = [s.conc for s in list]
        self.split = split
        self.concs_hi = [s for s in self.concs if s >= split_point]
        self.concs_lo = [s for s in self.concs if s <= split_point]

        self.area_weighted = [s.normalized_area for s in list]
        self.area_weighted_hi = [s.normalized_area for s in list if s.conc >= split_point]
        self.area_weighted_lo = [s.normalized_area for s in list if s.conc <= split_point]

        self.area = [s.area for s in list]
        self.area_hi = [s.area for s in list if s.conc >= split_point]
        self.area_lo = [s.area for s in list if s.conc <= split_point]
        self.weighted = weighted
        if weighted:
            if split:
                regr_hi = linear_model.LinearRegression(fit_intercept=True,copy_X=True)
                regr_hi.fit(np.array(self.concs_hi).reshape((-1, 1)), np.array(self.area_weighted_hi))
                self.slope_hi = regr_hi.coef_
                self.intercept_hi = regr_hi.intercept_
                self.r_squared_hi = (regr_hi.score(np.array(self.concs_hi).reshape((-1, 1)),
                                                   np.array(self.area_weighted_hi))) ** 2

                regr_lo = linear_model.LinearRegression(fit_intercept=True,copy_X=True)
                regr_lo.fit(np.array(self.concs_lo).reshape((-1, 1)), np.array(self.area_weighted_lo))
                self.slope_lo = regr_lo.coef_
                self.intercept_lo = regr_lo.intercept_
                self.r_squared_lo = (regr_lo.score(np.array(self.concs_lo).reshape((-1, 1)),
                                                   np.array(self.area_weighted_lo))) ** 2
                # self.slope_hi, self.intercept_hi, self.r_value_hi, self.p_value_hi, self.std_err_hi = scipy.stats.linregress(
                #     self.concs_hi, self.area_weighted_hi)
                # self.r_squared_hi = self.r_value_hi ** 2
                #
                # self.slope_lo, self.intercept_lo, self.r_value_lo, self.p_value_lo, self.std_err_lo = scipy.stats.linregress(
                #     self.concs_lo, self.area_weighted_lo)
                # self.r_squared_lo = self.r_value_lo ** 2

            else:
                regr = linear_model.LinearRegression(fit_intercept=intercept,copy_X=True)
                regr.fit(np.array(self.concs).reshape((-1, 1)), np.array(self.area_weighted))
                self.slope = regr.coef_[0]
                self.intercept = regr.intercept_
                self.r_squared = (regr.score(np.array(self.concs).reshape((-1, 1)), np.array(self.area_weighted))) ** 2
                # self.slope, self.intercept, self.r_value, self.p_value, self.std_err = scipy.stats.linregress(self.concs, self.area_weighted)
                # self.r_squared = self.r_value ** 2
        else:
            if split:
                regr_hi = linear_model.LinearRegression(fit_intercept=True,copy_X=True)
                regr_hi.fit(np.array(self.concs_hi).reshape((-1, 1)), np.array(self.area_hi))
                self.slope_hi = regr_hi.coef_
                self.intercept_hi = regr_hi.intercept_
                self.r_squared_hi = (regr_hi.score(np.array(self.concs_hi).reshape((-1,1)),np.array(self.area_hi)))**2

                regr_lo = linear_model.LinearRegression(fit_intercept=True,copy_X=True)
                regr_lo.fit(np.array(self.concs_lo).reshape((-1, 1)), np.array(self.area_lo))
                self.slope_lo = regr_lo.coef_
                self.intercept_lo = regr_lo.intercept_
                self.r_squared_lo = (regr_lo.score(np.array(self.concs_lo).reshape((-1, 1)),
                                                   np.array(self.area_lo))) ** 2
                # self.slope_hi, self.intercept_hi, self.r_value_hi, self.p_value_hi, self.std_err_hi = scipy.stats.linregress(
                #     self.concs_hi, self.area_hi)
                # self.r_squared_hi = self.r_value_hi ** 2
                #
                # self.slope_lo, self.intercept_lo, self.r_value_lo, self.p_value_lo, self.std_err_lo = scipy.stats.linregress(
                #     self.concs_lo, self.area_lo)
                # self.r_squared_lo = self.r_value_lo ** 2
            else:
                regr = linear_model.LinearRegression(fit_intercept=intercept,copy_X=True)
                regr.fit(np.array(self.concs).reshape((-1,1)),np.array(self.area))
                self.slope = regr.coef_
                self.intercept = regr.intercept_
                self.r_squared = (regr.score(np.array(self.concs).reshape((-1,1)),np.array(self.area)))**2
                # self.slope, self.intercept, self.r_value, self.p_value, self.std_err = scipy.stats.linregress(self.concs, self.area)
                # self.r_squared = self.r_value ** 2

    def plot(self):
        if self.split:
            y_m_hi = []  # initialze arrays to plot calibration curve
            x_m_hi = [self.concs_hi[0], self.concs_hi[-1]]

            for i in x_m_hi:
                y_m_hi.append((i * self.slope_hi) + self.intercept_hi)

            plt.plot(x_m_hi, y_m_hi, '--g')

            y_m_lo = []  # initialze arrays to plot colibration curve
            x_m_lo = [self.concs_lo[0], self.concs_lo[-1]]

            for i in x_m_lo:
                y_m_lo.append((i * self.slope_lo) + self.intercept_lo)

            plt.plot(x_m_lo, y_m_lo, '--r')

        else:
            y_m = []  # initialze arrays to plot colibration curve
            x_m = [self.concs[0], self.concs[-1]]

            for i in x_m:
                y_m.append((i * self.slope) + self.intercept)

            plt.plot(x_m, y_m, '--g')

        if self.weighted:
            if self.split:
                plt.scatter(self.concs, self.area_weighted)
                plt.text(5, 0.9 * max(self.area_weighted), '$R^2 = $ ' + str(np.around(self.r_squared_lo, decimals=4)),
                         fontsize=14)
                plt.text(5, 0.82 * max(self.area_weighted),
                         '$y = $' + str(np.around(self.slope_lo, decimals=4)) + '$x$' + ' + ' + str(
                             np.around(self.intercept_lo, decimals=4)),
                         fontsize=14)

                plt.scatter(self.concs, self.area_weighted)
                plt.text(60, 0.3 * max(self.area_weighted), '$R^2 = $ ' + str(np.around(self.r_squared_hi, decimals=4)),
                         fontsize=11)
                plt.text(60, 0.22 * max(self.area_weighted),
                         '$y = $' + str(np.around(self.slope_hi, decimals=4)) + '$x$' + ' + ' + str(
                             np.around(self.intercept_hi, decimals=4)),
                         fontsize=11)
                plt.ylabel('Area under MC-LR peak/IS peak')
                plt.xlabel('Concentration')
                plt.title('Linear regression of all points')
                plt.axes([0.2, 0.45, 0.2, 0.2])
                plt.scatter(self.concs, self.area_weighted)
                plt.plot(x_m_lo, y_m_lo, '--r')
                plt.xlim([0, 1.1])
                plt.ylim([0, max(self.area_weighted_lo) + 0.1 * max(self.area_weighted_lo)])
                plt.title('lower end')

            else:
                plt.scatter(self.concs, self.area_weighted)
                plt.text(5, 0.9 * max(self.area_weighted), '$R^2 = $ ' + str(np.around(self.r_squared, decimals=4)), fontsize=14)
                plt.text(5, 0.82 * max(self.area_weighted),
                         '$y = $' + str(np.around(self.slope, decimals=4)) + '$x$' + ' + ' + str(np.around(self.intercept, decimals=4)),
                         fontsize=14)
                plt.ylabel(f'Area under {self.comp} peak/{self.ISTD} peak')
                plt.xlabel('Concentration $[ng]$')
                plt.title(f'Calibration curve for {self.comp}')
                plt.grid(alpha=0.4)

                plt.axes([0.7,0.18,0.18,0.18])
                plt.scatter(self.concs, self.area_weighted)
                plt.plot(x_m, y_m, '--g')
                plt.xlim([0,max(self.concs_lo)+0.1*max(self.concs_lo)])
                plt.ylim([0,max(self.area_weighted_lo)+0.1*max(self.area_weighted_lo)])
                plt.title('lower end')
                plt.grid(alpha=0.4)
                fig = plt.gcf()
                fig.set_size_inches(7, 5)
                return fig
        else:
            if self.split:
                plt.scatter(self.concs, self.area)
                plt.text(5, 0.9 * max(self.area), '$R^2 = $ ' + str(np.around(self.r_squared_lo, decimals=4)),
                         fontsize=11)
                plt.text(5, 0.82 * max(self.area),
                         '$y = $' + str(np.around(self.slope_lo, decimals=4)) + '$x$' + ' + ' + str(
                             np.around(self.intercept_lo, decimals=4)),
                         fontsize=11)

                plt.scatter(self.concs, self.area)
                plt.text(60, 0.3 * max(self.area), '$R^2 = $ ' + str(np.around(self.r_squared_hi, decimals=4)),
                         fontsize=11)
                plt.text(60, 0.22 * max(self.area),
                         '$y = $' + str(np.around(self.slope_hi, decimals=4)) + '$x$' + ' + ' + str(
                             np.around(self.intercept_hi, decimals=4)),
                         fontsize=11)
                plt.ylabel('Area under MC-LR peak')
                plt.axes([0.2, 0.45, 0.2, 0.2])
                plt.scatter(self.concs, self.area)
                plt.plot(x_m_lo, y_m_lo, '--r')
                plt.xlim([0, 1.1])
                plt.ylim([0, max(self.area_lo) + 0.1 * max(self.area_lo)])
                plt.title('lower end')
            else:
                plt.scatter(self.concs, self.area)
                plt.text(5, 0.9 * max(self.area), '$R^2 = $ ' + str(np.around(self.r_squared, decimals=4)), fontsize=14)
                plt.text(5, 0.82 * max(self.area),
                         '$y = $' + str(np.around(self.slope, decimals=4)) + '$x$' + ' + ' + str(
                             np.around(self.intercept, decimals=4)))
                plt.ylabel('Area under MC-LR peak')
                plt.xlabel('Concentration')
                plt.title('Linear regression of all points')
                plt.axes([0.2, 0.45, 0.2, 0.2])
                plt.scatter(self.concs, self.area)
                plt.plot(x_m, y_m, '--g')
                plt.xlim([0, 1.1])
                plt.ylim([0, max(self.area_lo)+0.1*max(self.area_lo)])
                plt.title('lower end')





    def export(self):
        return pickle.dumps(self)


def import_calibration(path):
    a = pickle.load(open(path, 'rb'))
    return a

def predict(sample, cal):

    calibration = cal[sample.comp]

    if calibration.weighted:
        if calibration.split:
            if (sample.area/sample.ISTD_area) < min(calibration.area_weighted_hi):
                sample.predicted_conc = ((sample.MC_area / sample.IS_area) - calibration.intercept_lo) / calibration.slope_lo
            else:
                sample.predicted_conc = ((sample.MC_area / sample.IS_area) - calibration.intercept_hi) / calibration.slope_hi
        else:
            sample.conc = ((sample.area/sample.ISTD_area)-calibration.intercept)/calibration.slope
    else:
        if calibration.split:
            if sample.MC_area < min(calibration.area_hi):
                sample.predicted_conc = (sample.MC_area - calibration.intercept_lo) / calibration.slope_lo
            else:
                sample.predicted_conc = (sample.MC_area - calibration.intercept_hi) / calibration.slope_hi
        else:
            sample.predicted_conc = ((sample.MC_area) - calibration.intercept)/calibration.slope
    return sample.conc

def ICV(sample,calibration,ac_criteria):
    Hlimit = sample.conc + sample.conc*ac_criteria[1]
    Llimit = sample.conc - sample.conc*ac_criteria[0]
    if  Llimit <= predict(sample,calibration) <= Hlimit:
        a = str(np.around(predict(sample,calibration),2))+' is in range (' + str(Llimit) + ' ,' + str(Hlimit) + ')'
        b = True
        return (a,b)
    else:
        a = str(predict(sample, calibration)) + ' is not in range (' + str(Llimit) + ' ,' + str(Hlimit) + ' )'
        b = False
        return (a,b)

class sample:
    def __init__(self,name,comp,ISTD,conc,area, ISTD_area):
        self.name = name
        self.comp = comp
        self.ISTD = ISTD
        self.conc = conc
        self.area = area
        self.ISTD_area = ISTD_area

        self.normalized_area = self.area/self.ISTD_area


