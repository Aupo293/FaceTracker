#!/usr/bin/env python
# coding: utf-8
from matplotlib import pyplot as plt
import numpy as np
import os


def get_data(path_controc, path_discroc):
    path_controc = os.path.join("/Users/qiuxiaocong/Downloads/proroc", path_controc)
    path_discroc = os.path.join("/Users/qiuxiaocong/Downloads/proroc", path_discroc)

    with open(path_discroc, 'r') as fd:
        discROC = fd.readlines()
    # with open(path_controc, 'r') as fc:
    #     contROC = fc.readlines()
    # get my disc data x, y
    discROC = [line.split() for line in discROC]
    disc_x = [float(x[1]) for x in discROC]
    disc_y = [float(y[0]) for y in discROC]

    # get given cont data x, y
    # contROC = [line.split() for line in contROC]
    # cont_x = [float(x[1]) for x in contROC]
    # cont_y = [float(y[0]) for y in contROC]
    # return disc_x, disc_y, cont_x, cont_y
    return disc_x, disc_y


disc_x_my_mid_9, disc_y_my_mid_9 = get_data('rnet_0007ContROC.txt', 'pnet_0009DiscROC.txt')
disc_x_my_mid_8, disc_y_my_mid_8 = get_data('rnet_0008ContROC.txt', 'pnet_0008DiscROC.txt')
disc_x_my_mid_7, disc_y_my_mid_7 = get_data('rnet_0005ContROC.txt', 'pnet_0007DiscROC.txt')
disc_x_my_mid_6, disc_y_my_mid_6 = get_data('rnet_0019ContROC.txt', 'pnet_0006DiscROC.txt')
disc_x_my_mid_5, disc_y_my_mid_5 = get_data('rnet_0013ContROC.txt', 'pnet_0005DiscROC.txt')
disc_x_my_mid_4, disc_y_my_mid_4 = get_data('rnet_0007ContROC.txt', 'pnet_0004DiscROC.txt')
disc_x_my_mid_3, disc_y_my_mid_3 = get_data('rnet_0008ContROC.txt', 'pnet_0003DiscROC.txt')
disc_x_my_mid_2, disc_y_my_mid_2 = get_data('rnet_0005ContROC.txt', 'pnet_0002DiscROC.txt')
disc_x_my_mid_1, disc_y_my_mid_1 = get_data('rnet_0019ContROC.txt', 'pnet_0001DiscROC.txt')
disc_x_given, disc_y_given = get_data('DiscROC_pnet_given.txt', 'DiscROC_pnet_given.txt')


# disc_x_my_mid_8, disc_y_my_mid_8 = get_data('rnet_0013ContROC.txt', 'rnet_0013DiscROC.txt')
# disc_x_my_mid_7, disc_y_my_mid_7 = get_data('rnet_0007ContROC.txt', 'rnet_0007DiscROC.txt')
# disc_x_my_mid_6, disc_y_my_mid_6 = get_data('rnet_0008ContROC.txt', 'rnet_0008DiscROC.txt')
# disc_x_my_mid_5, disc_y_my_mid_5 = get_data('rnet_0005ContROC.txt', 'rnet_0005DiscROC.txt')
# disc_x_my_mid_4, disc_y_my_mid_4 = get_data('rnet_0019ContROC.txt', 'rnet_0019DiscROC.txt')
# disc_x_my_mid, disc_y_my_mid = get_data('rnet_0015ContROC.txt', 'rnet_0015DiscROC.txt')
# disc_x_my, disc_y_my = get_data('rnet_0010ContROC.txt', 'rnet_0010DiscROC.txt')
# disc_x_given, disc_y_given = get_data('ContROC_rnet_given.txt', 'DiscROC_rnet_given.txt')


# disc_x_my_mid_8, disc_y_my_mid_8 = get_data('onet_0015_givenPRnetContROC.txt', 'onet_0015_givenPRnetDiscROC.txt')
# disc_x_my_mid_7, disc_y_my_mid_7 = get_data('onet_0064ContROC.txt', 'onet_0064DiscROC.txt')
# disc_x_my_mid_6, disc_y_my_mid_6 = get_data('onet_0011ContROC.txt', 'onet_0011DiscROC.txt')
# disc_x_my_mid_5, disc_y_my_mid_5 = get_data('onet_0012ContROC.txt', 'onet_0012DiscROC.txt')
# disc_x_my_mid_4, disc_y_my_mid_4 = get_data('onet_0013ContROC.txt', 'onet_0013DiscROC.txt')
# disc_x_my_mid, disc_y_my_mid = get_data('onet_0010ContROC.txt', 'onet_0010DiscROC.txt')
# disc_x_my, disc_y_my = get_data('onet_0015ContROC.txt', 'onet_0015DiscROC.txt')
# disc_x_given, disc_y_given = get_data('ContROC.txt', 'DiscROC.txt')


# disc_x_my_mid_8, disc_y_my_mid_8 = get_data('onet_0015_givenPRnetContROC.txt', 'onet_0015_givenPRnetDiscROC.txt')
# disc_x_my_mid_7, disc_y_my_mid_7 = get_data('onet_0009_givenPRnetContROC.txt', 'onet_0009_givenPRnetDiscROC.txt')
# disc_x_my_mid_6, disc_y_my_mid_6 = get_data('onet_0008_givenPRnetContROC.txt', 'onet_0008_givenPRnetDiscROC.txt')
# disc_x_my_mid_5, disc_y_my_mid_5 = get_data('onet_0010_givenPRnetContROC.txt', 'onet_0010_givenPRnetDiscROC.txt')
# disc_x_my_mid_4, disc_y_my_mid_4 = get_data('onet_0012_givenPRnetContROC.txt', 'onet_0012_givenPRnetDiscROC.txt')
# disc_x_my_mid, disc_y_my_mid = get_data('onet_0009_givenPRnetContROC.txt', 'onet_0009_givenPRnetDiscROC.txt')
# disc_x_my, disc_y_my = get_data('onet_0012ContROC.txt', 'onet_0012DiscROC.txt')
# disc_x_given, disc_y_given = get_data('ContROC.txt', 'DiscROC.txt')

# plot data
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.figure(figsize=(10, 6))

# set y limite
plt.ylim((-0.05, 1))
# plt.xlim((-2,set_x_lim))
# print label
plt.xlabel('False Positive (FP)')
plt.ylabel('True Positive Rate (TPR)')




# plt.plot(disc_x_my_mid_9, disc_y_my_mid_9, color='cyan', linewidth=1, label='PNet_0009')
plt.plot(disc_x_my_mid_8, disc_y_my_mid_8, color='r', linewidth=2, label='Trained_PNet')
# plt.plot(disc_x_my_mid_7, disc_y_my_mid_7, color='dodgerblue', linewidth=1, label='PNet_0007')
# plt.plot(disc_x_my_mid_6, disc_y_my_mid_6, color='orange', linewidth=1, label='PNet_0006')
# plt.plot(disc_x_my_mid_5, disc_y_my_mid_5, color='black', linewidth=1, label='PNet_0005')
# plt.plot(disc_x_my_mid_4, disc_y_my_mid_4, color='tan', linewidth=1, label='PNet_0004')
# plt.plot(disc_x_my_mid_3, disc_y_my_mid_3, color='pink', linewidth=1, label='PNet_0003')
# plt.plot(disc_x_my_mid_2, disc_y_my_mid_2, color='blue', linewidth=1, label='PNet_0002')
# plt.plot(disc_x_my_mid_1, disc_y_my_mid_1, color='red', linewidth=1, label='PNet_0001')
plt.plot(disc_x_given, disc_y_given, color='b', linewidth=2, label='Given_PNet')


plt.scatter([disc_x_my_mid_8[0]], [disc_y_my_mid_8[0]], s=45, c='r',marker='x')  # stroke, colour
plt.scatter([disc_x_given[0]], [disc_y_given[0]], s=45, c='b',marker='x')  # stroke, colour

plt.text(disc_x_my_mid_8[0] - disc_x_my_mid_8[0] / 3,disc_y_my_mid_8[0] + 0.015 , 'Trained_PNet Disc Score: %.3f' %(disc_y_my_mid_8[0] * 100) + '%', color='r')
plt.text(disc_x_given[0] - disc_x_given[0] / 3 +3 ,disc_y_given[0] -0.06,'Given_PNet Disc Score: %.3f' %(disc_y_given[0] * 100) + '%',color='b')


# plt.scatter([disc_x_my_mid_8[0]], [disc_y_my_mid_8[0]], s=45, c='r',marker='x')  # stroke, colour
# plt.scatter([disc_x_given[0]], [disc_y_given[0]], s=45, c='b',marker='x')  # stroke, colour
# plt.text(disc_x_my_mid_8[0] - disc_x_my_mid_8[0] / 3,disc_y_my_mid_8[0] + 0.03, 'Trained_RNet Disc Score: %.3f' %(disc_y_my_mid_8[0] * 100) + '%', color='r')
# plt.text(disc_x_given[0] - disc_x_given[0] / 3 +1500 ,disc_y_given[0] - 0.09,'Given_RNet Disc Score: %.3f' %(disc_y_given[0] * 100) + '%',color='b')



# plt.scatter([disc_x_my_mid_8[0]], [disc_y_my_mid_8[0]], s=45, c='r',marker='x')  # stroke, colour
# plt.scatter([disc_x_given[0]], [disc_y_given[0]], s=45, c='b',marker='x')  # stroke, colour
#
# plt.text(disc_x_my_mid_8[0] - disc_x_my_mid_8[0] / 3,disc_y_my_mid_8[0] - 0.06, 'Trained_ONet Disc Score: %.3f' %(disc_y_my_mid_8[0] * 100) + '%', color='r')
# plt.text(disc_x_given[0] - disc_x_given[0] / 3 +3 ,disc_y_given[0] + 0.03,'Given_ONet Disc Score: %.3f' %(disc_y_given[0] * 100) + '%',color='b')


# print data text
plt.title('MTCNN-MXNet')
plt.grid()
# save img
# plt.figure(figsize=(10, 10))
plt.legend()
# plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
plt.savefig('/Users/qiuxiaocong/Downloads/proroc/pnet.png', bbox_inches='tight', dpi=600)
plt.show()












# path_ContROC_my_mid = "./FDDB-result/ContROC_rnet_my_0001.txt"
# path_DiscROC_my_mid = "./FDDB-result/DiscROC_rnet_my_0001.txt"
#
# path_ContROC_my_mid_4 = "./FDDB-result/ContROC_rnet_my_0004.txt"
# path_DiscROC_my_mid_4 = "./FDDB-result/DiscROC_rnet_my_0004.txt"
#
# path_ContROC_my = "./FDDB-result/ContROC_rnet_my.txt"
# path_DiscROC_my = "./FDDB-result/DiscROC_rnet_my.txt"
#
# path_ContROC_given = "./FDDB-result/ContROC_rnet_given.txt"
# path_DiscROC_given = "./FDDB-result/DiscROC_rnet_given.txt"
#
# path_imgSave = "./FDDB-result/result_rnet_comparison_0004.png"

# set_x_lim = 1000
# # get data
# with open(path_DiscROC_my_mid_4, 'r') as fd:
#     discROC_my_mid_4 = fd.readlines()
# with open(path_ContROC_my_mid_4, 'r') as fc:
#     contROC_my_mid_4 = fc.readlines()
#
# with open(path_DiscROC_my_mid, 'r') as fd:
#     discROC_my_mid = fd.readlines()
# with open(path_ContROC_my_mid, 'r') as fc:
#     contROC_my_mid = fc.readlines()
#
# with open(path_DiscROC_my, 'r') as fd:
#     discROC_my = fd.readlines()
# with open(path_ContROC_my, 'r') as fc:
#     contROC_my = fc.readlines()
#
# with open(path_DiscROC_given, 'r') as fd:
#     discROC_given = fd.readlines()
# with open(path_ContROC_given, 'r') as fc:
#     contROC_given = fc.readlines()
#
# # get my disc data x, y
# discROC_my_mid_4 = [line.split() for line in discROC_my_mid_4]
# disc_x_my_mid_4 = [float(x[1]) for x in discROC_my_mid_4]
# disc_y_my_mid_4 = [float(y[0]) for y in discROC_my_mid_4]
#
# # get given cont data x, y
# contROC_my_mid_4 = [line.split() for line in contROC_my_mid_4]
# cont_x_my_mid_4 = [float(x[1]) for x in contROC_my_mid_4]
# cont_y_my_mid_4 = [float(y[0]) for y in contROC_my_mid_4]
#
# # get my disc data x, y
# discROC_my_mid = [line.split() for line in discROC_my_mid]
# disc_x_my_mid = [float(x[1]) for x in discROC_my_mid]
# disc_y_my_mid = [float(y[0]) for y in discROC_my_mid]
#
# # get given cont data x, y
# contROC_my_mid = [line.split() for line in contROC_my_mid]
# cont_x_my_mid = [float(x[1]) for x in contROC_my_mid]
# cont_y_my_mid = [float(y[0]) for y in contROC_my_mid]
#
# # get my disc data x, y
# discROC_my = [line.split() for line in discROC_my]
# disc_x_my = [float(x[1]) for x in discROC_my]
# disc_y_my = [float(y[0]) for y in discROC_my]
#
# # get given cont data x, y
# contROC_my = [line.split() for line in contROC_my]
# cont_x_my = [float(x[1]) for x in contROC_my]
# cont_y_my = [float(y[0]) for y in contROC_my]
#
# # get disc data x, y
# discROC_given = [line.split() for line in discROC_given]
# disc_x_given = [float(x[1]) for x in discROC_given]
# disc_y_given = [float(y[0]) for y in discROC_given]
#
# # get cont data x, y
# contROC_given = [line.split() for line in contROC_given]
# cont_x_given= [float(x[1]) for x in contROC_given]
# cont_y_given = [float(y[0]) for y in contROC_given]

# get data we need to be print
# count_disc_my = len(discROC_my)
# count_cont_my = len(contROC_my)
#
# count_disc_given = len(discROC_given)
# count_cont_given = len(contROC_given)

# plt.text(disc_x_my_mid_4[0] - disc_x_my_mid_4[0] / 3,disc_y_my_mid[0] + 0.03,'MY 0004 Disc Score: %.3f' %(disc_y_my_mid_4[0] * 100) + '%',color='r')

# plt.text(disc_x_my_mid[0] - disc_x_my_mid[0] / 3,disc_y_my_mid[0] + 0.03,'MY 0001 Disc Score: %.3f' %(disc_y_my_mid[0] * 100) + '%',color='m')
# plt.text(cont_x_my_mid[0] - cont_x_my_mid[0] / 3,cont_y_my_mid[0] + 0.03,'My 0001 Cont Score: %.3f' %(cont_y_my_mid[0] * 100) + '%',color='k')

# plt.text(disc_x_my[0] - disc_x_my[0] / 3,disc_y_my[0] + 0.03,'MY Final Disc Score: %.3f' %(disc_y_my[0] * 100) + '%',color='b')
# plt.text(cont_x_my[0] - cont_x_my[0] / 3,cont_y_my[0] + 0.03,'My FinalCont Score: %.3f' %(cont_y_my[0] * 100) + '%',color='r')

# plt.text(disc_x_given[0] - disc_x_given[0] / 3,disc_y_given[0] + 0.03,'Given Disc Score: %.3f' %(disc_y_given[0] * 100) + '%',color='g')
# plt.text(cont_x_given[0] - cont_x_given[0] / 3,cont_y_given[0] + 0.03,'Given Cont Score: %.3f' %(cont_y_given[0] * 100) + '%',color='y')

# def plot(path_ContROC_file, path_DiscROC_file, path_imgSave_file, color_disc, color_cont):
#     path_ContROC = os.path.join("./FDDB-result", path_ContROC_file)
#     path_DiscROC = os.path.join("./FDDB-result", path_DiscROC_file)
#
#     with open(path_DiscROC, 'r') as fd:
#         discROC = fd.readlines()
#
#     with open(path_ContROC, 'r') as fc:
#         contROC = fc.readlines()
#
#     # get my disc data x, y
#     discROC = [line.split() for line in discROC]
#     disc_x = [float(x[1]) for x in discROC]
#     disc_y = [float(y[0]) for y in discROC]
#
#     # get given cont data x, y
#     contROC = [line.split() for line in contROC]
#     cont_x = [float(x[1]) for x in contROC]
#     cont_y = [float(y[0]) for y in contROC]
#
#     ### plot data
#     plt.figure()
#
#     # set y limite
#     plt.ylim((-0.07, 1))
#     # plt.xlim((-2,set_x_lim))
#     # print label
#     plt.xlabel('False Positive (FP)')
#     plt.ylabel('True Positive Rate (TPR)')
#
#     # plot data
#     plt.plot(disc_x, disc_y, color=color_disc, linewidth=3.0)
#     plt.plot(cont_x, cont_y, color=color_cont, linewidth=3.0)
#
#     # print data text
#     plt.title('MTCNN-MXNet')
#     plt.text(disc_x[0] - disc_x[0]/3, disc_y[0]+0.03, '% Disc Score: %.3f' % (path_DiscROC_file, disc_y[0]*100) + '%', color=color_disc)
#     plt.text(cont_x[0] - cont_x[0]/3, cont_y[0]+0.03, '% Cont Score: %.3f' % (path_ContROC_file, cont_y[0]*100) + '%', color=color_cont)
#
#     plt.grid()
#
#     # save img
#     # plt.figure(figsize=(10, 10))
#     # result_rnet_comparison_0001.png
#     path_imgSave = os.path.join("./FDDB-result", path_imgSave_file)
#     plt.savefig(path_imgSave)
#     plt.show()
#
#
# if __name__ == '__main__':
#     plot()
#     plot()





