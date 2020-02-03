

# this file is just for testing out stuff

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


# training_data = pd.read_csv("../digit-recognizer/train.csv", index_col=0)
# test_data = pd.read_csv("../digit-recognizer/test.csv")
# sample_submission = pd.read_csv("../digit-recognizer/sample_submission.csv", index_col='ImageId')


# print(training_data.shape) # 42000x784
#
#
# print(test_data.shape) # 28000, 784
#
# print(sample_submission.shape) # 28000, 2
#
# print(sample_submission.head())
#
#
# print('\b\b\b\b\b\b\b\b\b\b\b')
#
# print(str(0.6 * 70) + " " + str(0.2 * 70) + " " + str(0.2 * 70))
#
#
# temp = np.zeros((10,), dtype=int)
# print(temp)
#

# vars = []
# L = 10
#
# for i in range(1, 10):
#     print(i)
#     vars.append(i)
#
# print(vars)


grads = [(1, 2), (4, 5), (5, 6)]


# print(grads[0][1])


# for i in range(20):
#     d = np.random.randint(5)
#     print(d)

#
# f = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 6, 11])
#
# d = np.where(f == np.amax(f))
# print(d[0])


# a = np.arange(1, 11)
# print(f"a: {a}")
# b = np.array(a, copy=True)
#
# print(f"b: {b}")
#
# a = np.arange(5, 122)
#
# print(f"a now: {a}")
# print(f"b now: {b}")





# t = np.random.randint(1, 11, size=(3, 6))
# print(f"t: {t}")
# print(f"n: {t.mean()}")
# print(f"axis 0: {t.mean(axis=0)}")
# print(f"axis 1: {t.mean(axis=1)}")





# pd.DataFrame.from_dict(data=mydict, orient='index').to_csv('dict_file.csv', header=False)
# nowdict = pd.read_csv('dict_file.csv', header=None, index_col=0, squeeze=True).to_dict()
# for key in nowdict.keys():
#     temp = nowdict[key]
#     nowdict[key] = np.asarray(list(temp))
#
# print(nowdict["W1"].shape)




# a = np.arange(1, 11).reshape(2, 5)
#
# print(f"a: {a}")
#
# print(f"a[0][1]: {a[:, 4]}")
#



# a = np.array([1, 2, 3, 4, 5])
# b = np.array([1, 2, 3, 5, 6])
# acc = np.sum(a == b)
# print(int(acc * 100 / 5))


# d = np.zeros((10, 30))
#
# print(f"d: {d}")
#
# print(f"d[:, 10]: {np.where(d[:, 10] == 0)}")


prac = np.zeros((1, 10))
#
# for i in range(10):
#     print(f"prac[{i}] = {prac[:, i]}")
#
#
#


import PIL

# from PIL import Image
# # load the image
# image = Image.open('../digit3.png')
# # summarize some details about the image
# gs_image = image.convert(mode='L')
# # save in jpeg format
# gs_image.save('opera_house_grayscale.jpg')
# # load the image again and show it
# image2 = Image.open('opera_house_grayscale.jpg')
# # show the image
# # image2.show()
#
# image3 = image2.resize((28, 28))
# image3.show()
# # report the size of the thumbnail
# print(type(image3))
#
#
# from PIL import Image, ImageFilter
#
#
#
# def imageprepare(argv):
#     """
#     This function returns the pixel values.
#     The imput is a png file location.
#     """
#     im = Image.open(argv).convert('L')
#     width = float(im.size[0])
#     height = float(im.size[1])
#     newImage = Image.new('L', (28, 28), (255))  # creates white canvas of 28x28 pixels
#
#     if width > height:  # check which dimension is bigger
#         # Width is bigger. Width becomes 20 pixels.
#         nheight = int(round((20.0 / width * height), 0))  # resize height according to ratio width
#         if (nheight == 0):  # rare case but minimum is 1 pixel
#             nheight = 1
#             # resize and sharpen
#         img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wtop = int(round(((28 - nheight) / 2), 0))  # calculate horizontal position
#         newImage.paste(img, (4, wtop))  # paste resized image on white canvas
#     else:
#         # Height is bigger. Heigth becomes 20 pixels.
#         nwidth = int(round((20.0 / height * width), 0))  # resize width according to ratio height
#         if (nwidth == 0):  # rare case but minimum is 1 pixel
#             nwidth = 1
#             # resize and sharpen
#         img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
#         wleft = int(round(((28 - nwidth) / 2), 0))  # caculate vertical pozition
#         newImage.paste(img, (wleft, 4))  # paste resized image on white canvas
#
#     # newImage.save("sample.png
#
#     tv = list(newImage.getdata())  # get pixel values
#
#     # normalize pixels to 0 and 1. 0 is pure white, 1 is pure black.
#     tva = [(255 - x) * 1.0 / 255.0 for x in tv]
#     # print(tva)
#     return tva
#
# image3=np.asarray(imageprepare('../digit3.png')).reshape(784, 1)#file path here
# # print(len(image3))# mnist IMAGES are 28x28=784 pixels


import pickle
# gd_inc = open("costs_gd.pickle", "rb")
# gd_cv = open("cv_gd.pickle", "rb")
#
# gd_costs = pickle.load(gd_inc)
# gd_cv_costs = pickle.load(gd_cv)
#
# pickle_inc = open("costs_place.pickle", "rb")
# pickle_cv = open("cv_costs.pickle", "rb")
#
# train_costs = pickle.load(pickle_inc)
# cv_costs = pickle.load(pickle_cv)
#
# plt.plot(train_costs, label="Adam train")
# plt.plot(cv_costs, label="Adam validation")
# plt.plot(gd_costs)
# plt.plot(gd_cv_costs)
# plt.legend(loc="upper right")
#


# def make_batches(X, y, batch_size):
#     """
#     returns a list of batches of size passed in
#     :param data: training data passed in
#     :param batch_size: batch size
#     :return: list of batches
#     """
#
#     total = X.shape[1]
#
#     permutation = np.random.permutation(total)
#
#     shuffled_x = X[:, permutation]
#     shuffled_y = y[:, permutation].reshape(1, total)
#
#     whole_batches = total // batch_size  # considering data's second dimension contains all examples
#     batches = []
#
#     for i in range(whole_batches):
#         curr_x = X[:, i * batch_size: (i + 1) * batch_size]
#         curr_y = y[:, i * batch_size: (i + 1) * batch_size]
#         batch = (curr_x, curr_y)
#         batches.append(batch)
#
#     if total % 2 != 0:
#         curr_x = X[:, whole_batches * batch_size:]
#         curr_y = y[:, whole_batches * batch_size]
#         batch = (curr_x, curr_y)
#         batches.append(batch)
#
#     return batches
#
#
# datas = np.arange(100).reshape(2, 50)
# y = np.arange(50).reshape(1, 50)
#
# print(len(make_batches(datas, y, 10)[0]))
#
#
#





#
#
# import time
#
#
#
#
# start = time.time()
#
# for i in range(3):
#     time.sleep(1)
#
# end = time.time()
#
# print(end-start)
#
#
#




print("\\")


































