import numpy as np

data_1 = np.load('./data/progressed/dayClip7.npz')
print("data1 loaded")
data_2 = np.load('./data/progressed/nightClip4.npz')
print("data2 loaded")
#data_3 = np.load('./data/progressed/dayClip13.npz')
#print("data3 loaded")
#data_4 = np.load('./data/progressed/dayClip12.npz')
#print("data4 loaded")

images = np.concatenate((data_1['images'], data_2['images']), axis=0)
print("Images concatenated")
labels = np.concatenate((data_1['boxes'], data_2['boxes']), axis=0)
print("Labels concatenated")

np.random.seed(13)
indices = np.arange(len(images))
np.random.shuffle(indices)
images, labels = images[indices], labels[indices]
print("Data shuffled")

np.savez("./data/progressed/data_6", images=images, boxes=labels)
print("Done")