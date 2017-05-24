import tensorflow as tf
data_1 = [1, 2, 3, 4, 5, 6, 7, 8]
data_2 = [[1, 2, 3, 4, 5, 6, 7, 8],
	  [11, 12, 13, 14, 15, 16, 17, 18]]
data_3 = [[[1, 1, 1], [2, 2, 2]],
	  [[3, 3, 3], [4, 4, 4]],
	  [[5, 5, 5], [6, 6, 6]]]

x = tf.strided_slice(data_1, [0], [4]) # start from [0], total 4-0=4 numbers
y = tf.strided_slice(data_1, [1], [5]) # start from [1], total 5-1=4 numbers

# start from [0, 0]=1, total [1-0, 4-0]=[1, 4]
a = tf.strided_slice(data_2, [0, 0], [1, 4]) # [1, 2, 3, 4]
b = tf.strided_slice(data_2, [1, 1], [2, 5]) # [11, 12, 13, 14]

# Firstly, the first dimenssion is [1, 2], 1. The first dimension starts from index 1, length=1
# Secondly, the second dimenssion is [0, 1], 1. The second dimension starts from index 0, length=1
# Finally, the third dimenssion is [0, 3], 3. The third dimenssion starts from index 0, length=3
i = tf.strided_slice(data_3, [1, 0, 0], [2, 1, 3], [1, 1, 1]) # [[[3, 3, 3]]]

# Firstly, the first dimenssion is [1, 2], 1. The first dimension starts from index 1, length=1
# Secondly, the second dimenssion is [0, 2], 2. The second dimension starts from index 0, length=2
# Finally, the third dimenssion is [0, 3], 3. The third dimenssion starts from index 0, length=3
j = tf.strided_slice(data_3, [1, 0, 0], [2, 2, 3], [1, 1, 1]) #[[[3, 3, 3], [4, 4, 4]]]

# Firstly, the first dimenssion is [0, 3], 3. The first dimension starts from index 0, length=3
# Secondly, the second dimenssion is [0, 2], 2. The second dimension starts from index 0, length=2
# Finally, the third dimenssion is [1, 3], 2. The third dimenssion starts from index 1, length=2
k = tf.strided_slice(data_3, [0, 0, 1], [3, 2, 3], [1, 1, 1]) # [[[1, 1], [2, 2]],
							      # [[3, 3], [4, 4]],
							      #	 [[5, 5], [6, 6]]]

# Firstly, the first dimenssion is [1, 2], 1. The first dimension starts from index 1
# Secondly, the second dimenssion is [-1, -3], -2. The second dimension has length -2
# Finally, the third dimenssion is [0, 3], 3. The third dimenssion has length 3
l = tf.strided_slice(data_3, [1, -1, 0], [2, -3, 3], [1, -1, 1]) # [[[4, 4, 4], [3, 3, 3]]]

with tf.Session() as sess:
	print("x", sess.run(x))
	print("y", sess.run(y))
	print("a", sess.run(a))
	print("b", sess.run(b))
	print("i", sess.run(i))
	print("j", sess.run(j))
	print("k", sess.run(k))
	print("l", sess.run(l))
