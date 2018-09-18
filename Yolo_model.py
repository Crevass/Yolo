import tensorflow as tf
import numpy as np
#from tf_util import conv2d, leaky_relu, maxpooling, fc
slim = tf.contrib.slim

class MODEL(object):
	def __init__(self, scale_width, scale_height, final_w, final_h, class_num, box_num, is_training, batch_size):
		self.batch_size = batch_size
		self.scale_height = scale_height
		self.scale_width = scale_width
		self.final_w = final_w
		self.final_h = final_h
		self.class_num = class_num
		self.box_num = box_num

		self.boundary1 = final_w * final_h * class_num
		self.boundary2 = self.boundary1 + final_w * final_h * box_num

		self.alpha = 0.1
		self.class_scale = 1.5
		self.object_scale = 1.0
		self.noobject_scale = 0.5
		self.coord_scale = 5.0

		self.offset = np.transpose(np.reshape(
							np.array([np.arange(self.final_h)]*self.final_w*self.box_num),
							(self.box_num, self.final_h, self.final_w)),
									(1, 2, 0))

		self.images = tf.placeholder(
				dtype=tf.float32,
				shape=[None, scale_height, scale_width, 3],
				name="images")
			
		self.output_shape = (final_w * final_h) * (class_num + box_num*5)
		#self.prediction = self.build_network(self.images, self.output_shape, is_training, 'yolo')
		self.prediction = self.build_network(self.images, 
						num_outputs=self.output_shape, 
						alpha=self.alpha, 
						is_training=is_training)

		if is_training:
			self.labels = tf.placeholder(dtype=tf.float32,
				shape=[None, self.final_h, self.final_w, self.class_num + 5])
	
			self.create_loss(self.prediction, self.labels)
			self.total_loss = tf.losses.get_total_loss()
			tf.summary.scalar('total_loss', self.total_loss)

	def calc_iou(self, box1, box2, name='iou'):
		with tf.variable_scope(name):
			# (x, y, w, h)  -->  (x1, y1, x2, y2)
			box1_t = tf.stack([box1[..., 0] - box1[..., 2] / 2.0,
							   box1[..., 1] - box1[..., 3] / 2.0,
							   box1[..., 0] + box1[..., 2] / 2.0,
							   box1[..., 1] + box1[..., 3] / 2.0], axis=-1)

			box2_t = tf.stack([box2[..., 0] - box2[..., 2] / 2.0,
							   box2[..., 1] - box2[..., 3] / 2.0,
							   box2[..., 0] + box2[..., 2] / 2.0,
							   box2[..., 1] + box2[..., 3] / 2.0], axis=-1)
			
			left_up = tf.maximum(box1_t[..., :2], box2_t[..., :2])
			right_down = tf.minimum(box1_t[..., 2:], box2_t[..., 2:])

			intersection = tf.maximum(0.0, right_down - left_up)
			inter_square = intersection[..., 0] * intersection[..., 1]

			square1 = box1[..., 2] * box1[..., 3]
			square2 = box2[..., 2] * box2[..., 3]

			union_square = tf.maximum(square1 + square2 - inter_square, 1e-8)
		return tf.clip_by_value(inter_square / union_square, 0.0, 1.0)
	"""
	def build_network(self, images, output_shape, training_mask, name):
		with tf.variable_scope(name):
			
			net = tf.pad(images, 
						np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
						name='pad_1')

			net = leaky_relu(conv2d(net, 64, 7, 2, 'conv_2', 'VALID'))

			net = maxpooling(net, 2, 2, 'pool_3', padding='SAME')
			net = leaky_relu(conv2d(net, 192, 3, 1, 'conv_4'))
			net = maxpooling(net, 2, 2, 'pool_5', padding='SAME')
			net = leaky_relu(conv2d(net, 128, 1, 1, 'conv_6'))
			net = leaky_relu(conv2d(net, 256, 3, 1, 'conv_7'))
			net = leaky_relu(conv2d(net, 256, 1, 1, 'conv_8'))
			net = leaky_relu(conv2d(net, 512, 3, 1, 'conv_9'))
			net = maxpooling(net, 2, 2, 'pool_10', padding='SAME')
			net = leaky_relu(conv2d(net, 256, 1, 1, 'conv_11'))
			net = leaky_relu(conv2d(net, 512, 3, 1, 'conv_12'))
			net = leaky_relu(conv2d(net, 256, 1, 1, 'conv_13'))
			net = leaky_relu(conv2d(net, 512, 3, 1, 'conv_14'))
			net = leaky_relu(conv2d(net, 256, 1, 1, 'conv_15'))
			net = leaky_relu(conv2d(net, 512, 3, 1, 'conv_16'))
			net = leaky_relu(conv2d(net, 256, 1, 1, 'conv_17'))
			net = leaky_relu(conv2d(net, 512, 3, 1, 'conv_18'))
			net = leaky_relu(conv2d(net, 512, 1, 1, 'conv_19'))
			net = leaky_relu(conv2d(net, 1024, 3, 1, 'conv_20'))
			net = maxpooling(net, 2, 2, 'pool_21', padding='SAME')
			net = leaky_relu(conv2d(net, 512, 1, 1, 'conv_22'))
			net = leaky_relu(conv2d(net, 1024, 3, 1, 'conv_23'))
			net = leaky_relu(conv2d(net, 512, 1, 1, 'conv_24'))
			net = leaky_relu(conv2d(net, 1024, 3, 1, 'conv_25'))
			net = leaky_relu(conv2d(net, 1024, 3, 1, 'conv_26'))
			
			net = tf.pad(net,
						np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
						name='pad_27')

			net = leaky_relu(conv2d(net, 1024, 3, 2, 'conv_28', 'VALID'))
			net = leaky_relu(conv2d(net, 1024, 3, 1, 'conv_29'))
			net = leaky_relu(conv2d(net, 1024, 3, 1, 'conv_30'))

			net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')

			net = tf.layers.flatten(net, 'flat_32')

			net = leaky_relu(fc(net, 512, 'fc_33'))
			net = leaky_relu(fc(net, 4096, 'fc_34'))
			net = tf.layers.dropout(inputs=net, rate=0.5, training=training_mask, name='dropout_35')

			final = fc(net, output_shape, 'fc_36')

		return final
	"""

	def build_network(self,
					images,
					num_outputs,
					alpha,
					keep_prob=0.5,
					is_training=True,
					scope='yolo'):
		with tf.variable_scope(scope):
			with slim.arg_scope(
				[slim.conv2d, slim.fully_connected],
				activation_fn=leaky_relu(alpha),
				weights_regularizer=slim.l2_regularizer(0.0005),
				weights_initializer=tf.truncated_normal_initializer(0.0, 0.01)
			):
				net = tf.pad(
					images, np.array([[0, 0], [3, 3], [3, 3], [0, 0]]),
					name='pad_1')
				net = slim.conv2d(
					net, 64, 7, 2, padding='VALID', scope='conv_2')
				net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_3')
				net = slim.conv2d(net, 192, 3, scope='conv_4')
				net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_5')
				net = slim.conv2d(net, 128, 1, scope='conv_6')
				net = slim.conv2d(net, 256, 3, scope='conv_7')
				net = slim.conv2d(net, 256, 1, scope='conv_8')
				net = slim.conv2d(net, 512, 3, scope='conv_9')
				net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_10')
				net = slim.conv2d(net, 256, 1, scope='conv_11')
				net = slim.conv2d(net, 512, 3, scope='conv_12')
				net = slim.conv2d(net, 256, 1, scope='conv_13')
				net = slim.conv2d(net, 512, 3, scope='conv_14')
				net = slim.conv2d(net, 256, 1, scope='conv_15')
				net = slim.conv2d(net, 512, 3, scope='conv_16')
				net = slim.conv2d(net, 256, 1, scope='conv_17')
				net = slim.conv2d(net, 512, 3, scope='conv_18')
				net = slim.conv2d(net, 512, 1, scope='conv_19')
				net = slim.conv2d(net, 1024, 3, scope='conv_20')
				net = slim.max_pool2d(net, 2, padding='SAME', scope='pool_21')
				net = slim.conv2d(net, 512, 1, scope='conv_22')
				net = slim.conv2d(net, 1024, 3, scope='conv_23')
				net = slim.conv2d(net, 512, 1, scope='conv_24')
				net = slim.conv2d(net, 1024, 3, scope='conv_25')
				net = slim.conv2d(net, 1024, 3, scope='conv_26')
				net = tf.pad(
					net, np.array([[0, 0], [1, 1], [1, 1], [0, 0]]),
					name='pad_27')
				net = slim.conv2d(
					net, 1024, 3, 2, padding='VALID', scope='conv_28')
				net = slim.conv2d(net, 1024, 3, scope='conv_29')
				net = slim.conv2d(net, 1024, 3, scope='conv_30')
				net = tf.transpose(net, [0, 3, 1, 2], name='trans_31')
				net = slim.flatten(net, scope='flat_32')
				net = slim.fully_connected(net, 512, scope='fc_33')
				net = slim.fully_connected(net, 4096, scope='fc_34')
				net = slim.dropout(
					net, keep_prob=keep_prob, is_training=is_training,
					scope='dropout_35')
				net = slim.fully_connected(
					net, num_outputs, activation_fn=None, scope='fc_36')
		return net


	def create_loss(self, prediction, labels, name='loss_layer'):
		with tf.variable_scope(name):
			predict_class = tf.reshape(
					prediction[:, :self.boundary1],
					[self.batch_size, self.final_h, self.final_w, self.class_num])

			predict_scales = tf.reshape(
					prediction[:, self.boundary1:self.boundary2],
					[self.batch_size, self.final_h, self.final_w, self.box_num])

			predict_boxes = tf.reshape(
					prediction[:, self.boundary2:],
					[self.batch_size, self.final_h, self.final_w, self.box_num, 4])


			responsible = tf.reshape(
					labels[..., 0],
					[self.batch_size, self.final_h, self.final_w, 1])

			boxes = tf.reshape(
					labels[..., 1:5],
					[self.batch_size, self.final_h, self.final_w, 1, 4])

			boxes = tf.tile(boxes, [1, 1, 1, self.box_num, 1]) / self.scale_width

			classes = labels[..., 5:]

			offset = tf.reshape(tf.constant(self.offset, dtype=tf.float32),
						[1, self.final_h, self.final_w, self.box_num])
			offset = tf.tile(offset, [self.batch_size, 1, 1, 1])
			offset_trans = tf.transpose(offset, (0, 2, 1, 3))
			predict_box_trans = tf.stack(
				[(predict_boxes[..., 0] + offset) / self.final_w,
				 (predict_boxes[..., 1] + offset_trans) / self.final_h,
				 tf.square(predict_boxes[..., 2]),
				 tf.square(predict_boxes[..., 3])], axis=-1)

			iou = self.calc_iou(predict_box_trans, boxes)

			object_mask = tf.reduce_max(iou, 3, keepdims=True)
			# box with highest confidence is used for prediction
			# "responsible" is used to confirm whether there's an object in the cell 
			object_mask = tf.cast((iou >= object_mask), tf.float32) * responsible
			# opposite to object_mask
			noobject_mask = tf.ones_like(object_mask, dtype=tf.float32) - object_mask

			box_trans = tf.stack(
						[boxes[..., 0] * self.final_w - offset,
						 boxes[..., 1] * self.final_h - offset_trans,
						 tf.sqrt(boxes[..., 2]),
						 tf.sqrt(boxes[..., 3])], axis=-1)

			#class_loss
			class_delta = responsible * (predict_class - classes)
			class_loss = tf.reduce_mean(
						tf.reduce_sum(tf.square(class_delta), axis=[1,2,3]),
						name='class_loss') * self.class_scale

			# object_loss
			object_delta = object_mask * (predict_scales - iou)
			object_loss = tf.reduce_mean(
						tf.reduce_sum(tf.square(object_delta), axis=[1,2,3]),
						name='object_loss') * self.object_scale

			# noobject_loss
			noobject_delta = noobject_mask * predict_scales
			noobject_loss = tf.reduce_mean(
						tf.reduce_sum(tf.square(noobject_delta), axis=[1,2,3]),
						name='noobject_loss') * self.noobject_scale

			# coord_loss
			coord_mask = tf.expand_dims(object_mask, 4)
			boxes_delta = coord_mask * (predict_boxes - box_trans)
			coord_loss = tf.reduce_mean(
						tf.reduce_sum(tf.square(boxes_delta), axis=[1,2,3,4]),
						name='coord_loss') * self.coord_scale

			tf.losses.add_loss(class_loss)
			tf.losses.add_loss(object_loss)
			tf.losses.add_loss(noobject_loss)
			tf.losses.add_loss(coord_loss)

			tf.summary.scalar('class_loss', class_loss)
			tf.summary.scalar('object_loss', object_loss)
			tf.summary.scalar('noobject_loss', noobject_loss)
			tf.summary.scalar('coord_loss', coord_loss)

			tf.summary.histogram('boxes_delta_x', boxes_delta[..., 0])
			tf.summary.histogram('boxes_delta_y', boxes_delta[..., 1])
			tf.summary.histogram('boxes_delta_w', boxes_delta[..., 2])
			tf.summary.histogram('boxes_delta_h', boxes_delta[..., 3])
			tf.summary.histogram('iou', iou)
		
def leaky_relu(alpha):
	def op(inputs):
		return tf.nn.leaky_relu(inputs, alpha=alpha, name='leaky_relu')
	return op