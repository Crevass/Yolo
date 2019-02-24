import numpy as np
import tensorflow as tf
from Yolo_model import MODEL
from reader import Reader
import cv2
from datetime import datetime
class Trainer(object):
	def __init__(self, net, file_reader, batch_size):
		self.net = net
		self.file_reader = file_reader

		self.init_lr = 0.0001
		self.decay_steps = 30000
		self.decay_rate = 0.1
		self.staircase = True

		self.batch_size = batch_size

		self.variable_to_restore = tf.global_variables()
		self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None)

		date = datetime.now().strftime("%Y-%m-%d")
		self.weights_file = "./YOLO_small.ckpt"
		#self.weights_file = "./saver_dir/final_weights.ckpt"
		#self.weights_file = None
		self.retrain_yolo = True
		self.ckpt_path = "./saver_dir" + date + "/"
		self.log_dir = "./log" + date
		self.summary_op = tf.summary.merge_all()
		self.writer = tf.summary.FileWriter(self.log_dir, flush_secs=60)

		self.global_step = tf.train.create_global_step()
		
		self.learning_rate = tf.train.exponential_decay(
				self.init_lr, self.global_step, self.decay_steps, self.decay_rate, self.staircase, name='learning_rate')
		
		#self.learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate')
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, epsilon=1e-5)
		self.train_op = self.optimizer.minimize(
						loss=self.net.total_loss,
						global_step=self.global_step)

		gpu_option = tf.GPUOptions(allow_growth=True)
		self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_option))
		self.sess.run(tf.global_variables_initializer())

		if self.weights_file is not None:
			print("restoring weights_file from " + self.weights_file)
			self.saver.restore(self.sess, self.weights_file)
		else:
			print("no pretrained weights loaded.")
		
		if self.retrain_yolo:
			fc33 = tf.global_variables(scope='yolo/fc_33')
			fc34 = tf.global_variables(scope='yolo/fc_34')
			fc36 = tf.global_variables(scope='yolo/fc_36')
			var_list = []
			var_list.extend(fc33)
			var_list.extend(fc34)
			var_list.extend(fc36)
			self.sess.run(tf.variables_initializer(var_list))
			print("yolo layers initialzed")

		self.writer.add_graph(self.sess.graph)

	def train(self):
		data_num = len(self.file_reader.images_paths)
		EPISODE_LIMIT = data_num // self.batch_size
		MAX_EPOCH = 100

		print("Data: %i" %data_num)
		print("batch_size: %i" %self.batch_size)
		print("episode_limit: %i" %EPISODE_LIMIT)
		print("max_epoch: %i" %MAX_EPOCH)
		print("Start training...")

		step = 0
		#lr = 1e-3
		for epoch in range(MAX_EPOCH):
			#if epoch < 20:
			#	lr = 1e-4 + (1e-3 - 1e-4) / 20 * epoch
			#elif epoch >=30 and epoch < 60:
			#	lr = 1e-3
			#elif epoch >= 105 and epoch <135:
			#	lr = 1e-4
			#else:
			#	lr = 1e-4 / 2 

			for episode in range(EPISODE_LIMIT):
				x_batch, y_batch = self.file_reader.get_batch(self.batch_size)
				if step % 20 == 0:
					summary_str, loss, _ = self.sess.run(
						[self.summary_op, self.net.total_loss, self.train_op],
						feed_dict={self.net.images: x_batch, self.net.labels: y_batch
									})
				
					self.writer.add_summary(summary_str, step)
				else:
					loss, _ = self.sess.run(
						[self.net.total_loss, self.train_op],
						feed_dict={self.net.images: x_batch, self.net.labels: y_batch
									})
				
				print("epoch: %i, episode: %i, loss: %.4f, lr: %.5f" %(epoch, episode, loss, self.learning_rate.eval(session=self.sess)))
				step += 1

			if (epoch + 1) % 20 == 0:
				print("save weights of epoch %i" %epoch)
				prefix_path = self.saver.save(self.sess, self.ckpt_path+"yolo.ckpt", global_step=self.global_step)
				print("path: ", prefix_path)
		self.saver.save(self.sess, self.ckpt_path+"final_weights.ckpt")
		print("Training finished, weights saved at: "+self.ckpt_path+"final_weights.ckpt")


if __name__ == '__main__':
	path = '/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2007'
	#path = ['/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2012/',
	#		'/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2007']
	scale_width = 448
	scale_height = 448

	box_num = 2
	class_num = 20
	final_w = 7
	fianl_h = 7

	is_training = True
	batch_size = 45

	model = MODEL(scale_width, scale_height, final_w, fianl_h, class_num, box_num, is_training, batch_size)
	file_reader = Reader(path, box_num, class_num, scale_width, scale_height, final_w, fianl_h)
	trainer = Trainer(model, file_reader, batch_size)
	trainer.train()