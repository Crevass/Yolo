import numpy as np
import cv2
import tensorflow as tf
from Yolo_model import MODEL
from reader import Reader

class Detector(object):
	def __init__(self, net, weights_file):
		self.net = net
		self.weights_file = weights_file

		self.class_num = net.class_num
		self.scale_width = net.scale_width
		self.scale_height = net.scale_height
		self.final_w = net.final_w
		self.final_h = net.final_h
		self.box_num = net.box_num
		
		self.threshold = 0.17
		self.iou_threshold = 0.5
		
		self.boundary1 = self.final_h * self.final_w * self.class_num
		self.boundary2 = self.boundary1 + self.final_h * self.final_w * self.box_num

		self.session = tf.Session()
		self.session.run(tf.global_variables_initializer())
		"""
		self.class_map = [
				'person',
				'bird',
				'cat',
				'cow',
				'dog',
				'horse',
				'sheep',
				'aeroplane',
				'bicycle',
				'boat',
				'bus',
				'car',
				'motorbike',
				'train',
				'bottle',
				'chair',
				'diningtable',
				'pottedplant',
				'sofa',
				'tvmonitor'
			]
			"""
		self.class_map = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
			'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
			'train', 'tvmonitor']

		self.saver = tf.train.Saver()
		self.saver.restore(self.session, self.weights_file)
		print("weights restored")

	def draw_result(self, img, result):
		for i in range(len(result)):
			x = int(result[i][1])
			y = int(result[i][2])
			w = int(result[i][3] / 2)
			h = int(result[i][4] / 2)
			cv2.rectangle(img, (x - w, y - h), (x + w, y + h), (0, 255, 0), 2)
			cv2.rectangle(img, (x - w, y - h - 20),
						(x + w, y - h), (125, 125, 125), -1)
			lineType = cv2.LINE_AA if cv2.__version__ > '3' else cv2.CV_AA
			name = result[i][0]
			confidence = result[i][5]
			cv2.putText(
				img, name + ' : %.2f' % confidence,
				(x - w + 5, y - h - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
				(0, 0, 0), 1, lineType)
			


	def detect(self, img):
		img_h, img_w, _ = img.shape
		inputs = cv2.resize(img, (self.scale_width, self.scale_height))
		inputs = cv2.cvtColor(inputs, cv2.COLOR_BGR2RGB).astype(np.float32)
		inputs = (inputs / 255.0) * 2.0 - 1.0
		inputs = np.reshape(inputs, (1, self.scale_height, self.scale_width, 3))

		result = self.detect_from_cvmat(inputs)[0]
		for i in range(len(result)):
			result[i][1] *= (1.0 * img_w / self.scale_width)
			result[i][2] *= (1.0 * img_h / self.scale_height)
			result[i][3] *= (1.0 * img_w / self.scale_width)
			result[i][4] *= (1.0 * img_h / self.scale_height)

		return result

	def detect_from_cvmat(self, inputs):
		model_prediction = self.session.run(self.net.prediction, 
												feed_dict={self.net.images: inputs})
		result = []
		for i in range(model_prediction.shape[0]):
			result.append(self.interpret_output(model_prediction[i]))
		return result

	def interpret_output(self, output):
		probs = np.zeros((self.final_h, self.final_w, self.box_num, self.class_num))

		class_probs = np.reshape(
				output[0:self.boundary1],
				(self.final_h, self.final_w, self.class_num))

		scales = np.reshape(
				output[self.boundary1:self.boundary2],
				(self.final_h, self.final_w, self.box_num))

		boxes = np.reshape(
				output[self.boundary2:],
				(self.final_h, self.final_w, self.box_num, 4))

		offset = np.array(
				[np.arange(self.final_h)] * self.final_w * self.box_num)
		
		offset = np.transpose(
				np.reshape(
					offset,
					[self.box_num, self.final_h, self.final_w]),
				(1,2,0))
		boxes[:, :, :, 0] += offset
		boxes[:, :, :, 1] += np.transpose(offset, (1, 0, 2))
		boxes[:, :, :, 0:2] = 1.0 * boxes[:, :, :, 0:2] / self.final_h
		boxes[:, :, :, 2:] = np.square(boxes[:, :, :, 2:])

		boxes *= self.scale_height

		for i in range(self.box_num):
			for j in range(self.class_num):
				probs[:, :, i, j] = np.multiply(
					class_probs[:, :, j], scales[:, :, i])

		filter_mat_probs = np.array(probs >= self.threshold, dtype='bool')
		filter_mat_boxes = np.nonzero(filter_mat_probs)

		boxes_filtered = boxes[filter_mat_boxes[0],
								filter_mat_boxes[1],
								filter_mat_boxes[2]]

		probs_filtered = probs[filter_mat_probs]
		classes_num_filtered = np.argmax(
				filter_mat_probs, axis=3)[
				filter_mat_boxes[0], filter_mat_boxes[1], filter_mat_boxes[2]]

		argsort = np.array(np.argsort(probs_filtered))[::-1]
		boxes_filtered = boxes_filtered[argsort]
		probs_filtered = probs_filtered[argsort]
		classes_num_filtered = classes_num_filtered[argsort]
		for i in range(len(boxes_filtered)):
			if probs_filtered[i] == 0:
				continue
			for j in range(i + 1, len(boxes_filtered)):
				if self.iou(boxes_filtered[i], boxes_filtered[j]) > self.iou_threshold:
					probs_filtered[j] = 0.0

		filter_iou = np.array(probs_filtered > 0.0, dtype='bool')
		boxes_filtered = boxes_filtered[filter_iou]
		probs_filtered = probs_filtered[filter_iou]
		classes_num_filtered = classes_num_filtered[filter_iou]

		result = []
		for i in range(len(boxes_filtered)):
			result.append(
				[self.class_map[classes_num_filtered[i]],
				 boxes_filtered[i][0],
				 boxes_filtered[i][1],
				 boxes_filtered[i][2],
				 boxes_filtered[i][3],
				 probs_filtered[i]])

		return result

	def iou(self, box1, box2):
		top_bottom = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2]) - \
					max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])

		left_right = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3]) - \
					max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])

		inter = 0 if top_bottom < 0 or left_right < 0 else top_bottom * left_right
		return inter / (box1[2] * box1[3] + box2[2] * box2[3] - inter)

	def image_detector(self, img_path, wait=0):
		image = cv2.imread(img_path)
		result = self.detect(image)
		print(result)
		self.draw_result(image, result)
		cv2.imshow('Image', image)
		c = cv2.waitKey(wait)
		if c == 27:
			cv2.destroyAllWindows()
			return

	def camera_detect(self, cap, wait=10):
		ret, frame = cap.read()
		fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
		out = cv2.VideoWriter("record.mp4", fourcc, 25, (640, 480))
		while ret:
			#ret, frame = cap.read()
			result = self.detect(frame)
			self.draw_result(frame, result)
			out.write(frame)
			cv2.imshow("Video", frame)
			c = cv2.waitKey(wait)
			if c == 27:
				break
			ret, frame = cap.read()
		cap.release()
		out.release()
		cv2.destroyAllWindows()
		return

if __name__ == '__main__':
	#path = '/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2012/'
	path = ['/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2012/',
			'/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2007']
	scale_width = 448
	scale_height = 448

	box_num = 2
	class_num = 20
	final_w = 7
	fianl_h = 7

	learning_rate = 0.0001
	decay_steps = 30000
	decay_rate = 0.1
	is_training = False
	batch_size = 45

	weights_file = "./saver_dir2018-09-19/final_weights.ckpt"
	#weights_file = "YOLO_small.ckpt"
	model = MODEL(scale_width, scale_height, final_w, fianl_h, class_num, box_num, is_training, batch_size)
	file_reader = Reader(path, box_num, class_num, scale_width, scale_height, final_w, fianl_h)
	detector = Detector(model, weights_file)

	data_num = len(file_reader.images_paths)
	index = np.random.randint(0, data_num)

	img_path = file_reader.images_paths[index]

	# detect image
	#img_path = "./2007_001239.jpg"
	#print("img path: %s" %img_path)
	#detector.image_detector(img_path)
	
	# detect video from a web camera
	cap = cv2.VideoCapture(-1)
	detector.camera_detect(cap, wait=5)