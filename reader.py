import numpy as np
from os import listdir
from os.path import isfile, join
import xml.dom.minidom
import cv2


class Reader(object):
	def __init__(self, paths, box_num, class_num, scale_width, scale_height, final_width, final_height):
		def _get_list(path):
			images_list = listdir(join(path, 'images'))
			labels_list = []
			assert len(images_list) == len(listdir(join(path, 'Annotations')))
			for name in images_list:
				label = name[:-4] + ".xml"
				labels_list.append(label)
			images_path = [join(path, 'images', f) for f in images_list]
			labels_path = [join(path, 'Annotations', f) for f in labels_list]
			return images_path, labels_path
		
		if isinstance(paths, list):
			images_path = []
			labels_path = []
			for p in paths:
				ip, lp = _get_list(p)
				images_path.extend(ip)
				labels_path.extend(lp)
		else:
			images_path, labels_path = _get_list(paths)

		assert len(images_path) == len(labels_path)

		self.images_paths = images_path
		self.labels_paths = labels_path
		self.box_num = box_num
		self.class_num = class_num
		self.scale_width = scale_width
		self.scale_height = scale_height
		self.final_width = final_width
		self.final_height = final_height
		self.counter = 0
		self.data_num = len(images_path)
		self.index = np.zeros(self.data_num, dtype=np.int)
		self.index[:] = np.arange(self.data_num).copy()
		self.classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
			'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
			'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
			'train', 'tvmonitor']

		self.class_map = dict(zip(self.classes, range(len(self.classes))))

	def label_extractor(self, xml_path):
		label = np.zeros((self.final_height, self.final_width, 5+self.class_num))
		with xml.dom.minidom.parse(xml_path) as DOMTree:
			DOMRoot = DOMTree.documentElement
			size_xml = DOMRoot.getElementsByTagName('size')[0]

			width = size_xml.getElementsByTagName('width')[0].childNodes[0].data
			height = size_xml.getElementsByTagName('height')[0].childNodes[0].data
			width = int(width)
			height = int(height)
			h_ratio = 1.0 * self.scale_height / height
			w_ratio = 1.0 * self.scale_width / width

			objects_xml = DOMRoot.getElementsByTagName('object')
			for obj in objects_xml:
				name = obj.getElementsByTagName('name')[0].childNodes[0].data
				assert isinstance(name, str)
				bndbox = obj.getElementsByTagName('bndbox')[0]
				xmin = bndbox.getElementsByTagName('xmin')[0].childNodes[0].data
				ymin = bndbox.getElementsByTagName('ymin')[0].childNodes[0].data
				xmax = bndbox.getElementsByTagName('xmax')[0].childNodes[0].data
				ymax = bndbox.getElementsByTagName('ymax')[0].childNodes[0].data

				xmin = max(min((float(xmin) - 1) * w_ratio, self.scale_width - 1), 0)
				xmax = max(min((float(xmax) - 1) * w_ratio, self.scale_width - 1), 0)
				ymin = max(min((float(ymin) - 1) * h_ratio, self.scale_height - 1), 0)
				ymax = max(min((float(ymax) - 1) * h_ratio, self.scale_height - 1), 0)
				class_ind = self.class_map[name]
				box = [(xmin + xmax) / 2.0, (ymin + ymax) / 2.0, xmax - xmin, ymax - ymin]
				x_ind = int(box[0] * self.final_width / self.scale_width)
				y_ind = int(box[1] * self.final_height / self.scale_height)
				if label[y_ind, x_ind, 0] == 1:
					continue
				label[y_ind, x_ind, 0] = 1
				label[y_ind, x_ind, 1:5] = box
				label[y_ind, x_ind, 5+class_ind] = 1

		return label
	
	def image_extractor(self, image_path):
		img = cv2.imread(image_path)
		img = cv2.resize(img, (self.scale_width, self.scale_height))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
		img = (img / 255.0) * 2.0 - 1.0
		return img

	def get_batch(self, batch_size):
		labels = np.zeros((batch_size, self.final_height, self.final_width, 5+self.class_num))
		images = np.zeros((batch_size, self.scale_height, self.scale_width, 3))
		for i in range(batch_size):
			assert self.counter < self.data_num
			xml_path = self.labels_paths[self.index[self.counter]]
			img_path = self.images_paths[self.index[self.counter]]

			labels[i, :, :, :] = self.label_extractor(xml_path)
			images[i, :, :, :] = self.image_extractor(img_path)
			
			self.counter += 1
			if self.counter >= self.data_num:
				self.counter = 0
				np.random.shuffle(self.index)

		return images, labels

	"""
	def label_generator(self, objects):
		label = np.zeros([self.final_width, self.final_height, self.box_num*(5+self.class_num)])
		for obj in objects:
			class_label = self.class_map[obj[0]]
			x = obj[1] * self.scale_width
			y = obj[2] * self.scale_height
			grid_x_num = self.scale_width / self.final_width
			grid_y_num = self.scale_height / self.final_height
			index_x = x // grid_x_num
			index_y = y // grid_y_num
			if index_x == self.final_width:
				index_x = self.final_width - 1
			if index_y == self.final_height:
				index_y = self.final_height - 1
			assert index_x < self.final_width
			assert index_y < self.final_height
			x = (x - (index_x * grid_x_num)) / grid_x_num
			y = (y - (index_y * grid_y_num)) / grid_y_num

			for i in range(self.box_num):
				label[int(index_x), int(index_y), i*(5+self.class_num)] = x
				label[int(index_x), int(index_y), i*(5+self.class_num)+1] = y
				label[int(index_x), int(index_y), i*(5+self.class_num)+2] = obj[3]
				label[int(index_x), int(index_y), i*(5+self.class_num)+3] = obj[4]
				label[int(index_x), int(index_y), i*(5+self.class_num)+4] = 1
				label[int(index_x), int(index_y), i*(5+self.class_num)+5+class_label] = 1.0
		return label

	def class_label(self, xml_path):
		DOMTree = xml.dom.minidom.parse(xml_path)
		DOMRoot = DOMTree.documentElement

		objects_xml = DOMRoot.getElementsByTagName('object')
		label = np.zeros(20)
		for obj in objects_xml:
			name = obj.getElementsByTagName('name')[0].childNodes[0].data
			c = self.class_map[name]
			label[c] = 1.0
		return label


	def get_batches(self, index, batch_size):
		label_batch = np.zeros([batch_size, self.final_width, self.final_height, self.box_num*(5+self.class_num)], np.float32)
		image_batch = np.zeros([batch_size, self.scale_width, self.scale_height, 3])
		for i in range(batch_size):
			xml_path = self.labels_paths[index+i]
			img_path = self.images_paths[index+i]
			w, h, objs = self.xml_extractor(xml_path)
			label = self.label_generator(objs)
			label_batch[i] = label.copy()

			img = self.image_extractor(img_path)
			img = np.array(img, np.float32)
			img = np.divide(img, 255.0)
			image_batch[i] = img.copy()
		return image_batch, label_batch

	def CNN_batch(self, batch_size):
		label_batch = np.zeros([batch_size, self.class_num])
		image_batch = np.zeros([batch_size, self.scale_width, self.scale_height, 3])
		for i in range(batch_size):
			assert self.counter < self.data_num
			xml_path = self.labels_paths[self.index[self.counter]]
			img_path = self.images_paths[self.index[self.counter]]

			label = self.class_label(xml_path)
			
			img = self.image_extractor(img_path)
			img = np.array(img, np.float32)
			img = np.divide(img, 255.0)

			label_batch[i] = label.copy()
			image_batch[i] = img.copy()

			self.counter += 1
			if self.counter >= self.data_num:
				self.counter = 0
				np.random.shuffle(self.index)

		return image_batch, label_batch
	"""		


if __name__ == '__main__':
	path = ['/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2012/',
			'/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2007']
	#xml_path = '/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2012/Annotations/2007_003118.xml'
	#pic_path = '/home/wang/Research/MODELS/Yolo/dataset/VOCdevkit/VOC2012/images/2007_003118.jpg'
	
	scale_width = 448
	scale_height = 448

	box_num = 1
	class_num = 20
	final_w = 7
	final_h = 7

	learning_rate = 0.0001
	decay_steps = 30000
	decay_rate = 0.1


	file_reader = Reader(path, box_num, class_num, scale_width, scale_height, final_w, final_h)
	data_num = len(file_reader.images_paths)
	index = np.random.randint(0, data_num)

	xml_path = file_reader.labels_paths[index]
	img_path = file_reader.images_paths[index]

	print("xml path: %s" %xml_path)
	print("img path: %s" %img_path)

	img = cv2.imread(img_path)
	print(img.shape)