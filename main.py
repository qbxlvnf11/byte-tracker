import os
import cv2
import argparse
import numpy as np
import PIL.Image as pilimg
from configparser import ConfigParser

from detecting_multi_objects_tracking import Detecting_Multi_Objects_Tracking_Processor
from train import Yolov3_Train_Processor
from yolov3_pytorch.utils.detection_util import draw_id_boxes

def parse_args():
	parser = argparse.ArgumentParser(
		description='Yolo-v3 model train/inference')
	
	# Run mode
	parser.add_argument('--mode', help='yolov3 run mode', choices=['byte-multi-objects-tracking-video', 'byte-multi-objects-tracking-images', 'yolov3-train'], default='byte-multi-objects-tracking-images')
	
	# Video
	parser.add_argument("--video_path", default='./data/inf_video/test.mp4')
	parser.add_argument("--image_folder_path", default='./data/inf_image_folder/test_imgs')
	parser.add_argument("--fps", type=int, default=30)
	
	# CUDA or not
	parser.add_argument('--device', help='device of yolov3 (cpu, cuda:0, ...)', default='cuda:0')
	
	# Random seed
	parser.add_argument('--random_seed_num', help='number of random seed', type=int, default=99)
	
	# Config file path
	parser.add_argument('--yolov3_config_file_path', help='config file of yolov3', default='./config/yolov3_config.ini')
	parser.add_argument('--byte_tracker_config_file_path', help='config file of tensorrt', default='./config/byte_tracker_config.ini')
	parser.add_argument('--train_config_file_path', help='config file of train param', default='./config/train_config.ini')
	parser.add_argument('--tensorrt_config_file_path', help='config file of tensorrt', default='./config/tensorrt_config.ini')
	
	# Save path
	parser.add_argument('--save_path', help='save path of inference image', default='./data/inf_images')
		
	args = parser.parse_args()
	return args

def read_config(path):
	config = ConfigParser()
	config.read(path, encoding='utf-8') 
	
	return config

def get_config_dict(config_path_list):
	
	config_dict = {}
	
	for config_path in config_path_list:
		config_dict[config_path.split('/')[-1].split('.')[0]] = read_config(config_path)
	
	return config_dict
	
def main():
	# Arg parsing
	args = parse_args()
	
	# Device
	device = args.device
	
	# Random seed
	random_seed_num = args.random_seed_num
	
	# Config dict
	config_dict = get_config_dict([args.yolov3_config_file_path, args.byte_tracker_config_file_path, args.train_config_file_path, args.tensorrt_config_file_path])

	# Setting fps
	fps = args.fps	
	
	# Yolo
	detecting_multi_objects_tracking_processor = Detecting_Multi_Objects_Tracking_Processor(fps, config_dict, device)
	
	if args.mode == 'byte-multi-objects-tracking-video':
		
		# Video object
		print('-'*50)
		print('Read video path:', args.video_path)
		video = cv2.VideoCapture(args.video_path)		
		print('-'*50)
		
		# Check video
		assert video.isOpened(), 'Can not open the video ' + str(args.video_path) + '!'

		rfps = int(video.get(cv2.CAP_PROP_FPS))
		img_id = 0
		add_num = 0

		while True:

			# Read frame
			ret, img = video.read()
			#print('Size of read image:', img.shape)
				
			add_num += 1
			
			if ret and add_num % int(rfps / fps) == 0:
			
				print('-'*40)
				img_id += 1
				print('img_id:', img_id)
				
				# Detection
				id_list, box_list, class_list, conf_list, num_objects = detecting_multi_objects_tracking_processor.predict(img)
				
				print('-'*30)
				print('Object id list:', id_list)
				print('Object box list:', box_list)
				print('Class name list:', class_list)
				print('Conf score list:', conf_list)
				print('-'*30)
				
				# Draw id & objects
				draw_id_boxes(img, id_list, box_list, class_list, args.save_path, img_id)
				
				print('-'*40)
				
			if not ret:
				break
			
	if args.mode == 'byte-multi-objects-tracking-images':		
		
		image_names = [img for img in os.listdir(args.image_folder_path) if img.endswith(".png") or img.endswith(".jpg")]
		image_names.sort()
		
		for img_id, image_name in enumerate(image_names):

			# Read image
			img_path = os.path.join(args.image_folder_path, image_name)
			print('Image path:', img_path)
			img = np.array(pilimg.open(img_path))
			#print('Size of read image:', img.shape)
			
			print('-'*40)
			print('img_id:', img_id+1)
			
			# Detection
			id_list, box_list, class_list, conf_list, num_objects = detecting_multi_objects_tracking_processor.predict(img)
			
			print('-'*30)
			print('Object id list:', id_list)
			print('Object box list:', box_list)
			print('Class name list:', class_list)
			print('Conf score list:', conf_list)
			print('-'*30)
			
			# Draw id & objects
			draw_id_boxes(img, id_list, box_list, class_list, args.save_path, img_id+1)
			
			print('-'*40)			
			
	elif args.mode == 'yolov3-train':
		yolov3_train_processor = Yolov3_Train_Processor(config_dict, device, random_seed_num)	
		yolov3_train_processor.train_eval()	
	
if __name__ == '__main__':
	main()
