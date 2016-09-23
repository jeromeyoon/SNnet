import numpy as np 
import glob,sys,os
from sorting import natsorted
import pdb
import matplotlib.pyplot as plt
import argparse
import scipy.misc
		
def compute_average(path_,num_imgs):
	main_path = os.path.join('/home/yjyoon/Dropbox/ECCV_result/avg_view/%03d/' %num_imgs)
	if not os.path.exists(main_path):
		os.makedirs(main_path)
	dirs = os.listdir(path_)
	for d in range(len(dirs)):
		subp = os.path.join(path_,dirs[d])
		subdirs  = os.listdir(subp)
		for subd in range(len(subdirs)):
			images = glob.glob(os.path.join(path_,dirs[d],subdirs[subd],'*.bmp'))
			images = natsorted(images)
		        rand_idx = np.random.permutation(len(images))	
			avg_img = np.zeros((600,800,3)).astype(np.float)
			for idx in range(num_imgs):
				img = scipy.misc.imread(images[rand_idx[idx]])
				avg_img += img
	
			avg_img /= num_imgs
			savepath = main_path +'avg_img_%s_%03d.bmp' %(dirs[d],int(subdirs[subd]))
			scipy.misc.imsave(savepath,avg_img)
				
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", help='for draw plots: ~/work/ECCV16/NIR_single_L1_ang/L1_ang_loss_lights,')
	parser.add_argument("--num_imgs", help='num_imgs:number of images')
	
	args = parser.parse_args()
	print('num_imgs:%d path:%s \n' %(int(args.num_imgs),args.path))
	compute_average(args.path,int(args.num_imgs)) #ex:inputpath( ~/Dropbox/ECCV_result/L1)
	sys.exit(0)

