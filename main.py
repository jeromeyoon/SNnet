import numpy as np
import os
import tensorflow as tf
import random
import time 
import json
from model import DCGAN
from test import EVAL
from utils import pp, save_images, to_json, make_gif, merge, imread, get_image
import scipy.misc
from numpy import inf
import glob
from sorting import natsorted
import pdb
import matplotlib.image as mpimg
#import cv2
import time
flags = tf.app.flags
flags.DEFINE_integer("epoch", 1000, "Epoch to train [25]")
flags.DEFINE_float("g_learning_rate", 0.0001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("g_learning_rate_minimum", 0.00001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("d_learning_rate", 0.00001, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 16, "The size of batch images [64]")
flags.DEFINE_integer("image_size", 108, "The size of image to use (will be center cropped) [108]")
flags.DEFINE_string("dataset", "MSE_ang_hinge", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "output", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_integer("input_size", 64, "The size of image input size")
flags.DEFINE_float("gpu",0.5,"GPU fraction per process")
FLAGS = flags.FLAGS

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def rgb2ycbcr(rgb):
    return [rgb[:,:,0] * 65.481 + rgb[:,:,1] * 128.553 + rgb[:,:,2] * 24.966 + 16 ,rgb[:,:,0] * -37.797 + rgb[:,:,1] * -74.203 + rgb[:,:,2] * 112.0 + 128, 128+ 112.0 * rgb[:,:,0]  - 93.786*rgb[:,:,1] -18.214*rgb[:,:,2]]

def main(_):
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)
    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)
    if not os.path.exists(os.path.join('./logs',time.strftime('%d%m'))):
	os.makedirs(os.path.join('./logs',time.strftime('%d%m')))

    gpu_config = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_config)) as sess:
        if FLAGS.is_train:
            dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size, input_size=FLAGS.input_size,
                      dataset_name=FLAGS.dataset,
                      is_crop=FLAGS.is_crop, checkpoint_dir=FLAGS.checkpoint_dir)
        else:
   
            dcgan = EVAL(sess, input_size = 600, batch_size=1,ir_image_shape=[None,None,1],normal_image_shape=[None,None,3],dataset_name=FLAGS.dataset,\
                      is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)

        if FLAGS.is_train:
            dcgan.train(FLAGS)
        else:
            OPTION = 2 # for validation
            list_val = [11,16,21,22,33,36,38,53,59,92]
            VAL_OPTION =3
            """
            if OPTION == 1:
                data = json.load(open("/research2/IR_normal_small/json/traininput_single_224_ori_small.json"))
                data_label = json.load(open("/research2/IR_normal_small/json/traingt_single_224_ori_small.json"))
            
            elif OPTION == 2:
                data = json.load(open("/research2/IR_normal_small/json/testinput_single_224_ori_small.json"))
                data_label = json.load(open("/research2/IR_normal_small/json/testgt_single_224_ori_small.json"))
            """
            if VAL_OPTION ==1:
	    	model = 'DCGAN.model-10000'
	        dcgan.load(FLAGS.checkpoint_dir,model)
                list_val = [11,16,21,22,33,36,38,53,59,92]
                for idx in range(len(list_val)):
		    os.makedirs(os.path.join('L1_loss_result','%03d' %list_val[idx]))
                    for idx2 in range(1,10): 
                        print("Selected material %03d/%d" % (list_val[idx],idx2))
                        img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx],idx2)
                        input_ = scipy.misc.imread(img+'/3.bmp').astype(float)
                        gt_ = scipy.misc.imread('/research2/IR_normal_small/save016/1/12_Normal.bmp').astype(float)
                        input_ = scipy.misc.imresize(input_,[600,800])
			input_  = input_/255.0 -1.0 # normalize -1 ~1
                        gt_ = scipy.misc.imresize(gt_,[600,800])
                        #input_ = input_[240:840,515:1315]
                        #gt_ = gt_[240:840,515:1315]
                        input_ = np.reshape(input_,(1,600,800,1)) 
                        gt_ = np.reshape(gt_,(1,600,800,3)) 
                        input_ = np.array(input_).astype(np.float32)
                        gt_ = np.array(gt_).astype(np.float32)
                        start_time = time.time() 
                        sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
                        print('time: %.8f' %(time.time()-start_time))     
                        # normalization #
                        sample = np.squeeze(sample).astype(np.float32)
                        gt_ = np.squeeze(gt_).astype(np.float32)

                        output = np.zeros((600,800,3)).astype(np.float32)
                        output[:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                        output[:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                        output[:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
   
                        output[output ==inf] = 0.0
                        sample = (output+1.)/2.
			os.makedirs(os.path.join('L1_loss_result','%03d/%d' %(list_val[idx],idx2)))
                        savename = './L1_loss_result/%03d/%d/single_normal_L1_%s.bmp' % (list_val[idx],idx2,model)

                        scipy.misc.imsave(savename, sample)

            
            elif VAL_OPTION ==2: # arbitary dataset 
                print("Computing arbitary dataset ")
		trained_models = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		trained_models  = natsorted(trained_models)
		datapath = '/home/yjyoon/Dropbox/ECCV_result/smartphone/iphone/input/gray_*.bmp'
                savepath = '/home/yjyoon/Dropbox/ECCV_result/smartphone/iphone/output'
		fulldatapath = os.path.join(glob.glob(datapath))
		model = trained_models[-2]
		model = model.split('/')
		model = model[-1]
		dcgan.load(FLAGS.checkpoint_dir,model)
                for idx in xrange(len(fulldatapath)):
		    #input_ = cv2.imread(fulldatapath[idx])
		    #input_ = cv2.cvtColor(input_,cv2.COLOR_BGR2YCR_CB)
		    #input_ = cv2.resize(input_[:,:,0],(600,800))
		    input_= scipy.misc.imread(fulldatapath[idx]).astype(float)
	            input_  = (input_/127.5)-1. # normalize -1 ~1
                    input_ = np.reshape(input_,(1,input_.shape[0],input_.shape[1],1)) 
		    #[Y,Cr,Cb]= rgb2ycbcr(input_)
		    #input_= rgb2gray(input_)
		    #input_ = scipy.misc.imresize(Y,(600,800))
                    #input_ = np.reshape(input_,(1,600,800,1)) 
                    input_ = np.array(input_).astype(np.float32)
                    start_time = time.time() 
                    sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
                    print('time: %.8f' %(time.time()-start_time))     
                    # normalization #
                    sample = np.squeeze(sample).astype(np.float32)

                    output = np.zeros((sample.shape[0],sample.shape[1],3)).astype(np.float32)
                    output[:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                    output[:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
                    output[:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
   
                    output[output ==inf] = 0.0
                    sample = (output+1.)/2.
                    name = fulldatapath[idx].split('/')
		    name = name[-1].split('.')
                    name = name[0]
		    savename = savepath + '/normal_' + name +'.bmp' 
                    scipy.misc.imsave(savename, sample)

	    elif VAL_OPTION ==3: # light source fixed
                list_val = [11,16,21,22,33,36,38,53,59,92]
		save_files = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		save_files  = natsorted(save_files)
		savepath ='./L1__ang_loss_result'
		for model_idx in range(0,len(save_files),2):
		    model = save_files[model_idx]
		    model = model.split('/')
		    model = model[-1]
		    dcgan.load(FLAGS.checkpoint_dir,model)
            	    for idx in range(len(list_val)):
			if not os.path.exists(os.path.join(savepath,'%03d' %list_val[idx])):
		            os.makedirs(os.path.join(savepath,'%03d' %list_val[idx]))
		        for idx2 in range(1,10): 
			    print("Selected material %03d/%d" % (list_val[idx],idx2))
			    img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx],idx2)
			    input_ = scipy.misc.imread(img+'/3.bmp').astype(float)
			    gt_ = scipy.misc.imread('/research2/IR_normal_small/save016/1/12_Normal.bmp').astype(float)
			    input_ = scipy.misc.imresize(input_,[600,800])

			    input_  = (input_/127.5)-1. # normalize -1 ~1
			    gt_ = scipy.misc.imresize(gt_,[600,800])
			    #input_ = input_[240:840,515:1315]
			    #gt_ = gt_[240:840,515:1315]
			    input_ = np.reshape(input_,(1,600,800,1)) 
			    gt_ = np.reshape(gt_,(1,600,800,3)) 
			    input_ = np.array(input_).astype(np.float32)
			    gt_ = np.array(gt_).astype(np.float32)
			    start_time = time.time() 
			    sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
			    print('time: %.8f' %(time.time()-start_time))     
			    # normalization #
			    sample = np.squeeze(sample).astype(np.float32)
			    gt_ = np.squeeze(gt_).astype(np.float32)

			    output = np.zeros((600,800,3)).astype(np.float32)
			    output[:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
			    output[:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
			    output[:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
   
			    output[output ==inf] = 0.0
			    sample = (output+1.)/2.
			    if not os.path.exists(os.path.join(savepath,'%03d/%d' %(list_val[idx],idx2))):
			        os.makedirs(os.path.join(savepath,'%03d/%d' %(list_val[idx],idx2)))
			    savename = os.path.join(savepath, '%03d/%d/single_normal_%s.bmp' % (list_val[idx],idx2,model))
			    scipy.misc.imsave(savename, sample)


	    elif VAL_OPTION ==4: # depends on light sources 
                list_val = [11,16,21,22,33,36,38,53,59,92]
		save_files = glob.glob(os.path.join(FLAGS.checkpoint_dir,FLAGS.dataset,'DCGAN.model*'))
		save_files  = natsorted(save_files)
		savepath ='./L1_ang_loss_lights_result'
		if not os.path.exists(os.path.join(savepath)):
		    os.makedirs(os.path.join(savepath))
		model = save_files[-2]
		model = model.split('/')
		model = model[-1]
		dcgan.load(FLAGS.checkpoint_dir,model)
	        for idx in range(len(list_val)):
		    if not os.path.exists(os.path.join(savepath,'%03d' %list_val[idx])):
		        os.makedirs(os.path.join(savepath,'%03d' %list_val[idx]))
		    for idx2 in range(1,10): #tilt angles 1~9 
		        for idx3 in range(1,13): # light source 
			    print("Selected material %03d/%d" % (list_val[idx],idx2))
			    img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx],idx2)
			    input_ = scipy.misc.imread(img+'/%d.bmp' %idx3).astype(float) #input NIR image
			    input_ = scipy.misc.imresize(input_,[600,800])
			    input_  = input_/127.5 -1.0 # normalize -1 ~1
			    input_ = np.reshape(input_,(1,600,800,1)) 
			    input_ = np.array(input_).astype(np.float32)
			    start_time = time.time() 
			    sample = sess.run(dcgan.sampler, feed_dict={dcgan.ir_images: input_})
			    print('time: %.8f' %(time.time()-start_time))     
			    # normalization #
			    sample = np.squeeze(sample).astype(np.float32)

			    output = np.zeros((600,800,3)).astype(np.float32)
			    output[:,:,0] = sample[:,:,0]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
			    output[:,:,1] = sample[:,:,1]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
			    output[:,:,2] = sample[:,:,2]/(np.sqrt(np.power(sample[:,:,0],2) + np.power(sample[:,:,1],2) + np.power(sample[:,:,2],2)))
			    output[output ==inf] = 0.0
			    sample = (output+1.)/2.
			    if not os.path.exists(os.path.join(savepath,'%03d/%d' %(list_val[idx],idx2))):
			        os.makedirs(os.path.join(savepath,'%03d/%d' %(list_val[idx],idx2)))
			    savename = os.path.join(savepath,'%03d/%d/single_normal_%d.bmp' % (list_val[idx],idx2,idx3))
			    scipy.misc.imsave(savename, sample)



if __name__ == '__main__':
    tf.app.run()
