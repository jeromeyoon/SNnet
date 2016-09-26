import os
import time
from glob import glob
import numpy as np
from numpy import inf
import tensorflow as tf
import pdb
#from tensorflow.python.ops.script_ops import *
from ops import *
from utils import *
from compute_ei import *
from normal import norm_
import time
class Deep_DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,\
                 batch_size=32, input_size=64, sample_size=32, ir_image_shape=[64, 64,1], normal_image_shape=[64, 64, 3],\
	         light_shape=[64,64,3], gf_dim=64, df_dim=64,c_dim=3, dataset_name='default',checkpoint_dir=None):


        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.normal_image_shape = normal_image_shape
	self.light_shape = light_shape
        self.ir_image_shape = ir_image_shape
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.c_dim = c_dim
        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir
	
	self.lambda_ang = 1.0
        self.lambda_g = 0.001
        self.lambda_L2 = 1.0
	self.lambda_scale = 0.0
        self.lambda_hing = 0.0
        

	# batch normalization : deals with poor initialization helps gradient flow

        self.d_bn1 = batch_norm(batch_size, name='d_bn1')
        self.d_bn2 = batch_norm(batch_size, name='d_bn2')
        self.d_bn3 = batch_norm(batch_size, name='d_bn3')
        
	self.g_bn0 = batch_norm(batch_size, name='g_bn0')
        self.g_bn1 = batch_norm(batch_size, name='g_bn1')
        self.g_bn2 = batch_norm(batch_size, name='g_bn2')
        self.g_bn3 = batch_norm(batch_size, name='g_bn3')
        self.g_bn4 = batch_norm(batch_size, name='g_bn4')
        self.g_bn5 = batch_norm(batch_size, name='g_bn5')
        self.g_bn6 = batch_norm(batch_size, name='g_bn6')
        self.build_model()

    def build_model(self):

        self.ir_images = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='ir_images')
        self.normal_images = tf.placeholder(tf.float32, [self.batch_size] + self.normal_image_shape,
                                    name='normal_images')

        self.train_light = tf.placeholder(tf.float32, [self.batch_size] + self.light_shape,
                                    name='train_light')

        self.train_mask = tf.placeholder(tf.float32, [self.batch_size] + self.ir_image_shape,
                                    name='train_mask')


        self.ir_test = tf.placeholder(tf.float32, [1,600,800,1],name='ir_test')
        self.light_test = tf.placeholder(tf.float32, [1,600,800,1],name='light_test')
        self.gt_test = tf.placeholder(tf.float32, [1,600,800,3],name='gt_test')

        self.G = self.generator(self.ir_images)
        self.D = self.discriminator(self.normal_images) # real image output
        self.sampler = self.sampler(self.ir_test)
        self.D_ = self.discriminator(self.G, reuse=True) #fake image output
        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)

        self.d_loss_real = binary_cross_entropy_with_logits(tf.ones_like(self.D), self.D)
        self.d_loss_fake = binary_cross_entropy_with_logits(tf.zeros_like(self.D_), self.D_)


	##### reconstruct NIR from scale invariant(David eigen NIPS 2014) ######
	#### To reconver NIR, Surface normale should be -1~ 1
	self.scale_inv,self.recon_NIR = scale_invariant(self.G,self.ir_images,self.train_mask,self.train_light)
	# reconstructing should be positive NIR >0
	#maksing mask
	self.masked_NIR = tf.mul(self.train_mask,self.recon_NIR)
	self.hing_loss = tf.reduce_mean(tf.maximum(tf.neg(self.masked_NIR),0.))
	#normal error
	self.ang_loss = norm_(self.G,self.normal_images,self.train_mask)
        self.L2_loss = tf.sqrt(tf.reduce_mean(tf.square(tf.sub(self.G,self.normal_images))))

        self.g_loss = binary_cross_entropy_with_logits(tf.ones_like(self.D_), self.D_)
        self.gen_loss = self.g_loss * self.lambda_g + self.L2_loss*self.lambda_L2 + self.hing_loss * self.lambda_hing + self.scale_inv * self.lambda_scale + self.ang_loss * self.lambda_ang

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.ang_loss_sum = tf.scalar_summary("ang_loss", self.ang_loss)
        self.l2_loss_sum = tf.scalar_summary("l2_loss", self.L2_loss)
        self.scale_inv_sum = tf.scalar_summary('scale_inv_loss',self.scale_inv)
        self.hing_loss_sum = tf.scalar_summary('hing_loss',self.hing_loss)
        
        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=0)

    def train(self, config):
        """Train DCGAN"""

        global_step = tf.Variable(0,name='global_step',trainable=False)
	self.learning_rate_op = tf.maximum(config.g_learning_rate_minimum,\
				          tf.train.exponential_decay(config.g_learning_rate,global_step,12000,0.95,staircase=True))
	d_optim = tf.train.AdamOptimizer(config.d_learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, global_step=global_step,var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(self.learning_rate_op, beta1=config.beta1) \
                          .minimize(self.gen_loss, global_step=global_step,var_list=self.g_vars)
        
	tf.initialize_all_variables().run()
	
	self.g_sum = tf.merge_summary([self.d__sum,self.d_loss_fake_sum, self.g_loss_sum,self.l2_loss_sum,self.ang_loss_sum,self.scale_inv_sum,self.hing_loss_sum])
        self.d_sum = tf.merge_summary([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.train.SummaryWriter(os.path.join("./logs",time.strftime('%d%m')), self.sess.graph_def)
                
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        # loda training and validation dataset path
        data = json.load(open("/research2/ECCV_journal/with_light/json/traininput.json"))
        data_light = json.load(open("/research2/ECCV_journal/with_light/json/trainlight.json"))
        data_label = json.load(open("/research2/ECCV_journal/with_light/json/traingt.json"))
        datalist =[data[idx] for idx in xrange(0,len(data))]
        datalightlist =[data_light[idx] for idx in xrange(0,len(data))]
        labellist =[data_label[idx] for idx in xrange(0,len(data))]


        list_val = [11,16,21,22,33,36,38,53,59,92]

        for epoch in xrange(config.epoch):
            # loda training and validation dataset path
            shuffle = np.random.permutation(range(len(data)))
            batch_idxs = min(len(data), config.train_size)/config.batch_size
    
            for idx in xrange(0, batch_idxs):
                batch_files = shuffle[idx*config.batch_size:(idx+1)*config.batch_size]

		batches = [get_image(datalist[batch_file],datalightlist[batch_file], labellist[batch_file],self.image_size,np.random.randint(64,224-64),\
				np.random.randint(64,224-64), is_crop=self.is_crop) for batch_file in batch_files]

                batches = np.array(batches).astype(np.float32)
		batch_images = np.reshape(batches[:,:,:,0],[config.batch_size,64,64,1])
		batch_light = np.reshape(batches[:,:,:,1:4],[config.batch_size,64,64,3])
		batch_mask = np.reshape(batches[:,:,:,4],[config.batch_size,64,64,1])
                batchlabel_images = np.reshape(batches[:,:,:,5:],[config.batch_size,64,64,3])
                # Update D network
                _, summary_str= self.sess.run([d_optim, self.d_sum], feed_dict={self.normal_images: batchlabel_images,
                                                                                 self.ir_images: batch_images,self.train_light:batch_light })
                self.writer.add_summary(summary_str, global_step.eval())

                # Update G network
                _, summary_str,g_loss,L2_loss,ang_loss,hing_loss,scale_inv = self.sess.run([g_optim, self.g_sum,self.g_loss,self.L2_loss,\
		self.ang_loss,self.hing_loss,self.scale_inv], feed_dict={ self.ir_images: batch_images,self.normal_images: batchlabel_images,\
		self.train_light:batch_light,self.train_mask:batch_mask})
                self.writer.add_summary(summary_str, global_step.eval())

		"""
		_, summary_str,g_loss,L2_loss,ang_loss,hing_loss,scale_inv = self.sess.run([g_optim, self.g_sum,self.g_loss,self.L2_loss,\
		self.ang_loss,self.hing_loss,self.scale_inv], feed_dict={ self.ir_images: batch_images,self.normal_images: batchlabel_images,\
		self.train_light:batch_light,self.train_mask:batch_mask})
                self.writer.add_summary(summary_str, global_step.eval())
		"""
		print("Epoch: [%2d] [%4d/%4d] time: %4.4f g_loss: %.4f L2_loss:%.4f ang_loss:%.4f hing_loss:%.4f scale_inv:%.4f" \
		% (epoch, idx, batch_idxs,time.time() - start_time,g_loss,L2_loss,ang_loss,hing_loss,scale_inv))
            
 	    for idx2 in xrange(0,len(list_val)):
		for tilt in range(1,10):	
		    print("Epoch: [%2d] [%4d/%4d] " % (epoch, idx2, len(list_val)))
		    img = '/research2/IR_normal_small/save%03d/%d' % (list_val[idx2],tilt)
		    input_ = scipy.misc.imread(img+'/3.bmp').astype(float)
	            input_ = scipy.misc.imresize(input_,[600,800])
	            input_ = input_/127.5 - 1.0
		    input_ = np.reshape(input_,[1,600,800,1])
		    #mask_  = [input_ >-1.][0]*1.0
		    #mask_ = np.reshape(mask_,(600,800,1))
		    sample = self.sess.run([self.sampler],feed_dict={self.ir_test: input_})
                    sample = np.squeeze(sample).astype(np.float32)
	            #sample = (sample +1.0)/2.0
		    #sample = sample * mask_
		    #sample = np.clip(sample,1e-10,1.0)	
		    output = np.sqrt(np.sum(np.power(sample,2),axis=2))
		    output = np.expand_dims(output,axis=-1)
		    output = sample/output
                    output[output ==inf] = 0.0
		    
		    if not os.path.exists(os.path.join(config.sample_dir,'save%03d'%list_val[idx2],'%d' %tilt)):
                        os.makedirs(os.path.join(config.sample_dir,'save%03d'%list_val[idx2],'%d' %tilt))	
		    save_normal(output,os.path.join(config.sample_dir,'save%03d' %list_val[idx2],'%d' %tilt,'preditced_%06d.png' %epoch))

            self.save(config.checkpoint_dir,global_step)

    def discriminator(self, image, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()
	### yan lecun model (ICLR 2016)
        h0 = lrelu(conv2d(image, self.df_dim*2,k_h=5,k_w=5,padding='VALID', name='d_h0_conv'))
        h1 = lrelu(conv2d(h0, self.df_dim*4,k_h=5,k_w=5, padding='VALID', name='d_h1_conv'))
        h2 = lrelu(conv2d(h1, self.df_dim*8,k_h=5,k_w=5, padding='VALID', name='d_h2_conv'))
	h3 = linear(tf.reshape(h2,[-1,int(np.prod(h2.get_shape()[1:]))]),1024,'d_h3_fc')
	h4 = linear(h3,512,'d_h4_fc')
	h5 = linear(h4,1,'d_h5_fc')
	return tf.sigmoid(h5)
	"""
        h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
        h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
        h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
        h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
        h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
        return tf.nn.sigmoid(h4)
	"""
    def generator(self, real_image, y=None):

        h1 = conv2d(real_image,self.gf_dim*2,k_h=5,k_w=5,d_h=1,d_w=1, name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))
    
        h2 = conv2d(h1,self.gf_dim*4,d_h=1,d_w=1, name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))
   
        h3 = conv2d(h2,self.gf_dim*8,d_h=1,d_w=1, name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))
    
        h4 = conv2d(h3,self.gf_dim*4,d_h=1,d_w=1, name='g_h4')
        h4 = tf.nn.relu(self.g_bn4(h4))
        
	h5 = conv2d(h4,self.gf_dim*2,d_h=1,d_w=1, name='g_h5')
        h5 = tf.nn.relu(self.g_bn5(h5))
        
	h6 = conv2d(h5,3, d_h=1,d_w=1, name='g_h6')
    
        return tf.nn.tanh(h6)

    def sampler(self,images, y=None):

        tf.get_variable_scope().reuse_variables()    
	h1 = conv2d(images,self.gf_dim*2,k_h=5,k_w=5,d_h=1,d_w=1, name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1))
    
        h2 = conv2d(h1,self.gf_dim*4,d_h=1,d_w=1, name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2))
   
        h3 = conv2d(h2,self.gf_dim*8,d_h=1,d_w=1, name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3))
    
        h4 = conv2d(h3,self.gf_dim*4,d_h=1,d_w=1, name='g_h4')
        h4 = tf.nn.relu(self.g_bn4(h4))
        
	h5 = conv2d(h4,self.gf_dim*2,d_h=1,d_w=1, name='g_h5')
        h5 = tf.nn.relu(self.g_bn5(h5))
        
	h6 = conv2d(h5,3, d_h=1,d_w=1, name='g_h6')
    
        return tf.nn.tanh(h6)
    
    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

	    


