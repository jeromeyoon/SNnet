import numpy as np 
import glob,sys,os
from sorting import natsorted
import pdb
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import argparse


def drawplot(csvpath):
	files = glob.glob(os.path.join(csvpath,'*.csv'))
	files = natsorted(files)
	num_files = len(files)
	for f in range(num_files):
		read_files(files[f],csvpath)


def draw_mean(csvpath):
	min_idx = 16
	files = glob.glob(os.path.join(csvpath,'*.csv'))
	files = natsorted(files)
	num_files = len(files)
	color = iter(cm.rainbow(np.linspace(0,1,num_files)))
	mean_error=[]
	mean_ang=[]
	mean_deg=[]
	for f in range(num_files):
		err,ang,deg = mean_read_files(files[f],csvpath)
		mean_error.append(err)
		mean_ang.append(ang)
		mean_deg.append(deg)
		if len(err) < min_idx-1:
			print('Minimum epochs is %d, current file has %d epochs' %(min_idx,len(err)))
			
	pdb.set_trace()
	for d in range(num_files):	
		labelname = files[d]
		labelname = labelname.split('/')
		labelname = labelname[-1].split('.')
		c = next(color)
		plt.plot(range(min_idx),mean_error[d][:min_idx],color=c,label=labelname[0])
	plt.legend(loc='best')
	#plt.legend(loc='upper left',bbox_to_anchor=[0,1],ncol=1)
	plt.savefig(os.path.join(csvpath,'mean_err.png'))
	plt.close()

	color = iter(cm.rainbow(np.linspace(0,1,num_files)))
	for d in range(num_files):	
		labelname = files[d]
		labelname = labelname.split('/')
		labelname = labelname[-1].split('.')
		c = next(color)
		plt.plot(range(min_idx),mean_ang[d][:min_idx],color=c,label=labelname[0])
	plt.legend(loc='upper left',bbox_to_anchor=[0,1],ncol=1)
	plt.savefig(os.path.join(csvpath,'mean_ang.png'))
	plt.close()

	color = iter(cm.rainbow(np.linspace(0,1,num_files)))
	for d in range(num_files):	
		labelname = files[d]
		labelname = labelname.split('/')
		labelname = labelname[-1].split('.')
		c = next(color)
		plt.plot(range(min_idx),mean_deg[d][:min_idx],color=c,label=labelname[0])
	plt.legend(loc='upper left',bbox_to_anchor=[0,1],ncol=1)
	plt.savefig(os.path.join(csvpath,'good_deg.png'))
	plt.close()

def mean_read_files(file_,csvpath):
	min_err = 100.0
	min_arg = 100.0
	max_deg = 0.0
	err_epoch = 0
	arg_epoch = 0
	deg_epoch = 0
	err_val = []
	arg_val = []
	deg_val = []		
	index = []
	with open (file_,'r') as f:
		content  = f.readlines()
		for line in range(1,len(content)):
			values = content[line].strip().split(",")
			err = float(values[0])
			arg = float(values[1])	
			deg = float(values[2])
			err_val.append(err) 
			arg_val.append(arg)
			deg_val.append(deg)
		return err_val,arg_val,deg_val
		"""
			if err < min_err:
				min_err = err
				err_epoch = line
			if arg < min_arg:
				min_arg = arg
				arg_epoch = line
			if deg > max_deg:
				max_deg = deg
				deg_epoch = line
			index.append(line)
			err_val.append(err) 
			arg_val.append(arg)
			deg_val.append(deg)

		name = file_.strip().split("/")
		name = name[-1].strip().split(".")
		name = name[0]
		plt.plot(index,err_val,'r')
		plt.title('min abs error:%f epoch %d' %(min_err,err_epoch))
		plt.savefig(os.path.join(csvpath,name +'_err.png'))
		plt.close()
		plt.plot(index,arg_val,'g')
		plt.title('min ang error:%f epoch %d' %(min_arg,arg_epoch))
		plt.savefig(os.path.join(csvpath,name +'_arg.png'))
		plt.close()
		plt.plot(index,deg_val,'b')
		plt.title('max ang within 10 deg:%f epoch %d' %(max_deg,deg_epoch))
		plt.savefig(os.path.join(csvpath,name +'_deg.png'))
		plt.close()
		"""
def read_files(file_,csvpath):
	min_err = 100.0
	min_arg = 100.0
	max_deg = 0.0
	err_epoch = 0
	arg_epoch = 0
	deg_epoch = 0
	err_val = []
	arg_val = []
	deg_val = []		
	index = []
	with open (file_,'r') as f:
		content  = f.readlines()
		for line in range(len(content)):
			values = content[line].strip().split(",")
			err = float(values[0])
			arg = float(values[2])	
			deg = float(values[6])

			if err < min_err:
				min_err = err
				err_epoch = line
			if arg < min_arg:
				min_arg = arg
				arg_epoch = line
			if deg > max_deg:
				max_deg = deg
				deg_epoch = line
			index.append(line)
			err_val.append(err) 
			arg_val.append(arg)
			deg_val.append(deg)

		name = file_.strip().split("/")
		name = name[-1].strip().split(".")
		name = name[0]
		plt.plot(index,err_val,'r')
		plt.title('min abs error:%f epoch %d' %(min_err,err_epoch))
		plt.savefig(os.path.join(csvpath,name +'_err.png'))
		plt.close()
		plt.plot(index,arg_val,'g')
		plt.title('min ang error:%f epoch %d' %(min_arg,arg_epoch))
		plt.savefig(os.path.join(csvpath,name +'_arg.png'))
		plt.close()
		plt.plot(index,deg_val,'b')
		plt.title('max ang within 10 deg:%f epoch %d' %(max_deg,deg_epoch))
		plt.savefig(os.path.join(csvpath,name +'_deg.png'))
		plt.close()
		


def finding_bestlossfunction(mainpath): 

	min_err =100.0
	min_arg =100.0
	max_deg = 0.0
	err_view='err'
	arg_view='arg'
	deg_view='deg'
	dirs  = os.listdir(mainpath) 
	for d in range(len(dirs)):
		print('dir %d /%d \n' %(d,len(dirs)))
		p = os.path.join(mainpath,dirs[d])
		files = glob.glob(os.path.join(p,'*.csv'))
		files = natsorted(files)
		for f in range(len(files)):
			file_ = os.path.join(p,files[f])
			#[min_err,min_arg,max_deg,err_epoch,arg_epoch,deg_epoch,err_view,arg_view,deg_view] = searchminmax(file_,min_err,min_arg,max_deg,files[f])		
			min_err,min_arg,max_deg,err_view,arg_view,deg_view = searchminmax(file_,min_err,min_arg,max_deg,err_view,arg_view,deg_view,file_)		
	print('min_err:%f view:%s \n' %(min_err,err_view))
	print('min_arg:%f view:%s\n' %(min_arg,arg_view))
	print('max_deg:%f view:%s\n' %(max_deg,deg_view))



def finding_bestview(mainpath): 

	min_err =100.0
	min_arg =100.0
	max_deg = 0.0
	err_view='err'
	arg_view='arg'
	deg_view='deg'
	files = glob.glob(os.path.join(mainpath,'*.csv'))
	files = natsorted(files)
	for d in range(len(files)):
		print('dir %d /%d \n' %(d,len(files)))
		file_ = os.path.join(mainpath,files[d])
		#[min_err,min_arg,max_deg,arr_epoch,arg_epoch,deg_epoch,err_view,arg_view,deg_view] = searchminmax(file_,min_err,min_arg,max_deg,files[f])		
		min_err,min_arg,max_deg,err_view,arg_view,deg_view = searchminmax(file_,min_err,min_arg,max_deg,err_view,arg_view,deg_view,file_)		
	print('min_err:%f view:%s \n' %(min_err,err_view))
	print('min_arg:%f view:%s\n' %(min_arg,arg_view))
	print('max_deg:%f view:%s\n' %(max_deg,deg_view))


def diff_lights(mainpath): 
	sum_diff=0.0
	num  = 0
	files = glob.glob(os.path.join(mainpath,'*.csv'))
	files = natsorted(files)
	for d in range(len(files)):
		print('dir %d /%d \n' %(d,len(files)))
		file_ = os.path.join(mainpath,files[d])

		with open (file_,'r') as f:
			content  = f.readlines()
			for line in range(1,len(content)):
				values = content[line].strip().split(",")
				sum_diff += float(values[4])
				num +=1
	print('mean different lights: %0.6f \n' %(sum_diff/num))

		
def searchminmax(file_,min_err,min_arg,max_deg,err_view,arg_view,deg_view,view):
	with open (file_,'r') as f:
		content  = f.readlines()
		for line in range(len(content)):
			values = content[line].strip().split(",")
			err = float(values[0])
			arg = float(values[2])	
			deg = float(values[6])

			if err < min_err:
				min_err = err
				#err_epoch = line
				err_view = view
			if arg < min_arg:
				min_arg = arg
				#arg_epoch = line
				arg_view = view
			if deg > max_deg:
				max_deg = deg
				#deg_epoch = line
				deg_view = view
		return min_err,min_arg,max_deg,err_view,arg_view,deg_view
		#return min_err,min_arg,max_deg,err_epoch,arg_epoch,deg_epoch,err_view,arg_view,deg_view

	
	
				
if __name__=='__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--path", help='for draw plots: ~/Dropbox/ECCV_result/Light_fixed/L1, for finding minmax: ~/Dropbox/ECCV_result/Light_fixed')
	parser.add_argument("--type", help='draw: for draw plot,find: finding minmax')
	args = parser.parse_args()
	print('type:%s path:%s \n' %(args.type,args.path))
	if args.type == 'draw':
		drawplot(args.path) #ex:inputpath( ~/Dropbox/ECCV_result/L1)
	elif args.type == 'view':
		finding_bestview(args.path) # findind the best loss function, light source fixed
	elif args.type == 'find':
		finding_bestlossfunction(args.path) # findind the best loss function, light source fixed
	elif args.type == 'lights':
		diff_lights(args.path)
	if args.type == 'mean':
		draw_mean(args.path) #ex:inputpath( ~/Dropbox/ECCV_result/L1)
	else:
		print('insert correct arguments')
	sys.exit(0)

