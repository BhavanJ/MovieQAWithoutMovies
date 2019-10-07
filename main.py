import sys
# sys.path.append('/etc/VOLUME1/BJ_new/MovieQA/Layered-Memory-Network')
#import MovieQA_benchmark as MovieQA
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import os
import h5py
import math
from model import DataUtil
from model import ModelUtil
from model import SEModelUtil
import word2vec as w2v
import tensorflow as tf
from sklearn.decomposition import PCA
import cPickle as pickle
import time
import json
from collections import namedtuple
import pdb
import gensim
from gensim.models import Word2Vec
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

# from /data/bjasani/Backup_Aug2019/MovieQA/Layered-Memory-Network/Experiments/PyTorch_implementation/Tensorflow_version

# TODO subtitle version
# TODO..need to download google word2vec and store it at and also need to download the h5 file and store it at


EXP_NAME = '_ICCV_abalation'

QA_TRAIN_VAL_AUG_FILE_NAME = "./data/QA_augmented_wrt_ans_shuf_false.pkl" #Created using Task4_ver3.py #TODO verify these
# QA_TRAIN_VAL_AUG_FILE_NAME = "/etc/VOLUME1/BJ_new/MovieQA/Layered-Memory-Network/Experiments/PyTorch_implementation/Task_2_4/All_data/OUR_TRAIN_VAL_WITHOUT_TRAINING/QA_augmented_wrt_ans_shuf_false.pkl"
## QA_TRAIN_VAL_AUG_FILE_NAME = "/etc/VOLUME1/BJ_new/MovieQA/Layered-Memory-Network/Experiments/PyTorch_implementation/Task4/Data/Aug_QA/QA_augmented_wrt_ans_shuf_false.pkl"     

######################################

QA_SUBSET_TYPE = 'ORIG_DATASET'
# QA_SUBSET_TYPE = 'ONLY_AUG'
# QA_SUBSET_TYPE = 'ONLY_UNBIASED'
# QA_SUBSET_TYPE = 'FULL_NEW'
# QA_SUBSET_TYPE = 'ONLY_BIASED'
# QA_SUBSET_TYPE = 'COMB_TRAIN_ORIG_VAL_UNBIASED'
# QA_SUBSET_TYPE = 'COMB_TRAIN_ORIG_VAL_UNBIASED+AUG'
# QA_SUBSET_TYPE = 'COMB_TRAIN_UNBIASED_VAL_ORIG'
# QA_SUBSET_TYPE = 'COMB_TRAIN_BIASED_VAL_ORIG'

######################################
MODEL_TYPE = 'QA'
# MODEL_TYPE = 'subtitle'
# MODEL_TYPE = 'video'
# MODEL_TYPE = 'video+subtitle'

######################################
# W2V_FILE_TYPE = 'MAK'
# W2V_FILE_TYPE = 'GOOGLE'
# W2V_MODAL = None

W2V_FILE_TYPE = 'OURS'
# W2V_MODAL = 'val_only'
# W2V_MODAL = 'train_only'
W2V_MODAL = 'train_val'
# W2V_MODAL = 'gen_only'
# W2V_MODAL = 'gen_val'
# W2V_MODAL = 'gen_train'
# W2V_MODAL = 'gen_train_val'

#######################################
MOMENTUM = None
LEARNING_RATE = 0.01
BATCH_SIZE = 8
N_EPOCH = 100

##########################################################################################################

DATA_SAVE_FILE_NAME = QA_SUBSET_TYPE + '_' + MODEL_TYPE + '_' + W2V_FILE_TYPE + '_' + str(W2V_MODAL) + '_N_' + str(N_EPOCH) + '_BS_' + str(BATCH_SIZE) + '_LR_' + str(LEARNING_RATE).split('.')[-1] + EXP_NAME
DATA_SAVE_FOLDER_NAME = './MovieQAWithoutMovies_abalation/' + QA_SUBSET_TYPE + '_' + MODEL_TYPE + '_' + W2V_FILE_TYPE + '_' + str(W2V_MODAL) + '/'

if not os.path.exists(DATA_SAVE_FOLDER_NAME):    
    os.mkdir(DATA_SAVE_FOLDER_NAME)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

sys.stdout = Logger(DATA_SAVE_FOLDER_NAME+DATA_SAVE_FILE_NAME +'.txt')
## sys.stdout = open(DATA_SAVE_FILE_NAME +'.txt', 'w')


print('\n\n')
print(QA_SUBSET_TYPE)
print(MODEL_TYPE,W2V_FILE_TYPE,W2V_MODAL)
print(LEARNING_RATE,BATCH_SIZE,N_EPOCH,MOMENTUM)
print(DATA_SAVE_FILE_NAME)
print(DATA_SAVE_FOLDER_NAME)
print(EXP_NAME)
print(QA_TRAIN_VAL_AUG_FILE_NAME)
print('\n\n')

################################################################################################################################################




def build_model(input_video, input_stories, input_question, input_answer, answer_index, v2i, pca_mat = None, lr = 0.01,d_w2v=300,d_lproj=300, question_guided=False):

	# w2v_mqa_model_filename = './data/movie_plots_1364.d-300.mc1.w2v'
	# w2v_model_movie = w2v.load(w2v_mqa_model_filename, kind='bin')

	# w2v_mqa_model_filename_new = '/etc/VOLUME1/BJ_new/MovieQA/word_embedding/Glove/GoogleNews-vectors-negative300.bin'
	# w2v_model_google = gensim.models.KeyedVectors.load_word2vec_format(  w2v_mqa_model_filename_new, binary=True)

	# custom_word2vec_filename = '/etc/VOLUME1/BJ_new/MovieQA/word_embedding/MovieQA_1.bin'
	# w2v_model_custom = Word2Vec.load(custom_word2vec_filename)

	# custom_word2vec_filename = '/etc/VOLUME1/BJ_new/MovieQA/word_embedding/all_models/orig_w2v_movie.bin'
	# custom_word2vec_filename = '/etc/VOLUME1/BJ_new/MovieQA/word_embedding/all_models/MovieQA_gensim_shuffle_sentence.bin'

	# custom_word2vec_filename = '/etc/VOLUME1/BJ_new/MovieQA/word_embedding/all_models/MovieQA_gensim_single_sentence.bin'
	# w2v_model_custom = w2v.load(custom_word2vec_filename, kind='bin')


	with tf.variable_scope('video_subtitle_hierarchical_frame_clip') as scope:
		

		if W2V_FILE_TYPE == 'MAK':
			w2v_mqa_model_filename = './w2v_models/MovieQA/movie_plots_1364.d-300.mc1.w2v'
			w2v_model = w2v.load(w2v_mqa_model_filename, kind='bin')

		elif W2V_FILE_TYPE == 'OURS':
			custom_word2vec_filename = './w2v_models/gensim_models_splits/w2v_gensim_Makarand_' + W2V_MODAL
			w2v_model = Word2Vec.load(custom_word2vec_filename)

		elif W2V_FILE_TYPE == 'GOOGLE':
			w2v_mqa_model_filename_new = './w2v_models/Glove/GoogleNews-vectors-negative300.bin'
			w2v_model = gensim.models.KeyedVectors.load_word2vec_format(  w2v_mqa_model_filename_new, binary=True)			

		if MODEL_TYPE == 'video+subtitle':
			
			T_B, T_w2v, T_mask, pca_mat_ = ModelUtil.setWord2VecModelConfiguration(v2i,w2v_model,d_w2v,d_lproj, file_type = W2V_FILE_TYPE)

			embedded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
			embedded_question = SEModelUtil.getAverageRepresentation(embedded_question_words,T_B,d_lproj)
			embedded_stories_words, mask_s = ModelUtil.getEmbeddingWithWord2Vec(input_stories, T_w2v, T_mask)
			embeded_stories = SEModelUtil.getAverageRepresentation(embedded_stories_words, T_B, d_lproj)
			embedded_video = SEModelUtil.getVideoDualSemanticEmbeddingWithQuestionAttention_question_guid(embedded_stories_words, d_lproj,input_video, T_w2v, embeded_stories, embedded_question, T_B, pca_mat=pca_mat, return_sequences=True)
			embedded_answer_words, mask_a = ModelUtil.getEmbeddingWithWord2Vec(input_answer, T_w2v, T_mask)
			embedded_answer = SEModelUtil.getAverageRepresentation(embedded_answer_words,T_B,d_lproj)

			video_loss,video_scores = ModelUtil.getClassifierLoss(embedded_video, embedded_question, embedded_answer, answer_index=answer_index)
			loss = tf.reduce_mean(video_loss) 
			optimizer = tf.train.GradientDescentOptimizer(lr)
			train = optimizer.minimize(loss)
			return train,loss,video_scores
		
		elif MODEL_TYPE == 'subtitle':

			T_B, T_w2v, T_mask, pca_mat_ = ModelUtil.setWord2VecModelConfiguration(v2i,w2v_model,d_w2v,d_lproj, file_type = W2V_FILE_TYPE)
			
			embedded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
			embedded_question = SEModelUtil.getAverageRepresentation(embedded_question_words,T_B,d_lproj)			
			embedded_stories_words, mask_s = ModelUtil.getEmbeddingWithWord2Vec(input_stories, T_w2v, T_mask)
			embeded_stories = SEModelUtil.getAverageRepresentation(embedded_stories_words, T_B, d_lproj)
			embeded_stories_reduced = tf.reduce_sum(embeded_stories,reduction_indices=1) 
			embeded_stories_reduced = tf.nn.l2_normalize(embeded_stories_reduced,1)
			embedded_answer_words, mask_a = ModelUtil.getEmbeddingWithWord2Vec(input_answer, T_w2v, T_mask)
			embedded_answer = SEModelUtil.getAverageRepresentation(embedded_answer_words,T_B,d_lproj)
			
			video_loss,video_scores = ModelUtil.getClassifierLoss(embeded_stories_reduced, embedded_question, embedded_answer, answer_index=answer_index)
			loss = tf.reduce_mean(video_loss) 
			optimizer = tf.train.GradientDescentOptimizer(lr)
			train = optimizer.minimize(loss)
			return train,loss,video_scores

		
		elif MODEL_TYPE == 'video':

			T_B, T_w2v, T_mask, pca_mat_ = ModelUtil.setWord2VecModelConfiguration(v2i,w2v_model,d_w2v,d_lproj, file_type = W2V_FILE_TYPE)
			
			embedded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
			embedded_question = SEModelUtil.getAverageRepresentation(embedded_question_words,T_B,d_lproj)
			# embedded_stories_words, mask_s = ModelUtil.getEmbeddingWithWord2Vec(input_stories, T_w2v, T_mask)
			# embeded_stories = SEModelUtil.getAverageRepresentation(embedded_stories_words, T_B, d_lproj)
			embedded_video = ModelUtil.getVideoSemanticEmbedding(input_video,T_w2v,T_B, pca_mat=pca_mat) #....just using videos, doesn't use subtitles, and not even question for attention
			embedded_answer_words, mask_a = ModelUtil.getEmbeddingWithWord2Vec(input_answer, T_w2v, T_mask)
			embedded_answer = SEModelUtil.getAverageRepresentation(embedded_answer_words,T_B,d_lproj)
			
			video_loss,video_scores = ModelUtil.getClassifierLoss(embedded_video, embedded_question, embedded_answer, answer_index=answer_index)
			loss = tf.reduce_mean(video_loss) 
			optimizer = tf.train.GradientDescentOptimizer(lr)
			train = optimizer.minimize(loss)
			return train,loss,video_scores

		elif MODEL_TYPE == 'QA':

			T_B, T_w2v, T_mask, pca_mat_ = ModelUtil.setWord2VecModelConfiguration(v2i,w2v_model,d_w2v,d_lproj, file_type = W2V_FILE_TYPE)
			
			embedded_question_words, mask_q = ModelUtil.getEmbeddingWithWord2Vec(input_question, T_w2v, T_mask)
			embedded_question = SEModelUtil.getAverageRepresentation(embedded_question_words,T_B,d_lproj)
			embedded_answer_words, mask_a = ModelUtil.getEmbeddingWithWord2Vec(input_answer, T_w2v, T_mask)
			embedded_answer = SEModelUtil.getAverageRepresentation(embedded_answer_words,T_B,d_lproj)
			
			video_loss,video_scores = ModelUtil.getClassifierLoss_QA_only(embedded_question, embedded_answer, answer_index=answer_index)
			loss = tf.reduce_mean(video_loss) #shape=()
			optimizer = tf.train.GradientDescentOptimizer(lr)
			train = optimizer.minimize(loss)
			return train,loss,video_scores

		else:
			pdb.set_trace()	


def linear_project_pca_initialization(hf, feature_shape, d_w2v=300, output_path=None):

	print('--utilize PCA to initialize the embedding matrix of feature to d_w2v')
	samples = []
	for imdb_key in hf.keys():
		feature = hf[imdb_key][:]
		axis = [0,2,3,1]
		feature = np.transpose(feature, tuple(axis))
		feature = np.reshape(feature,(-1,feature_shape[1]))
		feature = np.random.permutation(feature)
		samples.extend(feature[:50])
	print('samples:',len(samples))

	pca = PCA(n_components=d_w2v, whiten=True)
	pca_mat = pca.fit_transform(np.asarray(samples).T)  # 1024 x 300

	pickle.dump(pca_mat,open(output_path,'w'))
	print('pca_amt dump to file:',output_path)
	return pca_mat


def exe_model(sess, data, batch_size, v2i, hf, feature_shape, stories, story_shape,
	loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=None, nql=25, nqa=32):
	if train is not None:
		np.random.shuffle(data)

	total_data = len(data)
	# num_batch = int(round(total_data*1.0/batch_size))
	num_batch = int(math.ceil(total_data*1.0/batch_size))
	
	model_pred = []
	model_batch_data = []

	total_correct_num = 0
	total_loss = 0.0
	for batch_idx in xrange(num_batch):

		batch_qa = data[batch_idx*batch_size:min((batch_idx+1)*batch_size,total_data)]
	   
		data_q,data_a,data_y = DataUtil.getBatchIndexedQAs_return(batch_qa,v2i, nql=nql, nqa=nqa, numOfChoices=numberOfChoices)
		data_s = DataUtil.getBatchIndexedStories(batch_qa,stories,v2i,story_shape)
		data_v = DataUtil.getBatchVideoFeatureFromQid(batch_qa, hf, feature_shape)

		if train is not None:
			_, l, s = sess.run([train,loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})
		else:
			l, s = sess.run([loss,scores],feed_dict={input_video:data_v, input_stories:data_s, input_question:data_q, input_answer:data_a, y:data_y})

		if train == None:
			for iii in range(len(batch_qa)):
				model_pred.append(s[iii])
				model_batch_data.append(batch_qa[iii])

		num_correct = np.sum(np.where(np.argmax(s,axis=-1)==np.argmax(data_y,axis=-1),1,0))
		total_correct_num += num_correct
		total_loss += l
	total_acc = total_correct_num*1.0/total_data
	total_loss = total_loss/num_batch
	return total_acc, total_loss, np.asarray(model_pred), model_batch_data



def train_model(train_stories,val_stories,v2i,trained_video_QAs,val_video_QAs,hf,f_type,nql=25,nqa=32,numberOfChoices=5,
		feature_shape=(16,1024,7,7),
		batch_size=8,total_epoch=100,
		lr=0.01,pretrained_model=False,pca_mat_init_file=None):
	

	print("model type: " + MODEL_TYPE)
	size_voc = len(v2i) #26033
	max_sentences = 3660
	max_words = 40
	story_shape = (max_sentences,max_words)
	size_voc = len(v2i)

	print('building model ...')    
	if os.path.exists(pca_mat_init_file):
		pca_mat = pickle.load(open(pca_mat_init_file,'r'))
	else:
		pca_mat = linear_project_pca_initialization(hf, feature_shape, d_w2v=300, output_path=pca_mat_init_file)

	print('pca_mat.shape:',pca_mat.shape) # pca_mat.shape 0--> (512, 300)

	input_video = tf.placeholder(tf.float32, shape=(None,)+feature_shape,name='input_video')
	input_stories = tf.placeholder(tf.int32, shape=(None, max_sentences, max_words),name='input_stories')
	input_question = tf.placeholder(tf.int32, shape=(None,nql), name='input_question')
	input_answer = tf.placeholder(tf.int32, shape=(None,numberOfChoices,nqa), name='input_answer')
	y = tf.placeholder(tf.float32,shape=(None, numberOfChoices))

	#w2v_model_movie, w2v_model_google, w2v_model_custom,
	# answer_index = y

	train,loss,scores = build_model(input_video, input_stories, input_question, input_answer, y, v2i, pca_mat = pca_mat,  lr = lr, d_w2v = 300,d_lproj = 300, question_guided = False)
	# train,loss,scores = build_model(input_video, input_stories, input_question, input_answer, yyy, v2i, pca_mat = pca_mat)

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.5
	config.log_device_placement=False
	sess = tf.Session(config=config)
	init = tf.global_variables_initializer()
	sess.run(init)

	with open('/etc/VOLUME1/BJ_new/MovieQA/Layered-Memory-Network/train_split.json') as fid:
		trdev = json.load(fid)

	def getTrainDevSplit(trained_video_QAs,trdev):
		train_data = []
		dev_data = []
		for k, qa in enumerate(trained_video_QAs):

			if qa.imdb_key in trdev['train']:
				train_data.append(qa)
			else:
				dev_data.append(qa)
		return train_data,dev_data

	train_data,dev_data = getTrainDevSplit(trained_video_QAs,trdev)


	with sess.as_default():
		saver = tf.train.Saver(sharded=True,max_to_keep=total_epoch)

		# if pretrained_model is not None:
		#     saver.restore(sess, pretrained_model)
		#     print('restore pre trained file:' + pretrained_model)

		best_train_acc = 0.0
		best_dev_acc = 0.0
		best_val_acc = 0.0
		stopping_epoch = 0.0

		highest_acc_train     = 0.0
		highest_acc_dev       = 0.0
		highest_acc_val       = 0.0
		highest_acc_train_e   = 0.0 
		highest_acc_dev_e     = 0.0
		highest_acc_val_e     = 0.0

		list_train_acc = []
		list_dev_acc = []
		list_val_acc = []

		model_pred_val_best_epoch        = []
		model_batch_data_val_best_epoch   = []

		for epoch in xrange(total_epoch):

			if epoch == 0:
				train_mode = None
				print('\n\nJust testing.........\n\n')
			else:
				train_mode = train	
			
			# # shuffle
			print('Epoch: %d/%d, Batch_size: %d' %(epoch+1,total_epoch,batch_size))
			
			# train phase
			tic = time.time()
			total_acc_train, total_loss_train, model_pred_train, model_batch_data_train = exe_model(sess, train_data, batch_size, v2i, hf, feature_shape, train_stories, story_shape,
				loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=train_mode, nql=25, nqa=32)
			print('    --Train--, Loss: %.5f, Acc: %.5f.......Time:%.3f' %(total_loss_train,total_acc_train,time.time()-tic))

			# dev phase
			tic = time.time()
			total_acc_dev, total_loss_dev, model_pred_dev, model_batch_data_dev = exe_model(sess, dev_data, batch_size, v2i, hf, feature_shape, train_stories, story_shape,
				loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=None, nql=25, nqa=32)
			print('    --Train-val--, Loss: %.5f, Acc: %.5f.......Time:%.3f' %(total_loss_dev,total_acc_dev,time.time()-tic))
			
			# eval phase
			tic = time.time()
			total_acc_val, total_loss_val, model_pred_val, model_batch_data_val = exe_model(sess, val_video_QAs, batch_size, v2i, hf, feature_shape, val_stories, story_shape,
				loss, scores, input_video, input_question, input_stories, input_answer, y, numberOfChoices=5, train=None, nql=25, nqa=32)
			print('    --Val--,  Loss: %.5f, Acc: %.5f.......Time:%.3f' %(total_loss_val,total_acc_val,time.time()-tic))

			# sys.stdout.flush()

			list_train_acc.append(total_acc_train)
			list_dev_acc.append(total_acc_dev)
			list_val_acc.append(total_acc_val)

			'''	
			if epoch%20 == 0:
				epoch_list = np.arange(0, len(list_train_acc), 1).tolist()
				plt.figure()
				plt.plot(epoch_list,list_train_acc, label="Train accuracy")
				plt.plot(epoch_list,list_dev_acc,   label="Dev accuracy")
				plt.plot(epoch_list,list_val_acc,   label="Val accuracy")
				plt.legend()
				plt.title(DATA_SAVE_FILE_NAME)	
				plt.savefig(DATA_SAVE_FOLDER_NAME + DATA_SAVE_FILE_NAME)
				plt.clf()
				plt.close()
			'''


			if total_acc_dev > highest_acc_dev:
				highest_acc_dev = total_acc_dev
				highest_acc_dev_e = epoch

			if total_acc_val > highest_acc_val:
				highest_acc_val = total_acc_val
				highest_acc_val_e = epoch

			if total_acc_train > highest_acc_train:
				highest_acc_train = total_acc_train
				highest_acc_train_e = epoch

			if total_acc_dev > best_dev_acc:
				best_dev_acc = total_acc_dev
				best_val_acc = total_acc_val
				best_train_acc = total_acc_train
				stopping_epoch = epoch
				model_pred_val_best_epoch         = model_pred_val
				model_batch_data_val_best_epoch   = model_batch_data_val
	


		print('\n\n\n\n')
		print('Best train acc       ' + str(best_train_acc))
		print('Best dev acc         ' + str(best_dev_acc))
		print('Best val acc         ' + str(best_val_acc))
		print('Stopping epoch       ' + str(stopping_epoch))
		print('Highest train acc    ' + str(highest_acc_train))     
		print('Highest train epoch  ' + str(highest_acc_train_e))    
		print('Highest dev acc      ' + str(highest_acc_dev))       
		print('Highest dev epoch    ' + str(highest_acc_dev_e))     
		print('Highest val acc      ' + str(highest_acc_val))       
		print('Highest val epoch    ' + str(highest_acc_val_e))     
		print('\n\n\n\n')

		if N_EPOCH > 40:
			stopping_intial_epoch = np.argmax(np.asarray(list_dev_acc[0:40]))
			print('\n\n')
			print('Considering only first 40 epochs: \n')
			print('Best train acc       ' + str(list_train_acc[stopping_intial_epoch]))
			print('Best dev acc         ' + str(list_dev_acc[stopping_intial_epoch]))
			print('Best val acc         ' + str(list_val_acc[stopping_intial_epoch]))
			
		pdb.set_trace()	

		dict_save_data = {
				'list_train_acc'	:list_train_acc,
				'list_dev_acc'		:list_dev_acc,
				'list_val_acc'		:list_val_acc,
				'best_train_acc'	:best_train_acc,
				'best_dev_acc'		:best_dev_acc,
				'best_val_acc'		:best_val_acc,
				'QA_SUBSET_TYPE'	:QA_SUBSET_TYPE,
				'MODEL_TYPE'		:MODEL_TYPE,
				'W2V_FILE_TYPE'		:W2V_FILE_TYPE,
				'W2V_MODAL'			:W2V_MODAL,
				'MOMENTUM'			:MOMENTUM,
				'LEARNING_RATE'		:LEARNING_RATE, 
				'BATCH_SIZE'		:BATCH_SIZE, 
				'N_EPOCH'			:N_EPOCH,
				'QA_TRAIN_VAL_AUG_FILE_NAME' : QA_TRAIN_VAL_AUG_FILE_NAME,
				'EXP_NAME'			:EXP_NAME, 

				'DATA_SAVE_FILE_NAME'	:DATA_SAVE_FILE_NAME,
				'DATA_SAVE_FOLDER_NAME'	:DATA_SAVE_FOLDER_NAME,
				'stopping_epoch'	:stopping_epoch,
				'model_pred_val_best_epoch'         :model_pred_val_best_epoch,
				'model_batch_data_val_best_epoch'   :model_batch_data_val_best_epoch

		}


		epoch_list = np.arange(0, len(list_train_acc), 1).tolist()
		plt.figure()
		plt.plot(epoch_list,list_train_acc, label="Train accuracy")
		plt.plot(epoch_list,list_dev_acc,   label="Dev accuracy")
		plt.plot(epoch_list,list_val_acc,   label="Val accuracy")
		plt.legend()
		plt.title(DATA_SAVE_FILE_NAME)	
		plt.savefig(DATA_SAVE_FOLDER_NAME + DATA_SAVE_FILE_NAME)
		plt.clf()
		plt.close()
		
		np.save(DATA_SAVE_FOLDER_NAME + DATA_SAVE_FILE_NAME,dict_save_data)
			# #save model
			# # export_path = '/data1/wb/saved_model/vqa_baseline/video+subtitle'+'/'+f_type+'_b'+str(batch_size)+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])
			# export_path = './saved_model/vqa_baseline/video+subtitle'+'/'+f_type+'_b'+str(batch_size)+'/'+'lr'+str(lr)+'_f'+str(feature_shape[0])

			# if not os.path.exists(export_path):
			#     os.makedirs(export_path)
			#     print('mkdir %s' %export_path)
			# save_path = saver.save(sess, export_path+'/'+'E'+str(epoch+1)+'_A'+str(total_acc)+'.ckpt')
			# print("Model saved in file: %s" % save_path)






def trans(all):

	qa_list = []
	for dicts in all:
		qa_list.append(
			QAInfo(dicts['qid'], dicts['questions'], dicts['answers'] , dicts['ground_truth'],
				   dicts['imdb_key'], dicts['video_clips']))
	return qa_list        


if __name__ == '__main__':

	nql=25 # sequences length for question.....i.e. no. of words in the question
	nqa=32 # sequences length for anwser.......i.e. no. of words in the answer
	numberOfChoices = 5 # for input choices, one for correct, one for wrong answer
	QAInfo = namedtuple('QAInfo','qid question answers correct_index imdb_key video_clips')
	

	v2i = pickle.load(open("./model_files/movieQA_v2i.pkl","rb"))
	## qa_train = trans(pickle.load(open("/etc/VOLUME1/BJ_new/MovieQA/Layered-Memory-Network/data/process_train.pkl","rb")))  #TODO..remove
	## qa_val = trans(pickle.load(open("/etc/VOLUME1/BJ_new/MovieQA/Layered-Memory-Network/data/process_val.pkl","rb"))) #TODO..remove
	train_stories = pickle.load(open("./data/train_stories.pkl","rb"))
	val_stories = pickle.load(open("./data/val_stories.pkl","rb"))

	qa_train_original = trans(pickle.load(open("./data/process_train.pkl","rb")))
	qa_val_original   = trans(pickle.load(open("./data/process_val.pkl","rb")))

	qa_train_val_augumented = pickle.load(open(QA_TRAIN_VAL_AUG_FILE_NAME,"rb"))    
	qa_val_augmented   = trans(qa_train_val_augumented['qa_val_augmented'])
	qa_val_unbiased    = trans(qa_train_val_augumented['qa_val_unbiased'])
	qa_train_augmented = trans(qa_train_val_augumented['qa_train_augmented'])
	qa_train_unbaised  = trans(qa_train_val_augumented['qa_train_unbaised'])  

	list_qid_biased_train_QAs = [qa_tupple[0] for qa_tupple in qa_train_augmented]
	list_qid_biased_val_QAs   = [qa_tupple[0] for qa_tupple in qa_val_augmented]
	qa_train_biased    = [qa_tupple for qa_tupple in qa_train_original if qa_tupple[0] in list_qid_biased_train_QAs] 
	qa_val_biased      = [qa_tupple for qa_tupple in qa_val_original   if qa_tupple[0] in list_qid_biased_val_QAs]

	
	if QA_SUBSET_TYPE == 'ORIG_DATASET':
		qa_train_custom     = qa_train_original 
		qa_val_custom       = qa_val_original	
	
	elif QA_SUBSET_TYPE == 'ONLY_AUG':
		qa_train_custom  	= qa_train_augmented
		qa_val_custom    	= qa_val_augmented
	
	elif QA_SUBSET_TYPE == 'ONLY_UNBIASED':
		qa_train_custom  	= qa_train_unbaised
		qa_val_custom    	= qa_val_unbiased
	
	elif QA_SUBSET_TYPE == 'FULL_NEW': 
		qa_train_custom  	= qa_train_unbaised + qa_train_augmented
		qa_val_custom    	= qa_val_unbiased + qa_val_augmented
	
	elif QA_SUBSET_TYPE == 'ONLY_BIASED':
		qa_train_custom     = qa_train_biased
		qa_val_custom       = qa_val_biased

	elif QA_SUBSET_TYPE == 'COMB_TRAIN_ORIG_VAL_UNBIASED':
		qa_train_custom     = qa_train_original
		qa_val_custom       = qa_val_unbiased
	
	elif QA_SUBSET_TYPE == 'COMB_TRAIN_ORIG_VAL_UNBIASED+AUG':
		qa_train_custom     = qa_train_original
		qa_val_custom       = qa_val_unbiased + qa_val_augmented

	elif QA_SUBSET_TYPE == 'COMB_TRAIN_UNBIASED_VAL_ORIG':	
		qa_train_custom     = qa_train_unbaised
		qa_val_custom       = qa_val_original

	elif QA_SUBSET_TYPE == 'COMB_TRAIN_BIASED_VAL_ORIG':
		qa_train_custom     = qa_train_biased
		qa_val_custom       = qa_val_original

	else:
		pdb.set_trace()   


	video_feature_dims=512
	timesteps_v=32 # sequences length for video
	hight = 7
	width = 7
	feature_shape = (timesteps_v,video_feature_dims,hight,width)

	f_type = '224x224_VGG'
	# feature_path = '/etc/VOLUME1/BJ_new/MovieQA/Layered-Memory-Network/data/224x224_movie_all_clips_vgg_'+str(timesteps_v)+'f.h5'    #TODO
	feature_path = './data/224x224_movie_all_clips_vgg_'+str(timesteps_v)+'f.h5'    #TODO
	pca_mat_init_file = './data/224x224_vgg_pca_mat.pkl'
	hf = h5py.File(feature_path,'r')

	pretrained_model = None
	train_model(train_stories,val_stories,v2i,qa_train_custom,qa_val_custom,hf,f_type,nql=25,nqa=32,numberOfChoices=5,
		feature_shape=feature_shape,lr=LEARNING_RATE,
		batch_size=BATCH_SIZE,total_epoch=N_EPOCH,
		pretrained_model=pretrained_model,pca_mat_init_file=pca_mat_init_file)



#########################################################################################################################



 #    w2v_mqa_model_filename = './data/movie_plots_1364.d-300.mc1.w2v'
 #    w2v_model_movie = w2v.load(w2v_mqa_model_filename, kind='bin')
 #    ## w2v_mqa_model_filename = '/home/wb/movie_plots_1364.d-300.mc1.w2v'
 #    ## <word2vec.wordvectors.WordVectors object at 0x7f19b4492290>

 #    w2v_mqa_model_filename_new = '/etc/VOLUME1/BJ_new/MovieQA/word_embedding/Glove/GoogleNews-vectors-negative300.bin'
 #    w2v_model_google = gensim.models.KeyedVectors.load_word2vec_format(  w2v_mqa_model_filename_new, binary=True)
 #    ## w2v_model = w2v.load(w2v_mqa_model_filename_new, kind='bin')
 #    ## new_model = gensim.models.Word2Vec.load(w2v_mqa_model_filename_new)
 #    ## <gensim.models.keyedvectors.Word2VecKeyedVectors object at 0x7fc6d9770510>

 #    custom_word2vec_filename = '/etc/VOLUME1/BJ_new/MovieQA/word_embedding/MovieQA_1.bin'
 #    w2v_model_custom = Word2Vec.load(custom_word2vec_filename)




