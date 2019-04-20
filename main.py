import time
import pickle
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


from ctrl import *
from utils import *
from dataset import *

from config import CONFIG
cfg = CONFIG()


class Processor():
	def __init__(self):
		self.load_data()
		self.load_model()
		self.load_optimizer()	

	def load_data(self):
		self.data_loader = dict()
		if cfg.phase == 'train':
			self.data_loader['train'] = torch.utils.data.DataLoader(
				dataset=TrainDataset(cfg.train_feature_dir, 
								  cfg.train_csv_path,
								  cfg.visual_dim,
								  cfg.sentence_embed_dim,
								  cfg.IoU,
								  cfg.nIoU,
								  cfg.context_num,
                 				  cfg.context_size,
								),
				batch_size=cfg.batch_size,
				shuffle=True,
				num_workers=cfg.num_worker)

		self.testDataset = TestingDataSet(cfg.test_feature_dir, cfg.test_csv_path, cfg.test_batch_size)
		# self.data_loader['test'] = torch.utils.data.DataLoader(
		# 	dataset=TestDataset(cfg.train_feature_dir, 
		# 						  cfg.train_csv_path,
		# 						  cfg.visual_dim,
		# 						  cfg.sentence_embed_dim,
		# 						  cfg.IoU,
		# 						  cfg.nIoU,
		# 						  cfg.context_num,
  #                				  cfg.context_size,
		# 						),
		# 	batch_size=cfg.test_batch_size,
		# 	shuffle=False,
		# 	num_workers=cfg.num_worker)

	def load_model(self):
		torch.manual_seed(cfg.seed)
		if torch.cuda.is_available():
			if type(cfg.device) is list and len(cfg.device) > 1:
				torch.cuda.manual_seed_all(cfg.seed)
			else:
				torch.cuda.manual_seed(cfg.seed)

			self.output_device = cfg.device[0] if type(cfg.device) is list else cfg.device

		self.model = CTRL(cfg.visual_dim, cfg.sentence_embed_dim, cfg.semantic_dim, cfg.middle_layer_dim)
		self.loss = CTRL_loss(cfg.lambda_reg)
		if torch.cuda.is_available():
			self.model.cuda(self.output_device)
			self.loss.cuda(self.output_device)

		if torch.cuda.is_available() and type(cfg.device) is list and len(cfg.device) > 1:				
			self.model = nn.DataParallel(self.model, device_ids=cfg.device, output_device=self.output_device)

	def load_optimizer(self):
		if cfg.optimizer == 'Adam':
			self.optimizer = optim.Adam(
				self.model.parameters(),
				lr=cfg.vs_lr,
				weight_decay=cfg.weight_decay,
				)
		else:
			raise ValueError()


	def train(self):
		losses = []
		for epoch in range(cfg.max_epoch):
			for step, data_torch in enumerate(self.data_loader['train']):
				self.model.train()
				self.record_time()

				# forward
				output = self.model(data_torch['vis'], data_torch['sent'])
				loss = self.loss(output, data_torch['offset'])

				# backward
				self.optimizer.zero_grad()
				loss.backward()
				self.optimizer.step()
				losses.append(loss.item())

				duration = self.split_time()

				if (step+1) % 5 == 0 or step == 0:
					self.print_log('Epoch %d, Step %d: loss = %.3f (%.3f sec)' % (epoch+1, step+1, losses[-1], duration))

				if (step+1) % 2000 == 0:
					self.print_log('Testing:')
					movie_length_info = pickle.load(open(cfg.movie_length_info_path, 'rb'), encoding='iso-8859-1')
					self.eval(movie_length_info, step + 1, cfg.test_output_path)



	def eval(self, movie_length_info, step, test_output_path):
		self.model.eval()		
		IoU_thresh = [0.1, 0.2, 0.3, 0.4, 0.5]
		all_correct_num_10 = [0.0] * 5
		all_correct_num_5  = [0.0] * 5
		all_correct_num_1  = [0.0] * 5
		all_retrievd = 0.0

		for movie_name in self.testDataset.movie_names:
			movie_length=movie_length_info[movie_name.split(".")[0]]
			self.print_log("Test movie: " + movie_name + "....loading movie data")
			movie_clip_featmaps, movie_clip_sentences = self.testDataset.load_movie_slidingclip(movie_name, 16)
			self.print_log("sentences: "+ str(len(movie_clip_sentences)))
			self.print_log("clips: "+ str(len(movie_clip_featmaps)))
			sentence_image_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
			sentence_image_reg_mat = np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
			for k in range(len(movie_clip_sentences)):
				sent_vec = movie_clip_sentences[k][1]
				sent_vec = np.reshape(sent_vec,[1,sent_vec.shape[0]])
				for t in range(len(movie_clip_featmaps)):
					featmap = movie_clip_featmaps[t][1]
					visual_clip_name = movie_clip_featmaps[t][0]
					start = float(visual_clip_name.split("_")[1])
					end = float(visual_clip_name.split("_")[2].split("_")[0])
					featmap = np.reshape(featmap, [1, featmap.shape[0]])

					output = self.model(torch.from_numpy(featmap), torch.from_numpy(sent_vec))
					output_np = output.detach().numpy()[0][0]

					sentence_image_mat[k,t] = output_np[0]
					reg_clip_length = (end - start) * (10 ** output_np[2])
					reg_mid_point = (start + end) / 2.0 + movie_length * output_np[1]
					reg_end = end + output_np[2]
					reg_start = start + output_np[1]

					sentence_image_reg_mat[k, t, 0] = reg_start
					sentence_image_reg_mat[k, t, 1] = reg_end

			iclips = [b[0] for b in movie_clip_featmaps]
			sclips = [b[0] for b in movie_clip_sentences]

			# calculate Recall@m, IoU=n
			for k in range(len(IoU_thresh)):
				IoU=IoU_thresh[k]
				correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
				correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
				correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
				self.print_log(movie_name+" IoU="+str(IoU)+", R@10: "+str(correct_num_10/len(sclips))+"; IoU="+str(IoU)+", R@5: "+str(correct_num_5/len(sclips))+"; IoU="+str(IoU)+", R@1: "+str(correct_num_1/len(sclips)))
				all_correct_num_10[k]+=correct_num_10
				all_correct_num_5[k]+=correct_num_5
				all_correct_num_1[k]+=correct_num_1
			all_retrievd += len(sclips)
			
		for k in range(len(IoU_thresh)):
			self.print_log("IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd))
			with open(test_output_path, "w") as f:
				f.write("Step "+str(iter_step)+": IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)+"\n")


	def record_time(self):
		self.cur_time = time.time()
		return self.cur_time

	def split_time(self):
		split_time = time.time() - self.cur_time
		self.record_time()
		return split_time

	def print_log(self, line, print_time=True):
		if print_time:
			localtime = time.asctime(time.localtime(time.time()))
			line = "[ " + localtime + ' ] ' + line
		print(line)
		if cfg.save_log:
			with open(cfg.log_dir, 'a') as f:
				print(line, file=f)

	def print_time(self):
		localtime = time.asctime(time.localtime(time.time()))
		self.print_log("Local current time :  " + localtime)


if __name__ == '__main__':
	processor = Processor()
	processor.train()	


