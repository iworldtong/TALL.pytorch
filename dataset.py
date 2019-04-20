#!/usr/bin/env python
import os
import sys
import numpy as np
import pickle

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms



def calc_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] -  inter[0]) / (union[1] - union[0])
    return iou


def calc_nIoL(base, sliding_clip):
    '''
    The reason we use nIoL is that we want the the most part of the sliding
    window clip to overlap with the assigned sentence, and simply increasing
    IoU threshold would harm regression layers ( regression aims to move the
    clip from low IoU to high IoU).
    '''
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1] - inter[0]
    sliding_l = sliding_clip[1] - sliding_clip[0]
    nIoL = 1.0 * (sliding_l - inter_l) / sliding_l
    return nIoL


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self,
                 sliding_dir,
                 it_path,
                 visual_dim,
                 sentence_embed_dim,
                 IoU=0.5,
                 nIoU=0.15,
                 context_num=1,
                 context_size=128,
                ):
        self.sliding_dir = sliding_dir
        self.it_path = it_path
        self.visual_dim = visual_dim
        self.sentence_embed_dim = sentence_embed_dim
        self.IoU = IoU
        self.nIoU = nIoU
        self.context_num = context_num
        self.context_size = context_size

        self.load_data()

    def load_data(self):
        '''
        Note:
            self.clip_sentence_pairs     : list of (ori_clip_name, sent_vec)
            self.clip_sentence_pairs_iou : list of (ori_clip_name, sent_vec, clip_name(with ".npy"), s_o, e_o) —— not all ground truth
        '''
        # movie_length_info = pickle.load(open("./video_allframes_info.pkl", 'rb'), encoding='iso-8859-1')
        print("Reading training data list from " + self.it_path)
        csv = pickle.load(open(self.it_path, 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs = []
        for l in csv:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))

        movie_names_set = set()
        self.movie_clip_names = {}
        # read groundtruth sentence-clip pairs
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        self.num_samples = len(self.clip_sentence_pairs)
        print(str(len(self.clip_sentence_pairs))+" clip-sentence pairs are readed")
        
        # read sliding windows, and match them with the groundtruths to make training samples
        sliding_clips_tmp = os.listdir(self.sliding_dir)
        self.clip_sentence_pairs_iou = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2]=="npy":
                movie_name = clip_name.split("_")[0]
                for clip_sentence in self.clip_sentence_pairs:
                    original_clip_name = clip_sentence[0] 
                    original_movie_name = original_clip_name.split("_")[0]
                    if original_movie_name == movie_name:
                        start = int(clip_name.split("_")[1])
                        end = int(clip_name.split("_")[2].split(".")[0])
                        o_start = int(original_clip_name.split("_")[1]) 
                        o_end = int(original_clip_name.split("_")[2].split(".")[0])
                        iou = calc_IoU((start, end), (o_start, o_end))
                        if iou > self.IoU:
                            nIoL = calc_nIoL((o_start, o_end), (start, end))
                            if nIoL < self.nIoU:
                                # movie_length = movie_length_info[movie_name.split(".")[0]]
                                start_offset = o_start - start
                                end_offset = o_end - end
                                self.clip_sentence_pairs_iou.append((clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
        self.num_samples_iou = len(self.clip_sentence_pairs_iou)
        print(str(len(self.clip_sentence_pairs_iou))+" iou clip-sentence pairs are readed")

    def __len__(self):
        return self.num_samples_iou

    def __getitem__(self, index):
        # read context features
        left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[index][2])
        feat_path = os.path.join(self.sliding_dir, self.clip_sentence_pairs_iou[index][2])
        featmap = np.load(feat_path)        
        vis = np.hstack((left_context_feat, featmap, right_context_feat))

        sent = self.clip_sentence_pairs_iou[index][1][:self.sentence_embed_dim]

        p_offset = self.clip_sentence_pairs_iou[index][3]
        l_offset = self.clip_sentence_pairs_iou[index][4]
        offset = np.array([p_offset, l_offset], dtype=np.float32)

        data_torch = {
            'vis'   : torch.from_numpy(vis),
            'sent'  : torch.from_numpy(sent),
            'offset': torch.from_numpy(offset),
        }
        return data_torch


    def get_context_window(self, clip_name):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        left_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        right_context_feats = np.zeros([self.context_num, self.visual_dim // 3], dtype=np.float32)
        last_left_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        last_right_feat = np.load(os.path.join(self.sliding_dir, clip_name))
        for k in range(self.context_num):
            left_context_start = start - self.context_size * (k + 1)
            left_context_end = start - self.context_size * k
            right_context_start = end + self.context_size * k
            right_context_end = end + self.context_size * (k + 1)
            left_context_name = movie_name + "_" + str(left_context_start) + "_" + str(left_context_end) + ".npy"
            right_context_name = movie_name + "_" + str(right_context_start) + "_" + str(right_context_end) + ".npy"

            left_context_path = os.path.join(self.sliding_dir, left_context_name)
            if os.path.exists(left_context_path):
                left_context_feat = np.load(left_context_path)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat

            right_context_path = os.path.join(self.sliding_dir, right_context_name)
            if os.path.exists(right_context_path):
                right_context_feat = np.load(right_context_path)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat

            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)



class TestingDataSet(object):
    def __init__(self, img_dir, csv_path, batch_size):
        #il_path: image_label_file path
        #self.index_in_epoch = 0
        #self.epochs_completed = 0
        self.batch_size = batch_size
        self.image_dir = img_dir
        print("Reading testing data list from "+csv_path)
        self.semantic_size = 4800
        csv = pickle.load(open(csv_path, 'rb'), encoding='iso-8859-1')
        self.clip_sentence_pairs = []
        for l in csv:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))
        print(str(len(self.clip_sentence_pairs))+" pairs are readed")
        movie_names_set = set()
        self.movie_clip_names = {}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        
        self.clip_num_per_movie_max = 0
        for movie_name in self.movie_clip_names:
            if len(self.movie_clip_names[movie_name])>self.clip_num_per_movie_max: self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])
        print("Max number of clips in a movie is "+str(self.clip_num_per_movie_max))
        
        self.sliding_clip_path = img_dir
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2]=="npy":
                movie_name = clip_name.split("_")[0]
                if movie_name in self.movie_clip_names:
                    self.sliding_clip_names.append(clip_name.split(".")[0]+"."+clip_name.split(".")[1])
        self.num_samples = len(self.clip_sentence_pairs)
        print("sliding clips number: "+str(len(self.sliding_clip_names)))
        assert self.batch_size <= self.num_samples
        

    def get_clip_sample(self, sample_num, movie_name, clip_name):
        length=len(os.listdir(self.image_dir+movie_name+"/"+clip_name))
        sample_step=1.0*length/sample_num
        sample_pos=np.floor(sample_step*np.array(range(sample_num)))
        sample_pos_str=[]
        img_names=os.listdir(self.image_dir+movie_name+"/"+clip_name)
        # sort is very important! to get a correct sequence order
        img_names.sort()
       # print img_names
        for pos in sample_pos:
            sample_pos_str.append(self.image_dir+movie_name+"/"+clip_name+"/"+img_names[int(pos)])
        return sample_pos_str
    
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = 128#end-start
        left_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)+".npy"
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)+".npy"
            if os.path.exists(self.sliding_clip_path+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)


    def load_movie(self, movie_name):
        movie_clip_sentences=[]
        for k in range(len(self.clip_names)):
            if movie_name in self.clip_names[k]:
                movie_clip_sentences.append((self.clip_names[k], self.sent_vecs[k][:2400], self.sentences[k]))

        movie_clip_imgs=[]
        for k in range(len(self.movie_frames[movie_name])):
           # print str(k)+"/"+str(len(self.movie_frames[movie_name]))            
            if os.path.isfile(self.movie_frames[movie_name][k][1]) and os.path.getsize(self.movie_frames[movie_name][k][1])!=0:
                img=load_image(self.movie_frames[movie_name][k][1])
                movie_clip_imgs.append((self.movie_frames[movie_name][k][0],img))
                    
        return movie_clip_imgs, movie_clip_sentences

    def load_movie_byclip(self,movie_name,sample_num):
        movie_clip_sentences=[]
        movie_clip_featmap=[]
        clip_set=set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0],self.clip_sentence_pairs[k][1][:self.semantic_size]))

                if not self.clip_sentence_pairs[k][0] in clip_set:
                    clip_set.add(self.clip_sentence_pairs[k][0])
                    # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                    visual_feature_path=self.image_dir+self.clip_sentence_pairs[k][0]+".npy"
                    feature_data=np.load(visual_feature_path)
                    movie_clip_featmap.append((self.clip_sentence_pairs[k][0],feature_data))
        return movie_clip_featmap, movie_clip_sentences
    
    def load_movie_slidingclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []
        clip_set = set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                visual_feature_path = self.sliding_clip_path+self.sliding_clip_names[k]+".npy"
                #context_feat=self.get_context(self.sliding_clip_names[k]+".npy")
                left_context_feat,right_context_feat = self.get_context_window(self.sliding_clip_names[k]+".npy",1)
                feature_data = np.load(visual_feature_path)
                #comb_feat=np.hstack((context_feat,feature_data))
                comb_feat = np.hstack((left_context_feat,feature_data,right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))
        return movie_clip_featmap, movie_clip_sentences

