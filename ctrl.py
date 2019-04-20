#!/usr/bin/env python

import torch
import torch.nn as nn


class CTRL(nn.Module):
    def __init__(self, 
                visual_dim, 
                sentence_embed_dim,
                semantic_dim,
                middle_layer_dim,
                dropout_rate=0., 
                ):
        super(CTRL, self).__init__()
        self.semantic_dim = semantic_dim

        self.v2s_fc = nn.Linear(visual_dim, semantic_dim)
        self.s2s_fc = nn.Linear(sentence_embed_dim, semantic_dim)
        self.fc1 = nn.Conv2d(semantic_dim * 4, middle_layer_dim, kernel_size=1, stride=1)
        self.fc2 = nn.Conv2d(middle_layer_dim, 3, kernel_size=1, stride=1)
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, visual_feature, sentence_embed):
        batch_size, _ = visual_feature.size()

        transformed_clip = self.v2s_fc(visual_feature)
        transformed_sentence = self.s2s_fc(sentence_embed)

        transformed_clip_norm = transformed_clip / transformed_clip.norm(2, dim=1, keepdim=True) # by row
        transformed_sentence_norm = transformed_sentence / transformed_sentence.norm(2, dim=1, keepdim=True) # by row

        # Cross modal combine: [mul, add, concat]
        vv_f = transformed_clip_norm.repeat(batch_size, 1).reshape(batch_size, batch_size, self.semantic_dim)
        ss_f = transformed_sentence_norm.repeat(1, batch_size).reshape(batch_size, batch_size, self.semantic_dim)
        mul_feature = vv_f * ss_f
        add_feature = vv_f + ss_f
        cat_feature = torch.cat((vv_f, ss_f), 2)
        cross_modal_vec = torch.cat((mul_feature, add_feature, cat_feature), 2)
        
        # vs_multilayer 
        out = cross_modal_vec.unsqueeze(0).permute(0,3,1,2)  # match conv op 
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = out.permute(0,2,3,1).squeeze(0)

        return out

class CTRL_loss(nn.Module):
    def __init__(self, lambda_reg):
        super(CTRL_loss, self).__init__()
        self.lambda_reg = lambda_reg

    def forward(self, net, offset_label):
        batch_size = net.size()[0]
        sim_score_mat, p_reg_mat, l_reg_mat = net.split(1, dim=2)
        sim_score_mat = sim_score_mat.reshape(batch_size, batch_size)
        p_reg_mat = p_reg_mat.reshape(batch_size, batch_size)
        l_reg_mat = l_reg_mat.reshape(batch_size, batch_size)

        # make mask mat
        I_2 = 2.0 * torch.eye(batch_size)
        all1 = torch.ones([batch_size, batch_size])
        mask = all1 - I_2

        # loss cls, not considering iou
        I = torch.eye(batch_size)
        batch_para_mat = torch.ones([batch_size, batch_size]) / batch_size
        para_mat = I + batch_para_mat

        loss_mat = torch.log(all1 + torch.exp(torch.mul(mask, sim_score_mat)))
        loss_mat = torch.mul(loss_mat, para_mat)
        loss_align = torch.mean(loss_mat)

        # regression loss
        l_reg_diag = torch.mm(torch.mul(l_reg_mat, I), torch.ones([batch_size, 1]))
        p_reg_diag = torch.mm(torch.mul(p_reg_mat, I), torch.ones([batch_size, 1]))
        offset_pred = torch.cat((p_reg_diag, l_reg_diag), 1)
        loss_reg = torch.mean(torch.abs(offset_pred - offset_label))

        loss = loss_align + self.lambda_reg * loss_reg
        
        return  loss
                



