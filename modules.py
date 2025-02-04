import copy
import datetime
import numpy as np
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import models
from itertools import compress
from config import cfg
from data import make_data_loader, make_batchnorm_stats, FixTransform, MixDataset
from .utils import init_param, make_batchnorm, loss_fn ,info_nce_loss, SimCLR_Loss,elr_loss
from utils import to_device, make_optimizer, collate, to_device
from train_centralDA_target import op_copy
from metrics import Accuracy
from net_utils import set_random_seed
from net_utils import init_multi_cent_psd_label,init_psd_label_shot_icml,init_psd_label_shot_icml_up
from net_utils import EMA_update_multi_feat_cent_with_feat_simi,get_final_centroids
from data import make_dataset_normal
import gc
from utils import save
import json
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import pickle
from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster
import timm

class Server:
    def __init__(self, model):
        # if cfg['pretrained_source']:
        #         print('loading pretrained resnet50 model ')
        #         path_source = '/home/sampathkoti/Downloads/A-20231219T043936Z-001/A/'

        #         F = torch.load(path_source + 'source_F.pt')
        #         B = torch.load(path_source + 'source_B.pt')
        #         C = torch.load(path_source + 'source_C.pt')
        #         # print(F.keys())
        #         # exit()
        #         # model.backbone_layer.load_state_dict(torch.load(path_source + 'source_F.pt'))
        #         # model.feat_embed_layer.load_state_dict(torch.load(path_source + 'source_B.pt'))
        #         # model.class_layer.load_state_dict(torch.load(path_source + 'source_C.pt'))
        #         model.backbone_layer.load_state_dict(F)
        #         model.feat_embed_layer.load_state_dict(B)
        #         model.class_layer.load_state_dict(C)
        self.target_domains = len(list(cfg['unsup_doms'].split('-')))
        self.num_clusters = None
        self.cluster_labels = []
        # print(self.target_domains)
        # exit()
        if cfg['multi_model']:
            print('creating multiple models')
            self.model_state_dict = {}
            for i in range(self.target_domains):
                # print(i)
                self.model_state_dict[i] = save_model_state_dict(model.state_dict())
            self.global_model_state_dict = save_model_state_dict(model.state_dict())
            # print(self.model_state_dict.keys())
        else:
            self.model_state_dict = save_model_state_dict(model.state_dict())
            self.global_model_state_dict = save_model_state_dict(model.state_dict())
        self.avg_cent = None
        self.avg_cent_ = None
        self.var_cent = None
        # self.decay = 0.9
        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
            global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
        # print(self.model_state_dict.keys())
        del model
        del optimizer
        del global_optimizer


    def compute_dist(self,w1_in,w2_in,crit):
        if crit == 'mse':
            # return torch.mean((w1_in.reshape(-1).detach() - w2_in.reshape(-1).detach())**2)
            return  torch.norm((w1_in.reshape(-1) - w2_in.reshape(-1)),0.9)
        elif crit == 'mae':
            return torch.mean(torch.abs(w1_in.reshape(-1).detach() - w2_in.reshape(-1).detach()))
            #num = torch.sum((prev_global_p - client_p)**2)
            
        


    def compute_l2d_ratio(self,prev_g,valid_client,avg_model,crit='mse'):
        ll_div_weights = {}
        for k, _ in prev_g.named_parameters():
            ll_div_weights[k] = []
        
        param_prev_g = {}
        for k,v in prev_g.named_parameters():
            param_prev_g[k] = v
        
        param_avg = {}
        for k,v in avg_model.named_parameters():
            param_avg[k] = v

        for k, v in prev_g.named_parameters():
            parameter_type = k.split('.')[-1]
            if 'weight' in parameter_type or 'bias' in parameter_type:
                for m in range(len(valid_client)):
                    num = self.compute_dist(valid_client[m].model_state_dict[k],param_prev_g[k],crit)
                    den = self.compute_dist(valid_client[m].model_state_dict[k],param_avg[k],crit)
                    ll_div_weights[k].append(torch.exp(cfg['tau']*(num-den)))
                    # ll_div_weights[k].append(torch.exp(0.1*num/(den+1e-5)))
            #print("ll_div_weights[k]:",ll_div_weights[k])
            ll_div_weights[k] = torch.div(torch.Tensor(ll_div_weights[k]),sum(ll_div_weights[k]))
            # ll_div_weights[k] = torch.div(torch.Tensor(ll_div_weights[k]),sum(ll_div_weights[k]))
            #print("ll_div_weights[k]:",ll_div_weights[k])
         
        return ll_div_weights

    def cluster(self,client,epoch,init_model=None):
        client_ids =[]
        domain_ids = []
        feat = []
        model = eval('models.{}()'.format(cfg['model_name']))
        for client_i in client:
            # print(client.client_id,client.domain_id)
            client_ids.append(client_i.client_id.item())
            domain_ids.append(client_i.domain_id)

            model.load_state_dict(client_i.model_state_dict)
    
            feat.append(np.array(model.state_dict()['feat_embed_layer.bn.running_mean']))
            
            # exit()
        # print(client_ids)
        # print(domain_ids)
        # exit()
        feat = np.array(feat)
        feat = feat/(1e-9+np.linalg.norm(feat,axis=1,keepdims = True))
        print(feat.shape)
        # if epoch == 1:
        # Define the range of cluster numbers to evaluate
        min_clusters = 2
        max_clusters = 5

        # Initialize lists to store silhouette scores
        silhouette_scores = []

        # Compute hierarchical clustering and silhouette score for each number of clusters
        for n_clusters in range(min_clusters, max_clusters + 1):
            Z = hierarchy.linkage(feat, method='ward')
            cluster_labels = hierarchy.cut_tree(Z, n_clusters=n_clusters).flatten()
            silhouette_avg = silhouette_score(feat, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        Z = hierarchy.linkage(feat, method='ward')
        # # Determine the number of clusters
        print(silhouette_scores)
        print(max(silhouette_scores))
        k_ = silhouette_scores.index(max(silhouette_scores))  # Example: Number of clusters
        k = list(range(2,6))[k_]
        print('number of clusters',k)
        self.num_clusters = k
        
        # Assign cluster labels
        cluster_labels = fcluster(Z, k, criterion='maxclust')
        #####################################################################
        print('creating multiple models')
        self.model_state_dict_cluster = {}
        for i in range(k):
            # print(i)
            self.model_state_dict_cluster[i] = self.model_state_dict
        # print(self.model_state_dict.keys())
        # else:
        #     Z = hierarchy.linkage(feat, method='ward')
        #     # # Determine the number of clusters
        #     k =  self.num_clusters
            
        #     # Assign cluster labels
        #     cluster_labels = fcluster(Z, k, criterion='maxclust')
        self.global_model_state_dict = self.model_state_dict
            
        cluster_labels = list(cluster_labels)
        self.cluster_labels = cluster_labels
        # Print cluster labels
        print("Cluster Labels:", cluster_labels)
        print('GT Labels',domain_ids)
        # Initialize a dictionary to store indices for each cluster label
        indices_by_label = {}       

        # Iterate over data points and cluster labels
        for idx, label in enumerate(cluster_labels):
            if label not in indices_by_label:
                indices_by_label[label] = []
            indices_by_label[label].append(idx)

        # Print indices for each cluster label
        for label, indices in indices_by_label.items():
            print(f"Cluster Label {label}: Indices {indices}")
        
        og_indices_by_label = {}   
        for idx, label in enumerate(domain_ids):
            if label not in og_indices_by_label:
                og_indices_by_label[label] = []
            og_indices_by_label[label].append(idx)

        # Print indices for each cluster label ground truth
        for label, indices in og_indices_by_label.items():
            print(f"Cluster Label {label} GT: Indices {indices}")
            
        for i, client_i in enumerate(client):
            id = client_i.client_id
            if id == client_ids[i]:
                client_i.cluster_id = int(cluster_labels[i]-1)
        
        
        
        del model
        # del init_model
        gc.collect()
        torch.cuda.empty_cache()
        
            
    def distribute_cluster(self, client,epoch,BN_stats=False, batchnorm_dataset=None):
        # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        if cfg['world_size']==1:
            model = eval('models.{}()'.format(cfg['model_name']))
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        elif cfg['world_size']>1:
            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = eval('models.{}()'.format(cfg['model_name']))
            model = torch.nn.DataParallel(model,device_ids = [0, 1])
            model.to(cfg["device"])
        
        if epoch == 2:
            BN_stats = True
        else:
            BN_stats = False
        # model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                domain_id = client[m].domain_id
                cluster_id = client[m].cluster_id
                # print(cluster_id,self.model_state_dict_cluster.keys())
                print(f'client{client[m].client_id} with domain {domain_id} and cluster id {cluster_id}')
                print('cluster labels',self.cluster_labels)
                # exit()
                model.load_state_dict(self.model_state_dict_cluster[cluster_id])
                model_state_dict = save_model_state_dict(model.state_dict())
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                
                model.load_state_dict(self.global_model_state_dict)
                model_state_dict = save_model_state_dict(model.state_dict())
                client[m].global_model_state_dict = copy.deepcopy(model_state_dict)
                
                # if BN_stats == False:
                #     print('distributing without bn stats')
                #     for key in model_state_dict.keys():
                #         if 'bn' not in key or  'running' not in key:
                #             client[m].model_state_dict[key].data.copy_(model_state_dict[key])
                # elif BN_stats == True:
                #     print('distributing with bn stats')
                #     client[m].model_state_dict = copy.deepcopy(model_state_dict)
                if cfg['avg_cent']:
                    if self.avg_cent is not None:
                        client[m].avg_cent = self.avg_cent
                    else:
                        client[m].avg_cent = None
                        print('Warning:server.avg_cent is None')
        
        del model
        # del init_model
        gc.collect()
        torch.cuda.empty_cache()
        return       
    def distribute_multi(self, client,epoch,BN_stats=False, batchnorm_dataset=None):
        # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        if cfg['world_size']==1:
            model = eval('models.{}()'.format(cfg['model_name']))
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        elif cfg['world_size']>1:
            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = eval('models.{}()'.format(cfg['model_name']))
            model = torch.nn.DataParallel(model,device_ids = [0, 1])
            model.to(cfg["device"])
        
        
        # model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        if epoch ==1:
            BN_stats = True
            for m in range(len(client)):
                # if client[m].active:
                print('distributing to all clients at epch 1')
                domain_id = client[m].domain_id
                client[m].cluster_id = domain_id
                # print(domain_id,self.model_state_dict.keys())
                # exit()
                model.load_state_dict(self.model_state_dict[domain_id])
                model_state_dict = save_model_state_dict(model.state_dict())
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                
                if cfg['global_reg'] == 1:
                    model.load_state_dict(self.global_model_state_dict)
                    model_state_dict = save_model_state_dict(model.state_dict())
                    client[m].global_model_state_dict = copy.deepcopy(model_state_dict)
                # if BN_stats == False:
                #     print('distributing without bn stats')
                #     for key in model_state_dict.keys():
                #         if 'bn' not in key or  'running' not in key:
                #             client[m].model_state_dict[key].data.copy_(model_state_dict[key])
                # elif BN_stats == True:
                #     print('distributing with bn stats')
                #     client[m].model_state_dict = copy.deepcopy(model_state_dict)
                if cfg['avg_cent']:
                    if self.avg_cent is not None:
                        client[m].avg_cent = self.avg_cent
                    else:
                        client[m].avg_cent = None
                        print('Warning:server.avg_cent is None')
        else:
            BN_stats = False
            for m in range(len(client)):
                if client[m].active:
                    domain_id = client[m].domain_id
                    # print(domain_id,self.model_state_dict.keys())
                    # exit()
                    model.load_state_dict(self.model_state_dict[domain_id])
                    model_state_dict = save_model_state_dict(model.state_dict())
                    # client[m].model_state_dict = copy.deepcopy(model_state_dict)
                    if BN_stats == False:
                        print('distributing without bn stats')
                        for key in model_state_dict.keys():
                            if 'bn' not in key or  'running' not in key:
                                client[m].model_state_dict[key].data.copy_(model_state_dict[key])
                    elif BN_stats == True:
                        print('distributing with bn stats')
                        client[m].model_state_dict = copy.deepcopy(model_state_dict)
                    if cfg['avg_cent']:
                        if self.avg_cent is not None:
                            client[m].avg_cent = self.avg_cent
                        else:
                            client[m].avg_cent = None
                            print('Warning:server.avg_cent is None')
        
        
        del model
        # del init_model
        gc.collect()
        torch.cuda.empty_cache()
        return
    
    def distribute(self, client, batchnorm_dataset=None,BN_stats=False):
        # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        if cfg['world_size']==1:
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model = eval('models.{}()'.format(cfg['model_name']))
        elif cfg['world_size']>1:
            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = eval('models.{}()'.format(cfg['model_name']))
            model = torch.nn.DataParallel(model,device_ids = [0, 1])
            model.to(cfg["device"])
        
        model.load_state_dict(self.model_state_dict)
        # if batchnorm_dataset is not None:
        #     model = make_batchnorm_stats(batchnorm_dataset, model, 'global')

        model_state_dict = save_model_state_dict(model.state_dict())
        # model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())

        for m in range(len(client)):
            if client[m].active:
                if BN_stats == False:
                    print('distributing without bn stats')
                    for key in model_state_dict.keys():
                        if 'bn' not in key or  'running' not in key:
                            client[m].model_state_dict[key].data.copy_(model_state_dict[key])
                elif BN_stats == True:
                    print('distributing with bn stats')
                    client[m].model_state_dict = copy.deepcopy(model_state_dict)
                if cfg['avg_cent']:
                    if self.avg_cent is not None:
                        client[m].avg_cent = self.avg_cent
                    else:
                        client[m].avg_cent = None
                        print('Warning:server.avg_cent is None')
            elif BN_stats == True:
                # elif BN_stats == True:
                print('distributing with bn stats')
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
        
        del model
        gc.collect()
        torch.cuda.empty_cache()    
        return
    
    def distribute_fix_model(self, client, batchnorm_dataset=None):
        # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        if cfg['world_size']==1:
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        elif cfg['world_size']>1:
            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = eval('models.{}()'.format(cfg['model_name']))
            model = torch.nn.DataParallel(model,device_ids = [0, 1])
            model.to(cfg["device"])
        model.load_state_dict(self.model_state_dict,strict= False)
        # if batchnorm_dataset is not None:
        #     model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        # model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        for m in range(len(client)):
            
            client[m].fix_model_state_dict = copy.deepcopy(model_state_dict)
        return
    
    def update(self, client):
        if ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0 and cfg['with_BN'] == 1):
            print('FedAvg with BN params')
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    # model = eval('models.{}()'.format(cfg['model_name']))
                    if cfg['world_size']==1:
                        model = eval('models.{}()'.format(cfg['model_name']))
                    elif cfg['world_size']>1:
                        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = eval('models.{}()'.format(cfg['model_name']))
                        model = torch.nn.DataParallel(model,device_ids = [0, 1])
                        model.to(cfg["device"])
                    # model.load_state_dict(self.model_state_dict)
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    # weight = weight / weight.sum()
                    if cfg['wt_avg']:
                        for i in range(len(valid_client)):
                            weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                    weight = weight / weight.sum()

                    # # Store the averaged batchnorm parameters
                    # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}
                    # print()
                    for k, v in model.named_parameters():
                        
                        isBatchNorm = True if  '.bn' in k else False
                        parameter_type = k.split('.')[-1]
                        # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                # print(valid_client[m].model_state_dict.keys())
                                if cfg['world_size']==1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                elif  cfg['world_size']>1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                            v.grad = (v.data - tmp_v).detach()

                    global_optimizer.step()

                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
                    # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        elif ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0 and cfg['with_BN'] == 0):
            print('FedAvg with out BN params')
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    # model = eval('models.{}()'.format(cfg['model_name']))
                    if cfg['world_size']==1:
                        model = eval('models.{}()'.format(cfg['model_name']))
                    elif cfg['world_size']>1:
                        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = eval('models.{}()'.format(cfg['model_name']))
                        model = torch.nn.DataParallel(model,device_ids = [0, 1])
                        model.to(cfg["device"])
                    # model.load_state_dict(self.model_state_dict)
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    # weight = weight / weight.sum()
                    if cfg['wt_avg']:
                        for i in range(len(valid_client)):
                            weight[i] = valid_client[i].data_len
                # print(weight.sum())
                    weight = weight / weight.sum()

                    # # Store the averaged batchnorm parameters
                    # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}

                    for k, v in model.named_parameters():
                        
                        isBatchNorm = True if  '.bn' in k else False
                        parameter_type = k.split('.')[-1]
                        # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                        if  not isBatchNorm and ('weight' in parameter_type or 'bias' in parameter_type):
                            print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                if cfg['world_size']==1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                elif  cfg['world_size']>1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                            v.grad = (v.data - tmp_v).detach()

                    global_optimizer.step()

                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
                    # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())

        elif ('fmatch' in cfg['loss_mode'] and cfg['adapt_wt'] == 0):
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    # weight = weight / weight.sum()
                    if cfg['wt_avg']:
                        for i in range(len(valid_client)):
                            weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        # print(k)
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    # self.model_state_dict = save_model_state_dict(model.state_dict())
                    self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())

        elif cfg['adapt_wt'] == 1:
            print('dynamic aggregation')
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    prev_model = copy.deepcopy(model)

                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    # weight = weight / weight.sum()
                    if cfg['wt_avg']:
                        for i in range(len(valid_client)):
                            weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
                    
                    avg_model = copy.deepcopy(model)
                    

                    ###### compute the adaptive weights ######
                    weight_dict = self.compute_l2d_ratio(prev_model,valid_client,avg_model,'mae')
                    # weight_dict = self.compute_l2d_ratio(prev_model,valid_client,avg_model,'mse')

                    global_optimizer = make_optimizer(prev_model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()

                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight_dict[k][m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        # for i in range(len(client)):
        #     client[i].active = False
        # return
        if cfg['avg_cent'] == 1:
            with torch.no_grad():
                count=0
                for i in range(len(client)):
                    # print(i)
                    # print(self.avg_cent,client[i].cent)
                    if client[i].active == True and client[i].cent is not None:
                        if count==0:
                            self.avg_cent_=client[i].cent/(1e-9+client[i].var_cent)
                            count+=1
                        else:
                            self.avg_cent_+=client[i].cent/(1e-9+client[i].var_cent)
                    elif client[i].active == True  and client[i].cent is None:
                        print('Warning:client centntroid is None')
                    
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                # sum_var  = torch.zeros_like()
                for i in range(len(valid_client)):
                    if i == 0:
                        sum_var  = torch.zeros_like(valid_client[i].var_cent)
                        # prod_var = torch.zeros_like(valid_client[i].var_cent)
                    sum_var += 1.0/(1e-9+valid_client[i].var_cent)
                    # sum_var+= valid_client[i].var_cent
                    # prod_var*= valid_client[i].var_cent
                # sum_var = sum_var/(1e-9+prod_var)
                # print(sum_var)
                if self.avg_cent_ is not None:
                    print('averaging centroids')
                    # exit()
                    self.avg_cent_=self.avg_cent_/(1e-9+sum_var)
                    # self.avg_cent_=self.avg_cent_/len(valid_client)
                    if self.avg_cent is None:
                        self.avg_cent = self.avg_cent_
                    # self.avg_cent = cfg['decay']*self.avg_cent+(1-cfg['decay'])*self.avg_cent_
        if cfg['save_cent'] == 1:
            with torch.no_grad():
                # count=0
                # cent_list=[]
                cent_info = {}
                for i in range(len(client)):    
                    # print(client[i].client_id.item())
                    # print(client[i].domain)
                    # print(client[i].domain_id)
                    cent_info[client[i].client_id.item()] = [client[i].domain_id,client[i].domain,client[i].cent.cpu(),client[i].var_cent.cpu()]
            print('saving_centroids')
            save(cent_info, './output/cent_info10_{}.pt'.format(cfg['cent_log']))
            cfg['cent_log']+=1
            # print(cent_info)
            # exit()
        # print('avg_cent',self.avg_cent.shape)
        # # print(torch.isnan(self.avg_cent).any())
        # exit()
        return
        
        
    def update_multi(self, client):
        if ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0 and cfg['with_BN'] == 1):
            print('FedAvg with BN params')
            with torch.no_grad():
                for d in range(self.target_domains):
                    valid_client = [client[i] for i in range(len(client)) 
                                     if (client[i].active and client[i].domain_id==d) ]
                    # for client in valid_client:
                    #     print('domain:',client)
                    if len(valid_client) > 0:
                        # model = eval('models.{}()'.format(cfg['model_name']))
                        if cfg['world_size']==1:
                            model = eval('models.{}()'.format(cfg['model_name']))
                        elif cfg['world_size']>1:
                            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = eval('models.{}()'.format(cfg['model_name']))
                            model = torch.nn.DataParallel(model,device_ids = [0, 1])
                            model.to(cfg["device"])
                        # model.load_state_dict(self.model_state_dict)
                        model.load_state_dict(self.model_state_dict[d])
                        global_optimizer = make_optimizer(model.parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        weight = torch.ones(len(valid_client))
                        # weight = weight / weight.sum()
                        if cfg['wt_avg']:
                            for i in range(len(valid_client)):
                                weight[i] = valid_client[i].data_len
                        # print(weight.sum())
                        weight = weight / weight.sum()

                        # # Store the averaged batchnorm parameters
                        # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}
                        # print()
                        for m in range(len(valid_client)):
                            # print(valid_client[m].model_state_dict.keys())
                            print(f'domain:{valid_client[m].domain},id:{valid_client[m].domain_id},d:{d}')
                        for k, v in model.named_parameters():
                            
                            isBatchNorm = True if  '.bn' in k else False
                            parameter_type = k.split('.')[-1]
                            # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    # print(valid_client[m].model_state_dict.keys())
                                    # print(f'domain:{valid_client[m].domain},id:{valid_client[m].domain_id}')
                                    if cfg['world_size']==1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                    elif  cfg['world_size']>1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                                v.grad = (v.data - tmp_v).detach()

                        global_optimizer.step()

                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict[d] = save_model_state_dict(model.state_dict())
                        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        elif ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0 and cfg['with_BN'] == 0):
            print('FedAvg with out BN params')
            with torch.no_grad():
                for d in range(self.target_domains):
                    valid_client = [client[i] for i in range(len(client)) 
                                    if (client[i].active and client[i].domain_id == d)]
                    if len(valid_client) > 0:
                        # model = eval('models.{}()'.format(cfg['model_name']))
                        if cfg['world_size']==1:
                            model = eval('models.{}()'.format(cfg['model_name']))
                        elif cfg['world_size']>1:
                            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = eval('models.{}()'.format(cfg['model_name']))
                            model = torch.nn.DataParallel(model,device_ids = [0, 1])
                            model.to(cfg["device"])
                        # model.load_state_dict(self.model_state_dict)
                        model.load_state_dict(self.model_state_dict[d])
                        global_optimizer = make_optimizer(model.parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        weight = torch.ones(len(valid_client))
                        # weight = weight / weight.sum()
                        if cfg['wt_avg']:
                            for i in range(len(valid_client)):
                                weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                        weight = weight / weight.sum()

                        # # Store the averaged batchnorm parameters
                        # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}

                        for k, v in model.named_parameters():
                            
                            isBatchNorm = True if  '.bn' in k else False
                            parameter_type = k.split('.')[-1]
                            # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                            if  not isBatchNorm and ('weight' in parameter_type or 'bias' in parameter_type):
                                print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    if cfg['world_size']==1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                    elif  cfg['world_size']>1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                                v.grad = (v.data - tmp_v).detach()

                        global_optimizer.step()

                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict[d] = save_model_state_dict(model.state_dict())
                        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())

        elif ('fmatch' in cfg['loss_mode'] and cfg['adapt_wt'] == 0):
            with torch.no_grad():
                for d in range(self.target_domains):
                    valid_client = [client[i] for i in range(len(client)) 
                                    if (client[i].active and client[i].domain_id == d)]
                    if len(valid_client) > 0:
                        model = eval('models.{}()'.format(cfg['model_name']))
                        model.load_state_dict(self.model_state_dict[d])
                        global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        weight = torch.ones(len(valid_client))
                        # weight = weight / weight.sum()
                        if cfg['wt_avg']:
                            for i in range(len(valid_client)):
                                weight[i] = valid_client[i].data_len
                        # print(weight.sum())
                        weight = weight / weight.sum()
                        for k, v in model.named_parameters():
                            # print(k)
                            parameter_type = k.split('.')[-1]
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                v.grad = (v.data - tmp_v).detach()
                        global_optimizer.step()
                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        # self.model_state_dict = save_model_state_dict(model.state_dict())
                        self.model_state_dict[d] = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())

        elif cfg['adapt_wt'] == 1:
            print('dynamic aggregation')
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    prev_model = copy.deepcopy(model)

                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    # weight = weight / weight.sum()
                    if cfg['wt_avg']:
                        for i in range(len(valid_client)):
                            weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
                    
                    avg_model = copy.deepcopy(model)
                    

                    ###### compute the adaptive weights ######
                    weight_dict = self.compute_l2d_ratio(prev_model,valid_client,avg_model,'mae')
                    # weight_dict = self.compute_l2d_ratio(prev_model,valid_client,avg_model,'mse')

                    global_optimizer = make_optimizer(prev_model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()

                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight_dict[k][m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        if cfg['avg_cent'] == 1:
            count=0
            for i in range(len(client)):
                # print(i)
                # print(self.avg_cent,client[i].cent)
                if client[i].active == True and client[i].cent is not None:
                    if count==0:
                        self.avg_cent_=client[i].cent
                        count+=1
                    else:
                        self.avg_cent_+=client[i].cent
                elif client[i].active == True  and client[i].cent is None:
                    print('Warning:client centntroid is None')
            if self.avg_cent_ is not None:
                print('averaging centroids')
                # exit()
                self.avg_cent_=self.avg_cent_/len(client)
                if self.avg_cent is None:
                    self.avg_cent = self.avg_cent_
                self.avg_cent = cfg['decay']*self.avg_cent+(1-cfg['decay'])*self.avg_cent_
                 

        # for i in range(len(client)):
        #     client[i].active = False
        
        del model
        # del init_model
        gc.collect()
        torch.cuda.empty_cache()
        return
    def update_global_model(self,client):
        if ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0 and cfg['with_BN'] == 1):
            print('FedAvg with BN params for global model')
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    # model = eval('models.{}()'.format(cfg['model_name']))
                    if cfg['world_size']==1:
                        model = eval('models.{}()'.format(cfg['model_name']))
                    elif cfg['world_size']>1:
                        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = eval('models.{}()'.format(cfg['model_name']))
                        model = torch.nn.DataParallel(model,device_ids = [0, 1])
                        model.to(cfg["device"])
                    # model.load_state_dict(self.model_state_dict)
                    model.load_state_dict(self.global_model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    # weight = weight / weight.sum()
                    if cfg['wt_avg']:
                        for i in range(len(valid_client)):
                            weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                    weight = weight / weight.sum()

                    # # Store the averaged batchnorm parameters
                    # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}
                    # print()
                    for k, v in model.named_parameters():
                        
                        isBatchNorm = True if  '.bn' in k else False
                        parameter_type = k.split('.')[-1]
                        # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                # print(valid_client[m].model_state_dict.keys())
                                if cfg['world_size']==1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                elif  cfg['world_size']>1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                            v.grad = (v.data - tmp_v).detach()

                    global_optimizer.step()

                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.global_model_state_dict = save_model_state_dict(model.state_dict())
        # weight = torch.ones(self.num_clusters)
        # weight = weight / weight.sum()
        # with torch.no_grad():
        #         for d in range(self.num_clusters):
        #             for k, v in model.named_parameters():
                            
        #                     isBatchNorm = True if  '.bn' in k else False
        #                     parameter_type = k.split('.')[-1]
        #                     # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
        #                     if 'weight' in parameter_type or 'bias' in parameter_type:
        #                         tmp_v = v.data.new_zeros(v.size())
        #                         for m in range(len(valid_client)):
        #                             # print(valid_client[m].model_state_dict.keys())
        #                             # print(f'domain:{valid_client[m].domain},id:{valid_client[m].domain_id}')
        #                             if cfg['world_size']==1:
        #                                 tmp_v += weight[m] * valid_client[m].model_state_dict[k]
        #                             elif  cfg['world_size']>1:
        #                                 tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
        #                         v.grad = (v.data - tmp_v).detach()

        #                 global_optimizer.step()

                        # self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        # self.model_state_dict_cluster[d] = save_model_state_dict(model.state_dict())
    def update_cluster(self, client):
        if ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0 and cfg['with_BN'] == 1):
            print('FedAvg with BN params')
            with torch.no_grad():
                for d in range(self.num_clusters):
                    print(f'FedAvg for cluster {d}')
                    valid_client = [client[i] for i in range(len(client)) 
                                     if (client[i].active and client[i].cluster_id==d) ]
                    # for client in valid_client:
                    #     print('domain:',client)
                    if len(valid_client) > 0:
                        # model = eval('models.{}()'.format(cfg['model_name']))
                        if cfg['world_size']==1:
                            model = eval('models.{}()'.format(cfg['model_name']))
                        elif cfg['world_size']>1:
                            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = eval('models.{}()'.format(cfg['model_name']))
                            model = torch.nn.DataParallel(model,device_ids = [0, 1])
                            model.to(cfg["device"])
                        # model.load_state_dict(self.model_state_dict)
                        model.load_state_dict(self.model_state_dict_cluster[d])
                        global_optimizer = make_optimizer(model.parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        weight = torch.ones(len(valid_client))
                        # weight = weight / weight.sum()
                        if cfg['wt_avg']:
                            for i in range(len(valid_client)):
                                weight[i] = valid_client[i].data_len
                        # print(weight.sum())
                        weight = weight / weight.sum()

                        # # Store the averaged batchnorm parameters
                        # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}
                        # print()
                        for m in range(len(valid_client)):
                            # print(valid_client[m].model_state_dict.keys())
                            print(f'domain:{valid_client[m].domain},cluster id:{valid_client[m].cluster_id},client id {valid_client[m].client_id}')
                        for k, v in model.named_parameters():
                            
                            isBatchNorm = True if  '.bn' in k else False
                            parameter_type = k.split('.')[-1]
                            # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    # print(valid_client[m].model_state_dict.keys())
                                    # print(f'domain:{valid_client[m].domain},id:{valid_client[m].domain_id}')
                                    if cfg['world_size']==1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                    elif  cfg['world_size']>1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                                v.grad = (v.data - tmp_v).detach()

                        global_optimizer.step()

                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict_cluster[d] = save_model_state_dict(model.state_dict())
                    
                        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        elif ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0 and cfg['with_BN'] == 0):
            print('FedAvg with out BN params')
            with torch.no_grad():
                for d in range(self.target_domains):
                    valid_client = [client[i] for i in range(len(client)) 
                                    if (client[i].active and client[i].cluster_id == d)]
                    if len(valid_client) > 0:
                        # model = eval('models.{}()'.format(cfg['model_name']))
                        if cfg['world_size']==1:
                            model = eval('models.{}()'.format(cfg['model_name']))
                        elif cfg['world_size']>1:
                            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                            model = eval('models.{}()'.format(cfg['model_name']))
                            model = torch.nn.DataParallel(model,device_ids = [0, 1])
                            model.to(cfg["device"])
                        # model.load_state_dict(self.model_state_dict)
                        model.load_state_dict(self.model_state_dict_cluster[d])
                        global_optimizer = make_optimizer(model.parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        weight = torch.ones(len(valid_client))
                        # weight = weight / weight.sum()
                        if cfg['wt_avg']:
                            for i in range(len(valid_client)):
                                weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                        weight = weight / weight.sum()

                        # # Store the averaged batchnorm parameters
                        # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}

                        for k, v in model.named_parameters():
                            
                            isBatchNorm = True if  '.bn' in k else False
                            parameter_type = k.split('.')[-1]
                            # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                            if  not isBatchNorm and ('weight' in parameter_type or 'bias' in parameter_type):
                                print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    if cfg['world_size']==1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                    elif  cfg['world_size']>1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                                v.grad = (v.data - tmp_v).detach()

                        global_optimizer.step()

                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        self.model_state_dict_cluster[d] = save_model_state_dict(model.state_dict())
                        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())

        elif ('fmatch' in cfg['loss_mode'] and cfg['adapt_wt'] == 0):
            with torch.no_grad():
                for d in range(self.target_domains):
                    valid_client = [client[i] for i in range(len(client)) 
                                    if (client[i].active and client[i].domain_id == d)]
                    if len(valid_client) > 0:
                        model = eval('models.{}()'.format(cfg['model_name']))
                        model.load_state_dict(self.model_state_dict[d])
                        global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
                        global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                        global_optimizer.zero_grad()
                        weight = torch.ones(len(valid_client))
                        # weight = weight / weight.sum()
                        if cfg['wt_avg']:
                            for i in range(len(valid_client)):
                                weight[i] = valid_client[i].data_len
                        # print(weight.sum())
                        weight = weight / weight.sum()
                        for k, v in model.named_parameters():
                            # print(k)
                            parameter_type = k.split('.')[-1]
                            if 'weight' in parameter_type or 'bias' in parameter_type:
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                v.grad = (v.data - tmp_v).detach()
                        global_optimizer.step()
                        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                        # self.model_state_dict = save_model_state_dict(model.state_dict())
                        self.model_state_dict[d] = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())

        elif cfg['adapt_wt'] == 1:
            print('dynamic aggregation')
            with torch.no_grad():
                valid_client = [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    prev_model = copy.deepcopy(model)

                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    # weight = weight / weight.sum()
                    if cfg['wt_avg']:
                        for i in range(len(valid_client)):
                            weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                    weight = weight / weight.sum()
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
                    
                    avg_model = copy.deepcopy(model)
                    

                    ###### compute the adaptive weights ######
                    weight_dict = self.compute_l2d_ratio(prev_model,valid_client,avg_model,'mae')
                    # weight_dict = self.compute_l2d_ratio(prev_model,valid_client,avg_model,'mse')

                    global_optimizer = make_optimizer(prev_model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()

                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                tmp_v += weight_dict[k][m] * valid_client[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        if cfg['avg_cent'] == 1:
            count=0
            for i in range(len(client)):
                # print(i)
                # print(self.avg_cent,client[i].cent)
                if client[i].active == True and client[i].cent is not None:
                    if count==0:
                        self.avg_cent_=client[i].cent
                        count+=1
                    else:
                        self.avg_cent_+=client[i].cent
                elif client[i].active == True  and client[i].cent is None:
                    print('Warning:client centntroid is None')
            if self.avg_cent_ is not None:
                print('averaging centroids')
                # exit()
                self.avg_cent_=self.avg_cent_/len(client)
                if self.avg_cent is None:
                    self.avg_cent = self.avg_cent_
                self.avg_cent = cfg['decay']*self.avg_cent+(1-cfg['decay'])*self.avg_cent_
                 

        # for i in range(len(client)):
        #     client[i].active = False
        
        del model
        del global_optimizer
        # del init_model
        gc.collect()
        torch.cuda.empty_cache()
        return
    def update_BNstats(self, client,stat_type='mean'):
        with torch.no_grad():
            for d in range(self.target_domains):
                valid_client = [client[i] for i in range(len(client)) 
                                if (client[i].active and client[i].domain_id == d)]
                if len(valid_client) > 0:
                    # model = eval('models.{}()'.format(cfg['model_name']))
                    if cfg['world_size']==1:
                        model = eval('models.{}()'.format(cfg['model_name']))
                    elif cfg['world_size']>1:
                        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        model = eval('models.{}()'.format(cfg['model_name']))
                        model = torch.nn.DataParallel(model,device_ids = [0, 1])
                        model.to(cfg["device"])
                    # model.load_state_dict(self.model_state_dict)
                    model.load_state_dict(self.model_state_dict[d])
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client))
                    for i in range(len(valid_client)):
                        weight[i] = valid_client[i].data_len
                    # print(weight.sum())
                    weight = weight / weight.sum()

                    # # Store the averaged batchnorm parameters
                    # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}
                    # print()
                    # for name, module in model.named_modules():
                    #     if isinstance(module, torch.nn.BatchNorm2d) or isinstance(module, torch.nn.BatchNorm1d) :
                    #         print(name)
                    # for k, v in self.model_state_dict.items():             
                    #         isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
                    #         istype  = True if f'.running_var'  in k else False 
                    #         # istype  = True if '.running_mean'  in k or '.running_var' in k else False 
                    #         # parameter_type = k.split('.')[-1]
                    #         # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                    #         if isBatchNorm and istype:
                    #             print(k,v)
                    if stat_type == 'var':
                        print('averaging var')
                        for k, v in model.state_dict().items():
                            # print(k)
                            # exit()
                            isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
                            istype  = True if f'.running_{stat_type}'  in k else False 
                            # parameter_type = k.split('.')[-1]
                            # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                            if isBatchNorm and istype:
                                # print(k)
                                
                                tmp_v = v.data.new_zeros(v.size()).to(cfg['device'])
                                for m in range(len(valid_client)):
                                    # print(valid_client[m].model_state_dict.keys())
                                    k_ = '.'.join(k.split('.')[:-1]) 
                                    # print(k,'.'.join(k.split('.')[:-1]) in valid_client[m].running_var.keys())
                                    # if valid_client[m].client_id == 50:
                                    

                                    if cfg['world_size']==1:
                                        tmp_v += weight[m].to(cfg['device']) * valid_client[m].running_var[k_].squeeze().to(cfg['device'])
                                    elif  cfg['world_size']>1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                                # v.grad = (v.data.cpu() - tmp_v.cpu()).detach()
                                self.model_state_dict[d][k] = tmp_v
                        # for k, v in self.model_state_dict.items():             
                        #     isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
                        #     istype  = True if f'.running_var'  in k else False 
                        #     # istype  = True if '.running_mean'  in k or '.running_var' in k else False 
                        #     # parameter_type = k.split('.')[-1]
                        #     # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                        #     if isBatchNorm and istype:
                        #         print(k,v)
                    else:
                        print('averaging mean')
                        for k, v in model.state_dict().items():
                            
                            # exit()
                            isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
                            istype  = True if f'.running_{stat_type}'  in k else False 
                            # istype  = True if '.running_mean'  in k or '.running_var' in k else False 
                            # parameter_type = k.split('.')[-1]
                            # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                            if isBatchNorm and istype:
                                # print(k)
                                
                                tmp_v = v.data.new_zeros(v.size())
                                for m in range(len(valid_client)):
                                    # print(valid_client[m].model_state_dict.keys())
                                    # print(k,'.'.join(k.split('.')[:-1]) in valid_client[m].running_mean.keys())
                                    if cfg['world_size']==1:
                                        # print(valid_client[m].model_state_dict[k].shape)
                                        # print(weight[m])
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                    elif  cfg['world_size']>1:
                                        tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                                # v.grad = (v.data - tmp_v).detach()
                                # print(k,tmp_v)
                                # print(v)
                                self.model_state_dict[d][k] = tmp_v
                                # print(v)
                    global_optimizer.step()
                    
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    # self.model_state_dict = save_model_state_dict(model.state_dict())
                    # for k, v in self.model_state_dict.items(): 
                    # # for k, v in model.state_dict().items():              
                    #     isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
                    #     # istype  = True if f'.running_mean'  in k else False 
                    #     istype  = True if '.running_mean'  in k or '.running_var' in k else False 
                    #     # parameter_type = k.split('.')[-1]
                    #     # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                    #     if isBatchNorm and istype:
                    #         print(k,v)
                    # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())



    def update_parallel(self, client):
        if 'frgd' not in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server))
                    weight = weight / (2 * (weight.sum() - 1))
                    weight[0] = 1 / 2 if len(valid_client_server) > 1 else 1
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client_server)):
                                tmp_v += weight[m] * valid_client_server[m].model_state_dict[k]
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        elif 'frgd' in cfg['loss_mode']:
            with torch.no_grad():
                valid_client_server = [self] + [client[i] for i in range(len(client)) if client[i].active]
                num_valid_client = len(valid_client_server) - 1
                if len(valid_client_server) > 0:
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model.load_state_dict(self.model_state_dict)
                    global_optimizer = make_optimizer(model.parameters(), 'global')
                    global_optimizer.load_state_dict(self.global_optimizer_state_dict)
                    global_optimizer.zero_grad()
                    weight = torch.ones(len(valid_client_server)) / (num_valid_client // 2 + 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v_1 = v.data.new_zeros(v.size())
                            tmp_v_1 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(1, num_valid_client // 2 + 1):
                                tmp_v_1 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v_2 = v.data.new_zeros(v.size())
                            tmp_v_2 += weight[0] * valid_client_server[0].model_state_dict[k]
                            for m in range(num_valid_client // 2 + 1, len(valid_client_server)):
                                tmp_v_2 += weight[m] * valid_client_server[m].model_state_dict[k]
                            tmp_v = (tmp_v_1 + tmp_v_2) / 2
                            v.grad = (v.data - tmp_v).detach()
                    global_optimizer.step()
                    self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
                    self.model_state_dict = save_model_state_dict(model.state_dict())
        else:
            raise ValueError('Not valid loss mode')
        for i in range(len(client)):
            client[i].active = False
        return
    def deac_client(self,client):
        for i in range(len(client)):
            client[i].active = False
        return
    
    def train(self, dataset, lr, metric, logger):
        if 'fmatch' not in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['world_size']==1:
                model.projection.requires_grad_(False)
            if cfg['world_size']>1:
                model.module.projection.requires_grad_(False)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            data_loader = make_data_loader({'train': dataset}, 'server')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['server']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][1])))
            else:
                num_batches = None
            for epoch in range(1, cfg['server']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    for k, v in model.named_parameters():
                        parameter_type = k.split('.')[-1]
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            v.grad[(v.grad.size(0) // 2):] = 0
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
        
class clientClassifier(nn.Module):
                def __init__(self, embed_dim, class_num, type="linear"):
                    super(clientClassifier, self).__init__()
                    
                    self.type = type
                    if type == 'wn':
                        self.fc = nn.utils.weight_norm(nn.Linear(embed_dim, class_num), name="weight")
                        self.fc.apply(init_weights)
                    else:
                        self.fc = nn.Linear(embed_dim, class_num)
                        self.fc.apply(init_weights)

                def forward(self, x):
                    x = self.fc(x)
                    return x
class clientEmbedding(nn.Module):
    
    def __init__(self, feature_dim, embed_dim=256, type="ori"):
    
        super(clientEmbedding, self).__init__()
        self.bn = nn.BatchNorm1d(embed_dim, affine=True)
        # self.bn = torch.nn.GroupNorm(2, embed_dim, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, embed_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        # print(self.bottleneck,x.shape)
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        return x     
    
class Adapt(nn.Module):
    
    def __init__(self):
        
        super(Adapt, self).__init__()
        ## Activation statistics ##
        self.act_stats = {}
        self.running_mean = {}
        self.running_var = {}
        self.backbone_arch = cfg['backbone_arch'] # resnet101
        self.embed_feat_dim = cfg['embed_feat_dim'] # 256
        self.class_num = cfg['target_size']          # 12 for VisDA

        # if "vit-small" in self.backbone_arch:   
        #     # self.backbone_layer = ResBase(self.backbone_arch) 
        #     self.backbone_layer = timm.create_model("vit_small_patch16_224", pretrained=True)
        #     self.backbone_layer.head = nn.Identity()
        #     # self.backbone_layer = ResNet(Bottleneck, [3,4,6,3], self.class_num)
        # elif "vgg" in self.backbone_arch:
        #     self.backbone_layer = VGGBase(self.backbone_arch)
        # else:
        #     raise ValueError("Unknown Feature Backbone ARCH of {}".format(self.backbone_arch))
        
        self.backbone_feat_dim = 384
        if cfg['vit_bn']:
            self.feat_embed_layer = clientEmbedding(self.backbone_feat_dim, self.embed_feat_dim, type="bn")
            self.class_layer = clientClassifier(self.embed_feat_dim, class_num=self.class_num, type="wn")
        else:
            self.feat_embed_layer = clientEmbedding(self.backbone_feat_dim, self.embed_feat_dim)
            # self.feat_embed_layer = Embedding(self.backbone_feat_dim, self.embed_feat_dim)
            
            # self.class_layer = Classifier(self.embed_feat_dim, class_num=self.class_num, type="wn")
            self.class_layer = clientClassifier(self.embed_feat_dim, class_num=self.class_num)
            # self.class_layer = Classifier(self.backbone_feat_dim, class_num=self.class_num)
    
    def get_emd_feat(self, input_imgs):
        # input_imgs [B, 3, H, W]
        backbone_feat = self.backbone_layer(input_imgs)
        embed_feat = self.feat_embed_layer(backbone_feat)
        return embed_feat
    
    def forward(self,backbone_feat, apply_softmax=True):
        embed_feat = self.feat_embed_layer(backbone_feat)
        cls_out = self.class_layer(embed_feat)
        # cls_out = self.class_layer(backbone_feat)
        if apply_softmax:
            cls_out = torch.softmax(cls_out, dim=1)
        else:
            pass
        if cfg['cls_ps']:
            return embed_feat, cls_out
        return embed_feat, cls_out
    
    
    
class Client:
    def __init__(self, client_id, model, data_split=None):
        self.client_id = client_id
        self.data_split = data_split
        self.data_len = 1
        # print(len(data_split['train']))
        self.model_state_dict = save_model_state_dict(model.state_dict())
        self.global_model_state_dict = save_model_state_dict(model.state_dict())
        if cfg['cls_ps']:
            self.Adapt = Adapt()
            self.adapt_copy = True
            
        
        self.tech_model_state_dict = None
        self.server_state_dict = None
        self.running_mean = None
        self.running_var = None
        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_phi_parameters(), 'local')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
        if cfg['kl_loss'] ==1:
            self.fix_model_state_dict = save_model_state_dict(model.state_dict())
            self.fix_model_state_dict = save_model_state_dict(model.state_dict())
        #     kl_optimizer = make_optimizer(model.parameters(), 'local')
        #     self.kl_optimizer_state_dict = save_optimizer_state_dict(kl_optimizer.state_dict())
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.active = False
        self.supervised= False
        self.domain = None
        self.domain_id = None
        self.cluster_id = None
        self.cent = None
        self.var_cent = None
        self.avg_cent = None
        self.beta = torch.distributions.beta.Beta(torch.tensor([cfg['alpha']]), torch.tensor([cfg['alpha']]))
        self.verbose = cfg['verbose']
        self.EL_loss = elr_loss(500)
        if 'crco' in cfg['loss_mode']:
            self.num_class = self.train_loaders[0].dataset.n_classes
            self.baseline_type = cfg['baseline_type']
            self.lambda_ent = cfg['lambda_ent']
            self.lambda_div = cfg['lambda_div']
            self.lambda_aad = cfg['lambda_aad']
            self.prob_threshold = cfg['prob_threshold']
            self.lambda_nce = cfg['lambda_nce']
            self.lambda_temp = cfg['lambda_temp']
            self.pseudo_update_interval = cfg['pseudo_update_interval']
            self.threshold = cfg['threshold']
            self.num_k = cfg['num_k']
            self.num_m = cfg['num_m']
            self.lambda_near = cfg['lambda_near']
            self.lambda_fixmatch = cfg['lambda_fixmatch']
            self.fixmatch_start = cfg['fixmatch_start']
            self.fixmatch_type = cfg['fixmatch_type']
            self.use_cluster_label_for_fixmatch = cfg['use_cluster_label_for_fixmatch']
            self.lambda_fixmatch_temp = cfg['lambda_fixmatch_temp']
            self.bank_size = cfg['bank_size']
            self.non_diag_alpha = cfg['non_diag_alpha']
            self.class_contrastive_simmat = None
            self.instance_contrastive_simmat = None
            self.add_current_data_for_instance = cfg['add_current_data_for_instance']
            self.use_only_current_batch_for_instance = cfg['use_only_current_batch_for_instance']
            self.max_iters = cfg['max_iters']
            self.beta = cfg['beta']
            #
            # rank, world_size = get_dist_info()
            rank, world_size = 0,1
            self.world_size = world_size
            if self.local_rank == 0:
                log_names = ['info_nce', 'mean_max_prob', 'mask', 'mask_acc', 'cluster_mask_acc']
                if cfg['baseline_type'] == "IM":
                    log_names.extend(['ent', 'div'])
                elif cfg['baseline_type'] == 'AaD':
                    log_names.extend(['aad_pos', 'aad_neg'])
                # loss_metrics = MetricsLogger(log_names=log_names, group_name='loss', log_interval=self.log_interval)
                # self.register_hook(loss_metrics)
            #
            # if fix_classifier:
            #     base_model = self.model_dict['base_model']
            #     for param in base_model.module.online_classifier.parameters():
            #         param.requires_grad = False
            #
            # num_image = len(self.train_loaders[0].dataset)
            # self.weak_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
            # self.weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
            # self.label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))
            # self.pseudo_label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))
            # self.class_prototype_bank = torch.randn(self.num_class, feat_dim).to('cuda:{}'.format(rank))
            # self.strong_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
            # self.strong_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
            # self.aad_weak_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
            # self.aad_weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
            # #
            # self.weak_negative_bank = torch.randn(bank_size, self.num_class).to('cuda:{}'.format(rank))
            # self.weak_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))
            # self.strong_negative_bank = torch.randn(bank_size, self.num_class).to('cuda:{}'.format(rank))
            # self.strong_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))
            # self.ngative_img_ind_bank = torch.zeros((bank_size,), dtype=torch.long).to('cuda:{}'.format(rank))
    def update_bank(self,stu_model,tech_model,test_data_loader):
        # self.set_eval_state()
        # base_model = self.model_dict['base_model']
        stu_model.eval()
        shape = 0
        emd_feat_stack = []
        cls_out_stack = []
        gt_label_stack = []
        id_stack  = []
        with torch.no_grad():
            for i, input in enumerate(test_data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    if input_size<=1:
                        break
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    # optimizer.zero_grad()
                    # all_psd_label = to_device(all_psd_label, cfg['device'])
                    # psd_label = all_psd_label[input['id']]
                    # psd_label_ = all_psd_label[input['id']]
                    # all_psd_label = all_psd_label.cpu()
                    f_weak,weak_logit,f_s1,strong1_logit,f_s2,strong2_logit = stu_model(input)
                    # with torch.no_grad():
                    #     t_f_weak,t_weak_logit,t_f_s1,t_strong1_logit,t_f_s2,t_strong2_logit = tech_model(input)
                    # emd_feat_stack.append(F.normalize(f_weak,dim=-1))
                    # cls_out_stack.append(F.softmax(weak_logit,dim=-1))
                    # gt_label_stack.append(input['target'])
                    # id_stack.append(input['id'])
            
                    self.weak_feat_bank[input['id']] = F.normalize(f_weak,dim=-1)
                    self.weak_score_bank[input['id']] = F.softmax(weak_logit,dim=-1)
                    self.label_bank[input['id']] = input['target']
            #
                    if self.iteration == 0:
                        target_feat = f_weak
                        target_score = F.softmax(weak_logit, dim=-1)
                        self.aad_weak_feat_bank[input['id']] = F.normalize(target_feat, dim=-1)
                        self.aad_weak_score_bank[input['id']] = target_score
        #     for data in self.train_loaders[0]:
        #         img = data[0]['img']
        #         img_ind = data[0]['image_ind']
        #         img_label = data[0]['gt_label']
        #         tmp_res = base_model(img)
        #         feat, logits, target_feat, target_logits = tmp_res
        #         #
        #         tmp_feat = feat
        #         tmp_score = F.softmax(logits, dim=-1)
        #         feat = concat_all_gather(tmp_feat)
        #         score = concat_all_gather(tmp_score)
        #         img_ind = concat_all_gather(img_ind.to('cuda:{}'.format(self.local_rank)))
        #         img_label = concat_all_gather(img_label.to('cuda:{}'.format(self.local_rank)))
        #         self.weak_feat_bank[img_ind] = F.normalize(feat, dim=-1)
        #         self.weak_score_bank[img_ind] = score
        #         self.label_bank[img_ind] = img_label.squeeze(1).to('cuda:{}'.format(self.local_rank))
        #         #
        #         if self.iteration == 0:
        #             target_feat = concat_all_gather(target_feat)
        #             target_score = concat_all_gather(F.softmax(target_logits, dim=-1))
        #             self.aad_weak_feat_bank[img_ind] = F.normalize(target_feat, dim=-1)
        #             self.aad_weak_score_bank[img_ind] = target_score
        #         #
        #         shape += img.shape[0]
        # print('rank {}, shape {}'.format(self.local_rank, shape))
        # self.set_train_state()

    def obtain_all_label(self):
        all_output = self.weak_score_bank
        all_fea = self.weak_feat_bank
        all_label = self.label_bank
        #
        _, predict = torch.max(all_output, 1)

        accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
        print('orig acc is {}'.format(accuracy))
        all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

        all_fea = all_fea.float().cpu().numpy()
        K = all_output.size(1)
        aff = all_output.float().cpu().numpy()
        #
        predict = predict.cpu().numpy()
        for _ in range(2):
            initc = aff.transpose().dot(all_fea)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            cls_count = np.eye(K)[predict].sum(axis=0)
            labelset = np.where(cls_count > self.threshold)
            labelset = labelset[0]

            dd = cdist(all_fea, initc[labelset], 'cosine')
            pred_label = dd.argmin(axis=1)
            predict = labelset[pred_label]

            aff = np.eye(K)[predict]

        acc = np.sum(predict == all_label.cpu().float().numpy()) / len(all_fea)
        # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
        print('acc is {}'.format(acc))
        self.pseudo_label_bank = torch.from_numpy(predict).to("cuda:{}".format(self.local_rank))
        prototype = torch.from_numpy(initc).to("cuda:{}".format(self.local_rank)).to(torch.float32)
        self.class_prototype_bank = F.normalize(prototype)

    def obtain_batch_label(self, feat, score, prototype=None):
        feat_1 = F.normalize(feat.detach())
        if prototype is None:
            prototype = F.normalize(self.class_prototype_bank.detach())
        else:
            prototype = F.normalize(prototype.detach())
        cos_similarity = torch.mm(feat_1, prototype.t())
        pred_label = torch.argmax(cos_similarity, dim=1)
        return pred_label

    def my_sim_compute(self, prob_1, prob_2, sim_mat, expand=True):
        """
        prob_1: B1xC
        prob_2: B2xC
        sim_mat: CxC
        expand: True, computation conducted between every element in prob_2 and prob_1; Fasle, need B1=B2
        """
        b1 = prob_1.shape[0]
        b2 = prob_2.shape[0]
        cls_num = prob_1.shape[1]
        if expand:
            prob_1 = prob_1.unsqueeze(2).unsqueeze(1).expand(-1, b2, -1, -1)  # B1xB2xCx1
            prob_2 = prob_2.unsqueeze(1).unsqueeze(0).expand(b1, -1, -1, -1)  # B1xB2x1xC
            prob_1 = prob_1.reshape(b1 * b2, cls_num, 1)
            prob_2 = prob_2.reshape(b1 * b2, 1, cls_num)
            sim = torch.sum(torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, -1), -1)
            sim = sim.reshape(b1, b2)
        else:
            prob_1 = prob_1.unsqueeze(2)  # BxCx1
            prob_2 = prob_2.unsqueeze(1)  # Bx1xC
            sim = torch.sum(torch.sum(torch.bmm(prob_1, prob_2) * sim_mat, -1), -1)
            sim = sim.reshape(b1, 1)
        return sim

    def my_sim_compute_2(self, query_prob, pos_prob, neg_prob, sim_mat):
        pos_logits = my_sim_compute(query_prob, pos_prob, sim_mat, expand=False)
        neg_logits = my_sim_compute(query_prob, neg_prob, sim_mat, expand=True)
        all_logits = torch.cat((pos_logits, neg_logits), dim=1)
        return all_logits

    def IM_loss(self, score):
        softmax_out = score
        loss_ent = -torch.mean(torch.sum(softmax_out * torch.log(softmax_out + 1e-5), 1)) * 0.5
        tensors_gather = [torch.ones_like(softmax_out) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, softmax_out, async_op=False)
        self_ind = self.local_rank
        msoftmax = 0
        for i in range(len(tensors_gather)):
            if i == self_ind:
                msoftmax += softmax_out.mean(dim=0)
            else:
                msoftmax += tensors_gather[i].mean(dim=0)
        loss_div = torch.sum(msoftmax * torch.log(msoftmax + 1e-5)) * 0.5
        return loss_ent, loss_div

    def AaD_loss(self, score, feat):
        with torch.no_grad():
            normalized_feat = F.normalize(feat, dim=-1)
            # normalized_feat = concat_all_gather(normalized_feat)
            normalized_feat = normalized_feat
            distance = normalized_feat @ self.aad_weak_feat_bank.T
            _, idx_near = torch.topk(distance,
                                     dim=-1,
                                     largest=True,
                                     k=self.num_k + 1)
            idx_near = idx_near[:, 1:]  # batch x K
            score_near = self.aad_weak_score_bank[idx_near]  # batch x K x C
        #
        tensors_gather = [torch.ones_like(score) for _ in range(torch.distributed.get_world_size())]
        torch.distributed.all_gather(tensors_gather, score, async_op=False)
        tensors_gather[self.local_rank] = score
        outputs = torch.cat(tensors_gather, dim=0)
        softmax_out_un = outputs.unsqueeze(1).expand(-1, self.num_k, -1)  # batch x K x C
        #
        mask = torch.ones((normalized_feat.shape[0], normalized_feat.shape[0]))
        diag_num = torch.diag(mask)
        mask_diag = torch.diag_embed(diag_num)
        mask = mask - mask_diag
        #####################
        loss_1 = torch.mean(
            (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))
        copy = outputs.T  # .detach().clone()#
        dot_neg = outputs @ copy  # batch x batch
        dot_neg = (dot_neg * mask.to("cuda:{}".format(self.local_rank))).sum(-1)  # batch
        neg_pred = torch.mean(dot_neg)
        return loss_1, neg_pred

    def baseline_loss(self, score, feat, batch_metrics, logits):
        if self.baseline_type == "IM":
            loss_ent, loss_div = self.IM_loss(score)
            batch_metrics['loss']['ent'] = loss_ent.item()
            batch_metrics['loss']['div'] = loss_div.item()
            return loss_ent * self.lambda_ent + loss_div * self.lambda_div
        elif self.baseline_type == 'AaD':
            loss_aad_pos, loss_aad_neg = self.AaD_loss(score, feat)
            batch_metrics['loss']['aad_pos'] = loss_aad_pos.item()
            batch_metrics['loss']['aad_neg'] = loss_aad_neg.item()
            tmp_lambda = (1 + 10 * self.iteration / self.max_iters) ** (-self.beta)
            return (loss_aad_pos + loss_aad_neg * tmp_lambda) * self.lambda_aad
        else:
            raise RuntimeError('wrong type of baseline')

    def class_contrastive_loss(self, score, label, mask):
        all_other_prob = torch.eye(self.num_class).to("cuda:{}".format(self.local_rank))
        new_logits = self.my_sim_compute(score, all_other_prob, self.class_contrastive_simmat,
                                         expand=True) / self.lambda_fixmatch_temp
        loss_consistency = (F.cross_entropy(new_logits, label, reduction='none') * mask).mean()
        return loss_consistency

    def instance_contrastive_loss(self, query_feat, key_feat, neg_feat, self_ind, neg_ind):
        pos_logits = self.my_sim_compute(query_feat, key_feat, self.instance_contrastive_simmat, expand=False) * 0.5
        neg_logits = self.my_sim_compute(query_feat, neg_feat, self.instance_contrastive_simmat, expand=True) * 0.5
        all_logits = torch.cat((pos_logits, neg_logits), dim=1) / self.lambda_temp
        #
        constrastive_labels = self.get_contrastive_labels(query_feat)
        info_nce_loss = F.cross_entropy(all_logits, constrastive_labels) * 0.5
        return info_nce_loss

    def get_contrastive_labels(self, query_feat):
        current_batch_size = query_feat.shape[0]
        constrastive_labels = torch.zeros((current_batch_size,), dtype=torch.long,
                                          device='cuda:{}'.format(self.local_rank))
        return constrastive_labels

    def obtain_neg_mask(self, self_ind, neg_ind):
        self_size = self_ind.shape[0]
        neg_size = neg_ind.shape[0]
        final_mask = torch.ones((self_size, neg_size)).to('cuda:{}'.format(self.local_rank))
        # 获取self_ind对应的特征
        self_feat = self.aad_weak_feat_bank[self_ind, :]
        # 计算self_feat的近邻
        distance = self_feat @ self.aad_weak_feat_bank.T
        _, near_ind = torch.topk(distance,
                                 dim=-1,
                                 largest=True,
                                 k=self.num_k + 1)
        #
        neg_ind = neg_ind.unsqueeze(0).unsqueeze(2).expand(self_size, -1, 1)
        near_ind = near_ind.unsqueeze(1)
        #
        mask_ind = (neg_ind == near_ind).sum(-1)
        final_mask[mask_ind > 0] = 0
        return final_mask

    def update_negative_bank(self, weak_score, strong_score, img_ind):
        """
        update score bank in trainer
        :param weak_score: weak score output by teacher model
        :param strong_score: strong score output by teacher model
        :img_ind: image index
        :return: None
        """

        def update_bank(new_score, bank, ptr, img_ind=None):
            # all_score = concat_all_gather(new_score)
            all_score = new_score
            batch_size = all_score.shape[0]
            start_point = int(ptr)
            end_point = min(start_point + batch_size, self.bank_size)
            real_size = end_point - start_point
            bank[start_point:end_point, :] = all_score[0:(end_point - start_point), :]
            if img_ind is not None:
                # img_ind = concat_all_gather(img_ind)
                img_ind = img_ind
                self.ngative_img_ind_bank[start_point:end_point] = img_ind[0:(end_point - start_point)]
            if end_point == self.bank_size:
                ptr[0] = 0
            else:
                ptr += batch_size

        update_bank(weak_score, self.weak_negative_bank, self.weak_negative_bank_ptr, img_ind)
        update_bank(strong_score, self.strong_negative_bank, self.strong_negative_bank_ptr)
        # print(self.weak_negative_bank_ptr, self.strong_negative_bank_ptr)

    def update_weak_bank_timely(self, feat, score, ind):
        with torch.no_grad():
            single_output_f_ = F.normalize(feat).detach().clone()
            tmp_softmax_out = score
            tmp_img_ind = ind
            # output_f_ = concat_all_gather(single_output_f_)
            # tmp_softmax_out = concat_all_gather(tmp_softmax_out)
            # tmp_img_ind = concat_all_gather(tmp_img_ind)
            #
            output_f_ = single_output_f_
            tmp_softmax_out = tmp_softmax_out
            tmp_img_ind = tmp_img_ind
            self.aad_weak_feat_bank[tmp_img_ind] = output_f_.detach().clone()
            self.aad_weak_score_bank[tmp_img_ind] = tmp_softmax_out.detach().clone()

    def obtain_sim_mat(self,tech_model, usage):
        # base_model = self.model_dict['base_model']
        # fc_weight = base_model.module.online_classifier.fc.weight_v.detach()
        fc_weight = tech_model.fc.weight_v.detach()
        normalized_fc_weight = F.normalize(fc_weight)
        sim_mat_orig = normalized_fc_weight @ normalized_fc_weight.T
        eye_mat = torch.eye(self.num_class).to("cuda:{}".format(self.local_rank))
        non_eye_mat = 1 - eye_mat
        sim_mat = (eye_mat + non_eye_mat * sim_mat_orig * self.non_diag_alpha).clone()
        return sim_mat
    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask
    
    def make_dataset(self, dataset, metric, logger):
        if 'sup' in cfg['loss_mode'] or 'bmd' in cfg['loss_mode'] or 'ladd' in cfg['loss_mode']:# or 'sim' in cfg['loss_mode']:
            return None,None,dataset
        
        elif 'sim' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                # model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                if cfg['world_size']==1:
                    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
                elif cfg['world_size']>1:
                    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model = torch.nn.DataParallel(model,device_ids = [0, 1])
                    model.to(cfg["device"])
                model.load_state_dict(self.model_state_dict,strict=False)
                model.train(False)
                cfg['pred'] = True
                output = []
                target = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    # input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    # input['supervised_mode'] = self.supervised
                    # input['batch_size'] = cfg['client']['batch_size']['train']
                    output_ = model(input)
                    output_i = output_['target']
                    target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                input_['target'] = torch.cat(target, dim=0)
                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                logger.append(evaluation, 'train', n=len(input_['target']))
                cfg['pred'] = False
                # print(f'{torch.any(mask)}entered')
                if torch.any(mask):
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset,dataset
                else:
                    return None,None,dataset
        elif 'fix' in cfg['loss_mode']:
            with torch.no_grad():
                data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
                # model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
                if cfg['world_size']==1:
                    model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
                elif cfg['world_size']>1:
                    cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    model = eval('models.{}()'.format(cfg['model_name']))
                    model = torch.nn.DataParallel(model,device_ids = [0, 1])
                    model.to(cfg["device"])
                # model.load_state_dict(self.model_state_dict,strict=False)
                model.load_state_dict(self.model_state_dict)
                model.train(False)
                cfg['pred'] = True
                output = []
                target = []
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    # input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    # input['supervised_mode'] = self.supervised
                    # input['batch_size'] = cfg['client']['batch_size']['train']
                    output_ = model(input)
                    output_i = output_['target']
                    target_i = input['target']
                    output.append(output_i.cpu())
                    target.append(target_i.cpu())
                output_, input_ = {}, {}
                output_['target'] = torch.cat(output, dim=0)
                input_['target'] = torch.cat(target, dim=0)
                output_['target'] = F.softmax(output_['target'], dim=-1)
                new_target, mask = self.make_hard_pseudo_label(output_['target'])
                output_['mask'] = mask
                evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                logger.append(evaluation, 'train', n=len(input_['target']))
                cfg['pred'] = False
                # print(f'{torch.any(mask)}entered')
                if torch.any(mask):
                    fix_dataset = copy.deepcopy(dataset)
                    fix_dataset.target = new_target.tolist()
                    mask = mask.tolist()
                    fix_dataset.data = list(compress(fix_dataset.data, mask))
                    fix_dataset.target = list(compress(fix_dataset.target, mask))
                    fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
                    if 'mix' in cfg['loss_mode']:
                        mix_dataset = copy.deepcopy(dataset)
                        mix_dataset.target = new_target.tolist()
                        mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
                    else:
                        mix_dataset = None
                    return fix_dataset, mix_dataset,dataset
                else:
                    return None,None,dataset
        else:
            raise ValueError('Not valid client loss mode')

    def train(self, dataset, lr, metric, logger):
        print("loss mode:",cfg['loss_mode'])
        if cfg['loss_mode'] == 'sup':
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            if cfg['world_size']==1:
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.data_len = len(dataset)
            model.train(True)
            # if cfg['world_size']==1:
            #     model.projection.requires_grad_(False)
            # if cfg['world_size']>1:
            #     model.module.projection.requires_grad_(False)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'sim' in cfg['loss_mode']:
            _,_,dataset = dataset
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            if cfg['world_size']==1:
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.data_len = len(dataset)
            model.train(True)
            if cfg['world_size']==1:
                if self.supervised == False:
                    model.linear.requires_grad_(False)
            elif cfg['world_size']>1:
                model.module.linear.requires_grad_(False)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            print("cfg['client']['num_epochs']:",cfg['client']['num_epochs'])
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    # print(type(input['data']))
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'sim'
                    input = to_device(input, cfg['device'])
                    input['supervised_mode'] = self.supervised
                    input['batch_size'] = cfg['client']['batch_size']['train']
                    optimizer.zero_grad()
                    # print(type(input['data']))
                    output = model(input)
                    # print(output.keys())
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    # evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' not in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, _ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.data_len = len(fix_dataset)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            print("cfg['client']['num_epochs']:",cfg['client']['num_epochs'])
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(fix_data_loader):
                    
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and 'batch' not in cfg[
            'loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            fix_dataset, mix_dataset = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.data_len = len(fix_dataset)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            print("cfg['client']['num_epochs']:",cfg['client']['num_epochs'])
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    input = {'data': fix_input['data'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                             'mix_data': mix_input['data'], 'mix_target': mix_input['target']}
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        elif 'batch' in cfg['loss_mode'] or 'frgd' in cfg['loss_mode'] or 'fmatch' in cfg['loss_mode']:
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.data_len = len(dataset)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            if 'fmatch' in cfg['loss_mode']:
                optimizer = make_optimizer(model.make_phi_parameters(), 'local')
            else:
                optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    with torch.no_grad():
                        model.train(False)
                        input_ = collate(input)
                        input_ = to_device(input_, cfg['device'])
                        output_ = model(input_)
                        output_i = output_['target']
                        output_['target'] = F.softmax(output_i, dim=-1)
                        new_target, mask = self.make_hard_pseudo_label(output_['target'])
                        output_['mask'] = mask
                        evaluation = metric.evaluate(['PAccuracy', 'MAccuracy', 'LabelRatio'], input_, output_)
                        logger.append(evaluation, 'train', n=len(input_['target']))
                    if torch.all(~mask):
                        continue
                    model.train(True)
                    input = {'data': input['data'][mask], 'aug': input['aug'][mask], 'target': new_target[mask]}
                    input = to_device(input, cfg['device'])
                    input_size = input['data'].size(0)
                    input['loss_mode'] = 'fix'
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        else:
            raise ValueError('Not valid client loss mode')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        return
    def get_stats(self,dataset, metric, logger,mode=None):
        # tag = 'global'
        tag = 'client'
        # _,_,dataset = dataset
        with torch.no_grad():
            # test_model = copy.deepcopy(model)
            test_model = eval('models.{}()'.format(cfg['model_name']))
            test_model.to(cfg['device'])
            test_model.load_state_dict(self.model_state_dict)
            # print(dataset)
            dataset, _transform = make_dataset_normal(dataset)
            # print(dataset)
            # exit()
            data_loader = make_data_loader({'train': dataset}, tag, shuffle={'train': False})['train']
            if mode =='mean':
                test_model.apply(lambda m: models.make_batchnorm(m, momentum=0.1, track_running_stats=True))
                test_model.train(True)
                cfg['compute_running_mean'] = True
                test_model.running_mean = {}
                test_model.running_var = {}
            elif mode == 'var':
                # print('getting var stats')
                cfg['compute_running_mean'] = False
                test_model.running_mean = self.running_mean
                test_model.running_var = {}
            
            
            # print(len(dataset))
            # exit()
            self.data_len = len(dataset)
            for i, input in enumerate(data_loader):
                input = collate(input)
                input = to_device(input, cfg['device'])
                input['loss_mode'] = cfg['loss_mode']
                input['supervised_mode'] = False
                input['test'] = True
                # print(input['batch_size'])
                if input['data'].shape[0]==1:
                    break
                input['batch_size'] = cfg['client']['batch_size']['train']
                
                test_model(input)
            if mode == 'mean':
                self.running_mean = test_model.running_mean
            self.running_var = test_model.running_var
            # print(self.running_mean.keys())
            # if mode == 'mean':
            #     for k, v in test_model.state_dict().items():
            #             isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
            #             istype  = True if f'.running_{mode}'  in k else False 
            #             # parameter_type = k.split('.')[-1]
            #             # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
            #             k_ = '.'.join(k.split('.')[:-1]) 
            #             if isBatchNorm and istype and self.client_id ==50:
            #                 print(k,self.running_mean[k_].squeeze()==v,self.client_id,)
            for k,v in self.running_var.items():
                v = v.squeeze()
                # if mode == 'var':
                #     if self.client_id ==50:
                #         print(k,v.shape,v,self.client_id)
            # exit()
            dataset.transform = _transform
        self.model_state_dict = save_model_state_dict(test_model.state_dict())
        return

    def trainntune(self, dataset, lr, metric, logger,epoch,fwd_pass=False,CI_dataset=None,client=None,scheduler = None):
        
        if 'sup' in cfg['loss_mode']:
            # print(cfg['loss_mode'])
            # print('sup' in cfg['loss_mode'])
            _,_,dataset = dataset
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            if cfg['world_size']==1:
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            # model.load_state_dict(self.model_state_dict, strict=False)
            # print(model.layer4[0].n2.running_mean)
            model.load_state_dict(self.model_state_dict)
            # print(model.layer4[0].n2.running_mean)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.data_len = len(dataset)
            model.train(True)
            # if cfg['world_size']==1:
            #     model.projection.requires_grad_(False)
            # if cfg['world_size']>1:
            #     model.module.projection.requires_grad_(False)
                
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            model.running_mean = {}
            model.running_var = {}
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    if input_size == 1:
                        break
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    # if cfg['new_mix']:
                    #     x_mix = input['augw']
                    #     lam = cfg['lam']
                    #     x_flipped = x_mix.flip(0).mul_(1-lam)
                    #     x_mix.mul_(lam).add_(x_flipped)
                    #     input['new_mix'] = x_mix
                    # output = model(input)
                    optimizer.zero_grad()
                    output = model(input)
                    # print(output.keys())
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    
                        
                    
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    # evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
            # print(model.layer4[0].n2.running_mean)
        elif 'sim' in cfg['loss_mode']:
            _,_,dataset = dataset
            data_loader = make_data_loader({'train': dataset}, 'client')['train']
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            if cfg['world_size']==1:
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.data_len = len(dataset)
            model.train(True)
            g_epoch = epoch
            # for v,k in model.named_parameters():
            #     print(f'nmae{v} grad required{k.requires_grad}')
            # if self.supervised == False:
            #     model.linear.requires_grad_(False)
            if cfg['world_size']==1:
                if 'ft' in cfg['loss_mode'] and 'bl' not in cfg['loss_mode']:
                    if epoch <= cfg['switch_epoch'] :
                        model.linear.requires_grad_(False)
                    elif epoch > cfg['switch_epoch']:
                        model.projection.requires_grad_(False)
                elif 'ft' in cfg['loss_mode'] and 'bl'  in cfg['loss_mode']:
                    if epoch > cfg['switch_epoch'] :
                        model.linear.requires_grad_(False)
                    elif epoch <= cfg['switch_epoch']:
                        model.projection.requires_grad_(False)
                elif 'at' in cfg['loss_mode']:
                    if cfg['srange'][0]<=epoch<=cfg['srange'][1] or cfg['srange'][2]<=epoch<=cfg['srange'][3] or cfg['srange'][4]<=epoch<=cfg['srange'][5] or cfg['srange'][6]<=epoch<=cfg['srange'][7]:
                        model.projection.requires_grad_(False)
                    else:
                        model.linear.requires_grad_(False)
            elif cfg['world_size']>1:
                if 'ft' in cfg['loss_mode'] and 'bl' not in cfg['loss_mode']:
                    if epoch <= cfg['switch_epoch'] :
                        model.module.linear.requires_grad_(False)
                    elif epoch > cfg['switch_epoch']:
                        model.module.projection.requires_grad_(False)
                elif 'ft' in cfg['loss_mode'] and 'bl'  in cfg['loss_mode']:
                    if epoch > cfg['switch_epoch'] :
                        model.module.linear.requires_grad_(False)
                    elif epoch <= cfg['switch_epoch']:
                        model.module.projection.requires_grad_(False)
                elif 'at' in cfg['loss_mode']:
                    if cfg['srange'][0]<=epoch<=cfg['srange'][1] or cfg['srange'][2]<=epoch<=cfg['srange'][3] or cfg['srange'][4]<=epoch<=cfg['srange'][5] or cfg['srange'][6]<=epoch<=cfg['srange'][7]:
                        model.module.projection.requires_grad_(False)
                    else:
                        model.module.linear.requires_grad_(False)
            # if 'ft' in cfg['loss_mode']:
            #     if epoch > cfg['switch_epoch'] :
            #         model.linear.requires_grad_(False)
            #     elif epoch <= cfg['switch_epoch']:
            #         model.projection.requires_grad_(False)
            # elif 'at' in cfg['loss_mode']:
            #     if epoch==21 or epoch==42 or epoch==63 or epoch==84 or 100<epoch<=105:
            #         model.projection.requires_grad_(False)
            #     else:
            #         model.linear.requires_grad_(False)
                    
            # for v,k in model.named_parameters():
            #     print(f'nmae{v} grad required{k.requires_grad}')
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    input['supervised_mode'] = self.supervised
                    input['batch_size'] = cfg['client']['batch_size']['train']
                    input['epoch'] = g_epoch
                    optimizer.zero_grad()
                    output = model(input)
                    # print(output.keys())
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    # evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break

        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and CI_dataset is not None:
            fix_dataset, mix_dataset,_ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            # print(mix_data_loader)
            ci_data_loader = make_data_loader({'train':CI_dataset},'client')['train']
            # print(ci_data_loader)
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            # model.load_state_dict(self.model_state_dict, strict=False)
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            self.data_len = len(fix_dataset)
            model.train(True)
            if cfg['world_size']==1:
                model.projection.requires_grad_(False)
            if cfg['world_size']>1:
                model.module.projection.requires_grad_(False)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, (fix_input,mix_input,ci_input) in enumerate(zip(fix_data_loader, mix_data_loader,ci_data_loader)):
                    # input = {'data': fix_input['aug'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                    #          'mix_data': mix_input['aug'], 'mix_target': mix_input['target']}
                    # input = {'data': fix_input['data'], 'augw':fix_input['augw'], 'target': fix_input['target'], 'augs': fix_input['augs'],
                    #          'mix_data': mix_input['augw'], 'mix_target': mix_input['target']}
                    input = {'data': fix_input['augw'], 'target': fix_input['target'], 'aug': fix_input['augs'],
                            'mix_data': mix_input['augs'], 'mix_target': mix_input['target'],'ci_data':ci_input['data'],'ci_target':ci_input['target']}
                    
                    input = collate(input)
                    # print(len(ci_input['data']))
                    input_size = input['data'].size(0)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    output = model(input)
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
        
        elif 'bmd' in cfg['loss_mode']:
            _,_,dataset = dataset
            train_data_loader = make_data_loader({'train': dataset}, 'client')['train']
            test_data_loader = make_data_loader({'train': dataset},'client',batch_size = {'train':50},shuffle={'train':False})['train']
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            self.data_len = len(dataset)
            if cfg['world_size']==1:
                # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
                model = eval('models.{}()'.format(cfg['model_name']))
                if cfg['global_reg'] == 1:
                    global_model = eval('models.{}()'.format(cfg['model_name']))
                # init_model = eval('models.{}()'.format(cfg['model_name']))
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            # model.load_state_dict(self.model_state_dict, strict=False)
            # print(self.model_state_dict.keys())
            # print(self.model_state_dict.backbone_layer.layer4.1.bn3.running_mean.shape)
            # exit()
            model.load_state_dict(self.model_state_dict)
            if cfg['global_reg'] == 1:
                print('global regularization')
                # print(self.global_model_state_dict)
                # exit()
                global_model.load_state_dict(self.global_model_state_dict)
                global_model.to(cfg["device"])
            # init_model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            # self.optimizer_state_dict['param_groups'][0]['lr'] = 0.001

            if ( cfg['model_name'] == 'resnet50' or cfg['model_name'] == 'VITs') and cfg['par'] == 1:
                print('freezing')
                cfg['local']['lr'] = lr
                # cfg['local']['lr'] = 0.001
                param_group_ = []
                for k, v in model.backbone_layer.named_parameters():
                    # print(k)
                    if "bn" in k:
                        # param_group += [{'params': v, 'lr': cfg['local']['lr']*2}]
                        param_group_ += [{'params': v, 'lr': cfg['local']['lr']*0.1}]
                        # v.requires_grad = False
                        # print(k)
                    else:
                        # v.requires_grad = False
                        param_group_ += [{'params': v, 'lr': cfg['local']['lr']*0.1}]

                for k, v in model.feat_embed_layer.named_parameters():
                    # print(k)
                    param_group_ += [{'params': v, 'lr': cfg['local']['lr']}]
                for k, v in model.class_layer.named_parameters():
                    v.requires_grad = False
                    # param_group += [{'params': v, 'lr': cfg['local']['lr']}]
                if cfg['cls_ps']:
                    for k, v in self.Adapt.feat_embed_layer.named_parameters():
                        param_group_ += [{'params': v, 'lr': cfg['local']['lr']}]
                        if self.adapt_copy:
                            print('copying embed layer')
                            for k_m,v_m in model.feat_embed_layer.named_parameters():
                                if k == k_m:
                                    v = v_m
                    for k, v in self.Adapt.class_layer.named_parameters():
                        param_group_ += [{'params': v, 'lr': cfg['local']['lr']}]
                        if self.adapt_copy:
                            print('copying classsifier layer')
                            for k_m,v_m in model.class_layer.named_parameters():
                                if k == k_m:
                                    v = v_m
                            self.adapt_copy = False
                    
                # print(self.Adapt.parameters())
                # exit()
                if cfg['cls_ps']:
                    # params = list(param_group_) + list(self.Adapt.parameters())
                    optimizer_ = make_optimizer(param_group_, 'local')
                else:
                    optimizer_ = make_optimizer(param_group_, 'local')
                optimizer = op_copy(optimizer_)
                del optimizer_
            # # elif cfg['model_name']=='resnet9':
            # #     cfg['local']['lr'] = lr
            # #     # print(model)
            # #     param_group = []
            # #     for k,v in model.named_parameters():
            # #         # print(k)
            # #         if 'n1' in k or 'n2' in k or 'n4' in k or 'bn' in k:
            # #             # print(k)
            # #             param_group += [{'params': v, 'lr': cfg['local']['lr']*0.1}]
            # #         elif 'feat_embed_layer' in k:
            # #             print(k)
            # #             # v.requires_grad = False
            # #             param_group += [{'params': v, 'lr': cfg['local']['lr']}]
            # #         else :
            # #             print('grad false',k)
            # #             v.requires_grad = False
            # #     # for k, v in model.feat_embed_layer.named_parameters():
            # #     #     # print(k)
            # #     #     if 'n1' not in k or 'n2' not in k or 'n4' not in k or 'bn' not in k:
            # #     #         print(k)
            # #     #         param_group += [{'params': v, 'lr': cfg['local']['lr']}]

            # #     # for k, v in model.class_layer.named_parameters():
            # #     #     v.requires_grad = False
            # #     optimizer = make_optimizer(param_group, 'local')
            # #     optimizer = op_copy(optimizer)
            #     # exit()
            else:
                # print('not freezing')
                optimizer = make_optimizer(model.parameters(), 'local')
                optimizer.load_state_dict(self.optimizer_state_dict)
            # print(model)
            # optimizer = make_optimizer(model.parameters(), 'local')
            # optimizer.load_state_dict(self.optimizer_state_dict)
            # model.to_device(cfg['device'])
            model.to(cfg["device"])
            if cfg['cls_ps']:
                # params = list(model.parameters()) + list(self.classifier.parameters())
                self.Adapt.to(cfg['device'])
            # init_model.to(cfg["device"])
            model.train(True)
            # print(scheduler)
            # exit()
            # if cfg['world_size']==1:
            #     model.projection.requires_grad_(False)
            # if cfg['world_size']>1:
            #     model.module.projection.requires_grad_(False)
            num_local_epochs = cfg['client']['num_epochs']    
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(train_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            # num_batches =None
            # for epoch in range(1, cfg['client']['num_epochs']+1 ):
            print(self.client_id,self.domain)
            if self.domain == 'clipart' and cfg['clip_t']:
                print('reducing threshold for clip to 60\% ,80 global model  ')
                thres = cfg['threshold']
                for k, v in model.named_parameters():
                    for k_g,v_g in global_model.named_parameters():
                        if k_g == k:
                            v = 0.2*v+0.8*v_g
            else:
                thres = cfg['threshold']
            # if self.domain == 'webcam':
            #     num_local_epochs = cfg['tde']
            # else:
            #     num_local_epochs = cfg['client']['num_epochs']
            if fwd_pass == True and cfg['cluster']:
                 num_local_epochs = 10                     #re 10
            #print(num_local_epochs)
            
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0)
            # print(len(train_data_loader))
            # exit()
            id = self.client_id
            domain = self.cluster_id
            print(num_local_epochs)
            # exit()
            for epoch in range(0, num_local_epochs ):
                print('epoch:',epoch)
                output = {}
                # lr = optimizer.param_groups[0]['lr']
                # print(self.cent)
                # print(self.cent,self.avg_cent)
                # loss,cent = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent)
                # print(len(client))
                # exit()
                # fwd_pass = 0
                if cfg['run_shot']:
                    if cfg['global_reg'] and cfg['cls_ps']==False:
                        # print('global regularization')
                        # print(global_model)
                        # exit()
                        loss,cent = shot_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent,id,domain,fwd_pass=fwd_pass,scheduler=scheduler,client=client,global_model=global_model,thres=thres)
                    
                    elif cfg['global_reg'] and cfg['cls_ps']==True:
                        loss,cent = shot_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent,id,domain,fwd_pass=fwd_pass,scheduler=scheduler,client=client,global_model=global_model,thres=thres,cls_ps=self.Adapt)
                    
                    else:
                        loss,cent = shot_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent,id,domain,fwd_pass=fwd_pass,scheduler=scheduler,client=client,thres=thres)
                else:
                    # loss,cent = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent)
                    # loss,cent,var_cent = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent,fwd_pass,scheduler)
                    loss = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent,fwd_pass,scheduler)
                    # self.var_cent = var_cent.cpu()
                    # self.cent = cent.cpu()
                # self.var_cent = var_cent.cpu()
                # print(self.cent)
                # del cent
                # del var_cent
                output['loss'] = loss
        
        
        elif 'crco' in cfg['loss_mode']:
            _,_,dataset = dataset
            train_data_loader = make_data_loader({'train': dataset}, 'client')['train']
            test_data_loader = make_data_loader({'train': dataset},'client',batch_size = {'train':50},shuffle={'train':False})['train']
            
            self.data_len = len(dataset)
            num_image = self.data_len
            rank = 0
            feat_dim = cfg['embed_feat_dim']
            self.weak_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
            self.weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
            self.label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))
            self.pseudo_label_bank = torch.zeros((num_image,), dtype=torch.long).to('cuda:{}'.format(rank))
            self.class_prototype_bank = torch.randn(self.num_class, feat_dim).to('cuda:{}'.format(rank))
            self.strong_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
            self.strong_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
            self.aad_weak_feat_bank = torch.randn(num_image, feat_dim).to('cuda:{}'.format(rank))
            self.aad_weak_score_bank = torch.randn(num_image, self.num_class).to('cuda:{}'.format(rank))
            #
            self.weak_negative_bank = torch.randn(cfg['bank_size'], self.num_class).to('cuda:{}'.format(rank))
            self.weak_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))
            self.strong_negative_bank = torch.randn(cfg['bank_size'], self.num_class).to('cuda:{}'.format(rank))
            self.strong_negative_bank_ptr = torch.zeros(1, dtype=torch.long).to('cuda:{}'.format(rank))
            self.ngative_img_ind_bank = torch.zeros((cfg['bank_size'],), dtype=torch.long).to('cuda:{}'.format(rank))
            if cfg['world_size']==1:
                
                stu_model = eval('models.{}()'.format(cfg['model_name']))
                tech_model = eval('models.{}()'.format(cfg['model_name']))
                
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            
            stu_model.load_state_dict(self.model_state_dict)
            tech_model.load_state_dict(self.model_state_dict)
    
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            # self.optimizer_state_dict['param_groups'][0]['lr'] = 0.001

            if cfg['model_name'] == 'resnet50_crco' and cfg['par'] == 1:
                print('freezing')
                cfg['local']['lr'] = lr
                # cfg['local']['lr'] = 0.001
                param_group_ = []
                for k, v in stu_model.backbone_layer.named_parameters():
                    # print(k)
                    if "bn" in k:
                        # param_group += [{'params': v, 'lr': cfg['local']['lr']*2}]
                        param_group_ += [{'params': v, 'lr': cfg['local']['lr']*0.1}]
                        # v.requires_grad = False
                        # print(k)
                    else:
                        v.requires_grad = False

                # for k, v in model.feat_embed_layer.named_parameters():
                #     # print(k)
                #     param_group_ += [{'params': v, 'lr': cfg['local']['lr']}]
                for k, v in stu_model.class_layer.named_parameters():
                    v.requires_grad = False
                    # param_group += [{'params': v, 'lr': cfg['local']['lr']}]

                optimizer_ = make_optimizer(param_group_, 'local')
                optimizer = op_copy(optimizer_)
                del optimizer_
            else:
                optimizer = make_optimizer(model.parameters(), 'local')
                optimizer.load_state_dict(self.optimizer_state_dict)
            
            stu_model.train(True)
            # print(scheduler)
            # exit()
            
                
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(train_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
        
            print(self.client_id,self.domain)
            # if self.domain == 'webcam':
            #     num_local_epochs = cfg['tde']
            # else:
            #     num_local_epochs = cfg['client']['num_epochs']
            if fwd_pass == True and cfg['cluster']:
                 num_local_epochs = 10                     #re 10
            #print(num_local_epochs)
            num_local_epochs = cfg['client']['num_epochs']
            self.update_bank(stu_model,tech_model,test_data_loader)
            self.obtain_all_label()
            if self.iteration == 0:
                self.class_contrastive_simmat = self.obtain_sim_mat(tech_model,usage='class_contrastive')
                self.instance_contrastive_simmat = self.obtain_sim_mat(tech_model,usage='instance_contrastive')
            for epoch in range(0, num_local_epochs ):
                output = {}
                loss_stack = []
                stu_model.to(cfg["device"])
                tech_model.to(cfg["device"])
                with torch.no_grad():
                    stu_model.eval()
                    tech_model.eval()
                    #want to update banks and get labels as per thr og code
                    all_psd_label ,all_emd_feat,all_cls_out= init_multi_cent_psd_label_crco(stu_model,tech_model,test_data_loader)
                stu_model.train()
                epoch_idx=epoch
                grad_bank = {}
                avg_counter = 0 
                

                for i, input in enumerate(train_data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    if input_size<=1:
                        break
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    all_psd_label = to_device(all_psd_label, cfg['device'])
                    psd_label = all_psd_label[input['id']]
                    psd_label_ = all_psd_label[input['id']]
                    all_psd_label = all_psd_label.cpu()
                    f_weak,weak_logit,f_s1,strong1_logit,f_s2,strong2_logit = stu_model(input)
                    with torch.no_grad():
                        t_f_weak,t_weak_logit,t_f_s1,t_strong1_logit,t_f_s2,t_strong2_logit = tech_model(input)
                    
                    weak_prob,s1_prob,s2_prob = F.softmax(weak_logit,dim=-1),F.softmax(strong1_logit,dim=-1),F.softmax(strong2_logit,dim=-1)
                    t_weak_prob,t_s1_prob,t_s2_prob = F.softmax(t_weak_logit,dim=-1),F.softmax(t_strong1_logit,dim=-1),F.softmaxt_(strong2_logit,dim=-1)
                    
                    input = to_device(input, 'cpu')
                    #
                    # timely updated weak bank
                    self.update_weak_bank_timely(f_weak, weak_prob, input['id'])
                    # baseline
                    loss = self.baseline_loss(t_weak_prob, f_weak,t_weak_logit)
                    # fixmatch
                    tgt_unlabeled_label = input['target']
                    pseudo_label = torch.softmax(weak_logit.detach(), dim=-1)
                    max_probs, tgt_u = torch.max(pseudo_label, dim=-1)
                    mask = max_probs.ge(cfg['prob_threshold']).float().detach()
                    pred_right = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)
                    if self.use_cluster_label_for_fixmatch:
                        tgt_u = self.obtain_batch_label(t_f_weak, None)
                        # tgt_u = self.obtain_batch_label(target_weak_feat, None)
                        cluster_acc = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)
                    else:
                        cluster_acc = torch.tensor(0)
                    mask_val = torch.sum(mask).item() / mask.shape[0]
                    self.high_ratio = mask_val
                    ###########
                    if self.fixmatch_type == 'orig':
                        strong_aug_pred = t_strong1_logit
                        loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
                        loss += loss_consistency * self.lambda_fixmatch * 0.5
                        strong_aug_pred = t_strong2_logit
                        loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
                        loss += loss_consistency * self.lambda_fixmatch * 0.5
                    #
                    elif self.fixmatch_type == 'class_relation':
                        #
                        loss_1 = self.class_contrastive_loss(t_strong1_logit, tgt_u, mask)
                        loss_2 = self.class_contrastive_loss(t_strong1_logit, tgt_u, mask)
                        loss += (loss_1 + loss_2) * self.lambda_fixmatch * 0.5
                    else:
                        raise RuntimeError('wrong fixmatch type')
                    # #
                    # # constrastive loss
                    tgt_img_ind = input['id']
                    # all_k_strong = target_strong_prob
                    all_k_strong = torch.cat((s1_prob, s2_prob), dim=0)
                    all_k_weak = weak_prob
                    # weak_feat_for_backbone = online_weak_prob
                    weak_feat_for_backbone = t_weak_prob
                    k_weak_for_backbone = all_k_weak
                    # k_strong_for_backbone = all_k_strong[0:tgt_unlabeled_size]
                    k_strong_for_backbone = s1_prob
                    # strong_feat_for_backbone = online_strong_prob[0:tgt_unlabeled_size]
                    strong_feat_for_backbone = t_s1_prob
                    # k_strong_2 = all_k_strong[tgt_unlabeled_size:]
                    k_strong_2 = s2_prob
                    # feat_strong_2 = online_strong_prob[tgt_unlabeled_size:]
                    feat_strong_2 = t_s2_prob
                    if self.use_only_current_batch_for_instance:
                        # tmp_weak_negative_bank = online_weak_prob
                        tmp_weak_negative_bank = t_weak_prob
                        tmp_strong_negative_bank = strong_feat_for_backbone
                        # neg_ind = tgt_img_ind 
                        neg_ind = tgt_img_ind    #input['ids']?
                        self.num_k = 1
                    else:
                        if self.add_current_data_for_instance:
                            # tmp_weak_negative_bank = torch.cat((self.weak_negative_bank, online_weak_prob), dim=0)
                            tmp_weak_negative_bank = torch.cat((self.weak_negative_bank, t_weak_prob), dim=0)
                            tmp_strong_negative_bank = torch.cat((self.strong_negative_bank, strong_feat_for_backbone), dim=0)
                            # neg_ind = torch.cat((self.ngative_img_ind_bank, tgt_img_ind))
                            neg_ind = torch.cat((self.ngative_img_ind_bank, tgt_img_ind))
                        else:
                            tmp_weak_negative_bank = self.weak_negative_bank
                            tmp_strong_negative_bank = self.strong_negative_bank
                            neg_ind = self.ngative_img_ind_bank
                    #
                    info_nce_loss_1 = self.instance_contrastive_loss(strong_feat_for_backbone, k_weak_for_backbone,
                                                                    tmp_weak_negative_bank,
                                                                    self_ind=tgt_img_ind, neg_ind=neg_ind)
                    info_nce_loss_3 = self.instance_contrastive_loss(strong_feat_for_backbone, k_strong_2,
                                                                    tmp_strong_negative_bank,
                                                                    self_ind=tgt_img_ind, neg_ind=neg_ind)
                    info_nce_loss_2 = self.instance_contrastive_loss(weak_feat_for_backbone, k_strong_for_backbone,
                                                                    tmp_strong_negative_bank,
                                                                    self_ind=tgt_img_ind, neg_ind=neg_ind)
                    info_nce_loss = (info_nce_loss_1 + info_nce_loss_2 + info_nce_loss_3) / 3.0
                    #
                    loss += info_nce_loss * self.lambda_nce
                    
                    optimizer.zero_grad()
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    
                    optimizer.step()
                    self.update_negative_bank(weak_prob, s1_prob, tgt_img_ind)
                    if scheduler is not None:
                        # print('scheduler step')
                        scheduler.step()
                    
        elif 'ladd' in cfg['loss_mode']:
            _,_,dataset = dataset
            train_data_loader = make_data_loader({'train': dataset}, 'client')['train']
            test_data_loader = make_data_loader({'train': dataset},'client',batch_size = {'train':50},shuffle={'train':False})['train']
            
            self.data_len = len(dataset)
            num_image = self.data_len
            rank = 0
            feat_dim = cfg['embed_feat_dim']
            
            if cfg['world_size']==1:
                
                stu_model = eval('models.{}()'.format(cfg['model_name']))
                tech_model = eval('models.{}()'.format(cfg['model_name']))
                global_model = eval('models.{}()'.format(cfg['model_name']))
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            
            stu_model.load_state_dict(self.model_state_dict)
            global_model.load_state_dict(self.global_model_state_dict)
            stu_model.to(cfg["device"])
            global_model.to(cfg["device"])
            tech_model.to(cfg["device"])
            if self.tech_model_state_dict is not None:
                tech_model.load_state_dict(self.tech_model_state_dict)
            else:
                tech_model.load_state_dict(self.model_state_dict)
                self.tech_model_state_dict = self.model_state_dict
    
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            # self.optimizer_state_dict['param_groups'][0]['lr'] = 0.001
            for k, v in stu_model.class_layer.named_parameters():
                for k_g,v_g in global_model.class_layer.named_parameters():
                    if k == k_g:
                        v = v_g
            # if cfg['model_name'] == 'resnet50' and cfg['par'] == 1:
            #     print('freezing')
            #     cfg['local']['lr'] = lr
            #     # cfg['local']['lr'] = 0.001
            #     param_group_ = []
            #     for k, v in stu_model.backbone_layer.named_parameters():
            #         # print(k)
            #         if "bn" in k:
            #             # param_group += [{'params': v, 'lr': cfg['local']['lr']*2}]
            #             param_group_ += [{'params': v, 'lr': cfg['local']['lr']*0.1}]
            #             # v.requires_grad = False
            #             # print(k)
            #         else:
            #             v.requires_grad = False

            #     # for k, v in model.feat_embed_layer.named_parameters():
            #     #     # print(k)
            #     #     param_group_ += [{'params': v, 'lr': cfg['local']['lr']}]
            #     for k, v in stu_model.class_layer.named_parameters():
            #         v.requires_grad = False
            #         # param_group += [{'params': v, 'lr': cfg['local']['lr']}]

            #     optimizer_ = make_optimizer(param_group_, 'local')
            #     optimizer = op_copy(optimizer_)
            #     del optimizer_
            # else:
            optimizer = make_optimizer(stu_model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            
            stu_model.train(True)
            # print(scheduler)
            # exit()
            print('freezing teacher')
    
            for k, v in tech_model.backbone_layer.named_parameters():    
                v.requires_grad = False
            for k, v in stu_model.class_layer.named_parameters():
                v.requires_grad = False

                
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(train_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
        
            print(self.client_id,self.domain)
            # if self.domain == 'webcam':
            #     num_local_epochs = cfg['tde']
            # else:
            #     num_local_epochs = cfg['client']['num_epochs']
            num_local_epochs = cfg['client']['num_epochs']
            if fwd_pass == True and cfg['cluster']:
                 num_local_epochs = 10                     #re 10
            #print(num_local_epochs)
            
            
            loss_stack = []
            num_classes = cfg['target_size']
            avg_label_feat = torch.zeros(num_classes,256)
            stu_model.to(cfg["device"])
            tech_model.to(cfg["device"])
            # with torch.no_grad():
            #     tech_model.eval()
            #     pred_label = init_psd_label_shot_icml(tech_model,test_data_loader)
            #     #print("len dataloader:",len(dataloader))
            #     print("len pred_label:",len(pred_label))
            # print('entered ladd')
            # exit()
            stu_model.train()
            epoch_idx=epoch
            # for i, input in enumerate(test_data_loader):
            for i, input in enumerate(train_data_loader):
                input = collate(input)
                input_size = input['data'].size(0)
                if input_size<=1:
                    break
                input['loss_mode'] = cfg['loss_mode']
                input = to_device(input, cfg['device'])
                optimizer.zero_grad()
                # pred_label = to_device(pred_label,cfg['device'])
                # psd_label = pred_label[input['id']]
                # psd_label_ = pred_label[input['id']]

                
                embed_feat, pred_cls = stu_model(input)
                
                with torch.no_grad():
                    tech_model.eval()
                    ft,yt = tech_model(input)
                    pred_cls_t = torch.softmax(yt, dim=1)
                    max_p, hard_pseudo_label = torch.max(pred_cls_t, dim=-1)
                    mask = max_p.ge(cfg['threshold'])
                    embed_feat_masked = embed_feat[mask]
                    
                    # psd_label = psd_label[mask]
                    psd_label = hard_pseudo_label[mask]
                    spred = yt
                    
                y_pred = pred_cls
                
                # label_weights = -torch.log(label_probs[batch_y.reshape(-1).long()])
                
                s_pred_temp = F.softmax((spred  - torch.max(spred))/cfg['temp'], dim=1)
                y_pred_temp = F.softmax((y_pred - torch.max(y_pred))/cfg['temp'], dim=1)
                s_pred_notemp = F.softmax(spred, dim=1)
                #l_pred_notemp = F.softmax(lpred, dim=1)
                # KL_loss = torch.sum(s_pred_temp * torch.log(s_pred_temp/y_pred_temp),axis = 1)   
                # KL_loss = torch.mean(torch.sum(s_pred_temp * torch.log(s_pred_temp/y_pred_temp),axis =1))    
                # KL_loss = # Compute KL divergence loss
                KL_loss = F.kl_div(torch.log(s_pred_temp),y_pred_temp, reduction='batchmean')
                # KL_loss = F.kl_div(torch.log(y_pred_temp),s_pred_temp, reduction='batchmean')
                    
                    
                pred_cls = pred_cls[mask]
                # print(psd_label,psd_label.shape,cfg['threshold'])
                # exit()
                if pred_cls.shape != psd_label.shape:
                    # psd_label is not one-hot like.
                    psd_label = to_device(psd_label,cfg['device'])
                    psd_label = torch.zeros_like(pred_cls).scatter(1, psd_label.unsqueeze(1), 1)
                
                # print(psd_label,psd_label.shape)
                # exit()
                # psd_loss = - torch.sum(torch.log(pred_cls) * psd_label, dim=1).mean()
                psd_loss = loss_fn(pred_cls, psd_label)
                # print(psd_loss)
                # exit()
                # client_labels = torch.from_numpy(np.squeeze(input['target']))
                
                # label_count = to_device(torch.bincount(client_labels),cfg['device'])
                # label_probs = (1.0*label_count)/torch.sum(label_count)
                # spred = yt
                # y_pred = pred_cls
                
                # # label_weights = -torch.log(label_probs[batch_y.reshape(-1).long()])
                
                # s_pred_temp = F.softmax((spred  - torch.max(spred))/cfg['temp'], dim=1)
                # y_pred_temp = F.softmax((y_pred - torch.max(y_pred))/cfg['temp'], dim=1)
                # s_pred_notemp = F.softmax(spred, dim=1)
                # #l_pred_notemp = F.softmax(lpred, dim=1)
                # KL_loss = torch.sum(s_pred_temp * torch.log(s_pred_temp/y_pred_temp),axis = 1)
                # server_entropy = -1.0*torch.sum(s_pred_temp * torch.log(s_pred_temp),axis = 1)
                
                # label_imbalance_loss = torch.exp(args.dist_beta * label_weights)
                # distill_weights = (torch.exp(-server_entropy)** args.dist_beta_kl) * label_imbalance_loss
                # distill_weights = distill_weights/torch.sum(distill_weights)
                # distill_loss = torch.sum(distill_weights*KL_loss)
                # loss += args.lamda*distill_loss
                #if epoch_idx >= 1.0:
                    # loss = 2.0 * psd_loss
                #    loss = ent_loss + 1.0 * psd_loss
                #else:
                # print(KL_loss)
                # exit()
                loss = psd_loss + cfg['kl_weight']*KL_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(stu_model.parameters(), 1)
                optimizer.step()
                if scheduler is not None:
                    # print('scheduler step')
                    scheduler.step()
                    # print('lr at client
            with torch.no_grad():
                for k,v in tech_model.named_parameters():
                    for ks,vs in stu_model.named_parameters():
                        if k==ks:
                            v = 0.5*v+0.5*vs
                        
                
                
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and CI_dataset is  None:
            fix_dataset, mix_dataset,_ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            # model.load_state_dict(self.model_state_dict, strict=False)
            model.load_state_dict(self.model_state_dict, strict=False)
            self.data_len = len(fix_dataset)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
            model.train(True)
            if cfg['kl_loss'] == 1:
                fix_model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
                # fix_model.load_state_dict(self.fix_model_state_dict, strict=False)
                fix_model.load_state_dict(self.model_state_dict, strict=False)
                # kl_optimizer = make_optimizer(fix_model.parameters(), 'local')
                # kl_optimizer.load_state_dict(self.kl_optimizer_state_dict)
            # if cfg['world_size']==1:
            #     model.projection.requires_grad_(False)
            # if cfg['world_size']>1:
            #     model.module.projection.requires_grad_(False)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            for epoch in range(1, cfg['client']['num_epochs'] + 1):
                for i, (fix_input, mix_input) in enumerate(zip(fix_data_loader, mix_data_loader)):
                    # print(fix_input['target'])
                    # input = {'data': fix_input['aug'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                    #          'mix_data': mix_input['aug'], 'mix_target': mix_input['target']}
                    # input = {'data': fix_input['data'], 'augw':fix_input['augw'], 'target': fix_input['target'], 'augs': fix_input['augs'],
                    #          'mix_data': mix_input['augw'], 'mix_target': mix_input['target']}
                    output = {}
                    if cfg['DA']==1:
                        input = {'data': fix_input['augw'], 'target': fix_input['target'], 'aug': fix_input['augs'],
                             'mix_data': mix_input['augs'], 'mix_target': mix_input['target'],'id':fix_input['id']}
                    else:
                        input = {'data': fix_input['augw'], 'target': fix_input['target'], 'aug': fix_input['augs'],
                             'mix_data': mix_input['augs'], 'mix_target': mix_input['target'],'id':fix_input['id']}
                    input = collate(input)
                    input_size = input['data'].size(0)
                    if input_size == 1:
                        break
                    # print(input['mix_data'].shape)
                    input['lam'] = self.beta.sample()[0]
                    input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                    input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                    input['loss_mode'] = cfg['loss_mode']
                    input = to_device(input, cfg['device'])
                    optimizer.zero_grad()
                    # output = model(input)
                    aug_output,mix_output,augw_output = model(input)
                    # print(aug_output,mix_output,augw_output)
                    output['target'] = augw_output
                    # print(aug_output.get_device(),mix_output.get_device(),input['id'],input['target'].detach().get_device())
                    # output['loss']  = self.EL_loss(input['id'].detach(),aug_output, input['target'].detach())
                    # output['loss'] += input['lam'] * self.EL_loss(input['id'].detach(),mix_output, input['mix_target'][:, 0].detach()) + (
                    #         1 - input['lam']) * self.EL_loss(input['id'].detach(),mix_output, input['mix_target'][:, 1].detach())
                    output['loss'] = loss_fn(aug_output, input['target'].detach())
                    output['loss'] += input['lam'] * loss_fn(mix_output, input['mix_target'][:, 0].detach()) + (
                        1 - input['lam']) * loss_fn(mix_output, input['mix_target'][:, 1].detach())
                    output['loss'].backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    optimizer.step()
                    evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    logger.append(evaluation, 'train', n=input_size)
                    if num_batches is not None and i == num_batches - 1:
                        break
                if cfg['kl_loss']==1:
                    tau = 4
                    for epoch in range(1):
                        for i, fix_input in enumerate(fix_data_loader):
                            # input = {'data': fix_input['aug'], 'target': fix_input['target'], 'aug': fix_input['aug'],
                            #          'mix_data': mix_input['aug'], 'mix_target': mix_input['target']}
                            # input = {'data': fix_input['data'], 'augw':fix_input['augw'], 'target': fix_input['target'], 'augs': fix_input['augs'],
                            #          'mix_data': mix_input['augw'], 'mix_target': mix_input['target']}
                            input = {'data': fix_input['augw'], 'target': fix_input['target'], 'aug': fix_input['augs'],
                                        }
                            input = collate(input)
                            input_size = input['data'].size(0)
                            input['lam'] = self.beta.sample()[0]
                            # input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
                            # input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
                            input['loss_mode'] = cfg['loss_mode']
                            
                            input = to_device(input, cfg['device'])
                            input['kl_loss'] = cfg['kl_loss']
                            optimizer.zero_grad()
                            output = F.softmax(tau*model(input),dim=1)
                            output_fix = F.softmax(tau*fix_model(input),dim=1)
                            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
                            output_loss = 1*kl_loss(output,output_fix)
                            output_loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                            optimizer.step()
                            # evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                            # logger.append(evaluation, 'train', n=input_size)
                            if num_batches is not None and i == num_batches - 1:
                                break
        else:
            raise ValueError('Not valid client loss mode')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.model_state_dict = save_model_state_dict(model.state_dict())
        # if 'ladd' == cfg['loss_mode']:
        #     self.model_state_dict = save_model_state_dict(stu_model.state_dict())
        #     self.tech_model_state_dict = save_model_state_dict(tech_model.state_dict())
        # else:
        #     self.model_state_dict = save_model_state_dict(model.state_dict())
        del optimizer
        # del optimizer_
        del model
        # del init_model
        gc.collect()
        torch.cuda.empty_cache()
        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        return

# Function to compute the correlation matrix K
def compute_correlation_matrix(W):
    # Normalize W to have zero mean
    W_centered = W - W.mean(dim=0)
    # Compute the covariance matrix
    covariance_matrix = torch.mm(W_centered.T, W_centered) / (W_centered.size(0) - 1)
    # Compute the correlation matrix
    std_dev = torch.sqrt(torch.diag(covariance_matrix))
    correlation_matrix = covariance_matrix / torch.outer(std_dev, std_dev)
    return correlation_matrix

def shot_train(model,train_data_loader,test_data_loader,optimizer,epoch,cent,avg_cent,id,domain,fwd_pass=False,scheduler = None,client = None,global_model = None,thres=None,cls_ps=None):
    loss_stack = []
    num_classes = cfg['target_size']
    avg_label_feat = torch.zeros(num_classes,256)
    model.to(cfg["device"])
    # print(fwd_pass)
    # exit()
    # print(global_model)
    # exit()
    with torch.no_grad():
        model.eval()
        pred_label = init_psd_label_shot_icml(model,test_data_loader)
        # if fwd_pass:
        #     pred_label = init_psd_label_shot_icml(model,test_data_loader)
        # else:
        #     # print('entered')
        #     # exit()
        #     # pred_label = init_psd_label_shot_icml(model,test_data_loader)
        #     pred_label = init_psd_label_shot_icml_up(model,test_data_loader,client,id,domain)
        # #print("len dataloader:",len(dataloader))
        # print("len pred_label:",len(pred_label))
    
    model.train()
    epoch_idx=epoch
    # print('starting runs okkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkkk')
    # for i, input in enumerate(test_data_loader):
    for i, input in enumerate(train_data_loader):
        # print(i)
        input = collate(input)
        input_size = input['data'].size(0)
        if input_size<=1:
            break
        input['loss_mode'] = cfg['loss_mode']
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        pred_label = to_device(pred_label,cfg['device'])
        psd_label = pred_label[input['id']]
        psd_label_ = pred_label[input['id']]

        if cfg['add_fix']==0:
            if cfg['cls_ps']:
                cls_ps.train()
                p,embed_feat, pred_cls = model(input)
                adapt_feat,ps_cls = cls_ps(p)
            else:
                embed_feat, pred_cls = model(input)
        elif cfg['add_fix']==1 and cfg['logit_div'] ==0:
            if cfg['cls_ps']:
                p,embed_feat, pred_cls,x_s = model(input)
                adapt_feat,ps_cls = cls_ps(p)
                # print(ps_cls.shape,pred_cls.shape)
                # exit()
            else:
                embed_feat, pred_cls,x_s = model(input)
    
        elif cfg['add_fix']==1 and cfg['logit_div'] ==1:
            embed_feat, pred_cls,x,x_s = model(input)
            x_in = torch.softmax(x/cfg['temp'],dim =1)
        # print('applying again')
        with torch.no_grad():
            if cfg['global_reg'] == 1:
                if cfg['cls_ps'] == True:
                    _,_,g_yw,g_ys = model(input)
                else:
                    _,yw,ys = model(input)
                    _,g_yw,g_ys = global_model(input)
                g_max_p, g_hard_pseudo_label = torch.max(g_yw, dim=-1)
                
                # g_mask = g_max_p.ge(cfg['threshold'])
                g_mask = g_max_p.ge(thres)
                g_yw = g_yw[g_mask]
                # lable_s = torch.softmax(x_s,dim=1)
                # g_lable_s = lable_s[g_mask]
                #########################################
                max_p2, hard_pseudo_label2 = torch.max(yw, dim=-1)
                
                # g_mask = g_max_p.ge(cfg['threshold'])
                mask2 = max_p2.ge(thres)
                
                # lable_s = torch.softmax(x_s,dim=1)
                
        # print(g_yw.shape)    
        # exit()
        #act_loss = sum([item['mean_norm'] for item in list(model.act_stats.values())])
        #print("psd_label:",psd_label )
        # print(embed_feat.shape)
        if cfg['var_reg']:
            # print(embed_feat.shape)
            # K = compute_correlation_matrix(embed_feat)
            # # print(K,K.shape)
            # # Compute the Frobenius norm of K
            # frobenius_norm = torch.norm(K, p='fro')
            # # print(frobenius_norm)
            # # Compute the loss function LFedDecorr
            # d_ = K.shape
            # loss_var = 1/d_[0]**2 * frobenius_norm ** 2
            # print(loss_var)
            # exit()
            var = torch.var(embed_feat, dim=0)
            # var = torch.var(embed_feat, dim=1)
            loss_var = torch.mean(var)
            # print(var.shape,loss_var)
            # exit()
        # exit()
        if pred_cls.shape != psd_label.shape:
            # psd_label is not one-hot like.
            psd_label = to_device(psd_label,cfg['device'])
            psd_label = torch.zeros_like(pred_cls).scatter(1, psd_label.unsqueeze(1), 1)
        
        #print("psd_label:",psd_label )
        mean_pred_cls = torch.mean(pred_cls, dim=0, keepdim=True) #[1, C]
        reg_loss = - torch.sum(torch.log(mean_pred_cls) * mean_pred_cls)
        ent_loss = - torch.sum(torch.log(pred_cls) * pred_cls, dim=1).mean()
        psd_loss = - torch.sum(torch.log(pred_cls) * psd_label, dim=1).mean()
        
        #if epoch_idx >= 1.0:
            # loss = 2.0 * psd_loss
        #    loss = ent_loss + 1.0 * psd_loss
        #else:
        loss = - reg_loss + ent_loss + 0.5*psd_loss
        # loss = 0
         #need to re aadd
        if cfg['cls_ps']:
            class_psd_loss = -torch.sum(torch.log(ps_cls) * psd_label, dim=1).mean()
            loss+=1*class_psd_loss
            # print(class_psd_loss)
            # exit()
        if cfg['var_reg']:
            loss += cfg['var_wt']*loss_var
            # print(cfg['var_wt'],'varience weight',loss_var)
        # print(loss)
        # exit()
        unique_labels = torch.unique(psd_label_).cpu().numpy() 
        class_cent = torch.zeros(num_classes,embed_feat.shape[0])
        #batch_centers = torch.zeros(len(unique_labels).embed_feat.shape[1])
        max_p, hard_pseudo_label = torch.max(pred_cls, dim=-1)
        # mask = max_p.ge(cfg['threshold'])
        mask = max_p.ge(thres)
        embed_feat_masked = embed_feat[mask]
        pred_cls = pred_cls[mask]
        psd_label = psd_label[mask]
        # dym_psd_label  = dym_psd_label[mask]

        #print("loss_reg_dyn:",loss)
        #==================================================================#
        # loss = ent_loss + 1* psd_loss + 0.1 * dym_psd_loss - reg_loss + cfg['wt_actloss']*act_loss
        #==================================================================#
        #==================================================================#
        #==================================================================#
        # lr_scheduler(optimizer, iter_idx, iter_max)
        # optimizer.zero_grad()
        #==================================================================#
        # print(cent.shape,avg_cent.shape)
        #print("cfg_avg_cent:",cfg['avg_cent'])
        #avg_cent = torch.zeros(num_classes,embed_feat.shape[1])
        if cfg['avg_cent'] and avg_cent is not None:
        #if True:    
            cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat)
            #print("clnt_cent:",cent_batch)
            cent_batch = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None])
            server_cent = torch.squeeze(torch.Tensor(avg_cent))
            #print("server_cent:",server_cent.shape)
            clnt_cent = cent_batch[unique_labels]/torch.norm(cent_batch[unique_labels],dim=1,keepdim=True)
            server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
            server_cent = server_cent.to(cfg['device'])
            server_cent = torch.transpose(server_cent,0,1)
            #print("server_cent:",server_cent.shape)
            #print("clnt_cent:",clnt_cent.shape)
            #server_cent = (server_cent, cfg['device'])

            similarity_mat = torch.matmul(clnt_cent,server_cent)
            temp = cfg['temp']
            similarity_mat = torch.exp(similarity_mat/temp)
            pos_m = torch.diag(similarity_mat)
            pos_neg_m = torch.sum(similarity_mat,axis = 1)
            nce_loss = -1.0*torch.sum(torch.log(pos_m/pos_neg_m))
            # print('nce_loss',nce_loss)
            loss += cfg['gamma']*nce_loss
            #print("reg_loss:",reg_loss,"ent_loss:",ent_loss,"psd_loss:",psd_loss,"nce_loss:",nce_loss)
        if cfg['add_fix'] ==1:
            # target_prob,target_= torch.max(dym_label, dim=-1)
            # target_ = dym_label
            target_l = hard_pseudo_label
            # print(target_.shape,x_s.shape)
            lable_s = torch.softmax(x_s,dim=1)
            target_l = target_l[mask]
            # lable_s = torch.softmax(x_s,dim=1)
            g_lable_s = lable_s[g_mask]
            lable_s = lable_s[mask]
            # lable_s2 = lable_s[mask2]
            # print(target_.shape,lable_s.shape)
            # if target_.shape[0] != 0 and lable_s.shape[0]!= 0 :
            #     # continue
            #     fix_loss = loss_fn(lable_s,target_.detach())
            #     # print(loss)
            #     loss+=cfg['lambda']*fix_loss
                
            if cfg['global_reg'] == 1:
                # target_prob,target_= torch.max(dym_label, dim=-1)
                # target_ = dym_label
                # lable_s = torch.softmax(x_s,dim=1)
                # g_lable_s = lable_s[g_mask]
                g_target_ = g_hard_pseudo_label
                target_ = hard_pseudo_label
                # print(target_.shape,x_s.shape)
                # g_lable_s = torch.softmax(g_ys,dim=1)
                # g_target_ = g_target_[mask]
                g_target_ = g_target_[g_mask]
                # g_lable_s = g_lable_s[mask]
                # target_ = target_[mask]
                target_ = target_[g_mask]
                yw = yw[mask2]
                l_yw = pred_cls
                # hg = -torch.sum(torch.log(g_yw.detach())*g_yw.detach(), dim=1).mean()
                # hk = -torch.sum(torch.log(l_yw.detach())*l_yw.detach(), dim=1).mean()
                # # print('hk',hk,'hg',hg)
                # # exit()
                # wl = 2*(1/(hk+1e-8))/((1/(hk+1e-8)+(1/(hg+1e-8)))+1e-8)
                # wg = 2*(1/(hg+1e-8))/((1/(hk+1e-8)+(1/(hg+1e-8)))+1e-8)
                # print('wl',wl,'wg',wg)
                # exit()
                # print(target_.shape,lable_s.shape)
                if g_target_.shape[0] != 0 and g_lable_s.shape[0]!= 0 :
                    # continue
                    # g_fix_loss = loss_fn(g_lable_s,g_target_.detach())
                    # g_fix_loss = loss_fn(g_lable_s,target_.detach())
                    # g_fix_loss = loss_fn(lable_s,g_target_.detach())
                    g_fix_loss = loss_fn(g_lable_s,g_target_.detach())
                    # print(g_fix_loss)
                    # exit()
                    loss+=cfg['g_lambda']*g_fix_loss
                    # loss+=wg*g_fix_loss
                #######################################
                # target_2 = hard_pseudo_label2
                # target_ = hard_pseudo_label
                
                # target_2 = target_2[mask]
                
                # target_ = target_[mask]
            
                # if target_2.shape[0] != 0 and lable_s.shape[0]!= 0 :
                    
                #     fix_loss2= loss_fn(lable_s,target_2.detach())
                
                #     loss+=1*fix_loss2
                # print(fix_loss2)
                ################################################
                # target_2 = hard_pseudo_label2
                # target_ = hard_pseudo_label
                
                # target_2 = target_2[mask2]
                
                # target_ = target_[mask2]
            
                # if target_2.shape[0] != 0 and lable_s.shape[0]!= 0 :
                    
                #     fix_loss2= loss_fn(lable_s2,target_2.detach())
                
                #     loss+=1*fix_loss2
            if target_l.shape[0] != 0 and lable_s.shape[0]!= 0 :
                # continue
                fix_loss = loss_fn(lable_s,target_l.detach())
                # print(fix_loss)
                # exit()
                loss+=cfg['lambda']*fix_loss
                # loss+=wl*fix_loss
        
            
        # print(loss)
        # exit()
        
                        
        optimizer.zero_grad()
        grad_sum = 0
        # for k, v in model.backbone_layer.named_parameters():
        #             # print(k)
        #             if "bn" in k:
        #                 pass
        #             else:
        #                 # G = v.grad
        #                 # grad_sum+=G.sum()
        #                 print(v.grad)
        # print(grad_sum)
        # exit()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        # grad_sum = 0
        # for k, v in model.backbone_layer.named_parameters():
        #             # print(k)
        #             if "bn" in k:
        #                 pass
        #             else:
        #                 G = v.grad[0]
        #                 grad_sum+=G.sum()
        # print(grad_sum)
        # exit()
        if scheduler is not None:
            # print('scheduler step')
            scheduler.step()
            # print('lr at client
    with torch.no_grad():
        loss_stack.append(loss.cpu().item())
        if cfg['cls_ps']:
            cent = None
        else:
            cent = get_final_centroids(model,test_data_loader,pred_label)
        #print("cent here:",cent.shape)
    
       
    train_loss = np.mean(loss_stack)

    return train_loss,cent




def bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,cent,avg_cent,fwd_pass=False,scheduler = None):
    loss_stack = []
    model.to(cfg["device"])
    # model = to_device(model, cfg['device'])
    with torch.no_grad():
        model.eval()
        glob_multi_feat_cent, all_psd_label ,all_emd_feat,all_cls_out= init_multi_cent_psd_label(model,test_data_loader)
        #init_psd_label_shot_icml(model,test_data_loader)
        
     
    # print(glob_multi_feat_cent)
    
    model.train()
    epoch_idx=epoch
    grad_bank = {}
    avg_counter = 0 
    
    # print(scheduler)
    # exit()
    for i, input in enumerate(train_data_loader):
        # print(i)
        input = collate(input)
        input_size = input['data'].size(0)
        if input_size<=1:
            break
        input['loss_mode'] = cfg['loss_mode']
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        all_psd_label = to_device(all_psd_label, cfg['device'])
        psd_label = all_psd_label[input['id']]
        psd_label_ = all_psd_label[input['id']]
        # all_psd_label = all_psd_label.cpu()
        all_psd_label = all_psd_label
        # print('psd label shape',psd_label.shape)
        # print(cfg['add_fix'],cfg['logit_div'])
        if cfg['add_fix']==0:
            embed_feat, pred_cls = model(input)
        elif cfg['add_fix']==1 and cfg['logit_div'] ==0:
            embed_feat, pred_cls,x_s = model(input)
        elif cfg['add_fix']==1 and cfg['logit_div'] ==1:
            embed_feat, pred_cls,x,x_s = model(input)
            x_in = torch.softmax(x/cfg['temp'],dim =1)
            # if cfg['logit_div'] == 1:
            #     # print('div logit')
            #     pred_cls =pred_cls/2
            #     pred_cls = torch.softmax(pred_cls,dim=1)
        
        if cfg['logit_div'] == 1:
            # print('div logit')
            with torch.no_grad():
                init_model.eval()
                # print(init_model.device)
                # print(input.device)
                init_embed_feat,_,init_pred_cls,init_x_s = init_model(input)
                # init_pred_cls = torch.softmax(init_pred_cls,dim=1)
                # print(torch.sum(init_pred_cls,dim=1))
                # exit()
                init_pred_cls = init_pred_cls/cfg['temp']
                init_pred_cls = torch.softmax(init_pred_cls,dim=1)
                # print(torch.sum(init_pred_cls,dim=1))
                # exit()
                
        # input = to_device(input, 'cpu')
        # act_loss = sum([item['mean_norm'] for item in list(model.act_stats.values())])
        ##########################################
        # # print('pred shape:',pred_cls.shape)
        # max_p, hard_pseudo_label = torch.max(pred_cls, dim=-1)
        # mask = max_p.ge(cfg['threshold'])
        # # print('max_p shape',max_p.shape)
        # # print('embd feat shape',embed_feat.shape)
        # # print('mask shape',mask.shape)
        # # print(mask)
        # # embed_feat = torch.tensor(compress(embed_feat,mask))
        # embed_feat = embed_feat[mask]
        # pred_cls = pred_cls[mask]
        # psd_label = psd_label[mask]
        # # pred_cls = torch.tensor(compress(pred_cls,mask))
        # # print('embd feat shape',embed_feat.shape)
        # print('pred cls shape',pred_cls.shape)
        # # # exit()
        # print('psd_lable shape',psd_label.shape)
        ##############################
        if pred_cls.shape != psd_label.shape:
            # psd_label is not one-hot like.
            psd_label = torch.zeros_like(pred_cls).scatter(1, psd_label.unsqueeze(1), 1)
        
        mean_pred_cls = torch.mean(pred_cls, dim=0, keepdim=True) #[1, C]
        reg_loss = - torch.sum(torch.log(mean_pred_cls) * mean_pred_cls)
        ent_loss = - torch.sum(torch.log(pred_cls) * pred_cls, dim=1).mean()
        psd_loss = - torch.sum(torch.log(pred_cls) * psd_label, dim=1).mean()
        
        # unique_labels = torch.unique(psd_label_).cpu().numpy() 
        # cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
        # if epoch_idx >= 1.0:
        #     # loss = 2.0 * psd_loss
        #     loss = ent_loss + 1.0 * psd_loss
        # else:
        #     loss = - reg_loss + ent_loss
        #print("loss_reg:",loss)
        #==================================================================#
        # SOFT FEAT SIMI LOSS
        #==================================================================#
        normed_emd_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        glob_multi_feat_cent = to_device(glob_multi_feat_cent,cfg['device'])
        dym_feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_emd_feat)
        dym_feat_simi, _ = torch.max(dym_feat_simi, dim=2) #[N, C]
        dym_label = torch.softmax(dym_feat_simi, dim=1)    #[N, C]
        dym_psd_loss = - torch.sum(torch.log(pred_cls) * dym_label, dim=1).mean() - torch.sum(torch.log(dym_label) * pred_cls, dim=1).mean()
        # if pred_cls.shape != psd_label.shape:
            # psd_label is not one-hot like.
        # print('dym lable shape',dym_label.shape)
        _, dym_label = torch.max(dym_label, dim=1)
        dym_psd_label = torch.zeros_like(pred_cls).scatter(1, dym_label.unsqueeze(1), 1)
        # if epoch_idx >= 1.0:
        #     loss += 0.5 * dym_psd_loss
        # glob_multi_feat_cent = glob_multi_feat_cent.cpu()
        glob_multi_feat_cent = glob_multi_feat_cent
        #print("loss_reg_dyn:",loss)
        #==================================================================#
        loss = ent_loss + 0.3* psd_loss + 0.1 * dym_psd_loss - reg_loss #+ cfg['wt_actloss']*act_loss
        #==================================================================#
        #==================================================================#
        #==================================================================#
        # print('bmd_loss',loss)
        # exit()
        # lr_scheduler(optimizer, iter_idx, iter_max)
        # optimizer.zero_grad()
        #==================================================================#
        # print(cent.shape,avg_cent.shape)
        #print("cfg_avg_cent:",cfg['avg_cent'])
        # if cfg['avg_cent'] and avg_cent is not None:
        #     #cent_loss = torch.nn.MSELoss()
        #     #loss+=cfg['gamma']*cent_loss(cent.squeeze(),avg_cent.squeeze())
        #     # loss += cfg['gamma']*dist/avg_cent.shape[0]

        #     # print(loss)
     
        #     batch_size = embed_feat.shape[0]
        #     class_num  = glob_multi_feat_cent.shape[0]
        #     multi_num  = glob_multi_feat_cent.shape[1]
    
        #     normed_embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        #     feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_embed_feat)
        #     feat_simi = feat_simi.flatten(1) #[N, C*M]
        #     feat_simi = torch.softmax(feat_simi, dim=1).reshape(batch_size, class_num, multi_num) #[N, C, M]
    
        #     curr_multi_feat_cent = torch.einsum("ncm, nd -> cmd", feat_simi, normed_embed_feat)
        #     curr_multi_feat_cent /= (torch.sum(feat_simi, dim=0).unsqueeze(2) + 1e-8)
        #     #print("cent:",cent.shape)
        #     clnt_cent = torch.squeeze(curr_multi_feat_cent)
        #     #print("embed_feat:",embed_feat.shape)
        #     #clnt_cent = torch.squeeze(embed_feat)
        #     #normed_emd_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        #     #dym_feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_emd_feat)
        #     server_cent = torch.squeeze(avg_cent)
            
        #     #print("clnt_cent:",clnt_cent) 

        #     clnt_cent = clnt_cent/torch.norm(clnt_cent,dim=1,keepdim=True)
        #     server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
            
        #     server_cent = torch.transpose(server_cent,0,1)
        #     similarity_mat = torch.matmul(clnt_cent,server_cent)
        #     #print("similarity_mat:",similarity_mat)
        #     temp = 8.0
        #     similarity_mat = torch.exp(similarity_mat/temp)
        #     pos_m = torch.diag(similarity_mat)
        #     pos_neg_m = torch.sum(similarity_mat,axis = 1)
            
        #     #print("pos_m:",pos_m,"\t","neg_m:",pos_neg_m)
        #     nce_loss = -1.0*torch.sum(torch.log(pos_m/pos_neg_m))
        #     #print("loss:", loss,"nce_loss:",nce_loss)
        #     if epoch_idx >= 1.0:
        #         loss += cfg['gamma']*nce_loss
        ############################################################################
        max_p, hard_pseudo_label = torch.max(pred_cls, dim=-1)
        # print(cfg['threshold'])
        # exit()
        mask = max_p.ge(cfg['threshold'])
        embed_feat_masked = embed_feat[mask]
        pred_cls = pred_cls[mask]
        psd_label = psd_label[mask]
        dym_psd_label  = dym_psd_label[mask]
        # print('psd shape',psd_label.shape)
        # print('t.psd shape',torch.transpose(psd_label,0,1).shape)
        # print('embd shape',embed_feat.shape)
        # cent_batch = torch.matmul(torch.transpose(dym_psd_label,0,1), embed_feat)
        ################################################################################################
        # cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat_masked)
        # # print("clnt_cent:",cent_batch.shape)
        # cent_batch_ = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None]) #C x 256
        # # cent_batch_ = cent_batch / (1e-9 + dym_psd_label.sum(axis=0)[:,None]) #C x 256
        # # print("clnt_cent:",cent_batch.shape)
        # # Calculate the class-wise variance of clnt_cent
        # variance_clnt_cent = torch.zeros(psd_label.shape[1], embed_feat_masked.shape[1]) 
        # # variance_clnt_cent = torch.zeros(dym_psd_label.shape[1], embed_feat.shape[1]) 
        # for i in range(psd_label.shape[1]):
        #     # class_indices = psd_label[:, i].nonzero().squeeze() 
        #     class_indices = (psd_label[:, i] == 1).nonzero(as_tuple=True)[0]
        #     if len(class_indices) > 0:
        #         class_squared_diff = (cent_batch[class_indices] - cent_batch_[i]) ** 2
        #         variance_clnt_cent[i] = torch.mean(class_squared_diff, dim=0)
        # # for i in range(dym_psd_label.shape[1]):
        # #     # class_indices = psd_label[:, i].nonzero().squeeze() 
        # #     class_indices = (dym_psd_label[:, i] == 1).nonzero(as_tuple=True)[0]
        # #     if len(class_indices) > 0:
        # #         class_squared_diff = (cent_batch[class_indices] - cent_batch_[i]) ** 2
        # #         variance_clnt_cent[i] = torch.mean(class_squared_diff, dim=0)     
                 
        # clnt_cent = cent_batch_/(torch.norm(cent_batch_,dim=1,keepdim=True)+1e-9)
        # clnt_cent = torch.squeeze(clnt_cent)
        # variance_clnt_cent= variance_clnt_cent/(torch.norm(variance_clnt_cent,dim=1,keepdim=True)+1e-9)
        # variance_clnt_cent = torch.squeeze(variance_clnt_cent)
        # # print('var cent shape',variance_clnt_cent.shape)
        # variance_clnt_cent = to_device(variance_clnt_cent, cfg['device'])
        # # exit()
        ######################################################################################################
        if cfg['avg_cent'] and avg_cent is not None:
            # # print('pred shape:',pred_cls.shape)
            # max_p, hard_pseudo_label = torch.max(pred_cls, dim=-1)
            # mask = max_p.ge(cfg['threshold'])
            # # print('max_p shape',max_p.shape)
            # # print('embd feat shape',embed_feat.shape)
            # # print('mask shape',mask.shape)
            # # print(mask)
            # # embed_feat = torch.tensor(compress(embed_feat,mask))
            # embed_feat = embed_feat[mask]
            # pred_cls = pred_cls[mask]
            # psd_label = psd_label[mask]
            # # pred_cls = torch.tensor(compress(pred_cls,mask))
            # # print('embd feat shape',embed_feat.shape)
            # # print('pred cls shape',pred_cls.shape)
        # # exit()
        #if True:    
            if cfg['loss_mse'] == 1:
                # cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
                # cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat)
                # print("clnt_cent:",cent_batch.shape)
                # cent_batch = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None])
                server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
                
                # clnt_cent = cent_batch[unique_labels]/torch.norm(cent_batch[unique_labels],dim=1,keepdim=True)
                server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
                server_cent = server_cent.to(cfg['device'])
                server_cent = torch.transpose(server_cent,0,1)
                # print(server_cent)
                server_cent = (server_cent, cfg['device'])
                clnt_cent = torch.squeeze(clnt_cent)
                cent_loss = torch.nn.MSELoss()
                print("server_cent:",server_cent[0].shape)
                print("server_cent:",clnt_cent.shape)
                print(clnt_cent.shape)
                cent_mse = cent_loss(clnt_cent.squeeze(),server_cent[0].squeeze())
                print('centroid_loss',cent_mse)
                print(loss)
                exit()
                loss+=cfg['gamma']*cent_mse
            else:
                # cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat)
                # print("clnt_cent:",cent_batch.shape)
                # cent_batch = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None]) #C x 256
                # print("clnt_cent:",cent_batch.shape)
                # clnt_cent = cent_batch/(torch.norm(cent_batch,dim=1,keepdim=True)+1e-9)
                # clnt_cent = torch.squeeze(clnt_cent)
                
                server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
                
                # print("server_cent:",server_cent.shape)
                
                # clnt_cent = cent_batch[unique_labels]/(torch.norm(cent_batch[unique_labels],dim=1,keepdim=True)+1e-9)
                # print('clnt shape',clnt_cent.shape)
                server_cent = server_cent/(1e-9+torch.norm(server_cent,dim=1,keepdim=True))
                # if torch.isnan(server_cent).any():
                #     print("Tensor contains NaN values.")
                # else:
                #     print("Tensor does not contain NaN values.")
                server_cent = server_cent.to(cfg['device'])
                server_cent = torch.transpose(server_cent,0,1)
                
                server_cent = (server_cent, cfg['device'])
                
                
                # print(type(server_cent))
                # print(type(clnt_cent))
                # print(clnt_cent)
                # print(server_cent)
                # if torch.isnan(clnt_cent).any():
                #     print("Tensor contains NaN values.")
                # else:
                #     print("Tensor does not contain NaN values.")
                
                similarity_mat = torch.matmul(clnt_cent,server_cent[0])
                ###############################################################
                # server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
                # server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
                # server_cent = torch.transpose(server_cent,0,1)
                # clnt_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
                # clnt_cent = torch.squeeze(clnt_cent)
                # print(server_cent)
                # server_cent = torch.tensor(server_cent)
                # # print("server_cent:",server_cent.shape)
                # # print("clnt_cent:",clnt_cent.shape)
                
                # similarity_mat = torch.matmul(clnt_cent.cpu(),server_cent)
                ################################################################
                temp = cfg['temp']
                similarity_mat = torch.exp(similarity_mat/temp)
                # print(similarity_mat)
                pos_m = torch.diag(similarity_mat)
                pos_neg_m = torch.sum(similarity_mat,axis = 1)
                nce_loss = -1.0*torch.sum(torch.log(pos_m/pos_neg_m))
                # nce_loss = -1.0*torch.mean(torch.log(pos_m/pos_neg_m))
                # print(nce_loss)
                # exit()
                loss += cfg['gamma']*nce_loss
                # cent = clnt_cent
                # loss = 0*loss+cfg['gamma']*nce_loss


        if cfg['kl'] == 1:
            # server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
            # server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
            # server_cent = torch.transpose(server_cent,0,1)
            cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat)
            cent_batch = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None])
            clnt_cent = cent_batch/(1e-8+torch.norm(cent_batch,dim=1,keepdim=True))

            # clnt_cent = cent_batch[unique_labels]/torch.norm(cent_batch[unique_labels],dim=1,keepdim=True)
            # clnt_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
            clnt_cent = torch.log_softmax(clnt_cent,dim = 1)
            clnt_cent = torch.squeeze(clnt_cent)
            mean_p = torch.mean(clnt_cent,axis = 0)
            std_p = torch.std(clnt_cent,axis = 0)
            epsi = 1e-8
            kl_loss = torch.sum(-torch.log(std_p+epsi)/2+torch.square(std_p)/2+torch.square(mean_p)/2-0.5)
            # Compute the KL divergence analytically
            # kl = np.log(sigma) + (sigma**2 + mu**2 - 1) / 2 - mu
            print('kl loss',kl_loss)
            exit()
            loss+=cfg['kl_weight']*kl_loss


        if cfg['kl_loss'] == 1 and avg_cent is not None:
            server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
            server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
            server_cent = torch.transpose(server_cent,0,1)
            clnt_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
            clnt_cent = torch.squeeze(clnt_cent)
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            # input should be a distribution in the log space
            print(clnt_cent.shape,server_cent.shape)
            server_cent = torch.transpose(server_cent,0,1)
            clnt_cent = clnt_cent.cpu()
            input_c = F.log_softmax(clnt_cent, dim=1)
            # Sample a batch of distributions. Usually this would come from the dataset
            target_s = F.log_softmax(server_cent, dim=1)
            print(input_c.shape,target_s.shape)
            # print('kl loss',kl_loss)
            # exit()
            loss+=cfg['kl_weight']*kl_loss(input_c, target_s)

        if cfg['add_fix'] ==1:
            # target_prob,target_= torch.max(dym_label, dim=-1)
            # target_ = dym_label
            target_ = hard_pseudo_label
            # print(target_.shape,x_s.shape)
            lable_s = torch.softmax(x_s,dim=1)
            target_ = target_[mask]
            lable_s = lable_s[mask]
            # print(target_.shape,lable_s.shape)
            if target_.shape[0] != 0 and lable_s.shape[0]!= 0 :
                # continue
                fix_loss = loss_fn(lable_s,target_.detach())
                # print(loss)
                loss+=cfg['lambda']*fix_loss
            # exit()
            ## tbd for testing
            # fix_loss_symmetric  = - torch.sum(torch.log(target_prob) * label_s, dim=1).mean() - torch.sum(torch.log(label_s) * target_prob, dim=1).mean()
            ##
            # fix_loss = loss_fn(x_s,target_.detach())
            
                # print('fix-loss',fix_loss)
                # exit()
            # fix_loss = loss_fn(label_s,target_.detach())
            # print(fix_loss)
        # print(loss)
        if cfg['logit_div']==1:
            # print('div logit')
            kl_loss = torch.nn.KLDivLoss(reduction="mean")
            init_pred_cls = init_pred_cls[mask] #no grad
            # print(pred_cls.shape,init_pred_cls.shape)
            
            # pred_cls = torch.log(pred_cls)
            # logit_div_loss = kl_loss(init_pred_cls,pred_cls)
            # print(torch.sum(init_pred_cls,dim=1))
            # print(torch.sum(pred_cls,dim=1))
            logit_div_loss = torch.mean(torch.sum(init_pred_cls*torch.log(init_pred_cls/pred_cls),dim=1))
            # logit_div_loss = kl_loss(pred_cls,init_pred_cls)
            # print(logit_div_loss)
            # exit()
            loss+=cfg['kl_weight']*logit_div_loss
            
                
        optimizer.zero_grad()
        
        loss.backward()
        # Check gradients
        # if cfg['avg_cent'] and avg_cent is not None:
        #     for name, param in model.named_parameters():
        #         print(f"Parameter: {name}, Gradient: {param.grad}")
        #     exit()
        # if cfg['trk_grad']:
        #     with torch.no_grad():
        #         norm_type = 2
        #         total_norm = torch.norm(
        #         torch.stack([torch.norm(p.grad.detach(), norm_type) for p in model.parameters()]), norm_type)
        #         # for idx, param in enumerate(model.parameters()):
        #         #     grad_bank[f"layer_{idx}"] += param.grad
        #         #     avg_counter += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # if fwd_pass==False:
        model.to(cfg["device"])
        optimizer.step()
        if scheduler is not None:
            # print('scheduler step')
            scheduler.step()
            # print('lr at client',scheduler.get_last_lr())
        with torch.no_grad():
            loss_stack.append(loss.cpu().item())
            # print(loss.cpu().item())
            # print(loss_stack)
            # print(glob_multi_feat_cent.shape,embed_feat.shape)
            glob_multi_feat_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
    
    # cent = glob_multi_feat_cent.clone()  
    # print('cent:',cent.shape)
    # print(loss_stack)
    # exit()
    train_loss = np.mean(loss_stack)
    if cfg['trk_grad']:
        with torch.no_grad():
            for key in grad_bank:
                grad_bank[key] = grad_bank[key] / avg_counter
            with open(f"./output/gradients/{cfg['tag']}_{epoch}.json", "w") as outfile:
                json.dump(grad_bank, outfile)
    print('train_loss:',train_loss)
    # return train_loss,cent
    # del variables
    # print(torch.cuda.memory_summary(device=cfg['device']))
    gc.collect()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=cfg['device']))
    # exit()
    # return train_loss,clnt_cent,variance_clnt_cent
    return train_loss

def crco_train(model,tech_model,train_data_loader,test_data_loader,optimizer,epoch,cent,avg_cent,fwd_pass=False,scheduler = None):
    loss_stack = []
    model.to(cfg["device"])
    
    model.train()
    epoch_idx=epoch
    grad_bank = {}
    avg_counter = 0 
    

    for i, input in enumerate(train_data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        if input_size<=1:
            break
        input['loss_mode'] = cfg['loss_mode']
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        # all_psd_label = to_device(all_psd_label, cfg['device'])
        # psd_label = all_psd_label[input['id']]
        # psd_label_ = all_psd_label[input['id']]
        # all_psd_label = all_psd_label.cpu()
        f_weak,weak_logit,f_s1,strong1_logit,f_s2,strong2_logit = model(input)
        with torch.no_grad():
            t_f_weak,t_weak_logit,t_f_s1,t_strong1_logit,t_f_s2,t_strong2_logit = tech_model(input)
        
        weak_prob,s1_prob,s2_prob = F.softmax(weak_logit,dim=-1),F.softmax(strong1_logit,dim=-1),F.softmax(strong2_logit,dim=-1)
        t_weak_prob,t_s1_prob,t_s2_prob = F.softmax(t_weak_logit,dim=-1),F.softmax(t_strong1_logit,dim=-1),F.softmaxt_(strong2_logit,dim=-1)
        
        input = to_device(input, 'cpu')
        
        # baseline
        loss = baseline_loss(t_weak_prob, f_weak,t_weak_logit)
        # fixmatch
        pseudo_label = torch.softmax(weak_logit.detach(), dim=-1)
        max_probs, tgt_u = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(cfg['prob_threshold']).float().detach()
        pred_right = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)
        if self.use_cluster_label_for_fixmatch:
            tgt_u = self.obtain_batch_label(online_weak_feat, None)
            # tgt_u = self.obtain_batch_label(target_weak_feat, None)
            cluster_acc = torch.sum((tgt_u == tgt_unlabeled_label.squeeze(1)) * mask) / torch.sum(mask)
        else:
            cluster_acc = torch.tensor(0)
        mask_val = torch.sum(mask).item() / mask.shape[0]
        high_ratio = mask_val
        if self.iteration >= self.fixmatch_start:
            ###########
            if self.fixmatch_type == 'orig':
                strong_aug_pred = online_strong_logits[0:tgt_unlabeled_size]
                loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
                loss += loss_consistency * self.lambda_fixmatch * 0.5
                strong_aug_pred = online_strong_logits[tgt_unlabeled_size:]
                loss_consistency = (F.cross_entropy(strong_aug_pred, tgt_u, reduction='none') * mask).mean()
                loss += loss_consistency * self.lambda_fixmatch * 0.5
            #
            elif self.fixmatch_type == 'class_relation':
                #
                loss_1 = self.class_contrastive_loss(online_strong_prob[0:tgt_unlabeled_size], tgt_u, mask)
                loss_2 = self.class_contrastive_loss(online_strong_prob[tgt_unlabeled_size:], tgt_u, mask)
                loss += (loss_1 + loss_2) * self.lambda_fixmatch * 0.5
            else:
                raise RuntimeError('wrong fixmatch type')
        # #
        # # constrastive loss
        all_k_strong = target_strong_prob
        all_k_weak = target_weak_prob
        weak_feat_for_backbone = online_weak_prob
        k_weak_for_backbone = all_k_weak
        k_strong_for_backbone = all_k_strong[0:tgt_unlabeled_size]
        strong_feat_for_backbone = online_strong_prob[0:tgt_unlabeled_size]
        k_strong_2 = all_k_strong[tgt_unlabeled_size:]
        feat_strong_2 = online_strong_prob[tgt_unlabeled_size:]
        if self.use_only_current_batch_for_instance:
            tmp_weak_negative_bank = online_weak_prob
            tmp_strong_negative_bank = strong_feat_for_backbone
            neg_ind = tgt_img_ind
            self.num_k = 1
        else:
            if self.add_current_data_for_instance:
                tmp_weak_negative_bank = torch.cat((self.weak_negative_bank, online_weak_prob), dim=0)
                tmp_strong_negative_bank = torch.cat((self.strong_negative_bank, strong_feat_for_backbone), dim=0)
                neg_ind = torch.cat((self.ngative_img_ind_bank, tgt_img_ind))
            else:
                tmp_weak_negative_bank = self.weak_negative_bank
                tmp_strong_negative_bank = self.strong_negative_bank
                neg_ind = self.ngative_img_ind_bank
        #
        info_nce_loss_1 = self.instance_contrastive_loss(strong_feat_for_backbone, k_weak_for_backbone,
                                                         tmp_weak_negative_bank,
                                                         self_ind=tgt_img_ind, neg_ind=neg_ind)
        info_nce_loss_3 = self.instance_contrastive_loss(strong_feat_for_backbone, k_strong_2,
                                                         tmp_strong_negative_bank,
                                                         self_ind=tgt_img_ind, neg_ind=neg_ind)
        info_nce_loss_2 = self.instance_contrastive_loss(weak_feat_for_backbone, k_strong_for_backbone,
                                                         tmp_strong_negative_bank,
                                                         self_ind=tgt_img_ind, neg_ind=neg_ind)
        info_nce_loss = (info_nce_loss_1 + info_nce_loss_2 + info_nce_loss_3) / 3.0
        #
        loss += info_nce_loss * self.lambda_nce
        if pred_cls.shape != psd_label.shape:
            # psd_label is not one-hot like.
            psd_label = torch.zeros_like(pred_cls).scatter(1, psd_label.unsqueeze(1), 1)
        
        mean_pred_cls = torch.mean(pred_cls, dim=0, keepdim=True) #[1, C]
        reg_loss = - torch.sum(torch.log(mean_pred_cls) * mean_pred_cls)
        ent_loss = - torch.sum(torch.log(pred_cls) * pred_cls, dim=1).mean()
        psd_loss = - torch.sum(torch.log(pred_cls) * psd_label, dim=1).mean()
        
        unique_labels = torch.unique(psd_label_).cpu().numpy() 
        # cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
        # if epoch_idx >= 1.0:
        #     # loss = 2.0 * psd_loss
        #     loss = ent_loss + 1.0 * psd_loss
        # else:
        #     loss = - reg_loss + ent_loss
        #print("loss_reg:",loss)
        #==================================================================#
        # SOFT FEAT SIMI LOSS
        #==================================================================#
        normed_emd_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        glob_multi_feat_cent = to_device(glob_multi_feat_cent,cfg['device'])
        dym_feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_emd_feat)
        dym_feat_simi, _ = torch.max(dym_feat_simi, dim=2) #[N, C]
        dym_label = torch.softmax(dym_feat_simi, dim=1)    #[N, C]
        dym_psd_loss = - torch.sum(torch.log(pred_cls) * dym_label, dim=1).mean() - torch.sum(torch.log(dym_label) * pred_cls, dim=1).mean()
        # if pred_cls.shape != psd_label.shape:
            # psd_label is not one-hot like.
        # print('dym lable shape',dym_label.shape)
        _, dym_label = torch.max(dym_label, dim=1)
        dym_psd_label = torch.zeros_like(pred_cls).scatter(1, dym_label.unsqueeze(1), 1)
        # if epoch_idx >= 1.0:
        #     loss += 0.5 * dym_psd_loss
        glob_multi_feat_cent = glob_multi_feat_cent.cpu()
        #print("loss_reg_dyn:",loss)
        #==================================================================#
        loss = ent_loss + 0.3* psd_loss + 0.1 * dym_psd_loss - reg_loss #+ cfg['wt_actloss']*act_loss
        #==================================================================#
        #==================================================================#
        #==================================================================#
        # print('bmd_loss',loss)
        # exit()
        # lr_scheduler(optimizer, iter_idx, iter_max)
        # optimizer.zero_grad()
        #==================================================================#
        # print(cent.shape,avg_cent.shape)
        #print("cfg_avg_cent:",cfg['avg_cent'])
        # if cfg['avg_cent'] and avg_cent is not None:
        #     #cent_loss = torch.nn.MSELoss()
        #     #loss+=cfg['gamma']*cent_loss(cent.squeeze(),avg_cent.squeeze())
        #     # loss += cfg['gamma']*dist/avg_cent.shape[0]

        #     # print(loss)
     
        #     batch_size = embed_feat.shape[0]
        #     class_num  = glob_multi_feat_cent.shape[0]
        #     multi_num  = glob_multi_feat_cent.shape[1]
    
        #     normed_embed_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        #     feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_embed_feat)
        #     feat_simi = feat_simi.flatten(1) #[N, C*M]
        #     feat_simi = torch.softmax(feat_simi, dim=1).reshape(batch_size, class_num, multi_num) #[N, C, M]
    
        #     curr_multi_feat_cent = torch.einsum("ncm, nd -> cmd", feat_simi, normed_embed_feat)
        #     curr_multi_feat_cent /= (torch.sum(feat_simi, dim=0).unsqueeze(2) + 1e-8)
        #     #print("cent:",cent.shape)
        #     clnt_cent = torch.squeeze(curr_multi_feat_cent)
        #     #print("embed_feat:",embed_feat.shape)
        #     #clnt_cent = torch.squeeze(embed_feat)
        #     #normed_emd_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        #     #dym_feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_emd_feat)
        #     server_cent = torch.squeeze(avg_cent)
            
        #     #print("clnt_cent:",clnt_cent) 

        #     clnt_cent = clnt_cent/torch.norm(clnt_cent,dim=1,keepdim=True)
        #     server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
            
        #     server_cent = torch.transpose(server_cent,0,1)
        #     similarity_mat = torch.matmul(clnt_cent,server_cent)
        #     #print("similarity_mat:",similarity_mat)
        #     temp = 8.0
        #     similarity_mat = torch.exp(similarity_mat/temp)
        #     pos_m = torch.diag(similarity_mat)
        #     pos_neg_m = torch.sum(similarity_mat,axis = 1)
            
        #     #print("pos_m:",pos_m,"\t","neg_m:",pos_neg_m)
        #     nce_loss = -1.0*torch.sum(torch.log(pos_m/pos_neg_m))
        #     #print("loss:", loss,"nce_loss:",nce_loss)
        #     if epoch_idx >= 1.0:
        #         loss += cfg['gamma']*nce_loss
        ############################################################################
        max_p, hard_pseudo_label = torch.max(pred_cls, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        embed_feat_masked = embed_feat[mask]
        pred_cls = pred_cls[mask]
        psd_label = psd_label[mask]
        dym_psd_label  = dym_psd_label[mask]
        # print('psd shape',psd_label.shape)
        # print('t.psd shape',torch.transpose(psd_label,0,1).shape)
        # print('embd shape',embed_feat.shape)
        # cent_batch = torch.matmul(torch.transpose(dym_psd_label,0,1), embed_feat)
        cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat_masked)
        # print("clnt_cent:",cent_batch.shape)
        cent_batch_ = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None]) #C x 256
        # cent_batch_ = cent_batch / (1e-9 + dym_psd_label.sum(axis=0)[:,None]) #C x 256
        # print("clnt_cent:",cent_batch.shape)
        # Calculate the class-wise variance of clnt_cent
        variance_clnt_cent = torch.zeros(psd_label.shape[1], embed_feat_masked.shape[1]) 
        # variance_clnt_cent = torch.zeros(dym_psd_label.shape[1], embed_feat.shape[1]) 
        for i in range(psd_label.shape[1]):
            # class_indices = psd_label[:, i].nonzero().squeeze() 
            class_indices = (psd_label[:, i] == 1).nonzero(as_tuple=True)[0]
            if len(class_indices) > 0:
                class_squared_diff = (cent_batch[class_indices] - cent_batch_[i]) ** 2
                variance_clnt_cent[i] = torch.mean(class_squared_diff, dim=0)
        # for i in range(dym_psd_label.shape[1]):
        #     # class_indices = psd_label[:, i].nonzero().squeeze() 
        #     class_indices = (dym_psd_label[:, i] == 1).nonzero(as_tuple=True)[0]
        #     if len(class_indices) > 0:
        #         class_squared_diff = (cent_batch[class_indices] - cent_batch_[i]) ** 2
        #         variance_clnt_cent[i] = torch.mean(class_squared_diff, dim=0)     
                 
        clnt_cent = cent_batch_/(torch.norm(cent_batch_,dim=1,keepdim=True)+1e-9)
        clnt_cent = torch.squeeze(clnt_cent)
        variance_clnt_cent= variance_clnt_cent/(torch.norm(variance_clnt_cent,dim=1,keepdim=True)+1e-9)
        variance_clnt_cent = torch.squeeze(variance_clnt_cent)
        # print('var cent shape',variance_clnt_cent.shape)
        variance_clnt_cent = to_device(variance_clnt_cent, cfg['device'])
        # exit()
        ######################################################################################################
        if cfg['avg_cent'] and avg_cent is not None:
            # # print('pred shape:',pred_cls.shape)
            # max_p, hard_pseudo_label = torch.max(pred_cls, dim=-1)
            # mask = max_p.ge(cfg['threshold'])
            # # print('max_p shape',max_p.shape)
            # # print('embd feat shape',embed_feat.shape)
            # # print('mask shape',mask.shape)
            # # print(mask)
            # # embed_feat = torch.tensor(compress(embed_feat,mask))
            # embed_feat = embed_feat[mask]
            # pred_cls = pred_cls[mask]
            # psd_label = psd_label[mask]
            # # pred_cls = torch.tensor(compress(pred_cls,mask))
            # # print('embd feat shape',embed_feat.shape)
            # # print('pred cls shape',pred_cls.shape)
        # # exit()
        #if True:    
            if cfg['loss_mse'] == 1:
                # cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
                # cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat)
                # print("clnt_cent:",cent_batch.shape)
                # cent_batch = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None])
                server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
                
                # clnt_cent = cent_batch[unique_labels]/torch.norm(cent_batch[unique_labels],dim=1,keepdim=True)
                server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
                server_cent = server_cent.to(cfg['device'])
                server_cent = torch.transpose(server_cent,0,1)
                # print(server_cent)
                server_cent = (server_cent, cfg['device'])
                clnt_cent = torch.squeeze(clnt_cent)
                cent_loss = torch.nn.MSELoss()
                print("server_cent:",server_cent[0].shape)
                print("server_cent:",clnt_cent.shape)
                print(clnt_cent.shape)
                cent_mse = cent_loss(clnt_cent.squeeze(),server_cent[0].squeeze())
                print('centroid_loss',cent_mse)
                print(loss)
                exit()
                loss+=cfg['gamma']*cent_mse
            else:
                # cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat)
                # print("clnt_cent:",cent_batch.shape)
                # cent_batch = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None]) #C x 256
                # print("clnt_cent:",cent_batch.shape)
                # clnt_cent = cent_batch/(torch.norm(cent_batch,dim=1,keepdim=True)+1e-9)
                # clnt_cent = torch.squeeze(clnt_cent)
                
                server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
                
                # print("server_cent:",server_cent.shape)
                
                # clnt_cent = cent_batch[unique_labels]/(torch.norm(cent_batch[unique_labels],dim=1,keepdim=True)+1e-9)
                # print('clnt shape',clnt_cent.shape)
                server_cent = server_cent/(1e-9+torch.norm(server_cent,dim=1,keepdim=True))
                # if torch.isnan(server_cent).any():
                #     print("Tensor contains NaN values.")
                # else:
                #     print("Tensor does not contain NaN values.")
                server_cent = server_cent.to(cfg['device'])
                server_cent = torch.transpose(server_cent,0,1)
                
                server_cent = (server_cent, cfg['device'])
                
                
                # print(type(server_cent))
                # print(type(clnt_cent))
                # print(clnt_cent)
                # print(server_cent)
                # if torch.isnan(clnt_cent).any():
                #     print("Tensor contains NaN values.")
                # else:
                #     print("Tensor does not contain NaN values.")
                
                similarity_mat = torch.matmul(clnt_cent,server_cent[0])
                ###############################################################
                # server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
                # server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
                # server_cent = torch.transpose(server_cent,0,1)
                # clnt_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
                # clnt_cent = torch.squeeze(clnt_cent)
                # print(server_cent)
                # server_cent = torch.tensor(server_cent)
                # # print("server_cent:",server_cent.shape)
                # # print("clnt_cent:",clnt_cent.shape)
                
                # similarity_mat = torch.matmul(clnt_cent.cpu(),server_cent)
                ################################################################
                temp = cfg['temp']
                similarity_mat = torch.exp(similarity_mat/temp)
                # print(similarity_mat)
                pos_m = torch.diag(similarity_mat)
                pos_neg_m = torch.sum(similarity_mat,axis = 1)
                nce_loss = -1.0*torch.sum(torch.log(pos_m/pos_neg_m))
                # nce_loss = -1.0*torch.mean(torch.log(pos_m/pos_neg_m))
                # print(nce_loss)
                # exit()
                loss += cfg['gamma']*nce_loss
                # cent = clnt_cent
                # loss = 0*loss+cfg['gamma']*nce_loss


        if cfg['kl'] == 1:
            # server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
            # server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
            # server_cent = torch.transpose(server_cent,0,1)
            cent_batch = torch.matmul(torch.transpose(psd_label,0,1), embed_feat)
            cent_batch = cent_batch / (1e-9 + psd_label.sum(axis=0)[:,None])
            clnt_cent = cent_batch/(1e-8+torch.norm(cent_batch,dim=1,keepdim=True))

            # clnt_cent = cent_batch[unique_labels]/torch.norm(cent_batch[unique_labels],dim=1,keepdim=True)
            # clnt_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
            clnt_cent = torch.log_softmax(clnt_cent,dim = 1)
            clnt_cent = torch.squeeze(clnt_cent)
            mean_p = torch.mean(clnt_cent,axis = 0)
            std_p = torch.std(clnt_cent,axis = 0)
            epsi = 1e-8
            kl_loss = torch.sum(-torch.log(std_p+epsi)/2+torch.square(std_p)/2+torch.square(mean_p)/2-0.5)
            # Compute the KL divergence analytically
            # kl = np.log(sigma) + (sigma**2 + mu**2 - 1) / 2 - mu
            print('kl loss',kl_loss)
            exit()
            loss+=cfg['kl_weight']*kl_loss


        if cfg['kl_loss'] == 1 and avg_cent is not None:
            server_cent = torch.squeeze(torch.Tensor(avg_cent.cpu()))
            server_cent = server_cent/torch.norm(server_cent,dim=1,keepdim=True)
            server_cent = torch.transpose(server_cent,0,1)
            clnt_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
            clnt_cent = torch.squeeze(clnt_cent)
            kl_loss = torch.nn.KLDivLoss(reduction="batchmean")
            # input should be a distribution in the log space
            print(clnt_cent.shape,server_cent.shape)
            server_cent = torch.transpose(server_cent,0,1)
            clnt_cent = clnt_cent.cpu()
            input_c = F.log_softmax(clnt_cent, dim=1)
            # Sample a batch of distributions. Usually this would come from the dataset
            target_s = F.log_softmax(server_cent, dim=1)
            print(input_c.shape,target_s.shape)
            # print('kl loss',kl_loss)
            # exit()
            loss+=cfg['kl_weight']*kl_loss(input_c, target_s)

        if cfg['add_fix'] ==1:
            # target_prob,target_= torch.max(dym_label, dim=-1)
            # target_ = dym_label
            target_ = hard_pseudo_label
            # print(target_.shape,x_s.shape)
            lable_s = torch.softmax(x_s,dim=1)
            target_ = target_[mask]
            lable_s = lable_s[mask]
            # print(target_.shape,lable_s.shape)
            if target_.shape[0] != 0 and lable_s.shape[0]!= 0 :
                # continue
                fix_loss = loss_fn(lable_s,target_.detach())
                # print(loss)
                loss+=cfg['lambda']*fix_loss
            # exit()
            ## tbd for testing
            # fix_loss_symmetric  = - torch.sum(torch.log(target_prob) * label_s, dim=1).mean() - torch.sum(torch.log(label_s) * target_prob, dim=1).mean()
            ##
            # fix_loss = loss_fn(x_s,target_.detach())
            
                # print('fix-loss',fix_loss)
                # exit()
            # fix_loss = loss_fn(label_s,target_.detach())
            # print(fix_loss)
        # print(loss)
        if cfg['logit_div']==1:
            # print('div logit')
            kl_loss = torch.nn.KLDivLoss(reduction="mean")
            init_pred_cls = init_pred_cls[mask] #no grad
            # print(pred_cls.shape,init_pred_cls.shape)
            
            # pred_cls = torch.log(pred_cls)
            # logit_div_loss = kl_loss(init_pred_cls,pred_cls)
            # print(torch.sum(init_pred_cls,dim=1))
            # print(torch.sum(pred_cls,dim=1))
            logit_div_loss = torch.mean(torch.sum(init_pred_cls*torch.log(init_pred_cls/pred_cls),dim=1))
            # logit_div_loss = kl_loss(pred_cls,init_pred_cls)
            # print(logit_div_loss)
            # exit()
            loss+=cfg['kl_weight']*logit_div_loss
            
                
        optimizer.zero_grad()
        loss.backward()
        # Check gradients
        # if cfg['avg_cent'] and avg_cent is not None:
        #     for name, param in model.named_parameters():
        #         print(f"Parameter: {name}, Gradient: {param.grad}")
        #     exit()
        if cfg['trk_grad']:
            with torch.no_grad():
                norm_type = 2
                total_norm = torch.norm(
                torch.stack([torch.norm(p.grad.detach(), norm_type) for p in model.parameters()]), norm_type)
                # for idx, param in enumerate(model.parameters()):
                #     grad_bank[f"layer_{idx}"] += param.grad
                #     avg_counter += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        # if fwd_pass==False:
        
        optimizer.step()
        if scheduler is not None:
            # print('scheduler step')
            scheduler.step()
            # print('lr at client',scheduler.get_last_lr())
        with torch.no_grad():
            loss_stack.append(loss.cpu().item())
            # print(loss.cpu().item())
            # print(loss_stack)
            # print(glob_multi_feat_cent.shape,embed_feat.shape)
            glob_multi_feat_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
    
    # cent = glob_multi_feat_cent.clone()  
    # print('cent:',cent.shape)
    # print(loss_stack)
    # exit()
    train_loss = np.mean(loss_stack)
    if cfg['trk_grad']:
        with torch.no_grad():
            for key in grad_bank:
                grad_bank[key] = grad_bank[key] / avg_counter
            with open(f"./output/gradients/{cfg['tag']}_{epoch}.json", "w") as outfile:
                json.dump(grad_bank, outfile)
    print('train_loss:',train_loss)
    # return train_loss,cent
    # del variables
    # print(torch.cuda.memory_summary(device=cfg['device']))
    gc.collect()
    torch.cuda.empty_cache()
    # print(torch.cuda.memory_summary(device=cfg['device']))
    # exit()
    return train_loss,clnt_cent,variance_clnt_cent

def save_model_state_dict(model_state_dict):
    # print(model_state_dict.keys())
    return {k: v.cpu() for k, v in model_state_dict.items()}


def save_optimizer_state_dict(optimizer_state_dict):
    optimizer_state_dict_ = {}
    for k, v in optimizer_state_dict.items():
        if k == 'state':
            optimizer_state_dict_[k] = to_device(optimizer_state_dict[k], 'cpu')
        else:
            optimizer_state_dict_[k] = copy.deepcopy(optimizer_state_dict[k])
    return optimizer_state_dict_

def baseline_loss(score, feat, batch_metrics, logits):
        if cfg['baseline_type'] == "IM":
            loss_ent, loss_div = self.IM_loss(score)
            # batch_metrics['loss']['ent'] = loss_ent.item()
            # batch_metrics['loss']['div'] = loss_div.item()
            return loss_ent * cfg['lambda_ent'] + loss_div * cfg['lambda_div']
        elif cfg['baseline_type'] == 'AaD':
            loss_aad_pos, loss_aad_neg = self.AaD_loss(score, feat)
            # batch_metrics['loss']['aad_pos'] = loss_aad_pos.item()
            # batch_metrics['loss']['aad_neg'] = loss_aad_neg.item()
            tmp_lambda = (1 + 10 * self.iteration / self.max_iters) ** (-self.beta)
            return (loss_aad_pos + loss_aad_neg * tmp_lambda) * cfg['lambda_aad']
        else:
            raise RuntimeError('wrong type of baseline')
