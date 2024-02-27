import copy
import datetime
import numpy as np
import sys
import time
import torch
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
from net_utils import init_multi_cent_psd_label,init_psd_label_shot_icml
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
            # print(self.model_state_dict.keys())
        else:
            self.model_state_dict = save_model_state_dict(model.state_dict())
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
        
            
    def distribute_cluster(self, client, batchnorm_dataset=None):
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
    def distribute_multi(self, client,epoch, batchnorm_dataset=None):
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
            for m in range(len(client)):
                # if client[m].active:
                print('distributing to all clients at epch 1')
                domain_id = client[m].domain_id
                # print(domain_id,self.model_state_dict.keys())
                # exit()
                model.load_state_dict(self.model_state_dict[domain_id])
                model_state_dict = save_model_state_dict(model.state_dict())
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                if cfg['avg_cent']:
                    if self.avg_cent is not None:
                        client[m].avg_cent = self.avg_cent
                    else:
                        client[m].avg_cent = None
                        print('Warning:server.avg_cent is None')
        else:
            for m in range(len(client)):
                if client[m].active:
                    domain_id = client[m].domain_id
                    # print(domain_id,self.model_state_dict.keys())
                    # exit()
                    model.load_state_dict(self.model_state_dict[domain_id])
                    model_state_dict = save_model_state_dict(model.state_dict())
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


class Client:
    def __init__(self, client_id, model, data_split=None):
        self.client_id = client_id
        self.data_split = data_split
        self.data_len = 1
        # print(len(data_split['train']))
        self.model_state_dict = save_model_state_dict(model.state_dict())
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
    def make_hard_pseudo_label(self, soft_pseudo_label):
        max_p, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        mask = max_p.ge(cfg['threshold'])
        return hard_pseudo_label, mask
    
    def make_dataset(self, dataset, metric, logger):
        if 'sup' in cfg['loss_mode'] or 'bmd' in cfg['loss_mode']:# or 'sim' in cfg['loss_mode']:
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

    def trainntune(self, dataset, lr, metric, logger,epoch,fwd_pass=False,CI_dataset=None,scheduler = None):
        
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
            # init_model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            # self.optimizer_state_dict['param_groups'][0]['lr'] = 0.001

            if cfg['model_name'] == 'resnet50' and cfg['par'] == 1:
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
                        v.requires_grad = False

                for k, v in model.feat_embed_layer.named_parameters():
                    # print(k)
                    param_group_ += [{'params': v, 'lr': cfg['local']['lr']}]
                for k, v in model.class_layer.named_parameters():
                    v.requires_grad = False
                    # param_group += [{'params': v, 'lr': cfg['local']['lr']}]

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
            # model.to(cfg["device"])
            # init_model.to(cfg["device"])
            model.train(True)
            # print(scheduler)
            # exit()
            # if cfg['world_size']==1:
            #     model.projection.requires_grad_(False)
            # if cfg['world_size']>1:
            #     model.module.projection.requires_grad_(False)
                
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(train_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
            # num_batches =None
            # for epoch in range(1, cfg['client']['num_epochs']+1 ):
            print(self.client_id,self.domain)
            if self.domain == 'webcam':
                num_local_epochs = cfg['tde']
            else:
                num_local_epochs = cfg['client']['num_epochs']
            if fwd_pass == True and cfg['cluster']:
                 num_local_epochs = 10                               #re 10
            #print(num_local_epochs)
            
            # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_data_loader), eta_min=0)
            # print(len(train_data_loader))
            # exit()
            for epoch in range(0, num_local_epochs ):
                output = {}
                # lr = optimizer.param_groups[0]['lr']
                # print(self.cent)
                # print(self.cent,self.avg_cent)
                # loss,cent = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent)
                if cfg['run_shot']:
                    loss,cent = shot_train(model,init_model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent)
                else:
                    # loss,cent = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent)
                    loss,cent,var_cent = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent,fwd_pass,scheduler)
                self.cent = cent.cpu()
                self.var_cent = var_cent.cpu()
                # print(self.cent)
                del cent
                del var_cent
                output['loss'] = loss
                    
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
        del optimizer
        # del optimizer_
        del model
        # del init_model
        gc.collect()
        torch.cuda.empty_cache()
        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        return


def shot_train(model,train_data_loader,test_data_loader,optimizer,epoch,cent,avg_cent):
    loss_stack = []
    num_classes = 31
    avg_label_feat = torch.zeros(num_classes,256)
    with torch.no_grad():
        model.eval()
        pred_label = init_psd_label_shot_icml(model,test_data_loader)
        #print("len dataloader:",len(dataloader))
        print("len pred_label:",len(pred_label))
    
    model.train()
    epoch_idx=epoch
    for i, input in enumerate(test_data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        if input_size<=1:
            break
        input['loss_mode'] = cfg['loss_mode']
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        psd_label = pred_label[input['id']]
        psd_label_ = pred_label[input['id']]

        if cfg['add_fix']==0:
            embed_feat, pred_cls = model(input)
        elif cfg['add_fix']==1:
            embed_feat, pred_cls,x_s = model(input)
        
        #act_loss = sum([item['mean_norm'] for item in list(model.act_stats.values())])
        #print("psd_label:",psd_label )
        
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
        unique_labels = torch.unique(psd_label_).cpu().numpy() 
        class_cent = torch.zeros(num_classes,embed_feat.shape[0])
        #batch_centers = torch.zeros(len(unique_labels).embed_feat.shape[1])
            

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
            print('nce_loss',nce_loss)
            loss += cfg['gamma']*nce_loss
            #print("reg_loss:",reg_loss,"ent_loss:",ent_loss,"psd_loss:",psd_loss,"nce_loss:",nce_loss)
        if cfg['add_fix'] ==1:
            target_prob,target_= torch.max(dym_label, dim=-1)
            label_s = torch.softmax(x_s,dim=1)
            fix_loss = loss_fn(x_s,target_.detach())
            # print(loss)
            loss+=fix_loss
     
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
    
    with torch.no_grad():
        loss_stack.append(loss.cpu().item())
        cent = get_final_centroids(model,test_data_loader,pred_label)
        #print("cent here:",cent.shape)
    
       
    train_loss = np.mean(loss_stack)

    return train_loss,cent




def bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,cent,avg_cent,fwd_pass=False,scheduler = None):
    loss_stack = []
    model.to(cfg["device"])
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
        all_psd_label = all_psd_label.cpu()
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
                
        input = to_device(input, 'cpu')
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
