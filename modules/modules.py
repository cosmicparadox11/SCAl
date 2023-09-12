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
from net_utils import init_multi_cent_psd_label
from net_utils import EMA_update_multi_feat_cent_with_feat_simi

class Server:
    def __init__(self, model):
        self.model_state_dict = save_model_state_dict(model.state_dict())
        self.avg_cent = None
        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        if 'fmatch' in cfg['loss_mode']:
            optimizer = make_optimizer(model.make_sigma_parameters(), 'local')
            global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
        else:
            optimizer = make_optimizer(model.parameters(), 'local')
            global_optimizer = make_optimizer(model.parameters(), 'global')
        self.optimizer_state_dict = save_optimizer_state_dict(optimizer.state_dict())
        self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
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
                    ll_div_weights[k].append(torch.exp(0.9*(num-den)))
                    # ll_div_weights[k].append(torch.exp(0.1*num/(den+1e-5)))
            #print("ll_div_weights[k]:",ll_div_weights[k])
            ll_div_weights[k] = torch.div(torch.Tensor(ll_div_weights[k]),sum(ll_div_weights[k]))
            # ll_div_weights[k] = torch.div(torch.Tensor(ll_div_weights[k]),sum(ll_div_weights[k]))
            #print("ll_div_weights[k]:",ll_div_weights[k])
         
        return ll_div_weights


    def distribute(self, client, batchnorm_dataset=None):
        # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        if cfg['world_size']==1:
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        elif cfg['world_size']>1:
            cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = eval('models.{}()'.format(cfg['model_name']))
            model = torch.nn.DataParallel(model,device_ids = [0, 1])
            model.to(cfg["device"])
        # model.load_state_dict(self.model_state_dict,strict= False)
        model.load_state_dict(self.model_state_dict)
        # if batchnorm_dataset is not None:
        #     model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
        model_state_dict = save_model_state_dict(model.state_dict())
        # model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        for m in range(len(client)):
            if client[m].active:
                client[m].model_state_dict = copy.deepcopy(model_state_dict)
                if cfg['avg_cent']:
                    if self.avg_cent is not None:
                        client[m].avg_cent = self.avg_cent
                    else:
                        client[m].avg_cent = None
                        print('Warning:server.avg_cent is None')
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
    # def update(self, client):
    #     if 'fmatch' not in cfg['loss_mode']:
    #         with torch.no_grad():
    #             valid_client = [client[i] for i in range(len(client)) if client[i].active]
    #             if len(valid_client) > 0:
    #                 # model = eval('models.{}()'.format(cfg['model_name']))
    #                 if cfg['world_size']==1:
    #                     model = eval('models.{}()'.format(cfg['model_name']))
    #                 elif cfg['world_size']>1:
    #                     cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #                     model = eval('models.{}()'.format(cfg['model_name']))
    #                     model = torch.nn.DataParallel(model,device_ids = [0, 1])
    #                     model.to(cfg["device"])
    #                 # model.load_state_dict(self.model_state_dict)
    #                 model.load_state_dict(self.model_state_dict)
    #                 global_optimizer = make_optimizer(model.parameters(), 'global')
    #                 global_optimizer.load_state_dict(self.global_optimizer_state_dict)
    #                 global_optimizer.zero_grad()
    #                 weight = torch.ones(len(valid_client))
    #                 weight = weight / weight.sum()
    #                 for k, v in model.named_parameters():
    #                     # print(k)
    #                     parameter_type = k.split('.')[-1]
    #                     # print(f'{k} with parameter type {parameter_type}')
    #                     if 'weight' in parameter_type or 'bias' in parameter_type:
    #                         tmp_v = v.data.new_zeros(v.size())
    #                         for m in range(len(valid_client)):
    #                             if cfg['world_size']==1:
    #                                 tmp_v += weight[m] * valid_client[m].model_state_dict[k]
    #                             elif  cfg['world_size']>1:
    #                                 tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
    #                         v.grad = (v.data - tmp_v).detach()
    #                 # module = model.layer1[0].n1
    #                 # print(list(module.named_buffers()))
    #                 global_optimizer.step()
    #                 self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
    #                 self.model_state_dict = save_model_state_dict(model.state_dict())
    #                 # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
    #     elif 'fmatch' in cfg['loss_mode']:
    #         with torch.no_grad():
    #             valid_client = [client[i] for i in range(len(client)) if client[i].active]
    #             if len(valid_client) > 0:
    #                 model = eval('models.{}()'.format(cfg['model_name']))
    #                 model.load_state_dict(self.model_state_dict)
    #                 global_optimizer = make_optimizer(model.make_phi_parameters(), 'global')
    #                 global_optimizer.load_state_dict(self.global_optimizer_state_dict)
    #                 global_optimizer.zero_grad()
    #                 weight = torch.ones(len(valid_client))
    #                 weight = weight / weight.sum()
    #                 for k, v in model.named_parameters():
    #                     parameter_type = k.split('.')[-1]
    #                     if 'weight' in parameter_type or 'bias' in parameter_type:
    #                         tmp_v = v.data.new_zeros(v.size())
    #                         for m in range(len(valid_client)):
    #                             tmp_v += weight[m] * valid_client[m].model_state_dict[k]
    #                         v.grad = (v.data - tmp_v).detach()
    #                 global_optimizer.step()
    #                 self.global_optimizer_state_dict = save_optimizer_state_dict(global_optimizer.state_dict())
    #                 # self.model_state_dict = save_model_state_dict(model.state_dict())
    #                 self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
    def update(self, client):
        if ('fmatch' not in cfg['loss_mode'] and cfg['adapt_wt'] == 0):
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
                    weight = weight / weight.sum()

                    # # Store the averaged batchnorm parameters
                    # bn_parameters = {k: None for k, v in model.named_parameters() if isinstance(v, torch.nn.BatchNorm2d)}

                    for k, v in model.named_parameters():
                        # print(k)
                        isBatchNorm = isinstance(v, torch.nn.BatchNorm2d)
                        parameter_type = k.split('.')[-1]
                        # print(f'{k} with parameter type {parameter_type}')
                        if 'weight' in parameter_type or 'bias' in parameter_type:
                            tmp_v = v.data.new_zeros(v.size())
                            for m in range(len(valid_client)):
                                if cfg['world_size']==1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                                elif  cfg['world_size']>1:
                                    tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                            v.grad = (v.data - tmp_v).detach()

                    # for k, v in model.named_buffers():
                    #     # print(k)
                    #     parameter_type = k.split('.')[-1]
                    #     # print(f'{k} with parameter type {parameter_type}')
                    #     if 'running_mean' in parameter_type or 'running_mean' in parameter_type:
                    #         # print(k)
                    #         tmp_v = v.data.new_zeros(v.size())
                    #         for m in range(len(valid_client)):
                    #             if cfg['world_size']==1:
                    #                 tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                    #             elif  cfg['world_size']>1:
                    #                 tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                    #         v.grad = (v.data - tmp_v).detach()
                    #         v.data = v.data-1*v.grad
                        # else :
                        #     print(k)
                        #     tmp_v = v.data.new_zeros(v.size())
                        #     for m in range(len(valid_client)):
                        #         if cfg['world_size']==1:
                        #             tmp_v += valid_client[m].model_state_dict[k]
                        #         elif  cfg['world_size']>1:
                        #             tmp_v += valid_client[m].model_state_dict[k].to(cfg["device"])
                        #     # print(tmp_v)
                        #     v.data = (tmp_v//len(valid_client)).detach()


                        # isBatchNorm = isinstance(v, torch.nn.BatchNorm2d)
                        # parameter_type = k.split('.')[-1]
                        # # print(f'{k} with parameter type {parameter_type}')
                        # if 'weight' in parameter_type or 'bias' in parameter_type:
                        #     tmp_v = v.data.new_zeros(v.size())
                        #     for m in range(len(valid_client)):
                        #         if cfg['world_size']==1:
                        #             tmp_v += weight[m] * valid_client[m].model_state_dict[k]
                        #         elif  cfg['world_size']>1:
                        #             tmp_v += weight[m] * valid_client[m].model_state_dict[k].to(cfg["device"])
                        #     v.grad = (v.data - tmp_v).detach()

                        # elif isBatchNorm:
                        #     print(k)
                        #     # Accumulate BatchNorm parameters for averaging
                        #     if bn_parameters[k] is None:
                        #         bn_parameters[k] = weight[0] * valid_client[0].model_state_dict[k].clone()
                        #     for m in range(1, len(valid_client)):
                        #         bn_parameters[k] += weight[m] * valid_client[m].model_state_dict[k].clone()

                    # module = model.layer1[0].n1
                    # print(list(module.named_buffers()))
                    global_optimizer.step()

                    # # Update the averaged batchnorm parameters back to the server model
                    # for k, v in model.named_parameters():
                    #     if isinstance(v, torch.nn.BatchNorm2d):
                    #         # print(bn_parameters[k])

                    #         v.data = bn_parameters[k]

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
                        self.avg_cent=client[i].cent
                        count+=1
                    else:
                        self.avg_cent+=client[i].cent
                elif client[i].active == True  and client[i].cent is None:
                    print('Warning:client centntroid is None')
            if self.avg_cent is not None:
                self.avg_cent=self.avg_cent/len(client)

        for i in range(len(client)):
            client[i].active = False
        return

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
        # print(len(data_split['train']))
        self.model_state_dict = save_model_state_dict(model.state_dict())
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
        self.cent = None
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
        # elif 'fix' in cfg['loss_mode']:
        #     with torch.no_grad():
        #         data_loader = make_data_loader({'train': dataset}, 'global', shuffle={'train': False})['train']
        #         model = eval('models.{}(track=True).to(cfg["device"])'.format(cfg['model_name']))
        #         # model = eval('models.{}()'.format(cfg['model_name']))
        #         # model = torch.nn.DataParallel(model,device_ids = [0, 1])
        #         # model.to(cfg["device"])
        #         model.load_state_dict(self.model_state_dict,strict=False)
        #         model.train(False)
        #         output = []
        #         target = []
        #         for i, input in enumerate(data_loader):
        #             input = collate(input)
        #             input = to_device(input, cfg['device'])
        #             output_ = model(input)
        #             output_i = output_['target']
        #             target_i = input['target']
        #             output.append(output_i.cpu())
        #             target.append(target_i.cpu())
        #         output_, input_ = {}, {}
        #         output_['target'] = torch.cat(output, dim=0)
        #         input_['target'] = torch.cat(target, dim=0)
        #         evaluation = metric.evaluate(['PAccuracy'], input_, output_)
        #         output_['target'] = F.softmax(output_['target'], dim=-1)
        #         new_target, mask = self.make_hard_pseudo_label(output_['target'])
        #         output_['mask'] = mask
        #         evaluation = metric.evaluate(['MAccuracy', 'LabelRatio'], input_, output_)
        #         logger.append(evaluation, 'train', n=len(input_['target']))
        #         if torch.any(mask):
        #             fix_dataset = copy.deepcopy(dataset)
        #             fix_dataset.target = new_target.tolist()
        #             mask = mask.tolist()
        #             fix_dataset.data = list(compress(fix_dataset.data, mask))
        #             fix_dataset.target = list(compress(fix_dataset.target, mask))
        #             fix_dataset.other = {'id': list(range(len(fix_dataset.data)))}
        #             if 'mix' in cfg['loss_mode']:
        #                 mix_dataset = copy.deepcopy(dataset)
        #                 mix_dataset.target = new_target.tolist()
        #                 mix_dataset = MixDataset(len(fix_dataset), mix_dataset)
        #             else:
        #                 mix_dataset = None
        #             return fix_dataset, mix_dataset
        #         else:
        #             return None
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
        else:
            raise ValueError('Not valid client loss mode')

    def train(self, dataset, lr, metric, logger):
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
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
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
            model.train(True)
            if cfg['client']['num_epochs'] == 1:
                num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
            else:
                num_batches = None
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
    def trainntune(self, dataset, lr, metric, logger,epoch,CI_dataset=None):
        
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
        # elif 'fix' in cfg['loss_mode'] and 'mix' not in cfg['loss_mode']:
        #     # _,_,dataset = dataset
        #     # data_loader = make_data_loader({'train': dataset}, 'client')['train']
        #     fix_dataset, _ ,_ = dataset
        #     fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
        #     # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        #     if cfg['world_size']==1:
        #         model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        #     elif cfg['world_size']>1:
        #         cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #         model = eval('models.{}()'.format(cfg['model_name']))
        #         model = torch.nn.DataParallel(model,device_ids = [0, 1])
        #         model.to(cfg["device"])
        #     model.load_state_dict(self.model_state_dict, strict=False)
        #     self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        #     optimizer = make_optimizer(model.parameters(), 'local')
        #     optimizer.load_state_dict(self.optimizer_state_dict)
        #     model.train(True)
        #     if cfg['world_size']==1:
        #         model.projection.requires_grad_(False)
        #     if cfg['world_size']>1:
        #         model.module.projection.requires_grad_(False)
        #     g_epoch = epoch
        #     # for v,k in model.named_parameters():
        #     #     print(f'nmae{v} grad required{k.requires_grad}')
        #     # if self.supervised == False:
        #     #     model.linear.requires_grad_(False)
        #     if cfg['world_size']==1:
        #         if 'ft' in cfg['loss_mode'] and 'bl' not in cfg['loss_mode']:
        #             if epoch <= cfg['switch_epoch'] :
        #                 model.linear.requires_grad_(False)
        #             elif epoch > cfg['switch_epoch']:
        #                 model.projection.requires_grad_(False)
        #         elif 'ft' in cfg['loss_mode'] and 'bl'  in cfg['loss_mode']:
        #             if epoch > cfg['switch_epoch'] :
        #                 model.linear.requires_grad_(False)
        #             elif epoch <= cfg['switch_epoch']:
        #                 model.projection.requires_grad_(False)
        #         elif 'at' in cfg['loss_mode']:
        #             if cfg['srange'][0]<=epoch<=cfg['srange'][1] or cfg['srange'][2]<=epoch<=cfg['srange'][3] or cfg['srange'][4]<=epoch<=cfg['srange'][5] or cfg['srange'][6]<=epoch<=cfg['srange'][7]:
        #                 model.projection.requires_grad_(False)
        #             else:
        #                 model.linear.requires_grad_(False)
        #     elif cfg['world_size']>1:
        #         if 'ft' in cfg['loss_mode'] and 'bl' not in cfg['loss_mode']:
        #             if epoch <= cfg['switch_epoch'] :
        #                 model.module.linear.requires_grad_(False)
        #             elif epoch > cfg['switch_epoch']:
        #                 model.module.projection.requires_grad_(False)
        #         elif 'ft' in cfg['loss_mode'] and 'bl'  in cfg['loss_mode']:
        #             if epoch > cfg['switch_epoch'] :
        #                 model.module.linear.requires_grad_(False)
        #             elif epoch <= cfg['switch_epoch']:
        #                 model.module.projection.requires_grad_(False)
        #         elif 'at' in cfg['loss_mode']:
        #             if cfg['srange'][0]<=epoch<=cfg['srange'][1] or cfg['srange'][2]<=epoch<=cfg['srange'][3] or cfg['srange'][4]<=epoch<=cfg['srange'][5] or cfg['srange'][6]<=epoch<=cfg['srange'][7]:
        #                 model.module.projection.requires_grad_(False)
        #             else:
        #                 model.module.linear.requires_grad_(False)
        #     # if 'ft' in cfg['loss_mode']:
        #     #     if epoch > cfg['switch_epoch'] :
        #     #         model.linear.requires_grad_(False)
        #     #     elif epoch <= cfg['switch_epoch']:
        #     #         model.projection.requires_grad_(False)
        #     # elif 'at' in cfg['loss_mode']:
        #     #     if epoch==21 or epoch==42 or epoch==63 or epoch==84 or 100<epoch<=105:
        #     #         model.projection.requires_grad_(False)
        #     #     else:
        #     #         model.linear.requires_grad_(False)
                    
        #     # for v,k in model.named_parameters():
        #     #     print(f'nmae{v} grad required{k.requires_grad}')
        #     if cfg['client']['num_epochs'] == 1:
        #         num_batches = int(np.ceil(len(data_loader) * float(cfg['local_epoch'][0])))
        #     else:
        #         num_batches = None
        #     for epoch in range(1, cfg['client']['num_epochs'] + 1):
        #         for i, input in enumerate(fix_data_loader):
        #             input = collate(input)
        #             input_size = input['data'].size(0)
        #             input['loss_mode'] = cfg['loss_mode']
        #             input = to_device(input, cfg['device'])
        #             input['supervised_mode'] = self.supervised
        #             input['batch_size'] = cfg['client']['batch_size']['train']
        #             input['epoch'] = g_epoch
        #             optimizer.zero_grad()
        #             output = model(input)
        #             # print(output.keys())
        #             output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        #             output['loss'].backward()
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #             optimizer.step()
        #             # evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        #             evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
        #             logger.append(evaluation, 'train', n=input_size)
        #             if num_batches is not None and i == num_batches - 1:
        #                 break
        # elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and CI_dataset is not None:
        #     fix_dataset, mix_dataset,_ = dataset
        #     fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
        #     mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
        #     # print(mix_data_loader)
        #     ci_data_loader = make_data_loader({'train':CI_dataset},'client')['train']
        #     # print(ci_data_loader)
        #     model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        #     model.load_state_dict(self.model_state_dict, strict=False)
        #     self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        #     optimizer = make_optimizer(model.parameters(), 'local')
        #     optimizer.load_state_dict(self.optimizer_state_dict)
        #     model.train(True)
        #     if cfg['world_size']==1:
        #         model.projection.requires_grad_(False)
        #     if cfg['world_size']>1:
        #         model.module.projection.requires_grad_(False)
        #     if cfg['client']['num_epochs'] == 1:
        #         num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
        #     else:
        #         num_batches = None
            
        #     for epoch in range(1, cfg['client']['num_epochs'] + 1):
        #         for i, (fix_input,mix_input,ci_input) in enumerate(zip(fix_data_loader, mix_data_loader,ci_data_loader)):
        #             # input = {'data': fix_input['aug'], 'target': fix_input['target'], 'aug': fix_input['aug'],
        #             #          'mix_data': mix_input['aug'], 'mix_target': mix_input['target']}
        #             # input = {'data': fix_input['data'], 'augw':fix_input['augw'], 'target': fix_input['target'], 'augs': fix_input['augs'],
        #             #          'mix_data': mix_input['augw'], 'mix_target': mix_input['target']}
        #             input = {'data': fix_input['augw'], 'target': fix_input['target'], 'aug': fix_input['augs'],
        #                     'mix_data': mix_input['augs'], 'mix_target': mix_input['target'],'ci_data':ci_input['data'],'ci_target':ci_input['target']}
                    
        #             input = collate(input)
        #             # print(len(ci_input['data']))
        #             input_size = input['data'].size(0)
        #             input['lam'] = self.beta.sample()[0]
        #             input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
        #             input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
        #             input['loss_mode'] = cfg['loss_mode']
        #             input = to_device(input, cfg['device'])
        #             optimizer.zero_grad()
        #             aug_output,mix_output = model(input)
        #             output['loss']  = self.EL_loss(input['id'].detach().tolist(),aug_output, input['target'].detach())
        #             output['loss'] += input['lam'] * self.EL_loss(input['id'].detach(),mix_output, input['mix_target'][:, 0].detach()) + (
        #                     1 - input['lam']) * self.EL_loss(input['id'].detach(),mix_output, input['mix_target'][:, 1].detach())
        #             output['loss'].backward()
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #             optimizer.step()
        #             evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
        #             logger.append(evaluation, 'train', n=input_size)
        #             if num_batches is not None and i == num_batches - 1:
        #                 break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and CI_dataset is not None:
            fix_dataset, mix_dataset,_ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            # print(mix_data_loader)
            ci_data_loader = make_data_loader({'train':CI_dataset},'client')['train']
            # print(ci_data_loader)
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr
            optimizer = make_optimizer(model.parameters(), 'local')
            optimizer.load_state_dict(self.optimizer_state_dict)
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
        # elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and CI_dataset is not None:
        #     fix_dataset, mix_dataset,_ = dataset
        #     fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
        #     mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
        #     # print(mix_data_loader)
        #     ci_data_loader = make_data_loader({'train':CI_dataset},'client')['train']
        #     # print(ci_data_loader)
        #     model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        #     model.load_state_dict(self.model_state_dict, strict=False)
        #     self.optimizer_state_dict['param_groups'][0]['lr'] = lr
        #     optimizer = make_optimizer(model.parameters(), 'local')
        #     optimizer.load_state_dict(self.optimizer_state_dict)
        #     model.train(True)
        #     if cfg['world_size']==1:
        #         model.projection.requires_grad_(False)
        #     if cfg['world_size']>1:
        #         model.module.projection.requires_grad_(False)
        #     if cfg['client']['num_epochs'] == 1:
        #         num_batches = int(np.ceil(len(fix_data_loader) * float(cfg['local_epoch'][0])))
        #     else:
        #         num_batches = None
            
        #     for epoch in range(1, cfg['client']['num_epochs'] + 1):
        #         for i, (fix_input,mix_input,ci_input) in enumerate(zip(fix_data_loader, mix_data_loader,ci_data_loader)):
        #             # input = {'data': fix_input['aug'], 'target': fix_input['target'], 'aug': fix_input['aug'],
        #             #          'mix_data': mix_input['aug'], 'mix_target': mix_input['target']}
        #             # input = {'data': fix_input['data'], 'augw':fix_input['augw'], 'target': fix_input['target'], 'augs': fix_input['augs'],
        #             #          'mix_data': mix_input['augw'], 'mix_target': mix_input['target']}
        #             input = {'data': fix_input['augw'], 'target': fix_input['target'], 'aug': fix_input['augs'],
        #                     'mix_data': mix_input['augs'], 'mix_target': mix_input['target'],'ci_data':ci_input['data'],'ci_target':ci_input['target']}
                    
        #             input = collate(input)
        #             # print(len(ci_input['data']))
        #             input_size = input['data'].size(0)
        #             input['lam'] = self.beta.sample()[0]
        #             input['mix_data'] = (input['lam'] * input['data'] + (1 - input['lam']) * input['mix_data']).detach()
        #             input['mix_target'] = torch.stack([input['target'], input['mix_target']], dim=-1)
        #             input['loss_mode'] = cfg['loss_mode']
        #             input = to_device(input, cfg['device'])
        #             optimizer.zero_grad()
        #             output = model(input)
        #             output['loss'].backward()
        #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        #             optimizer.step()
        #             evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
        #             logger.append(evaluation, 'train', n=input_size)
        #             if num_batches is not None and i == num_batches - 1:
                        break
        
        elif 'bmd' in cfg['loss_mode']:
            _,_,dataset = dataset
            train_data_loader = make_data_loader({'train': dataset}, 'client')['train']
            test_data_loader = make_data_loader({'train': dataset},'client',batch_size = {'train':50},shuffle={'train':False})['train']
            # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            if cfg['world_size']==1:
                model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            elif cfg['world_size']>1:
                cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = eval('models.{}()'.format(cfg['model_name']))
                model = torch.nn.DataParallel(model,device_ids = [0, 1])
                model.to(cfg["device"])
            # model.load_state_dict(self.model_state_dict, strict=False)
            model.load_state_dict(self.model_state_dict)
            self.optimizer_state_dict['param_groups'][0]['lr'] = lr

            if cfg['model_name'] == 'SFDA':
                cfg['local']['lr'] = lr
                param_group = []
                for k, v in model.backbone_layer.named_parameters():
                    # print(k)
                    if "bn" in k:
                        param_group += [{'params': v, 'lr': cfg['local']['lr']*0.1}]
                    else:
                        v.requires_grad = False

                for k, v in model.feat_embed_layer.named_parameters():
                    # print(k)
                    param_group += [{'params': v, 'lr': cfg['local']['lr']}]
                for k, v in model.class_layer.named_parameters():
                    v.requires_grad = False

                optimizer = make_optimizer(param_group, 'local')
                optimizer = op_copy(optimizer)

            # elif cfg['model_name']=='resnet9':
            #     cfg['local']['lr'] = lr
            #     # print(model)
            #     param_group = []
            #     for k,v in model.named_parameters():
            #         # print(k)
            #         if 'n1' in k or 'n2' in k or 'n4' in k or 'bn' in k:
            #             # print(k)
            #             param_group += [{'params': v, 'lr': cfg['local']['lr']*0.1}]
            #         elif 'feat_embed_layer' in k:
            #             print(k)
            #             # v.requires_grad = False
            #             param_group += [{'params': v, 'lr': cfg['local']['lr']}]
            #         else :
            #             print('grad false',k)
            #             v.requires_grad = False
            #     # for k, v in model.feat_embed_layer.named_parameters():
            #     #     # print(k)
            #     #     if 'n1' not in k or 'n2' not in k or 'n4' not in k or 'bn' not in k:
            #     #         print(k)
            #     #         param_group += [{'params': v, 'lr': cfg['local']['lr']}]

            #     # for k, v in model.class_layer.named_parameters():
            #     #     v.requires_grad = False
            #     optimizer = make_optimizer(param_group, 'local')
            #     optimizer = op_copy(optimizer)
                # exit()
            else:
                # print('not freezing')
                optimizer = make_optimizer(model.parameters(), 'local')
                optimizer.load_state_dict(self.optimizer_state_dict)
            # print(model)
            model.train(True)
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
            print(num_local_epochs)
            for epoch in range(0, num_local_epochs ):
                # data_loader = make_data_loader({'train': dataset}, 'client')['train']
                # with torch.no_grad():
                #     model.eval()
                #     print("update psd label bank!")
                #     glob_multi_feat_cent, all_psd_label = init_multi_cent_psd_label(model,data_loader)
                    
                #     model.train()
                # epoch_idx=epoch
                # for i, input in enumerate(data_loader):
                #     # print(i)
                #     input = collate(input)
                #     input_size = input['data'].size(0)
                #     input['loss_mode'] = cfg['loss_mode']
                #     input = to_device(input, cfg['device'])
                #     optimizer.zero_grad()
                #     # iter_idx += 1
                #     # imgs_train = imgs_train.cuda()
                #     # imgs_idx = imgs_idx.cuda() 
                    
                #     psd_label = all_psd_label[input['id']]
                    
                #     embed_feat, pred_cls = model(input)
                    
                #     if pred_cls.shape != psd_label.shape:
                #         # psd_label is not one-hot like.
                #         psd_label = torch.zeros_like(pred_cls).scatter(1, psd_label.unsqueeze(1), 1)
                    
                #     mean_pred_cls = torch.mean(pred_cls, dim=0, keepdim=True) #[1, C]
                #     reg_loss = - torch.sum(torch.log(mean_pred_cls) * mean_pred_cls)
                #     ent_loss = - torch.sum(torch.log(pred_cls) * pred_cls, dim=1).mean()
                #     psd_loss = - torch.sum(torch.log(pred_cls) * psd_label, dim=1).mean()
                    
                #     if epoch_idx >= 1.0:
                #         loss = ent_loss + 2.0 * psd_loss
                #     else:
                #         loss = - reg_loss + ent_loss
                    
                #     #==================================================================#
                #     # SOFT FEAT SIMI LOSS
                #     #==================================================================#
                #     normed_emd_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
                #     dym_feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_emd_feat)
                #     dym_feat_simi, _ = torch.max(dym_feat_simi, dim=2) #[N, C]
                #     dym_label = torch.softmax(dym_feat_simi, dim=1)    #[N, C]
                    
                #     dym_psd_loss = - torch.sum(torch.log(pred_cls) * dym_label, dim=1).mean() - torch.sum(torch.log(dym_label) * pred_cls, dim=1).mean()
                    
                #     if epoch_idx >= 1.0:
                #         loss += 0.5 * dym_psd_loss
                #     #==================================================================#
                #     #==================================================================#
                #     # lr_scheduler(optimizer, iter_idx, iter_max)
                #     # optimizer.zero_grad()
                #     loss.backward()
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                #     optimizer.step()
                #     with torch.no_grad():
                #         # loss_stack.append(loss.cpu().item())
                #         glob_multi_feat_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
                #     # output = model(input)
                #     # print(output.keys())
                    output = {}
                    # print(self.cent)
                    loss,cent = bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,self.cent,self.avg_cent)
                    self.cent = cent
                    output['loss'] = loss
                    # output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    # output['loss'].backward()
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                    # optimizer.step()
                    # evaluation = metric.evaluate(metric.metric_name['train'], input, output)
                    # evaluation = metric.evaluate(['Loss', 'Accuracy'], input, output)
                    # logger.append(evaluation, 'train', n=input_size)
                    # if num_batches is not None and i == num_batches - 1:
                    #     break
        elif 'fix' in cfg['loss_mode'] and 'mix' in cfg['loss_mode'] and CI_dataset is  None:
            fix_dataset, mix_dataset,_ = dataset
            fix_data_loader = make_data_loader({'train': fix_dataset}, 'client')['train']
            mix_data_loader = make_data_loader({'train': mix_dataset}, 'client')['train']
            model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
            model.load_state_dict(self.model_state_dict, strict=False)
            
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
        # self.model_state_dict = save_model_state_dict(model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict())
        return
def bmd_train(model,train_data_loader,test_data_loader,optimizer,epoch,cent,avg_cent):
    loss_stack = []
    with torch.no_grad():
        model.eval()
        # print("update psd label bank!")
        glob_multi_feat_cent, all_psd_label = init_multi_cent_psd_label(model,test_data_loader)
        # print(glob_multi_feat_cent.squeeze().shape)
        # print(all_psd_label.shape)
        # if epoch%2==0:
        #     print(glob_multi_feat_cent.shape)
        #     num_cent = glob_multi_feat_cent.shape[1]
        #     temp = []
        #     from sklearn.manifold import TSNE
        #     import matplotlib.pyplot as plt
        #     from matplotlib import cm
        #     tsne = TSNE(2,perplexity = num_cent, verbose=1)
        #     # for i in range(glob_multi_feat_cent.shape[0]):
        #     #     tsne_proj = tsne.fit_transform(glob_multi_feat_cent[i].cpu()) 
        #     #     temp.append(tsne_proj)
        #     k_ = glob_multi_feat_cent.cpu().reshape(-1,512)
        #     print(k_.shape)
        #     tsne_proj = tsne.fit_transform(k_) 
        #     print(f'shape value{tsne_proj.shape}')
        #     cmap = cm.get_cmap('tab20')
        #     fig, ax = plt.subplots(figsize=(8,8))
        #     l=0
        #     for i in range(tsne_proj.shape[0]//num_cent):
        #         indices = slice(l,l+num_cent)
        #         l+=num_cent
        #         print(indices)
        #         ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(i)).reshape(1,4), label = i ,alpha=0.5)
        #     ax.legend(fontsize='large', markerscale=2)
        #     plt.show()
        #     exit()
    model.train()
    cent = glob_multi_feat_cent
    epoch_idx=epoch
    for i, input in enumerate(train_data_loader):
        # print(i)
        input = collate(input)
        input_size = input['data'].size(0)
        if input_size<=1:
            break
        input['loss_mode'] = cfg['loss_mode']
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        # iter_idx += 1
        # imgs_train = imgs_train.cuda()
        # imgs_idx = imgs_idx.cuda() 
        
        psd_label = all_psd_label[input['id']]
        # print(input)
        embed_feat, pred_cls = model(input)
        
        if pred_cls.shape != psd_label.shape:
            # psd_label is not one-hot like.
            psd_label = torch.zeros_like(pred_cls).scatter(1, psd_label.unsqueeze(1), 1)
        
        mean_pred_cls = torch.mean(pred_cls, dim=0, keepdim=True) #[1, C]
        reg_loss = - torch.sum(torch.log(mean_pred_cls) * mean_pred_cls)
        ent_loss = - torch.sum(torch.log(pred_cls) * pred_cls, dim=1).mean()
        psd_loss = - torch.sum(torch.log(pred_cls) * psd_label, dim=1).mean()
        # print(epoch_idx)
        # if epoch_idx >= 1.0:
        #     loss = 2.0 * psd_loss
        #     # loss = ent_loss + 1.0 * psd_loss
        # else:
        #     loss = - reg_loss + ent_loss
        # print(loss)
        #==================================================================#
        # SOFT FEAT SIMI LOSS
        #==================================================================#
        normed_emd_feat = embed_feat / torch.norm(embed_feat, p=2, dim=1, keepdim=True)
        dym_feat_simi = torch.einsum("cmd, nd -> ncm", glob_multi_feat_cent, normed_emd_feat)
        dym_feat_simi, _ = torch.max(dym_feat_simi, dim=2) #[N, C]
        dym_label = torch.softmax(dym_feat_simi, dim=1)    #[N, C]
        dym_psd_loss = - torch.sum(torch.log(pred_cls) * dym_label, dim=1).mean() - torch.sum(torch.log(dym_label) * pred_cls, dim=1).mean()
        
        # if epoch_idx >= 1.0:
        #     loss += 0.5 * dym_psd_loss

        #==================================================================#
        loss = ent_loss + 1* psd_loss + 0.1 * dym_psd_loss - reg_loss
        #==================================================================#
        #==================================================================#
        #==================================================================#
        # lr_scheduler(optimizer, iter_idx, iter_max)
        # optimizer.zero_grad()
        #==================================================================#
        # print(cent.shape,avg_cent.shape)
        if cfg['avg_cent'] and avg_cent is not None:
            dist=0
            # print(avg_cent.shape,cent.shape)
            for avg_ci,ci in zip(avg_cent.squeeze(),cent.squeeze()):
                # print(avg_ci.shape,ci.shape)
                # dist += np.sqrt(np.sum((avg_ci-ci)**2,axis=0))
                dist+=torch.norm((avg_ci.detach().reshape(-1) - ci.detach().reshape(-1)),0.9)
            loss += cfg['avg_cent_alpha'] * dist/avg_cent.shape[0]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            loss_stack.append(loss.cpu().item())
            glob_multi_feat_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
        # output = model(input)
        # print(output.keys())
        cent = glob_multi_feat_cent
    train_loss = np.mean(loss_stack)

    return train_loss,cent
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
