import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
# import numpy as np
import matplotlib.pyplot as plt
import random
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset,separate_dataset_DA, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats , split_class_dataset,split_class_dataset_DA,make_data_loader_DA,make_batchnorm_stats_DA,fetch_dataset_full_test
from metrics import Metric
from modules import Server, Client
from utils_eigenplot import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate,resume_DA,process_dataset_multi
from logger import make_logger
import gc
from pyhessian import hessian # Hessian computation
from density_plot import get_esd_plot # ESD plot
from itertools import islice


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    if k == 'control_name':
        continue
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
# args['contral_name']
args = vars(parser.parse_args())
process_args(args)


def main():

    

    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    cfg['unsup_list'] = cfg['unsup_doms'].split('-')
    print(cfg['unsup_list'])
    exp_num = cfg['control_name'].split('_')[0]
    if cfg['domain_s'] in ['amazon','dslr','webcam']:
        cfg['data_name'] = 'office31'
    elif cfg['domain_s'] in ['art', 'clipart','product','realworld']:
        cfg['data_name'] = 'OfficeHome'
    elif cfg['domain_s'] in ['MNIST','SVHN','USPS']:
        cfg['data_name'] = cfg['domain_s']
    for i in range(cfg['num_experiments']):
        cfg['domain_tag'] = '_'.join([x for x in cfg['unsup_list'] if x])
        model_tag_list = [str(seeds[i]), cfg['domain_s'],'to',cfg['domain_tag'], cfg['model_name'],exp_num]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    print('cfg:',cfg)
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    seed_val =  cfg['seed']
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_val)
    random.seed(seed_val)
    torch.cuda.empty_cache()
    # print(cfg['gm'])
    #server_dataset = fetch_dataset(cfg['data_name'])
    print('supervised source data name:',cfg['data_name'])
    client_dataset_sup = fetch_dataset(cfg['data_name'],domain=cfg['domain_s'])
    # print(cfg['data_name'])
    # print(client_dataset_sup)
    # exit()
    # client_dataset_unsup = fetch_dataset(cfg['data_name_unsup'],domain=cfg['domain_u'])
    #############
    print('list of un supervised domain',cfg['unsup_list'])
    # print(cfg['unsup_list'])
    client_dataset_unsup = {}
    for i ,domain in enumerate(cfg['unsup_list']):
        # print(i,domain)
        print('fetching unsupervised domains')
        if domain in ['MNIST','USPS','SVHN','MNIST_M', 'SYN32']:
            cfg['data_name_unsup'] = domain
            client_dataset_unsup[i] = fetch_dataset(cfg['data_name_unsup'])
        elif domain in ['dslr','webcam','amazon']:
            cfg['data_name_sup'] = 'office31'
            # client_dataset_unsup[i] = fetch_dataset(cfg['data_name_unsup'],domain=domain)
            client_dataset_unsup[i] = fetch_dataset_full_test(cfg['data_name_unsup'],domain=domain)
        elif domain in ['art','clipart','product','realworld']:
            cfg['data_name_unsup'] = 'OfficeHome'
            cfg['data_name_sup'] = 'OfficeHome'
            client_dataset_unsup[i] = fetch_dataset_full_test(cfg['data_name_unsup'],domain=domain)
            # client_dataset_unsup[i] = fetch_dataset(cfg['data_name_unsup'],domain=domain)
            # print(client_dataset_unsup[i])
    ##############
    # exit()
    # print(client_dataset_unsup.keys())
    # print(len(server_dataset['train'].data))
    # print(len(client_dataset['train'].data))
    # for i in range(2):
    #     print(server_dataset['train'][i])


    # process_dataset(client_dataset_sup,client_dataset_unsup)
    ####
    process_dataset_multi(client_dataset_sup,client_dataset_unsup)
    ####
    #server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
                                                                                        #    client_dataset['train'])
    # print(len(server_dataset['train'].data))
    # print(len(client_dataset['train'].data))
    #data_loader = make_data_loader(server_dataset, 'global')
    data_loader_sup = make_data_loader_DA(client_dataset_sup, 'global')
    # print(client_dataset_sup['test'])
    # exit()
    # data_loader_unsup = make_data_loader_DA(client_dataset_unsup, 'global')

    ####
    data_loader_unsup = {}
    for domain_id,dataset_unsup in client_dataset_unsup.items():
        # domain = cfg['unsup_list'][domain_id]
        data_loader_unsup[domain_id] = make_data_loader_DA(dataset_unsup, 'global')
    ####
    # print(cfg)
    # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    if cfg['world_size']==1:
        # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        # test_model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        model = eval('models.{}()'.format(cfg['model_name']))
        test_model = eval('models.{}()'.format(cfg['model_name']))
    elif cfg['world_size']>1:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = eval('models.{}()'.format(cfg['model_name']))
        model = torch.nn.DataParallel(model,device_ids = [0, 1])
        model.to(cfg["device"])
    # print(model)
    # exit()
    if cfg['pretrained_source']:
        print('loading pretrained resnet50 model ')
        # path_source = '/home/sampathkoti/Downloads/A-20231219T043936Z-001/A/'
        path_source = '/home/sampathkoti/codes/shot/SHOT/object/ckps/source/pda/office-home/A/'

        F = torch.load(path_source + 'source_F.pt')
        B = torch.load(path_source + 'source_B.pt')
        C = torch.load(path_source + 'source_C.pt')
        # print(F.keys())
        # exit()
        model.backbone_layer.load_state_dict(torch.load(path_source + 'source_F.pt'))
        model.feat_embed_layer.load_state_dict(torch.load(path_source + 'source_B.pt'))
        model.class_layer.load_state_dict(torch.load(path_source + 'source_C.pt'))
        # print(model.feat_embed_layer.state_dict())
        # # exit()
        # print(B)
        # exit()
        # model.backbone_layer.load_state_dict(F)
        # model.feat_embed_layer.load_state_dict(B)
        # model.class_layer.load_state_dict(C)
        # print(model.feat_embed_layer.state_dict())
        # exit()
    cfg['local']['lr'] = cfg['var_lr']
    optimizer = make_optimizer(model.parameters(), 'local')
    cfg['global']['scheduler_name'] = cfg['scheduler_name']
    # print(cfg['global']['scheduler_name'])
    scheduler = make_scheduler(optimizer, 'global')
    # print(scheduler)
    # exit()
    # if cfg['sbn'] == 1:
        # batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
    # elif cfg['sbn'] == 0:
    #     batchnorm_dataset = client_dataset['train']
    # else:
    #     raise ValueError('Not valid sbn')
    # print(len(batchnorm_dataset))
    # batchnorm_dataset = client_dataset['train']
    # data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    # data_split = split_class_dataset(client_dataset,cfg['data_split_mode'])
    if cfg['d_mode'] == 'old':
        data_split = split_dataset(client_dataset, cfg['num_clients'], cfg['data_split_mode'])
    elif cfg['d_mode'] == 'new':
        # split_len_sup = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        split_len_sup = int(np.ceil(cfg['num_sup'] * cfg['num_clients']))
        # split_len_unsup = int(cfg['num_clients'])-int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        split_len_unsup = int(cfg['num_clients'])-int(np.ceil(cfg['num_sup'] * cfg['num_clients']))
        #####
        unsup_dom = len(cfg['unsup_list'])
        print('number of unsupervised domains:',unsup_dom)
        # print(unsup_dom)
        split_len = []
        for i in range(int(unsup_dom)-1):
            split_len.append(split_len_unsup//unsup_dom)

        k=0
        for i in split_len:
            k+=i
        # print(f'k is {k}')
        last = split_len_unsup-k
        split_len.append(last)
        # print(split_len)
        # exit()
        ####
        # print(split_len_sup,split_len_unsup)
        print('data len list of sup clients', split_len_sup)
        if split_len_sup != 0 :
            data_split_sup = split_class_dataset_DA(client_dataset_sup,cfg['data_split_mode'],split_len_sup)
        else:
            data_split_sup = []
        
        # print(len(data_split_sup[1]))
        # data_split_unsup = split_class_dataset_DA(client_dataset_unsup,cfg['data_split_mode'],split_len_unsup)
        # print(len(data_split_unsup))
        ####
        data_split_unsup = {}
        print(split_len)
        for j,(domain_id,dataset_unsup) in enumerate(client_dataset_unsup.items()):
            print(f'domain id :{domain_id},j:{j}')
            data_split_unsup[domain_id] = split_class_dataset_DA(dataset_unsup,cfg['data_split_mode'],split_len[j])
            # print(data_split_unsup[domain_id])
        # for k,v in data_split_unsup.items():
        #     print(k,len(v['train']),len(v['test']))
        # exit()
        ####
    #     print(data_split_unsup[0].keys())
    #     for k,v in data_split_unsup.items():
    #         # print(k,v)
    #         for i,j in v.items():
    #             print(i,len(j))
    #             for a,b in j.items():
    #                 print(a,len(b))
    # exit()
    if cfg['loss_mode'] != 'sup':
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # if cfg['loss_mode'] == 'sim':
    #     metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # print(metric.metric_name['train'])
    if cfg['resume_mode'] == 1:
        # result = resume_DA(cfg['model_tag'])
        # result = resume_DA(cfg['model_tag'],load_tag='best')
        # tag_  = '0_dslr_to_amazon_webcam_resnet50_02'
        #train_2023_clipart_0.001_resnet50_4_sup-ft-fix
        # tag_ = '2023_clipart_0.001_resnet50_4_sup-ft-fix'
        tag_ = cfg['tag_']
        pick = cfg['pick']
        # tag_ = '0_dslr_to_amazon_resnet50_01'
        # result = resume_DA(tag_,'checkpoingt')
        # result = resume(tag_,'best')
        # result = resume(tag_,'checkpoint')
        result = resume(tag_,pick)
        # import pickle
        # path = "/home/sampathkoti/Downloads/R-50-GN.pkl"
        # # m = pickle.load(open(path, 'rb'))
        # m = torch.load(path)
        # print(m.keys())
        last_epoch = result['epoch']
        # exit()
        if last_epoch > 1:
            server = make_server(model)
            client,supervised_clients  = make_client_DA(model, data_split_sup,data_split_unsup,split_len)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
            # server_pre = result['server']
            # server.model_state_dict = server_pre.model_state_dict
            if cfg['multi_model']:
                for k,_ in server.model_state_dict.items():
                    server.model_state_dict[k]=result['model_state_dict']
            else:
                server.model_state_dict=result['model_state_dict']
            last_epoch = 1
            
            # data_split_sup = result['data_split_sup']
            # data_split_unsup = result['data_split_unsup']
            # split_len = result['split_len']
            # # supervised_idx = result['supervised_idx']
            # server = result['server']
            # client = result['client']
            # supervised_clients = result['supervised_clients']
            # optimizer.load_state_dict(result['optimizer_state_dict'])
            # scheduler.load_state_dict(result['scheduler_state_dict'])
            # if cfg['new_lr'] == 1:
            #     optimizer.param_groups[0]['lr']=cfg['var_lr']
            # logger = result['logger']
            # logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
            # # cfg['loss_mode'] = 'alt-fix'
        else:
            server = make_server(model)
            client,supervised_clients  = make_client_DA(model, data_split_sup,data_split_unsup,split_len)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    elif cfg['resume_mode_target'] == 1:
        print('resume target')
        # exit()
        pick = cfg['pick']
        result = resume_DA(cfg['model_tag'],pick)
    
        last_epoch = result['epoch']
        # exit()
        if last_epoch > 1:
            
            # server_pre = result['server']
            # server.model_state_dict = server_pre.model_state_dict
            # if cfg['multi_model']:
            #     for k,_ in server.model_state_dict.items():
            #         server.model_state_dict[k]=result['model_state_dict']
            # else:
            #     server.model_state_dict=result['model_state_dict']
            # last_epoch = 1
            
            data_split_sup = result['data_split_sup']
            data_split_unsup = result['data_split_unsup']
            split_len = result['split_len']
            # supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            supervised_clients = result['supervised_clients']
            # optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            if cfg['new_lr'] == 1:
                optimizer.param_groups[0]['lr']=cfg['var_lr']
            logger = result['logger']
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
            # # cfg['loss_mode'] = 'alt-fix'
        else:
            server = make_server(model)
            client,supervised_clients  = make_client_DA(model, data_split_sup,data_split_unsup,split_len)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        client,supervised_clients = make_client_DA(model, data_split_sup,data_split_unsup,split_len)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    cfg['global']['num_epochs'] = cfg['cycles']  
    mode = cfg['loss_mode']
    # print(model)
    print(last_epoch)
    # exit()
    for epoch in range(last_epoch, cfg['global']['num_epochs'] + 1):
        if mode == 'sim-ft-fix' or mode == 'sup-ft-fix':
            # print('entered fix-mix',epoch)
            if epoch<=cfg['switch_epoch_pred']:
                if 'sim' in mode:
                    cfg['loss_mode'] = 'sim-ft'
                elif 'sup' in mode:
                    cfg['loss_mode'] = 'sup-ft'
            elif epoch > cfg['switch_epoch_pred']:
                # print('entered fix-mix',epoch)
                cfg['loss_mode'] = 'alt-fix_'
                # cfg['loss_mode'] = 'fix-mix'
        print(cfg['loss_mode'])
        # if epoch == 1 and cfg['cluster'] == 1 :
            # get_clusters(client_dataset_sup['train'], client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode,scheduler)
        #     # server.cluster(client,model)
            # server.cluster(client)
            # exit()
        if True:

            # model.load_state_dict(server.model_state_dict)
            #====#
            test_model.load_state_dict(server.model_state_dict,strict=False)
            # for k,v in test_model.named_parameters():
            #     print(k)
            #     v=server.model_state_dict[k]
            # print(test_model.feat_embed_layer.state_dict())
            # exit()
            # ====#
            if ( cfg['model_name'] == 'resnet50' or cfg['model_name'] == 'VITs') and cfg['par'] == 1:
                print('freezing')
                cfg['local']['lr'] = 0.03
                # cfg['local']['lr'] = 0.001
                param_group_ = []
                for k, v in test_model.backbone_layer.named_parameters():
                    # print(k)
                    if "bn" in k:
                        # param_group += [{'params': v, 'lr': cfg['local']['lr']*2}]
                        param_group_ += [{'params': v, 'lr': cfg['local']['lr']*0.1}]
                        # v.requires_grad = False
                        # print(k)
                    else:
                        # v.requires_grad = False
                        param_group_ += [{'params': v, 'lr': cfg['local']['lr']*0.1}]

                for k, v in test_model.feat_embed_layer.named_parameters():
                    # print(k)
                    param_group_ += [{'params': v, 'lr': cfg['local']['lr']}]
                for k, v in test_model.class_layer.named_parameters():
                    v.requires_grad = False
                    # param_group += [{'params': v, 'lr': cfg['local']['lr']}]
            test_model.to(cfg["device"])
            E_list = []
            trace_list = []
            # test_DA(data_loader_sup['test'], test_model, metric, logger, epoch=0,sup=True)
            for domain_id,data_loader_unsup_ in data_loader_unsup.items():
                # print(data_loader_unsup_)
                domain = cfg['unsup_list'][domain_id]
                print(domain,domain_id)
                t = 'FedSFDAsingle_rerun_'+domain
                # t = 'ourmethod_rerun'+domain
                if domain  != 'realworld':
                    continue
                test_DA(data_loader_unsup_['test'], test_model, metric, logger, epoch=0,domain=domain)
                # create loss function
                criterion = torch.nn.CrossEntropyLoss()
                
                num_batches = 20
                limited_loader = islice(data_loader_unsup_['test'], num_batches)
                
                # create the hessian computation module
                hessian_comp = hessian(test_model, criterion, dataloader=data_loader_unsup_['test'], cuda=True)
                # hessian_comp = hessian(test_model, criterion, dataloader=limited_loader, cuda=True)
                
                # Now let's compute the top eigenvalue. This only takes a few seconds.
                top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
                trace = hessian_comp.trace()
                E_list.append(top_eigenvalues)
                trace_list.append(np.mean(trace))
                np.save(f'top_eigen_values_list_{t}.npy',np.array(E_list))
                print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])
                
                
                np.save(f'trace_list_{t}.npy',np.array(trace_list))
                print('\n***Trace: ', np.mean(trace))
                
                density_eigen, density_weight = hessian_comp.density()
                name1 = f'density_eigen_{t}.npy'
                name2 = f'density_weight_{t}.npy'
                np.save(name1, np.array(density_eigen))
                np.save(name2,np.array(density_weight))
                # get_esd_plot(density_eigen, density_weight,t)
                
            exit()
        # train_client(client_dataset_sup['train'], client_dataset_unsup['train'], server, client, supervised_clients, optimizer, metric, logger, epoch,mode)
        # train_client_multi(client_dataset_sup['train'], client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode)
        # exit()
        # else:
        # if epoch == 1 and cfg['cluster'] == 1 :
        #     train_client_multi(client_dataset_sup['train'], client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode,scheduler)
        # else:
        # train_client_multi(client_dataset_sup['train'], client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode,scheduler)
        # if 'ft' in cfg and cfg['ft'] == 0:
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        #     logger.reset()
        #     server.update_parallel(client)
        # else:
            # logger.reset()
            # server.update(client)
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        if epoch == 1 and cfg['cluster'] ==1:
            server.cluster(client,epoch)
            server.deac_client(client)
            result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'data_split_sup': data_split_sup,'data_split_unsup' : data_split_unsup, 'logger': logger,'supervised_clients':supervised_clients,'split_len' : split_len }
            save(result, './output/model/target/{}_checkpoint{}.pt'.format(cfg['model_tag'],epoch))
            continue
        logger.reset()

        if cfg['with_BN'] == 0:
            if epoch==cfg['cycles']:
                cfg['with_BN'] =1
        if cfg['multi_model']:
            server.update_multi(client)
            if cfg['global_reg']:
                server.update_global_model(client)
        elif cfg['cluster'] == 1:
            server.update_cluster(client)
        else:
            server.update(client)
        # scheduler.step()

        # model.load_state_dict(server.model_state_dict)
        # print(model)
        #needs to be removed for final clean up
        # print(server.model_state_dict.keys())
        # test_model = make_batchnorm_stats(client_dataset_sup['train'],model, 'global')
        # print(cfg)
        #====#
        # test_model.load_state_dict(server.model_state_dict)
        #====#
        
        # test_DA(data_loader_sup['test'], test_model, metric, logger, epoch,sup=True)
        if cfg['client_test'] and epoch%1== 0:
        # if cfg['client_test'] and epoch>12:
            
            print('testing on client models')
            # valid_client = [client[i] for i in range(len(client)) if client[i].active]
            valid_client = [client[i] for i in range(len(client)) ]
            domain_accu,domain_loss,domain_count={},{},{}
            print('num of active clients :',len(valid_client))
            for domain_id,data_loader_unsup_ in data_loader_unsup.items():
                    domain = cfg['unsup_list'][domain_id]
                    domain_accu[domain],domain_loss[domain],domain_count[domain] = 0,0,0
            for m in range(len(valid_client)):
                # print('testing client number:',valid_client[m].client_id)
                # print('clinet_domain',valid_client[m].domain)
                # test_model.load_state_dict(valid_client[m].model_state_dict)
                for domain_id,data_loader_unsup_ in data_loader_unsup.items():
                    # if cfg['multi_model']:
                    #     test_model.load_state_dict(server.model_state_dict[domain_id])
                    # else:
                    #     test_model.load_state_dict(server.model_state_dict)
                    domain = cfg['unsup_list'][domain_id]
                    if domain == valid_client[m].domain:
                        print('testing client number:',valid_client[m].client_id)
                        print('clinet_domain',valid_client[m].domain)
                        test_model.load_state_dict(valid_client[m].model_state_dict)
                        print(f'domain:{domain},id:{domain_id},client_domain:{valid_client[m].domain}')
                        test_DA(data_loader_unsup_['test'], test_model, metric, logger, epoch,domain=domain)
                        avg_accuracy=0
                        avg_loss = 0 
                        count = 0
                        log_dict = logger.mean
                        # for j in logger.mean:
                        #     print(j,logger.mean[j])
                        # print(log_dict)
                        for k in log_dict:
                            # print(k)
                            if 'test_sup' not in k:
                                # print(k)
                                
                                if 'Accuracy' in k :
                                    count+=1
                                    avg_accuracy+=logger.mean[k]
                                elif 'Loss' in k :
                                    avg_loss+=logger.mean[k]
                        print('accuracy',avg_accuracy)
                        avg_accuracy,avg_loss = avg_accuracy/count,avg_loss/count
                        print('accuracy',avg_accuracy)
                        domain_accu[domain]+=avg_accuracy
                        domain_loss[domain]+=avg_loss
                        domain_count[domain]+=1
                        logger.reset()
            print(domain_accu)
            for k ,v in domain_count.items():
                print('domian',k)
                domain_accu[k]/=v
                domain_loss[k]/=v
                eval_avg = {'Accuracy' : domain_accu[k],'Loss':domain_loss[k]}
                logger.safe(True)
                tag = f'clientside_test_average_of_{k}:domain'
                logger.append(eval_avg, tag)
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
                logger.append(info, tag, mean=False)
                # print(logger.mean)
                print(logger.write(tag, metric.metric_name['test']))
                logger.safe(False)

        else:
            for domain_id,data_loader_unsup_ in data_loader_unsup.items():
                if cfg['multi_model']:
                    test_model.load_state_dict(server.model_state_dict[domain_id])
                else:
                    test_model.load_state_dict(server.model_state_dict)
                domain = cfg['unsup_list'][domain_id]
                print(f'domain:{domain},id:{domain_id}')
                test_DA(data_loader_unsup_['test'], test_model, metric, logger, epoch,domain=domain)
        # print(logger.mean)
        # exit()
        avg_accuracy=0
        avg_loss = 0 
        count = 0
        log_dict = logger.mean
        # for j in logger.mean:
        #     print(j,logger.mean[j])
        # print(log_dict)
        for k in log_dict:
            # print(k)
            if 'test_sup' not in k:
                # print(k)
                
                if 'Accuracy' in k :
                    count+=1
                    avg_accuracy+=logger.mean[k]
                elif 'Loss' in k :
                    avg_loss+=logger.mean[k]
        avg_accuracy,avg_loss = avg_accuracy/count,avg_loss/count
        print(avg_accuracy,avg_loss)
        eval_avg = {'Accuracy' : avg_accuracy,'Loss':avg_loss}
        logger.safe(True)
        tag = f'test_average_of_{count}_domains'
        logger.append(eval_avg, tag)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, tag, mean=False)
        # print(logger.mean)
        print(logger.write(tag, metric.metric_name['test']))
        logger.safe(False)
        server.deac_client(client)
        # print(logger.mean[f'{tag}/Accuracy'])
        # print(logger.write(tag, metric.metric_name['test'])
        # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'scheduler_state_dict': scheduler.state_dict(),
        #           'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
        # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
        #           'optimizer_state_dict': optimizer.state_dict(),
        #           'scheduler_state_dict': scheduler.state_dict(),
        #           'data_split': data_split, 'logger': logger}
        if epoch<=cfg['switch_epoch_pred']:
            result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'data_split_sup': data_split_sup,'data_split_unsup' : data_split_unsup, 'logger': logger,'supervised_clients':supervised_clients ,'split_len' : split_len}
            # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
            #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
            # if epoch%1==0:
            print('saving_source')
            save(result, './output/model/source/{}_checkpoint.pt'.format(cfg['model_tag']))
            if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                shutil.copy('./output/model/source/{}_checkpoint.pt'.format(cfg['model_tag']),
                            './output/model/source/{}_best.pt'.format(cfg['model_tag']))
            
        else :
            result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'data_split_sup': data_split_sup,'data_split_unsup' : data_split_unsup, 'logger': logger,'supervised_clients':supervised_clients,'split_len' : split_len }
            # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
            #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
            if epoch%1==0:
                print('saving')
                save(result, './output/model/target/{}_checkpoint.pt'.format(cfg['model_tag']))
                if cfg['save_epoch'] == 1:
                    save(result, './output/model/target/{}_checkpoint{}.pt'.format(cfg['model_tag'],epoch))
                if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                    metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                    shutil.copy('./output/model/target/{}_checkpoint.pt'.format(cfg['model_tag']),
                                './output/model/target/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
        gc.collect()
        torch.cuda.empty_cache()

    if cfg['register_hook_BN']:
        test_model.load_state_dict(server.model_state_dict)
        cfg['post_conv'] = 1
        for epoch in range(2):
            
            if epoch==1:
                get_train_stat_DA(client_dataset_sup, client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode='var')
                server.update_BNstats(client,'var')
                # for k, v in server.model_state_dict.items():             
                #     isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
                #     istype  = True if f'.running_var'  in k else False 
                #     # istype  = True if '.running_mean'  in k or '.running_var' in k else False 
                #     # parameter_type = k.split('.')[-1]
                #     # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                #     if isBatchNorm and istype:
                #         print(k,v)
            else:
                get_train_stat_DA(client_dataset_sup, client_dataset_unsup,server, client, supervised_clients, optimizer, metric, logger, epoch,mode='mean')
                server.update_BNstats(client,'mean')
                # for k, v in server.model_state_dict.items():             
                #     isBatchNorm = True if  '.bn' in k or 'bn4' in k else False
                #     istype  = True if f'.running_mean'  in k else False 
                #     # istype  = True if '.running_mean'  in k or '.running_var' in k else False 
                #     # parameter_type = k.split('.')[-1]
                #     # # print(f'{k} with parameter type {parameter_type},is batchnorm {isBatchNorm}')
                #     if isBatchNorm and istype:
                #         print(k,v)
            # print(test_model.state_dict().keys())
            test_model.load_state_dict(server.model_state_dict)
            # if epoch==cfg['cycles']  and cfg['train_pass']:
            #     print('extracting train BN stats')
            #     test_model = BN_stats(batchnorm_dataset, model, 'global')
            # else:
            #     test_model = make_batchnorm_stats(batchnorm_dataset, model, 'global')
            # test(data_loader['test'], test_model, metric, logger, epoch)
            test_DA(data_loader_sup['test'], test_model, metric, logger, epoch,sup=True)
            for domain_id,data_loader_unsup_ in data_loader_unsup.items():
                domain = cfg['unsup_list'][domain_id]
                test_DA(data_loader_unsup_['test'], test_model, metric, logger, epoch,domain=domain)
            # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
            #           'optimizer_state_dict': optimizer.state_dict(),
            #           'scheduler_state_dict': scheduler.state_dict(),
            #           'supervised_idx': supervised_idx, 'data_split': data_split, 'logger': logger}
            # result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
            #           'optimizer_state_dict': optimizer.state_dict(),
            #           'scheduler_state_dict': scheduler.state_dict(),
            #           'data_split': data_split, 'logger': logger}
            result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'data_split_sup': data_split_sup,'data_split_unsup' : data_split_unsup, 'logger': logger,'supervised_clients':supervised_clients,'split_len' : split_len }
            # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
            #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
            if epoch%1==0 and False:
                print('saving')
                save(result, './output/model/{}_post_checkpoint.pt'.format(cfg['model_tag']))
                if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                    metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                    shutil.copy('./output/model/{}_post_checkpoint.pt'.format(cfg['model_tag']),
                                './output/model/{}_post_best.pt'.format(cfg['model_tag']))
            logger.reset()
    return


def make_server(model):
    server = Server(model)
    return server


def make_client(model, data_split):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model, {'train': data_split['train'][m], 'test': data_split['test'][m]})
    num_supervised_clients = int(cfg['num_supervised_clients'])
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_supervised_clients]].tolist()
    # print(client_id)
    for i in range(num_supervised_clients):
        client[client_id[i]].supervised = True
    
        
    return client , client_id
def make_client_DA(model, data_split_sup,data_split_unsup,split_len=None):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model)
        # , {'train': data_split['train'][m], 'test': data_split['test'][m]}
    # num_supervised_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    num_supervised_clients = int(np.ceil(cfg['num_sup'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_supervised_clients]].tolist()
    unsup_client_id = list(set(range(cfg['num_clients']))-set(client_id))
    # print(len(unsup_client_id))
    # print(split_len)
    # exit()
    unsup_client_id_list = []
    for i,num in enumerate(split_len):
        # print(len(unsup_client_id))
        unsup_client_id_list.append(random.sample(unsup_client_id,num))
        unsup_client_id = list(set(unsup_client_id)-set(unsup_client_id_list[i]))
        # print(unsup_client_id)
    # print(data_split_unsup.keys())
    # exit()
    # print(len(unsup_client_id))
    # print(client_id)
    for i in range(num_supervised_clients):
        # print(client_id[i])
        client[client_id[i]].supervised = True
        client[client_id[i]].domian = cfg['domain_s']
        client[client_id[i]].data_split = {'train': data_split_sup['train'][i], 'test': data_split_sup['test'][i]}
    for j,unsup_client_id in enumerate(unsup_client_id_list):
        print(j,len(unsup_client_id))
        for i in range(len(unsup_client_id)):
            if client[unsup_client_id[i]].supervised == False:
                domain_= cfg['unsup_list'][j]
                client[unsup_client_id[i]].domain = domain_
                client[unsup_client_id[i]].domain_id = j
                # client[client_id[i]].domian_id = j
                # print(len(data_split_unsup[j]['train'][i]),'necc')
                client[unsup_client_id[i]].data_split = {'train': data_split_unsup[j]['train'][i], 'test': data_split_unsup[j]['test'][i]}
    
    # for i in range(100):
    #     # print(i,client[i].supervised)
    #     if client[i].supervised == False:
    #         print(i,client[i].domain,client[i].domain_id)
        
    # exit()
    return client , client_id

def get_train_stat_DA(client_dataset_sup, client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode='mean'):
    logger.safe(True)

    num_active_clients = cfg['num_clients']
    ACL = torch.arange(cfg['num_clients']).tolist()
    client_id = random.sample(ACL,num_active_clients)
    print(client_id,supervised_clients)
    client_id = list(set(client_id)-set(supervised_clients))
    print(client_id)
    num_active_clients = len(client_id)
    for i in range(num_active_clients):
        # if client[client_id[i]].supervised:
        #     continue
        client[client_id[i]].active = True
    print(f'getting stats for  the following clients {client_id}')
    if cfg['multi_model']:
        server.distribute_multi(client,epoch,client_dataset_unsup)
    else:
        server.distribute(client,client_dataset_unsup)
    if cfg['kl_loss'] ==1 and epoch==cfg['switch_epoch']:
        server.distribute(client,client_dataset_unsup)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        

        m = client_id[i]
        # if client[m].supervised:
        #     continue
        # print(m)
        # print(type(client[m].data_split['train']))
        if client[m].supervised ==  True:
            dataset_m = separate_dataset_DA(client_dataset_sup, client[m].data_split['train'],cfg['data_name'])
        elif client[m].supervised ==  False:
            #print('entered false')
            domain_id = client[m].domain_id
            # print(client_dataset_unsup.keys())

            print('datasplit_len',len(client[m].data_split['train']))
            dataset_m = separate_dataset_DA(client_dataset_unsup[domain_id]['train'], client[m].data_split['train'],cfg['data_name_unsup'])
        
        if dataset_m is not None:
            client[m].active = True
            client[m].get_stats(dataset_m, metric, logger,mode)

        else:
            client[m].active = False
    logger.safe(False)
    return
def train_client(client_dataset_sup, client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode):
    logger.safe(True)
    if 'ft' in cfg['loss_mode']:
        if epoch <= cfg['switch_epoch']:
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = ACL-set(supervised_clients)
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_unsup)
        else:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_unsup)
            # if epoch == cfg['change_epoch']
    elif 'at' in cfg['loss_mode']:
        cfg['srange'] = [21,31,51,61,81,91,111,121]
        if cfg['srange'][0]<=epoch<=cfg['srange'][1] or cfg['srange'][2]<=epoch<=cfg['srange'][3] or cfg['srange'][4]<=epoch<=cfg['srange'][5] or cfg['srange'][6]<=epoch<=cfg['srange'][7]:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_sup)
        else:
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = ACL-set(supervised_clients)
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_unsup)

    elif 'fix' in cfg['loss_mode'] and 'alt' not in cfg['loss_mode']:
        if epoch >cfg['switch_epoch_pred']:
            cfg['loss_mode'] = 'fix-mix'
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = ACL-set(supervised_clients)
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_unsup)
        else:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_sup)
    # elif 'alt-fix' in cfg['loss_mode']:
    #     print('entered alt-fix mode')
    #     if epoch %2 != 0:# or epoch >270:
    #         cfg['loss_mode'] = 'bmd'
    #         # cfg['loss_mode'] = 'fix-mix'
    #         print(cfg['loss_mode'])
    #         num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    #         ACL=set(torch.arange(cfg['num_clients']).tolist())
    #         ACL = ACL-set(supervised_clients)
    #         # print(ACL)
    #         # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
    #         # ran_CL = [ran_CL-set(supervised_clients)]
    #         client_id = random.sample(ACL,num_active_clients)
    #         # print(client_id)
    #         # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True
    #         server.distribute(client,client_dataset_unsup)
    #     elif epoch % 2 == 0:# and epoch <=270:
    #         cfg['loss_mode'] = 'sup'
    #         print(cfg['loss_mode'])
    #         num_active_clients = len(supervised_clients)
    #         client_id = supervised_clients
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True
    #         server.distribute(client,client_dataset_sup)
    ####################################
    # elif 'alt-fix_' in cfg['loss_mode']:
    #     print('eeentered entered alt-fix mode')
    #     if epoch %2 != 0: # or epoch % 2 == 0:# or epoch >270:
    #         cfg['loss_mode'] = 'bmd'
    #         # cfg['loss_mode'] = 'fix-mix'
    #         print(cfg['loss_mode'])
    #         num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    #         ACL=set(torch.arange(cfg['num_clients']).tolist())
    #         ACL = ACL-set(supervised_clients)
    #         # print(ACL)
    #         # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
    #         # ran_CL = [ran_CL-set(supervised_clients)]
    #         client_id = random.sample(ACL,num_active_clients)
    #         # print(client_id)
    #         # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True
    #         server.distribute(client,client_dataset_unsup)
    #     elif epoch % 2 == 0:# and epoch <=270:
    #         cfg['loss_mode'] = 'sup'
    #         print(cfg['loss_mode'])
    #         num_active_clients = len(supervised_clients)
    #         client_id = supervised_clients
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True
    #         server.distribute(client,client_dataset_sup)
    ################################
    elif 'alt-fix_' in cfg['loss_mode']:
        print('entered entered alt-fix mode')
        print(epoch%4)
        if epoch%4 == 0:# and epoch <=270:
            cfg['loss_mode'] = 'sup'
            print(cfg['loss_mode'])
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_sup)

        else : # or epoch % 2 == 0:# or epoch >270:
            domains = []
            cfg['loss_mode'] = 'bmd'
            # cfg['loss_mode'] = 'fix-mix'
            print(cfg['loss_mode'])
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = ACL-set(supervised_clients)
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
                domains.append(client[client_id[i]].domain)
            server.distribute(client,client_dataset_unsup)
    # elif 'alt-fix' in cfg['loss_mode']:
    #     print('entered alt-fix mode')
    #     if epoch %2 == 0:
    #         cfg['loss_mode'] = 'fix-mix'
    #         print(cfg['loss_mode'])
    #         num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    #         ACL=set(torch.arange(cfg['num_clients']).tolist())
    #         ACL = ACL-set(supervised_clients)
    #         # print(ACL)
    #         # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
    #         # ran_CL = [ran_CL-set(supervised_clients)]
    #         client_id = random.sample(ACL,num_active_clients)
    #         # print(client_id)
    #         # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True
    #     elif epoch % 2 != 0:
    #         cfg['loss_mode'] = 'sup'
    #         print(cfg['loss_mode'])
    #         num_active_clients = len(supervised_clients)
    #         client_id = supervised_clients
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True

    # server.distribute(client, batchnorm_dataset)
    print(f'traning the following clients {client_id}')
    print(f'domains of the respective clients{domains}')
    # server.distribute(client,batchnorm_dataset)
    if cfg['kl_loss'] ==1 and epoch==cfg['switch_epoch']:
        server.distribute_fix_model(client,batchnorm_dataset)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        print(f'traning client {m}')
        # print(type(client[m].data_split['train']))
        # print(client[m].supervised)
        if client[m].supervised ==  True:
            dataset_m = separate_dataset_DA(client_dataset_sup, client[m].data_split['train'],cfg['data_name'])
        elif client[m].supervised ==  False:
            #print('entered false')
            dataset_m = separate_dataset_DA(client_dataset_unsup, client[m].data_split['train'],cfg['data_name_unsup'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            # cfg['pred'] = True
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
            # cfg['pred'] = False
        # print(cfg)
        # print(dataset_m is not None)
        if dataset_m is not None:
            # print(cfg)
            # print(dataset_m)
            # print(cfg['loss_mode'])
            if cfg['loss_mode'] == 'fix-mix' and dataset_m[0] is not None and dataset_m[1] is not None:
                client[m].active = True
                client[m].trainntune(dataset_m, lr, metric, logger, epoch)
            elif 'sim' in cfg['loss_mode'] or 'sup' in cfg['loss_mode'] or 'bmd' in cfg['loss_mode']:
                client[m].active = True
                client[m].trainntune(dataset_m, lr, metric, logger, epoch)
            else:
                client[m].active = False

        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return

def train_client_multi(client_dataset_sup, client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode,scheduler = None):
    logger.safe(True)
    # print(scheduler)
    # exit()
    if epoch == 1  and  cfg['cluster'] ==1:
        init_activity_rate = cfg['active_rate']
        cfg['active_rate'] = 1
    domains=[]
    if 'ft' in cfg['loss_mode']:
        if epoch <= cfg['switch_epoch']:
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = list(ACL-set(supervised_clients))
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # client_id = np.random.choice(ACL,num_active_clients)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            elif cfg['cluster'] and epoch != 1:
                server.distribute_cluster(client,epoch,client_dataset_unsup)
            else:
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)
        else:
            
        
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            elif cfg['cluster'] and epoch != 1:
                server.distribute_cluster(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)
            # if epoch == cfg['change_epoch']
    # elif 'at' in cfg['loss_mode']:
    #     cfg['srange'] = [21,31,51,61,81,91,111,121]
    #     if cfg['srange'][0]<=epoch<=cfg['srange'][1] or cfg['srange'][2]<=epoch<=cfg['srange'][3] or cfg['srange'][4]<=epoch<=cfg['srange'][5] or cfg['srange'][6]<=epoch<=cfg['srange'][7]:
    #         num_active_clients = len(supervised_clients)
    #         client_id = supervised_clients
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True
    #         server.distribute(client,client_dataset_sup)
    #     else:
    #         num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    #         ACL=set(torch.arange(cfg['num_clients']).tolist())
    #         ACL = ACL-set(supervised_clients)
    #         # print(ACL)
    #         # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
    #         # ran_CL = [ran_CL-set(supervised_clients)]
    #         client_id = random.sample(ACL,num_active_clients)
    #         # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
    #         for i in range(num_active_clients):
    #             client[client_id[i]].active = True
    #         server.distribute(client,client_dataset_unsup)

    elif 'fix' in cfg['loss_mode'] and 'alt' not in cfg['loss_mode']:
        if epoch >cfg['switch_epoch_pred']:
            cfg['loss_mode'] = 'fix-mix'
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = list(ACL-set(supervised_clients))
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # client_id = np.random.choice(ACL,num_active_clients)
            # print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            elif cfg['cluster'] and epoch != 1:
                server.distribute_cluster(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)
        else:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            else:
                server.distribute(client,client_dataset_unsup)

    elif 'alt-fix_' in cfg['loss_mode']:
        print('entered entered alt-fix mode')
        if epoch ==0:# and epoch <=270: // alternate training
            domains=[]
            cfg['loss_mode'] = 'sup'
            print(cfg['loss_mode'])
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            elif cfg['cluster'] and epoch != 1:
                server.distribute_cluster(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)
            domains.append(client[client_id[i]].domain)

        else : # or epoch % 2 == 0:# or epoch >270:
            if cfg['unsup_mode'] =='bmd':
                cfg['loss_mode'] = 'bmd'
            elif cfg['unsup_mode'] == 'fix-mix':
                print('changing local mode to fix-mix')
                cfg['loss_mode'] = 'fix-mix'
            else:
                print('Error:Undefined mode')
            domains=[]
            # cfg['loss_mode'] = 'fix-mix'
            print(cfg['loss_mode'])
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            # inc_seed = 0
            # while(True):
            #                 # Fix randomness in client selection
            #                 np.random.seed(epoch + inc_seed)
            #                 act_list    = np.random.uniform(size=cfg['num_clients'])
            #                 print(act_list)
            #                 act_clients = act_list <= cfg['active_rate']
            #                 selected_clnts = np.sort(np.where(act_clients)[0])
            #                 inc_seed += 1
            #                 print(selected_clnts)
            #                 if len(selected_clnts) != 0:
            #                     break
        
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = list(ACL-set(supervised_clients))
            print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            # client_id = list(set(selected_clnts)-set(supervised_clients))
            # print(client_id)
            # exit()
            # client_id = np.random.choice(np.array(ACL),num_active_clients)
            # print(client_id)
            client_id = random.sample(ACL,num_active_clients)
            print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
                domains.append(client[client_id[i]].domain)
            if cfg['multi_model']:
                # print(type(server.model_state_dict))
                # exit()
                server.distribute_multi(client,epoch,client_dataset_unsup)
            elif cfg['cluster'] and epoch != 1:
                server.distribute_cluster(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)

    print(f'traning the following clients {client_id}')
    print(f'corresponding domains{domains}')
    # server.distribute(client,batchnorm_dataset)
    if cfg['kl_loss'] ==1 and epoch==cfg['switch_epoch']:
        server.distribute_fix_model(client,batchnorm_dataset)
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        print(f'traning client {m}')
        # print(type(client[m].data_split['train']))
        # print(client[m].supervised)
        if client[m].supervised ==  True:
            # domain_ = client[m].domain
            # print(domain_)
            dataset_m = separate_dataset_DA(client_dataset_sup, client[m].data_split['train'],cfg['data_name'])
        elif client[m].supervised ==  False:
            print('unsupervised training in progess')
            domain_id = client[m].domain_id
            # print(client_dataset_unsup.keys())

            print('datasplit_len',len(client[m].data_split['train']))
            dataset_m = separate_dataset_DA(client_dataset_unsup[domain_id]['train'], client[m].data_split['train'],cfg['data_name_unsup'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            # cfg['pred'] = True
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
            # cfg['pred'] = False
        # print(cfg)
        # print(dataset_m is not None)
        # exit()
        if dataset_m is not None:
            # print(cfg)
            # print(dataset_m)
            # print(cfg['loss_mode'])
            if cfg['loss_mode'] == 'fix-mix' and dataset_m[0] is not None and dataset_m[1] is not None:
                client[m].active = True
                client[m].trainntune(dataset_m, lr, metric, logger, epoch)
            elif 'sim' in cfg['loss_mode'] or 'sup' in cfg['loss_mode'] or 'bmd' in cfg['loss_mode']:
                client[m].active = True
                # print(len(dataset_m))
                # print(scheduler)
                # exit()
                if epoch == 1 and cfg['cluster']:
                    # client[m].trainntune(dataset_m, lr, metric, logger, epoch,fwd_pass=True,scheduler =scheduler)
                    
                    client[m].trainntune(dataset_m, lr, metric, logger, epoch,client=client,fwd_pass=True,scheduler =scheduler)
                else:
                    client[m].trainntune(dataset_m, lr, metric, logger, epoch,client=client,scheduler =scheduler)
                # client[m].trainntune(dataset_m, lr, metric, logger, epoch)
            else:
                client[m].active = False

        else:
            client[m].active = False
        if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
            exp_progress = 100. * i / num_active_clients
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
                             'Learning rate: {:.6f}'.format(lr),
                             'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
                             'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    gc.collect()
    torch.cuda.empty_cache()
    if epoch == 1  and  cfg['cluster'] ==1:
        cfg['active_rate'] = init_activity_rate
    return
def get_clusters(client_dataset_sup, client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode,scheduler = None):
    init_activity_rate = cfg['active_rate']
    cfg['active_rate'] = 1
    domains=[]
    if 'ft' in cfg['loss_mode']:
        if epoch <= cfg['switch_epoch']:
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = list(ACL-set(supervised_clients))
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # client_id = np.random.choice(ACL,num_active_clients)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            else:
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)
        else:
            
        
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)

    elif 'fix' in cfg['loss_mode'] and 'alt' not in cfg['loss_mode']:
        if epoch >cfg['switch_epoch_pred']:
            cfg['loss_mode'] = 'fix-mix'
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = list(ACL-set(supervised_clients))
            # print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            client_id = random.sample(ACL,num_active_clients)
            # client_id = np.random.choice(ACL,num_active_clients)
            # print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)
        else:
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            else:
                server.distribute(client,client_dataset_unsup)

    elif 'alt-fix_' in cfg['loss_mode']:
        print('entered entered alt-fix mode')
        if epoch ==0:# and epoch <=270: // alternate training
            domains=[]
            cfg['loss_mode'] = 'sup'
            print(cfg['loss_mode'])
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            if cfg['multi_model']:
                server.distribute_multi(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)
            domains.append(client[client_id[i]].domain)

        else : # or epoch % 2 == 0:# or epoch >270:
            if cfg['unsup_mode'] =='bmd':
                cfg['loss_mode'] = 'bmd'
            elif cfg['unsup_mode'] == 'fix-mix':
                print('changing local mode to fix-mix')
                cfg['loss_mode'] = 'fix-mix'
            else:
                print('Error:Undefined mode')
            domains=[]
            # cfg['loss_mode'] = 'fix-mix'
            print(cfg['loss_mode'])
            num_active_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
            
            ACL=set(torch.arange(cfg['num_clients']).tolist())
            ACL = list(ACL-set(supervised_clients))
            print(ACL)
            # ran_CL = set(torch.randperm(cfg['num_clients']).tolist())
            # ran_CL = [ran_CL-set(supervised_clients)]
            # client_id = list(set(selected_clnts)-set(supervised_clients))
            # print(client_id)
            # exit()
            # client_id = np.random.choice(np.array(ACL),num_active_clients)
            # print(client_id)
            client_id = random.sample(ACL,num_active_clients)
            print(client_id)
            # client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_active_clients]].tolist()
            for i in range(num_active_clients):
                client[client_id[i]].active = True
                domains.append(client[client_id[i]].domain)
            if cfg['multi_model']:
                # print(type(server.model_state_dict))
                # exit()
                server.distribute_multi(client,epoch,client_dataset_unsup)
            else:
                # server.distribute(client,client_dataset_unsup)
                if epoch == 1:
                    server.distribute(client,client_dataset_unsup,BN_stats=True)
                else:
                    server.distribute(client,client_dataset_unsup)

    print(f'clustering the following clients {client_id}')
    print(f'corresponding domains{domains}')
    
    num_active_clients = len(client_id)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    for i in range(num_active_clients):
        m = client_id[i]
        print(f'client {m}')
        # print(type(client[m].data_split['train']))
        # print(client[m].supervised)
        if client[m].supervised ==  True:
            # domain_ = client[m].domain
            # print(domain_)
            dataset_m = separate_dataset_DA(client_dataset_sup, client[m].data_split['train'],cfg['data_name'])
        elif client[m].supervised ==  False:
            print('unsupervised training in progess')
            domain_id = client[m].domain_id
            # print(client_dataset_unsup.keys())

            print('datasplit_len',len(client[m].data_split['train']))
            dataset_m = separate_dataset_DA(client_dataset_unsup[domain_id]['train'], client[m].data_split['train'],cfg['data_name_unsup'])
        if 'batch' not in cfg['loss_mode'] and 'frgd' not in cfg['loss_mode'] and 'fmatch' not in cfg['loss_mode']:
            # cfg['pred'] = True
            dataset_m = client[m].make_dataset(dataset_m, metric, logger)
            # cfg['pred'] = False
        # print(cfg)
        # print(dataset_m is not None)
        # exit()
        if dataset_m is not None:
            # print(cfg)
            # print(dataset_m)
            # print(cfg['loss_mode'])
            if cfg['loss_mode'] == 'fix-mix' and dataset_m[0] is not None and dataset_m[1] is not None:
                client[m].active = True
                client[m].trainntune(dataset_m, lr, metric, logger, epoch)
            elif 'sim' in cfg['loss_mode'] or 'sup' in cfg['loss_mode'] or 'bmd' in cfg['loss_mode']:
                client[m].active = True
                # print(len(dataset_m))
                # print(scheduler)
                # exit()
                # client[m].cluster_pass(dataset_m, lr, metric, logger, epoch,scheduler =scheduler)
                client[m].trainntune(dataset_m, lr, metric, logger, epoch,fwd_pass=True)
                torch.cuda.memory_summary(device=None, abbreviated=False)
                exit()
            else:
                client[m].active = False

        else:
            client[m].active = False
        # if i % int((num_active_clients * cfg['log_interval']) + 1) == 0:
        #     _time = (time.time() - start_time) / (i + 1)
        #     epoch_finished_time = datetime.timedelta(seconds=_time * (num_active_clients - i - 1))
        #     exp_finished_time = epoch_finished_time + datetime.timedelta(
        #         seconds=round((cfg['global']['num_epochs'] - epoch) * _time * num_active_clients))
        #     exp_progress = 100. * i / num_active_clients
        #     info = {'info': ['Model: {}'.format(cfg['model_tag']),
        #                      'Train Epoch (C): {}({:.0f}%)'.format(epoch, exp_progress),
        #                      'Learning rate: {:.6f}'.format(lr),
        #                      'ID: {}({}/{})'.format(client_id[i], i + 1, num_active_clients),
        #                      'Epoch Finished Time: {}'.format(epoch_finished_time),
        #                      'Experiment Finished Time: {}'.format(exp_finished_time)]}
        #     logger.append(info, 'train', mean=False)
        #     print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    
    gc.collect()
    torch.cuda.empty_cache()
    cfg['active_rate'] = init_activity_rate
    return
def train_server(dataset, server, optimizer, metric, logger, epoch):
    logger.safe(True)
    start_time = time.time()
    lr = optimizer.param_groups[0]['lr']
    server.train(dataset, lr, metric, logger)
    _time = (time.time() - start_time)
    epoch_finished_time = datetime.timedelta(seconds=round((cfg['global']['num_epochs'] - epoch) * _time))
    info = {'info': ['Model: {}'.format(cfg['model_tag']),
                     'Train Epoch (S): {}({:.0f}%)'.format(epoch, 100.),
                     'Learning rate: {:.6f}'.format(lr),
                     'Epoch Finished Time: {}'.format(epoch_finished_time)]}
    logger.append(info, 'train', mean=False)
    print(logger.write('train', metric.metric_name['train']))
    logger.safe(False)
    return


def test(data_loader, model, metric, logger, epoch,sup=False):
    logger.safe(True)
    with torch.no_grad():
        model.train(False)
        for i, input in enumerate(data_loader):
            input = collate(input)
            input_size = input['data'].size(0)
            # print(input_size)
            input = to_device(input, cfg['device'])
            input['loss_mode'] = cfg['loss_mode']
            input['supervised_mode'] = False
            input['test'] = True
            output = model(input)
            output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            if sup:
                logger.append(evaluation, 'test_sup', input_size)
            else :
                logger.append(evaluation, 'test_unsup', input_size)
        info = {'info': ['Model: {}{}'.format(cfg['model_tag'],'sup' if sup else 'unsup'), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        if sup:
            logger.append(info, 'test_sup', mean=False)
            print(logger.write('test_sup', metric.metric_name['test']))
        else:
            logger.append(info, 'test_unsup', mean=False)
            print(logger.write('test_unsup', metric.metric_name['test']))
    logger.safe(False)
    return

def test_DA(data_loader, model, metric, logger, epoch,sup=False,domain=None):
    logger.safe(True)
    model.eval()
    if sup:
        tag = 'test_sup'
    else:
        tag = f'test_unsup_{domain}'
    with torch.no_grad():
        model.train(False)
        if cfg['test_10_crop']:
            # iter_test = [iter(data_loader[i]) for i in range(10)]
            for j in range(10):
                # print(type(data_loader[j]))
                for i, input in enumerate(data_loader[j]):
                    # print(type(input))
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    input['loss_mode'] = cfg['loss_mode']
                    input['supervised_mode'] = False
                    input['test'] = True
                    output = model(input)
                    input['test'] = True
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                    logger.append(evaluation, tag, input_size)
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
            logger.append(info, tag, mean=False)
            print(logger.write(tag, metric.metric_name['test']))
            #         logger.append(evaluation, 'test', input_size)
            
                
            # info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
            # # logger.append(info, 'test', mean=False)
            # # print(logger.write('test', metric.metric_name['test']))
            
        else :
            feat = []
            with torch.no_grad():
                model.to(cfg['device'])
                model.train(False)
                for i, input in enumerate(data_loader):
                    input = collate(input)
                    input_size = input['data'].size(0)
                    input = to_device(input, cfg['device'])
                    input['loss_mode'] = cfg['loss_mode']
                    input['supervised_mode'] = False
                    input['test'] = True
                    output = model(input)
                    input['test'] = True
                    # print(output['embd_feat'].shape)
                    # exit()
                    feat.append(output['embd_feat'].cpu())
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    # print(output['loss'])
                    evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                    # print(evaluation)
                
                #     logger.append(evaluation, 'test', input_size)
                # info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
                # logger.append(info, 'test', mean=False)
                # print(logger.write('test', metric.metric_name['test']))
                    logger.append(evaluation, tag, input_size)
                M = torch.cat(feat, dim=0)
                # print(M.shape)
                # # Compute the covariance matrix
                # cov_matrix = np.cov(M, rowvar=False)

                # # Perform SVD
                # U, S, Vt = np.linalg.svd(cov_matrix)

                # # S contains the singular values
                # singular_values = S
                # print(S.shape)
                # np.save(f'product_sfda_embd_{domain}_S.npy', singular_values)
                # np.save(f'product_sfda_feat_{domain}.npy', M)
                #C our method ,S sfda
                
                # Plotting the singular values
                
# Plotting the singular values on a log scale
                # plt.figure(figsize=(10, 6))
                # plt.plot(singular_values, marker='o')
                # plt.yscale('log')
                # plt.title('Singular Values of the Covariance Matrix (Log Scale)')
                # plt.xlabel('Index')
                # plt.ylabel('Singular Value (Log Scale)')
                # plt.grid(True, which="both", ls="--")

                # # Save the plot
                # plt.savefig('singular_values_log_scale.png')  # You can change the file format and name as needed

                # # Close the plot to free up memory
                # plt.close()
                # exit()
                #######################################
                # change the model to eval mode to disable running stats upate
                
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
                logger.append(info, tag, mean=False)
                # print(logger.mean[f'{tag}/Accuracy'])
                print(logger.write(tag, metric.metric_name['test']))

                # print(metric.metric_name['test'])
                # if sup:
                #     logger.append(info, 'test_sup', mean=False)
                #     print(logger.write('test_sup', metric.metric_name['test']))
                # else:
                #     # print(9999)
                #     print(domain)
                #     tag = f'test_unsup_{domain}'
                #     logger.append(info, tag , mean=False)
                #     print(logger.write(tag, metric.metric_name['test']))
    model.eval()

    # # create loss function
    # criterion = torch.nn.CrossEntropyLoss()
    # # create the hessian computation module
    # hessian_comp = hessian(model, criterion, dataloader=data_loader, cuda=True)
    
    # # Now let's compute the top eigenvalue. This only takes a few seconds.
    # top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues()
    # print("The top Hessian eigenvalue of this model is %.4f"%top_eigenvalues[-1])
    
    # density_eigen, density_weight = hessian_comp.density()
    # get_esd_plot(density_eigen, density_weight)
    logger.safe(False)
    gc.collect()
    torch.cuda.empty_cache()
    return

if __name__ == "__main__":
    main()
