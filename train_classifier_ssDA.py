import argparse
import datetime
import models
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import random
from config import cfg, process_args
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset,separate_dataset_DA, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats , split_class_dataset,split_class_dataset_DA,make_data_loader_DA,make_batchnorm_stats_DA
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate,resume_DA,process_dataset_multi
from logger import make_logger

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
    print(cfg)
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    # print(cfg['gm'])
    #server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset_sup = fetch_dataset(cfg['data_name'],domain=cfg['domain_s'])
    

    # client_dataset_unsup = fetch_dataset(cfg['data_name_unsup'],domain=cfg['domain_u'])
    #############
    print('list of un supervised domain')
    print(cfg['unsup_list'])
    client_dataset_unsup = {}
    for i ,domain in enumerate(cfg['unsup_list']):
        # print(i,domain)
        print('fetching unsupervised domains')
        if domain in ['MNIST','USPS','SVHN','MNIST_M', 'SYN32']:
            cfg['data_name_unsup'] = domain
            client_dataset_unsup[i] = fetch_dataset(cfg['data_name_unsup'])
        elif domain in ['dslr','webcam','amazon']:
            cfg['data_name_unsup'] = 'office31'
            client_dataset_unsup[i] = fetch_dataset(cfg['data_name_unsup'],domain=domain)
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
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
        test_model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    elif cfg['world_size']>1:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = eval('models.{}()'.format(cfg['model_name']))
        model = torch.nn.DataParallel(model,device_ids = [0, 1])
        model.to(cfg["device"])
    # print(model)
    cfg['local']['lr'] = cfg['var_lr']
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
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
        split_len_sup = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        split_len_unsup = int(cfg['num_clients'])-int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
        #####
        unsup_dom = len(cfg['unsup_list'])
        print('number of unsupervised domains')
        print(unsup_dom)
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
        data_split_sup = split_class_dataset_DA(client_dataset_sup,cfg['data_split_mode'],split_len_sup)
        
        # print(len(data_split_sup[1]))
        # data_split_unsup = split_class_dataset_DA(client_dataset_unsup,cfg['data_split_mode'],split_len_unsup)
        # print(len(data_split_unsup))
        ####
        data_split_unsup = {}
        
        for j,(domain_id,dataset_unsup) in enumerate(client_dataset_unsup.items()):
            print(domain_id,j)
            data_split_unsup[domain_id] = split_class_dataset_DA(dataset_unsup,cfg['data_split_mode'],split_len[j])
            # print(data_split_unsup[domain_id])
        # for k,v in data_split_unsup.items():
        #     print(k,len(v['train']),len(v['test']))
        # exit()
        ####
        # print(data_split_unsup.keys())
        # for k,v in data_split_unsup.items():
        #     # print(k,v)
        #     for i,j in v.items():
        #         print(i,len(j))
        #         for a,b in j.items():
        #             print(a,len(b))

    if cfg['loss_mode'] != 'sup':
        metric = Metric({'train': ['Loss', 'Accuracy', 'PAccuracy', 'MAccuracy', 'LabelRatio'],
                         'test': ['Loss', 'Accuracy']})
    else:
        metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # if cfg['loss_mode'] == 'sim':
    #     metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # print(metric.metric_name['train'])
    if cfg['resume_mode'] == 1:
        result = resume_DA(cfg['model_tag'])
        # result = resume_DA(cfg['model_tag'],load_tag='best')
        # tag_  = '0_dslr_to_amazon_webcam_resnet50_02'
        # result = resume_DA(tag_,'checkpoint')
        # import pickle
        # path = "/home/sampathkoti/Downloads/R-50-GN.pkl"
        # # m = pickle.load(open(path, 'rb'))
        # m = torch.load(path)
        # print(m.keys())
        last_epoch = result['epoch']
        # exit()
        if last_epoch > 1:
            data_split_sup = result['data_split_sup']
            data_split_unsup = result['data_split_unsup']
            split_len = result['split_len']
            # supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            supervised_clients = result['supervised_clients']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
            # logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
            # cfg['loss_mode'] = 'alt-fix'
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
        # train_client(client_dataset_sup['train'], client_dataset_unsup['train'], server, client, supervised_clients, optimizer, metric, logger, epoch,mode)
        train_client_multi(client_dataset_sup['train'], client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode)
        # if 'ft' in cfg and cfg['ft'] == 0:
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        #     logger.reset()
        #     server.update_parallel(client)
        # else:
            # logger.reset()
            # server.update(client)
        #     train_server(server_dataset['train'], server, optimizer, metric, logger, epoch)
        logger.reset()
        server.update(client)
        scheduler.step()

        model.load_state_dict(server.model_state_dict)
        # print(model)
        #needs to be removed for final clean up
        # print(server.model_state_dict.keys())
        # test_model = make_batchnorm_stats(client_dataset_sup['train'],model, 'global')
        #====#
        test_model.load_state_dict(model.state_dict())
        #====#
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
                if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                    metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                    shutil.copy('./output/model/target/{}_checkpoint.pt'.format(cfg['model_tag']),
                                './output/model/target/{}_best.pt'.format(cfg['model_tag']))
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
    num_supervised_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_supervised_clients]].tolist()
    unsup_client_id = list(set(range(cfg['num_clients']))-set(client_id))
    # print(len(unsup_client_id))
    # print(split_len)
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
            # print('entered false')
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

def train_client_multi(client_dataset_sup, client_dataset_unsup, server, client, supervised_clients, optimizer, metric, logger, epoch,mode):
    logger.safe(True)
    domains=[]
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
            server.distribute(client,client_dataset_sup)
            domains.append(client[client_id[i]].domain)

        else : # or epoch % 2 == 0:# or epoch >270:
            cfg['loss_mode'] = 'bmd'
            domains=[]
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
            print('entered false')
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
            with torch.no_grad():
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
                    output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
                    # print(output['loss'])
                    evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                    # print(evaluation)

                #     logger.append(evaluation, 'test', input_size)
                # info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
                # logger.append(info, 'test', mean=False)
                # print(logger.write('test', metric.metric_name['test']))
                    logger.append(evaluation, tag, input_size)
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
                logger.append(info, tag, mean=False)
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

    logger.safe(False)
    return

if __name__ == "__main__":
    main()
