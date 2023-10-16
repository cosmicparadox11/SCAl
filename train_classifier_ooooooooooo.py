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
    make_batchnorm_dataset_su, make_batchnorm_stats , split_class_dataset,split_class_dataset_DA
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate,resume_DA
from logger import make_logger

cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)


def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    exp_num = cfg['control_name'].split('_')[0]
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'],cfg['data_name_unsup'], cfg['model_name'],exp_num]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return


def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    print(cfg['gm'])
    #server_dataset = fetch_dataset(cfg['data_name'])
    client_dataset_sup = fetch_dataset(cfg['data_name'])
    # print(cfg['data_name_unsup'])
    client_dataset_unsup = fetch_dataset(cfg['data_name_unsup'])
    # print(len(server_dataset['train'].data))
    # print(len(client_dataset['train'].data))
    # for i in range(2):
    #     print(server_dataset['train'][i])
    process_dataset(client_dataset_sup,client_dataset_unsup)
    #server_dataset['train'], client_dataset['train'], supervised_idx = separate_dataset_su(server_dataset['train'],
                                                                                        #    client_dataset['train'])
    # print(len(server_dataset['train'].data))
    # print(len(client_dataset['train'].data))
    #data_loader = make_data_loader(server_dataset, 'global')
    data_loader_sup = make_data_loader(client_dataset_sup, 'global')
    data_loader_unsup = make_data_loader(client_dataset_unsup, 'global')
    # print(cfg)
    # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    if cfg['world_size']==1:
        model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    elif cfg['world_size']>1:
        cfg["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = eval('models.{}()'.format(cfg['model_name']))
        model = torch.nn.DataParallel(model,device_ids = [0, 1])
        model.to(cfg["device"])
    # print(model)
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    # if cfg['sbn'] == 1:
    #     batchnorm_dataset = make_batchnorm_dataset_su(server_dataset['train'], client_dataset['train'])
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
        print(split_len_sup,split_len_unsup)
        data_split_sup = split_class_dataset_DA(client_dataset_sup,cfg['data_split_mode'],split_len_sup)
        data_split_unsup = split_class_dataset_DA(client_dataset_unsup,cfg['data_split_mode'],split_len_unsup)
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
        last_epoch = result['epoch']
        if last_epoch > 1:
            data_split_sup = result['data_split_sup']
            data_split_unsup = result['data_split_unsup']
            # supervised_idx = result['supervised_idx']
            server = result['server']
            client = result['client']
            supervised_clients = result['supervised_clients']
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
            # cfg['loss_mode'] = 'alt-fix'
        else:
            server = make_server(model)
            client,supervised_clients  = make_client_DA(model, data_split_sup,data_split_unsup)
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        server = make_server(model)
        client,supervised_clients = make_client_DA(model, data_split_sup,data_split_unsup)
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    cfg['global']['num_epochs'] = cfg['cycles']  
    mode = cfg['loss_mode']
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
        train_client(client_dataset_sup['train'], client_dataset_unsup['train'], server, client, supervised_clients, optimizer, metric, logger, epoch,mode)
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
        test_model = make_batchnorm_stats(client_dataset_unsup['train'], model, 'global')
        test(data_loader_sup['test'], test_model, metric, logger, epoch,sup=True)
        test(data_loader_unsup['test'], test_model, metric, logger, epoch)
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
                    'data_split_sup': data_split_sup,'data_split_unsup' : data_split_unsup, 'logger': logger,'supervised_clients':supervised_clients }
            # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
            #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
            if epoch%2==0:
                print('saving')
                save(result, './output/model/source/{}_checkpoint.pt'.format(cfg['model_tag']))
                if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                    metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                    shutil.copy('./output/model/source/{}_checkpoint.pt'.format(cfg['model_tag']),
                                './output/model/source/{}_best.pt'.format(cfg['model_tag']))
            
        else :
            result = {'cfg': cfg, 'epoch': epoch + 1, 'server': server, 'client': client,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'data_split_sup': data_split_sup,'data_split_unsup' : data_split_unsup, 'logger': logger,'supervised_clients':supervised_clients }
            # save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            # if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
            #     metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
            #     shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
            #                 './output/model/{}_best.pt'.format(cfg['model_tag']))
            if epoch%2==0:
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
    print(client_id)
    for i in range(num_supervised_clients):
        client[client_id[i]].supervised = True
    
        
    return client , client_id
def make_client_DA(model, data_split_sup,data_split_unsup):
    client_id = torch.arange(cfg['num_clients'])
    client = [None for _ in range(cfg['num_clients'])]
    for m in range(len(client)):
        client[m] = Client(client_id[m], model)
        # , {'train': data_split['train'][m], 'test': data_split['test'][m]}
    num_supervised_clients = int(np.ceil(cfg['active_rate'] * cfg['num_clients']))
    client_id = torch.arange(cfg['num_clients'])[torch.randperm(cfg['num_clients'])[:num_supervised_clients]].tolist()
    unsup_client_id = list(set(range(cfg['num_clients']))-set(client_id))
    # print(len(unsup_client_id))
    # print(client_id)
    for i in range(num_supervised_clients):
        # print(client_id[i])
        client[client_id[i]].supervised = True
        client[client_id[i]].data_split = {'train': data_split_sup['train'][i], 'test': data_split_sup['test'][i]}
    
    for i in range(len(unsup_client_id)):
        if client[unsup_client_id[i]].supervised == False:
            client[unsup_client_id[i]].data_split = {'train': data_split_unsup['train'][i], 'test': data_split_unsup['test'][i]}
    
    # for i in range(100):
    #     print(i,client[i].supervised)

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
        print('eeentered entered alt-fix mode')
        if epoch % 5 == 0:# and epoch <=270:
            cfg['loss_mode'] = 'sup'
            print(cfg['loss_mode'])
            num_active_clients = len(supervised_clients)
            client_id = supervised_clients
            for i in range(num_active_clients):
                client[client_id[i]].active = True
            server.distribute(client,client_dataset_sup)

        else : # or epoch % 2 == 0:# or epoch >270:
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
            print('entered false')
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


if __name__ == "__main__":
    main()
