import argparse
import copy
import datetime
import models
import os
import shutil
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, separate_dataset_su, make_batchnorm_stats,FixTransform,fetch_dataset_full_test
from data import fetch_dataset, split_dataset, make_data_loader, separate_dataset,separate_dataset_DA, separate_dataset_su, \
    make_batchnorm_dataset_su, make_batchnorm_stats , split_class_dataset,split_class_dataset_DA,make_data_loader_DA
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
from net_utils import set_random_seed
from net_utils import init_multi_cent_psd_label
from net_utils import EMA_update_multi_feat_cent_with_feat_simi
import numpy as np
# from pytorch_adapt.datasets import DataloaderCreator, get_office31
# from utils import init_param

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    exp_name = cfg['control_name'].split('_')[1]
    if cfg['domain_s'] in ['amazon','dslr','webcam']:
        cfg['data_name'] = 'office31'
    elif cfg['domain_s'] in ['art','clipart','product','realworld']:
        cfg['data_name'] = cfg['data_name_unsup'] = 'OfficeHome'
    elif cfg['domain_s'] in ['MNIST','SVHN','USPS']:
        cfg['data_name'] = cfg['domain_s']
    for i in range(cfg['num_experiments']):
        if cfg['data_name'] in ['office31', 'OfficeHome']:
            model_tag_list = [str(seeds[i]), cfg['domain_s'],str(cfg['var_lr']), cfg['model_name'],exp_num,exp_name]
        else:
            model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'],exp_num,exp_name]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        # exit()
        runExperiment()
    return


def runExperiment():
    # cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    # torch.manual_seed(cfg['seed'])
    # torch.cuda.manual_seed(cfg['seed'])
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    seed_val =  cfg['seed']
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_val)
    # random.seed(seed_val)
    torch.cuda.empty_cache()
    # dataset = fetch_dataset(cfg['data_name'])
    ####
    client_dataset_sup = fetch_dataset(cfg['data_name'],domain=cfg['domain_s'])
    print(len(client_dataset_sup['train']),client_dataset_sup['train'].transform)
    print(len(client_dataset_sup['test']),client_dataset_sup['test'].transform)
    # exit()
    # print(cfg['data_name_unsup'])
    client_dataset_unsup = fetch_dataset_full_test(cfg['data_name_unsup'],domain=cfg['domain_u'])
    # print(len(client_dataset_unsup['test']),client_dataset_unsup['test'].transform)
    print(client_dataset_unsup)
    
    # exit()
    #####
    # domains=['amazon','dslr']
    # k = get_office31(["amazon","dslr"],[],folder='data',download =True)
    # print(k.keys())
    # print(len(dataset['test']))
    # client_dataset_sup,client_dataset_unsup=k,k
    # process_dataset(dataset)
    print(cfg['domain_s'])

    process_dataset(client_dataset_sup,client_dataset_unsup)
    # cfg['num_supervised'] == -1
    # client_dataset_sup['train'], _, supervised_idx_sup = separate_dataset_su(client_dataset_sup['train'])
    # cfg['data_name'] = 'SVHN'
    # client_dataset_unsup['train'], _, supervised_idx_sup = separate_dataset_su(client_dataset_unsup['train'])
    # cfg['data_name'] = 'MNIST'
    # data_loader_sup = make_data_loader(client_dataset_sup['train'], 'global')
    # data_loader_unsup = make_data_loader(client_dataset_unsup['train'], 'global')
    # data_loader_sup_t = make_data_loader(client_dataset_sup['test'], 'global')
    # data_loader_unsup_t = make_data_loader(client_dataset_unsup['test'], 'global')
    # # print(len(supervised_idx))
    # data_loader = make_data_loader(dataset, 'global')
    transform_sup = FixTransform(cfg['data_name'])
    client_dataset_sup['train'].transform = transform_sup
    # # print(client_dataset_sup['train'].dataset.datasets.transform)
    # # print(client_dataset_unsup)
    # # print(client_dataset_sup['train'].__dict__.keys())
    # # exit()
    if not cfg['test_10_crop']:
        client_dataset_sup['test'].transform = transform_sup
    # transform_unsup = FixTransform(cfg['data_name_unsup'])
    # client_dataset_unsup['train'].transform = transform_unsup
    # print(cfg)
    # bt = cfg['bt']
    bt = 64
    cfg['global']['batch_size']={'train':bt,'test':2*64}
    print(cfg['global']['batch_size'])
    # print(client_dataset_sup.keys())
    data_loader_sup = make_data_loader_DA(client_dataset_sup, 'global')
    # for input in data_loader_sup['train']:
    #     print(type(input['data'][0]))
    #     print(input['augw'][0].shape)
    #     # for i in input['data']:
    #     #     print(i.shape)
    #     # exit()
    # exit()
    # data_loader_unsup = make_data_loader(client_dataset_unsup, 'global')
    model = eval('models.{}()'.format(cfg['model_name']))
    model_t = eval('models.{}()'.format(cfg['model_name']))
    # print(model)
    # model = convert_layers(model, torch.nn.BatchNorm2d, torch.nn.GroupNorm, num_groups = 2,convert_weights=False)
    # model_t = convert_layers(model_t, torch.nn.BatchNorm2d, torch.nn.GroupNorm, num_groups = 2,convert_weights=False)
    # model = convert_layers(model, torch.nn.BatchNorm1d, torch.nn.GroupNorm, num_groups = 2,convert_weights=False)
    # model_t = convert_layers(model_t, torch.nn.BatchNorm1d, torch.nn.GroupNorm, num_groups = 2,convert_weights=False)
    # model.apply(init_param)
    # model_t.apply(init_param)
    model = model.to(cfg['device'])
    model_t = model_t.to(cfg['device'])
    # print(model)
    # pre_model = torch.load('/home/sampathkoti/Downloads/R-50-GN.pkl',encoding= "latin1")
    # print(pre_model)
    ####
    # import pickle
    # path = "/home/sampathkoti/Downloads/R-50-GN.pkl"
    # # m = pickle.load(open(path, 'rb'))
    # # print(m)
    # with open(path, 'rb') as f:
    #     m=pickle.load(f,encoding="latin1")
    # print(model.state_dict().keys())
    # # print(m['blobs'].keys())
    # print(len(model.state_dict().keys()))
    # cou=0
    # for k,v in m['blobs'].items():
    #     if 'res2' in k:
    #         cou+=1
    #         print(k,v.shape)
    # # for k,v in model.state_dict().items():
    # #     print(k,v.shape)
    # print(cou)
    # exit()
    ######
    # print(model)
    # exit()
    # for name, module in model.named_modules():
    #     print(name)
    #     if isinstance(module, torch.nn.BatchNorm2d):
    #         print(module.num_features)
    #         # Get current bn layer
    #         # bn = getattr(model, name)
    #         # # Create new gn layer
    #         # print(bn)
    #         # k = eval('{}.{}.num_features'.format(model,name))
    #         # print(k)
    #         # gn = torch.nn.GroupNorm(2, bn.num_features)
    #         # # Assign gn
    #         # print('Swapping {} with {}'.format(bn, gn))
    #         # setattr(model, name, gn)
    # exit()
    # print(model)
    # net = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    # print(model)
    # print(cfg['local'].keys())
    # print(model.state_dict().keys())
    # exit()
    # if cfg['pretrained_source']:
    #     path_source = '/home/sampathkoti/Downloads/A-20231219T043936Z-001/A/'

    #     F = torch.load(path_source + 'source_F.pt')
    #     B = torch.load(path_source + 'source_B.pt')
    #     C = torch.load(path_source + 'source_C.pt')
    #     # print(F.keys())
    #     # exit()
    #     # model.backbone_layer.load_state_dict(torch.load(path_source + 'source_F.pt'))
    #     # model.feat_embed_layer.load_state_dict(torch.load(path_source + 'source_B.pt'))
    #     # model.class_layer.load_state_dict(torch.load(path_source + 'source_C.pt'))
    #     model.backbone_layer.load_state_dict(F)
    #     model.feat_embed_layer.load_state_dict(B)
    #     model.class_layer.load_state_dict(C)
        # # exit()
        # model.backbone_layer.conv1 = F.conv1
        # model.backbone_layer.bn1 = F.bn1
        # # model.backbone_layerbn1 = torch.nn.GroupNorm(2, 64)
        # model.backbone_layer.relu = F.relu
        # model.backbone_layer.maxpool = F.maxpool
        # model.backbone_layer.layer1 = F.layer1
        # model.backbone_layer.layer2 = F.layer2
        # model.backbone_layer.layer3 = F.layer3
        # model.backbone_layer.layer4 = F.layer4
        # model.backbone_layer.avgpool = F.avgpool
        # # model.backbone_layer.backbone_feat_dim = F.fc.in_features
    # print(model)
    # exit()
    cfg['local']['lr'] = cfg['var_lr']
    # print(cfg['global']['scheduler_name'])
    cfg['global']['scheduler_name'] = cfg['scheduler_name']
    # print(cfg['global']['scheduler_name'])
    optimizer = make_optimizer(model.parameters(), 'local')
    scheduler = make_scheduler(optimizer, 'global')
    metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    if cfg['resume_mode'] == 1:
        result = resume(cfg['model_tag'],'checkpoint')
        last_epoch = result['epoch']
        if last_epoch > 1:
            model.load_state_dict(result['model_state_dict'])
            optimizer.load_state_dict(result['optimizer_state_dict'])
            scheduler.load_state_dict(result['scheduler_state_dict'])
            logger = result['logger']
        else:
            logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    else:
        last_epoch = 1
        logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    if cfg['world_size'] > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    cfg['model_name'] = 'global'
    # print(list(model.buffers()))
    cfg['global']['num_epochs'] = cfg['cycles']
    cfg['local']['lr'] = cfg['var_lr']
    param_group = []
    learning_rate = cfg['var_lr']
    print('learning rate',learning_rate)
    for k, v in model.backbone_layer.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate*0.1}]
    for k, v in model.feat_embed_layer.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]
    for k, v in model.class_layer.named_parameters():
        param_group += [{'params': v, 'lr': learning_rate}]   
    optimizer = torch.optim.SGD(param_group)
    optimizer = op_copy(optimizer)
    # scheduler = make_scheduler(optimizer, 'global')
    # print(len(data_loader_sup['train']))
    # exit()
    cfg['iter_num'] =0 
    for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
        # cfg['model_name'] = 'local'
        logger.safe(True)
        # torch.cuda.empty_cache()
        
        train(data_loader_sup['train'], model, optimizer, metric, logger, epoch)
        # module = model.layer1[0].n1
        # print(list(module.named_buffers()))
        # print(list(model.buffers()))
        #===#
        # model_t = make_batchnorm_stats(client_dataset_sup['train'], model, cfg['model_name'])
        #====#
        # print(list(model.buffers()))
        # module = model.layer1[0].n1
        # print(list(module.named_buffers()))
        #====#
        model_t.load_state_dict(model.state_dict())
        #====#
        test_DA(data_loader_sup['test'], model_t, metric, logger, epoch)
        # print(list(model.buffers()))
        # module = model.layer1[0].n1
        # print(list(module.named_buffers()))
        # scheduler.step()
        logger.safe(False)
        model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
        result = {'cfg': cfg, 'epoch': epoch + 1,
                  'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler_state_dict': scheduler.state_dict(), 'logger': logger}
        if epoch%1==0:
            print('saving')
            print(cfg['model_tag'])
            save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
            if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
                metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
                shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
                            './output/model/{}_best.pt'.format(cfg['model_tag']))
        logger.reset()
    logger.safe(False)




    # test_model = make_batchnorm_stats(client_dataset_unsup['train'], model_t, cfg['model_name'])
    #     # print(list(model.buffers()))
    #     # module = model.layer1[0].n1
    #     # print(list(module.named_buffers()))
    # test(data_loader_unsup['test'], test_model, metric, logger, epoch)





    
    # cfg['loss_mode'] = 'bmd'
    # # cfg['data_name'] = 'MNIST'
    # last_epoch = 0
    # logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    # model_t = make_batchnorm_stats(client_dataset_unsup['train'], model, cfg['model_name'])
    # optimizer = make_optimizer(model_t.parameters(), 'local')
    # scheduler = make_scheduler(optimizer, 'global')
    # metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # # if cfg['resume_mode'] == 1:
    # #     result = resume(cfg['model_tag'],'best')
    # #     last_epoch = result['epoch']
    # #     if last_epoch > 1:
    # #         model.load_state_dict(result['model_state_dict'])
    # #         optimizer.load_state_dict(result['optimizer_state_dict'])
    # #         scheduler.load_state_dict(result['scheduler_state_dict'])
    # #         logger = result['logger']
    # #     else:
    # #         logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    # # else:
    # #     last_epoch = 1
    # #     logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    # if cfg['world_size'] > 1:
    #     model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    # cfg['model_name'] = 'global'
    # # print(list(model.buffers()))
    # cfg['global']['num_epochs'] = cfg['cycles']  
    # for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
    #     # cfg['model_name'] = 'local'
    #     logger.safe(True)
    #     train_da(client_dataset_unsup['train'], model_t, optimizer, metric, logger, epoch)
    #     # module = model.layer1[0].n1
    #     # print(list(module.named_buffers()))
    #     # print(list(model.buffers()))
    #     test_model = make_batchnorm_stats(client_dataset_unsup['train'], model_t, cfg['model_name'])
    #     # print(list(model.buffers()))
    #     # module = model.layer1[0].n1
    #     # print(list(module.named_buffers()))
    #     test(data_loader_unsup['test'], test_model, metric, logger, epoch)
    #     # print(list(model.buffers()))
    #     # module = model.layer1[0].n1
    #     # print(list(module.named_buffers()))
    #     scheduler.step()
    #     logger.safe(False)
    #     model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
    #     result = {'cfg': cfg, 'epoch': epoch + 1,
    #               'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer.state_dict(),
    #               'scheduler_state_dict': scheduler.state_dict(), 'logger': logger}
    #     if epoch%5==0:
    #         print('saving')
    #         save(result, './output/model_t/{}_checkpoint.pt'.format(cfg['model_tag']))
    #         if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
    #             metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
    #             shutil.copy('./output/model_t/{}_checkpoint.pt'.format(cfg['model_tag']),
    #                         './output/model_t/{}_best.pt'.format(cfg['model_tag']))
    #     logger.reset()
    # logger.safe(False)
    #########################################################################################################################
    #########################################################################################################################
    # end of central da
    # net.load_state_dict(result['model_state_dict'])
    # # net.load_state_dict(result['model_state_dict'])
    # net.to(cfg['device'])
    # net.eval()
    #########################################################################################################################
    # # reserved to compute test accuracy on generated images by different networks
    # net_verifier = None
    # args = dict()
    # args['verifier'] = False
    # args['adi_scale'] = 0
    # # if args.verifier and args.adi_scale == 0:
    # #     # if multiple GPUs are used then we can change code to load different verifiers to different GPUs
    # #     if args.local_rank == 0:
    # #         print("loading verifier: ", args.verifier_arch)
    # #         net_verifier = models.__dict__[args.verifier_arch](pretrained=True).to(cfg['device'])
    # #         net_verifier.eval()

    # #         if use_fp16:
    # #             net_verifier = net_verifier.half()

    # # if args.adi_scale != 0.0:
    # #     student_arch = "resnet18"
    # #     net_verifier = models.__dict__[student_arch](pretrained=True).to(device)
    # #     net_verifier.eval()

    # #     if use_fp16:
    # #         net_verifier, _ = amp.initialize(net_verifier, [], opt_level="O2")

    # #     net_verifier = net_verifier.to(device)
    # #     net_verifier.train()

    # #     if use_fp16:
    # #         for module in net_verifier.modules():
    # #             if isinstance(module, nn.BatchNorm2d):
    # #                 module.eval().half()

    # from deepinversion import DeepInversionClass
    # from cifar10inversion import get_images
    # # exp_name = args.exp_name
    # exp_name = 'rn9_test3_inversion'
    # # final images will be stored here:
    # adi_data_path = "./final_images/%s"%exp_name
    # # temporal data and generations will be stored here
    # exp_name = "generations/%s"%exp_name

    # args['iterations'] = 200000
    # args['start_noise'] = True
    # # args.detach_student = False

    # args['resolution'] = 32
    # # bs = args.bs
    # bs = 256
    # jitter = 30
    # parameters = dict()
    # parameters["resolution"] = 32
    # parameters["random_label"] = False
    # parameters["start_noise"] = True
    # parameters["detach_student"] = False
    # parameters["do_flip"] = False
    # # parameters["do_flip"] = args.do_flip
    # # parameters["random_label"] = args.random_label
    # parameters["random_label"] = True
    # parameters["store_best_images"] = True
    # # parameters["store_best_images"] = args.store_best_images

    # criterion = torch.nn.CrossEntropyLoss()

    # # coefficients = dict()
    # # coefficients["r_feature"] = args.r_feature
    # # coefficients["first_bn_multiplier"] = args.first_bn_multiplier
    # # coefficients["tv_l1"] = args.tv_l1
    # # coefficients["tv_l2"] = args.tv_l2
    # # coefficients["l2"] = args.l2
    # # coefficients["lr"] = args.lr
    # # coefficients["main_loss_multiplier"] = args.main_loss_multiplier
    # # coefficients["adi_scale"] = args.adi_scale
    # coefficients = dict()
    # coefficients["r_feature"] = 0.01    
    # coefficients["first_bn_multiplier"] = 10
    # coefficients["tv_l1"] =0.0
    # coefficients["tv_l2"] =0.0001
    # coefficients["l2"] = 0.00001  
    # coefficients["lr"] = 0.25
    # coefficients["main_loss_multiplier"] = 1
    # coefficients["adi_scale"] = 0.0
    # network_output_function = lambda x: x

    # # check accuracy of verifier
    # # if args.verifier:
    # #     hook_for_display = lambda x,y: validate_one(x, y, net_verifier)
    # # else:
    # #     hook_for_display = Nonee?
    # hook_for_display = None
    # ###################################################################
    # criterion = torch.nn.CrossEntropyLoss()
    # discp = 'cifar10_50k'
    # prefix = "runs/data_generation/"+discp+"/"

    # for create_folder in [prefix, prefix+"/best_images/"]:
    #     if not os.path.exists(create_folder):
    #         os.makedirs(create_folder)
    # # place holder for inputs
    # data_type = torch.float
    # inputs = torch.randn((bs, 3, 32, 32), requires_grad=True, device=cfg['device'], dtype=data_type)
    # net_student=None
    # train_writer = None  # tensorboard writter
    # global_iteration = 0
    # print("Starting model inversion")
    # inputs = get_images(net=net, bs=bs, epochs=args['iterations'], idx=0,
    #                     net_student=net_student, prefix=prefix, competitive_scale=0.0,
    #                     train_writer=train_writer, global_iteration=global_iteration, use_amp= False,
    #                     optimizer=torch.optim.Adam([inputs], lr=0.1), inputs=inputs, bn_reg_scale=10,
    #                     var_scale=25e-6, random_labels=False, l2_coeff=0.0)
    return

def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True

def train(data_loader, model, optimizer, metric, logger, epoch):
    model.train(True)
    start_time = time.time()
    # model.projection.requires_grad_(False)
    
    for i, input in enumerate(data_loader):
        # torch.cuda.empty_cache()
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        input['loss_mode'] = cfg['loss_mode']
        if input_size == 1:
            break
        # print(input['data'].shape)
        # exit()
        cfg['iter_num']+=1
        max_iter = cfg['cycles']*len(data_loader)
        lr_scheduler(optimizer, iter_num=cfg['iter_num'], max_iter=max_iter)
        output = model(input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
        # print(output['loss'])
        output['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        evaluation = metric.evaluate(metric.metric_name['train'], input, output)
        logger.append(evaluation, 'train', n=input_size)
        if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
            _time = (time.time() - start_time) / (i + 1)
            lr = optimizer.param_groups[0]['lr']
            epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
            exp_finished_time = epoch_finished_time + datetime.timedelta(
                seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
            info = {'info': ['Model: {}'.format(cfg['model_tag']),
                             'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
                             'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
                             'Experiment Finished Time: {}'.format(exp_finished_time)]}
            logger.append(info, 'train', mean=False)
            print(logger.write('train', metric.metric_name['train']))
    return

def train_da(dataset, model, optimizer, metric, logger, epoch):
    train_data_loader = make_data_loader({'train': dataset}, 'client')['train']
    test_data_loader = make_data_loader({'train': dataset},'client',batch_size = {'train':500},shuffle={'train':False})['train']
    # model.train(True)
    start_time = time.time()
    loss_stack = []
    with torch.no_grad():
        model.eval()
        # print("update psd label bank!")
        glob_multi_feat_cent, all_psd_label = init_multi_cent_psd_label(model,test_data_loader)
        
    model.train()
    epoch_idx=epoch
    print(epoch)
    for i, input in enumerate(train_data_loader):
        # print(i)
        input = collate(input)
        input_size = input['data'].size(0)
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
        
        if epoch_idx >= 1.0:
            loss = ent_loss + 2.0 * psd_loss
        else:
            loss = -reg_loss + ent_loss
        
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
        #==================================================================#
        # lr_scheduler(optimizer, iter_idx, iter_max)
        # optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        with torch.no_grad():
            loss_stack.append(loss.cpu().item())
            glob_multi_feat_cent = EMA_update_multi_feat_cent_with_feat_simi(glob_multi_feat_cent, embed_feat, decay=0.9999)
        # output = model(input)
        # print(output.keys())
    train_loss = np.mean(loss_stack)
    print(train_loss)
    # model.projection.requires_grad_(False)
    # for i, input in enumerate(data_loader):
    #     input = collate(input)
    #     input_size = input['data'].size(0)
    #     input = to_device(input, cfg['device'])
    #     optimizer.zero_grad()
    #     input['loss_mode'] = cfg['loss_mode']
    #     output = model(input)
    #     output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
    #     output['loss'].backward()
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
    #     optimizer.step()
    #     evaluation = metric.evaluate(metric.metric_name['train'], input, output)
    #     logger.append(evaluation, 'train', n=input_size)
    #     if i % int((len(data_loader) * cfg['log_interval']) + 1) == 0:
    #         _time = (time.time() - start_time) / (i + 1)
    #         lr = optimizer.param_groups[0]['lr']
    #         epoch_finished_time = datetime.timedelta(seconds=round(_time * (len(data_loader) - i - 1)))
    #         exp_finished_time = epoch_finished_time + datetime.timedelta(
    #             seconds=round((cfg[cfg['model_name']]['num_epochs'] - epoch) * _time * len(data_loader)))
    #         info = {'info': ['Model: {}'.format(cfg['model_tag']),
    #                          'Train Epoch: {}({:.0f}%)'.format(epoch, 100. * i / len(data_loader)),
    #                          'Learning rate: {:.6f}'.format(lr), 'Epoch Finished Time: {}'.format(epoch_finished_time),
    #                          'Experiment Finished Time: {}'.format(exp_finished_time)]}
    #         logger.append(info, 'train', mean=False)
    #         print(logger.write('train', metric.metric_name['train']))
    return

def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer

def test(data_loader, model, metric, logger, epoch):
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
            evaluation = metric.evaluate(metric.metric_name['test'], input, output)
            logger.append(evaluation, 'test', input_size)
        info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
        logger.append(info, 'test', mean=False)
        print(logger.write('test', metric.metric_name['test']))
    return
def test_DA(data_loader, model, metric, logger, epoch):
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
                    logger.append(evaluation, 'test', input_size)
            
                
            info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
            logger.append(info, 'test', mean=False)
            print(logger.write('test', metric.metric_name['test']))
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
                    evaluation = metric.evaluate(metric.metric_name['test'], input, output)
                    logger.append(evaluation, 'test', input_size)
                info = {'info': ['Model: {}'.format(cfg['model_tag']), 'Test Epoch: {}({:.0f}%)'.format(epoch, 100.)]}
                logger.append(info, 'test', mean=False)
                print(logger.write('test', metric.metric_name['test']))

    return

def convert_layers(model, layer_type_old, layer_type_new, num_groups,convert_weights=False):
    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = convert_layers(module, layer_type_old, layer_type_new, convert_weights)

        if type(module) == layer_type_old:
            layer_old = module
            # layer_new = layer_type_new(module.num_features if num_groups is None else num_groups, module.num_features, module.eps, module.affine) 
            layer_new = layer_type_new(32, module.num_features, module.eps, module.affine) 


            if convert_weights:
                layer_new.weight = layer_old.weight
                layer_new.bias = layer_old.bias

            model._modules[name] = layer_new

    return model
def init_param(m):
    if isinstance(m, nn.Conv2d) and isinstance(m, models.DecConv2d):
        nn.init.kaiming_normal_(m.sigma_weight, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(m.phi_weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, torch.nn.GroupNorm):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.bias.data.zero_()
    return m
if __name__ == "__main__":
    main()
