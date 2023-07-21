import argparse
import copy
import datetime
import models
from models.LSTM import Forecast
import os
import shutil
import time
import torch
import torch.backends.cudnn as cudnn
from config import cfg, process_args
from data import fetch_dataset, make_data_loader, separate_dataset_su, make_batchnorm_stats
from metrics import Metric
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate
from logger import make_logger
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# from tsf_test import series_to_supervised,predict_iteration
from models.RNN_models import *
from sklearn.metrics import mean_squared_error,mean_absolute_error


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
cudnn.benchmark = True
parser = argparse.ArgumentParser(description='cfg')
for k in cfg:
    exec('parser.add_argument(\'--{0}\', default=cfg[\'{0}\'], type=type(cfg[\'{0}\']))'.format(k))
parser.add_argument('--control_name', default=None, type=str)
args = vars(parser.parse_args())
process_args(args)

# class Forecast(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.lstm = torch.nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
#         # self.lstm = torch.nn.LSTM(input_size=1, hidden_size=50, num_layers=1, batch_first=True)
#         self.linear = torch.nn.Linear(50, 1)
#     def forward(self, x):
#         x, _ = self.lstm(x)
#         x = self.linear(x)
#         return x
def main():
    process_control()
    seeds = list(range(cfg['init_seed'], cfg['init_seed'] + cfg['num_experiments']))
    for i in range(cfg['num_experiments']):
        model_tag_list = [str(seeds[i]), cfg['data_name'], cfg['model_name'], cfg['control_name']]
        cfg['model_tag'] = '_'.join([x for x in model_tag_list if x])
        print('Experiment: {}'.format(cfg['model_tag']))
        runExperiment()
    return
def predict_iteration(net, testX, lookAhead, RNN=True, use_cuda=True):

    testBatchSize = testX.shape[0]
    # print(testX.shape)
    ans = []

    for i in range(lookAhead):

        testX_torch = torch.from_numpy(testX)
        testX_torch = Variable(testX_torch)
        if use_cuda:
            testX_torch = testX_torch.to(cfg['device'])
        pred = net(testX_torch)
        if use_cuda:
            pred = pred.cpu().data.numpy()
        else:
            pred = pred.data.numpy()
        pred = np.squeeze(pred)
        ans.append(pred)
        # print(pred[-1])
        testX = testX[:, lookAhead:]  # drop the head
        # print(testX.shape)
        # print(pred.shape)
        # print('###############')
        if RNN:
            pred = pred.reshape((testBatchSize, lookAhead, 1))
            testX = np.append(testX, pred, axis=1)  # add the prediction to the tail
        else:
            pred = pred.reshape((testBatchSize, 1))
            testX = np.append(testX, pred, axis=1)  # add the prediction to the tail

    ans = np.array(ans)
    # print(ans.shape)
    # ans = ans.transpose([1,0])
    # print(ans.shape)
    return ans[0]
def get_dataset(path,filename):
    # df = pd.read_csv('path')
    # print(path)
    # elec = pd.read_parquet(path+'/'+filename)
    # elec.to_csv(path+'/15228-10.csv')
    elec = pd.read_csv(path+'/'+filename,index_col=1)
    elec.index = pd.to_datetime(elec.index,format='%Y-%m-%d %H:%M:%S')
    # elec.index = pd.to_datetime(elec.index,format='%Y-%m-%d %H:00:00')
    # print(elec.head())
    timeseries = elec[["out.site_energy.total.energy_consumption"]].values.astype('float32')
    datetime = elec.index
    # print(type(datetime))
    # plt.scatter(list(range(len(timeseries[:1000]))),timeseries[:1000])
    # plt.plot(timeseries[:1000])
    # plt.show()
    # print(len(timeseries))
    # for cl in elec.columns:
    #     print(cl)
    # out.site_energy.total.energy_consumption_intensity
    # timeseries = elec[["Passengers"]].values.astype('float32')
    return timeseries,datetime,elec
def create_dataset(dataset, lookback,pred_ahead):
    """Transform a time series into a prediction dataset
    
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    print(len(dataset)-lookback-pred_ahead,len(dataset))
    for i in range(len(dataset)-lookback-pred_ahead):
        feature = dataset[i:i+lookback]
        # target = dataset[i+1:i+lookback+1]
        # target = dataset[i+1:i+lookback+pred_ahead]
        target = dataset[i+lookback:i+lookback+pred_ahead]
        # print(i,feature,target)
        # print(target.T.shape)
        X.append(feature)
        y.append(target)
    # print(X[0],y[0])
    # print(X[1],y[1])
    return torch.tensor(X), torch.tensor(y)
#  convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = pd.DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = pd.concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
def runExperiment():
    cfg['seed'] = int(cfg['model_tag'].split('_')[0])
    torch.manual_seed(cfg['seed'])
    torch.cuda.manual_seed(cfg['seed'])
    path  = '/home/sampathkoti/Downloads'
    filename = '15228-10.csv'
    timeseries,datetime,df = get_dataset(path,filename)
    lookback = 5
    pred_ahead = 2
    #################################################
    # normalise features
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(timeseries)
    # scaled = timeseries
    n_mins = 5
    n_features = 1
    reframed = series_to_supervised(scaled, lookback,pred_ahead)
    # drop columns we don't want to predict
    #reframed.drop(reframed.columns[-24:-1], axis=1, inplace=True)
    print(reframed.shape)
    print(reframed.head())
    # timeseries_= timeseries
    # timeseries = scaled
    # print(df["out.site_energy.total.energy_consumption_intensity"].describe)
    # train-test split for time series
    timeseries= reframed.values
    # print(timeseries[0][-pred_ahead:])
    ####################################################################
    train_size = int(len(timeseries) * 0.67)
    test_size = len(timeseries) - train_size
    train, test = timeseries[:train_size], timeseries[train_size:]
    #####################################################################
    print('train shape and test shape')
    # print(train.shape,test.shape)
    # split into input and outputs
    n_obs = n_mins * n_features
    train,test = torch.tensor(train), torch.tensor(test)
    # train_X, train_y = train[:, :n_obs], train[:, -1]
    # test_X,  test_y =test[:, :n_obs], test[:,-1]
    train_X, train_y = train[:, :n_obs], train[:, -pred_ahead:]
    test_X,  test_y =test[:, :n_obs], test[:,-pred_ahead:]
    # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_mins, n_features))
    test_X = test_X.reshape((test_X.shape[0], n_mins, n_features))
    # train_y = train_y.reshape((train_y.shape[0], pred_ahead, n_features))
    # test_y = test_y.reshape((test_y.shape[0], pred_ahead, n_features))

    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)
    # print(train_X[0].shape[0], train_y[0].shape[0], test_X[0].shape[0], test_y[0].shape[0])
    ######################################################################
    # X_train, y_train = create_dataset(train, lookback=lookback,pred_ahead=pred_ahead)
    # X_test, y_test = create_dataset(test, lookback=lookback,pred_ahead=pred_ahead)
    # print(X_train.shape,y_train.shape)
    # input_size=lookback
    input_size = n_features
    output_size=pred_ahead
    hidden_dim_l1=128
    hidden_dim_l2=256
    n_layers =1
    # model = Forecast(input_size,output_size,hidden_dim_l1,hidden_dim_l2,n_layers).to(cfg['device'])
    model = LSTMModel(inputDim=1, hiddenNum=hidden_dim_l1, outputDim=pred_ahead, layerNum=n_layers, cell="LSTM", use_cuda=True)
    model = model.to(cfg['device'])
    lr = 0.005
    batch_size = 50
    optimizer = torch.optim.Adam(model.parameters(),lr=lr)
    criterion = torch.nn.MSELoss()
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(train_X, train_y), shuffle=False, batch_size=50)
    test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(test_X,test_y), shuffle=False, batch_size=50)

    epochs = 2000
    counter = 0
    print_every = 1000
    clip = 5
    # valid_loss_min = np.Inf

    model.train()
    for i in range(epochs):
        # h1 = model.init_hidden(batch_size)
        for inputs, labels in train_loader:
            counter += 1
            # print(inputs.shape,labels.shape)
            # if inputs.shape[0]!=batch_size:
            #     h1 = model.init_hidden(inputs.shape[0])
            # h1 = tuple([e1.data for e1 in h1])
            # print(h1[0].shape)
            # h2 = tuple([e2.data for e2 in h2])
            # print(h1,h1)
            # h1,h2=h1.to(cfg['device']),h2.to(cfg['device'])
            inputs, labels = inputs.to(cfg['device']), labels.to(cfg['device'])
            model.zero_grad()
            # output, h1 = model(inputs, h1)
            output = model(inputs)
            # print('################')
            # print(inputs,labels.squeeze())
            # print(output[0],labels[0])
            # print(output.shape,labels.shape)
            # exit()
            loss = criterion(output.squeeze(), labels.squeeze().float())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            
            if counter%print_every == 0:
                # val_h = model.init_hidden(batch_size)
                # val_losses = []
                # model.eval()
                # for inp, lab in val_loader:
                #     val_h = tuple([each.data for each in val_h])
                #     inp, lab = inp.to(cfg['device']), lab.to(cfg['device'])
                #     out, val_h = model(inp, val_h)
                #     val_loss = criterion(out.squeeze(), lab.float())
                #     val_losses.append(val_loss.item())
                    
                model.train()
                print("Epoch: {}/{}...".format(i+1, epochs),
                    "Step: {}...".format(counter),
                    "Loss: {:.10f}...".format(loss.item()))
                    # "Val Loss: {:.6f}".format(np.mean(val_losses)))
                # if np.mean(val_losses) <= valid_loss_min:
                #     torch.save(model.state_dict(), './state_dict.pt')
                #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,np.mean(val_losses)))
                #     valid_loss_min = np.mean(val_losses)
    # print(X_train[0],y_train[[0]])
    # print(X_train[1],y_train[[1]])
    #############################################################################################################
    test_X_ = test_X.data.cpu().numpy()
    testY = test_y.data.cpu().numpy()
    # print(test_X_[-2])
    # print(testY[-2])
    # print(test_X_[-1])
    # print(testY[-1])
    testPred = predict_iteration(model, test_X_, pred_ahead, use_cuda=True, RNN=True)
    # trainPred = predict_iteration(net, trainX, h_train, use_cuda=use_cuda, RNN=flag)
    # print("train pred shape:", trainPred.shape)
    print("test pred shape:", testPred.shape)

    testPred = scaler.inverse_transform(testPred)
    testY = scaler.inverse_transform(testY)

    # evaluation
    MAE = mean_absolute_error(testY, testPred)
    print("test MAE", MAE)
    MRSE =  np.sqrt(mean_squared_error(testY, testPred))
    print("test RMSE", MRSE)
    # MAPE = eval.calcMAPE(testY, testPred)
    # print("test MAPE", MAPE)
    # SMAPE = eval.calcSMAPE(testY, testPred)
    # print("test SMAPE", SMAPE)

    ########################################################################################################
    test_losses = []
    num_correct = 0
    batch_size = 50
    # h = model.init_hidden(batch_size)
    act_load,pred_load=[],[]
    model.eval()
    for inputs, labels in test_loader:
        # if inputs.shape[0]!=batch_size:
        #         h = model.init_hidden(inputs.shape[0])
        # h = tuple([each.data for each in h])
        inputs, labels = inputs.to(cfg['device']), labels.to(cfg['device'])
        output= model(inputs)
        output = scaler.inverse_transform(output.data.cpu().numpy())
        labels = scaler.inverse_transform(labels.data.cpu().numpy())
        # test_loss = criterion(output, labels.float())
        test_loss = np.sqrt(mean_squared_error(output, labels))
        # print(output.shape,labels.shape)
        # if inputs.shape[0]!=batch_size:
        #     plt.plot(range(len(labels[inputs.shape[0]-1].squeeze().float())),labels[inputs.shape[0]-1].squeeze().float().cpu().detach().numpy())
        #     plt.plot(range(len(labels[inputs.shape[0]-1].squeeze().float())),output[inputs.shape[0]-1].squeeze().cpu().detach().numpy())
        #     plt.show()
        # act_load.extend(labels.cpu().detach().numpy())
        # pred_load.extend(output.cpu().detach().numpy())
        test_losses.append(test_loss.item())
        # pred = torch.round(output.squeeze())  # Rounds the output to 0/1
        # correct_tensor = pred.eq(labels.float().view_as(pred))
        # correct = np.squeeze(correct_tensor.cpu().numpy())
        # num_correct += np.sum(correct)

    print("Test loss: {:.10f}".format(np.mean(test_losses)))
    # test_acc = num_correct/len(test_loader.dataset)
    # print("Test accuracy: {:.10f}%".format(test_acc*100))
    # print(len(act_load),len(pred_load))
    # print(act_load,pred_load)
    # plt.plot(range(len(act_load)),act_load)
    # plt.plot(range(len(act_load)),pred_load)
    # plt.show()
    
    # dataset = fetch_dataset(cfg['data_name'])
    # # print(len(dataset['test']))
    # process_dataset(dataset)
    # dataset['train'], _, supervised_idx = separate_dataset_su(dataset['train'])
    # # print(len(supervised_idx))
    # data_loader = make_data_loader(dataset, 'global')
    # model = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    # net = eval('models.{}().to(cfg["device"])'.format(cfg['model_name']))
    # # print(model)
    # # print(cfg['local'].keys())
    # optimizer = make_optimizer(model.parameters(), 'local')
    # scheduler = make_scheduler(optimizer, 'global')
    # metric = Metric({'train': ['Loss', 'Accuracy'], 'test': ['Loss', 'Accuracy']})
    # if cfg['resume_mode'] == 1:
    #     result = resume(cfg['model_tag'],'best')
    #     last_epoch = result['epoch']
    #     if last_epoch > 1:
    #         model.load_state_dict(result['model_state_dict'])
    #         optimizer.load_state_dict(result['optimizer_state_dict'])
    #         scheduler.load_state_dict(result['scheduler_state_dict'])
    #         logger = result['logger']
    #     else:
    #         logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    # else:
    #     last_epoch = 1
    #     logger = make_logger(os.path.join('output', 'runs', 'train_{}'.format(cfg['model_tag'])))
    # if cfg['world_size'] > 1:
    #     model = torch.nn.DataParallel(model, device_ids=list(range(cfg['world_size'])))
    # cfg['model_name'] = 'global'
    # # print(list(model.buffers()))
    # cfg['global']['num_epochs'] = cfg['cycles']  
    # for epoch in range(last_epoch, cfg[cfg['model_name']]['num_epochs'] + 1):
    #     # cfg['model_name'] = 'local'
    #     logger.safe(True)
    #     train(data_loader['train'], model, optimizer, metric, logger, epoch)
    #     # module = model.layer1[0].n1
    #     # print(list(module.named_buffers()))
    #     # print(list(model.buffers()))
    #     test_model = make_batchnorm_stats(dataset['train'], model, cfg['model_name'])
    #     # print(list(model.buffers()))
    #     # module = model.layer1[0].n1
    #     # print(list(module.named_buffers()))
    #     test(data_loader['test'], test_model, metric, logger, epoch)
    #     # print(list(model.buffers()))
    #     # module = model.layer1[0].n1
    #     # print(list(module.named_buffers()))
    #     scheduler.step()
    #     logger.safe(False)
    #     model_state_dict = model.module.state_dict() if cfg['world_size'] > 1 else model.state_dict()
    #     result = {'cfg': cfg, 'epoch': epoch + 1, 'supervised_idx': supervised_idx,
    #               'model_state_dict': model_state_dict, 'optimizer_state_dict': optimizer.state_dict(),
    #               'scheduler_state_dict': scheduler.state_dict(), 'logger': logger}
    #     save(result, './output/model/{}_checkpoint.pt'.format(cfg['model_tag']))
    #     if metric.compare(logger.mean['test/{}'.format(metric.pivot_name)]):
    #         metric.update(logger.mean['test/{}'.format(metric.pivot_name)])
    #         shutil.copy('./output/model/{}_checkpoint.pt'.format(cfg['model_tag']),
    #                     './output/model/{}_best.pt'.format(cfg['model_tag']))
    #     logger.reset()
    # logger.safe(False)
    # #Get Class impressions
    # cfg['loss_mode'] = 'gen'
    
    # net.load_state_dict(result['model_state_dict'])
    # # net.load_state_dict(result['model_state_dict'])
    # net.to(cfg['device'])
    # net.eval()

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


def train(data_loader, model, optimizer, metric, logger, epoch):
    model.train(True)
    start_time = time.time()
    model.projection.requires_grad_(False)
    for i, input in enumerate(data_loader):
        input = collate(input)
        input_size = input['data'].size(0)
        input = to_device(input, cfg['device'])
        optimizer.zero_grad()
        input['loss_mode'] = cfg['loss_mode']
        output = model(input)
        output['loss'] = output['loss'].mean() if cfg['world_size'] > 1 else output['loss']
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


if __name__ == "__main__":
    main()
