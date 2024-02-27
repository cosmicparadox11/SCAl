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
    make_batchnorm_dataset_su, make_batchnorm_stats , split_class_dataset,split_class_dataset_DA,make_data_loader_DA,make_batchnorm_stats_DA,fetch_dataset_full_test
from metrics import Metric
from modules import Server, Client
from utils import save, to_device, process_control, process_dataset, make_optimizer, make_scheduler, resume, collate,resume_DA,process_dataset_multi,load_Cent
from logger import make_logger
import gc
import faiss
import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
import pickle
from sklearn.cluster import KMeans, SpectralClustering, DBSCAN, MeanShift
from sklearn.metrics import adjusted_rand_score
from sklearn_extra.cluster import KMedoids
from scipy.cluster import hierarchy
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import fcluster


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

def load(path, mode='torch'):
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
        # return torch.load(path, map_location= torch.device(cfg['device']))
    
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return

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
    
    
    
    
    
    cfg['target_size'] = 65
    feat_all = []
    # model_tag = '2020_art_to_product_clipart_realworld_resnet50_1000102'
    # model_tag = '2020_art_to_product_clipart_realworld_resnet50_88888001'
    # model_tag = '2020_realworld_to_product_art_clipart_resnet50_88888002'
    # model_tag = '2020_clipart_to_product_art_realworld_resnet50_88888002'
    # model_tag = '2020_product_to_clipart_art_realworld_resnet50_88888003'
    for i in range(1,11):
        model_tag = '2020_product_to_clipart_art_realworld_resnet50_0000'
        load_tag = f'checkpoint{i}'
        result = load('./output/model/target/{}_{}.pt'.format(model_tag, load_tag))
        clients = result['client']
        model = eval('models.{}()'.format(cfg['model_name']))
        feat = []
        client_ids =[]
        domain_ids = []
        for client in clients:
            # print(client.client_id,client.domain_id)
            client_ids.append(client.client_id)
            domain_ids.append(client.domain_id)

            model.load_state_dict(client.model_state_dict)
            # print(model.state_dict()['feat_embed_layer.bn.running_mean'].shape)
            # print(model.state_dict().keys())
            # f1 = model.state_dict()['feat_embed_layer.bn.running_mean'].reshape(-1,1)
            # f2 = model.state_dict()['backbone_layer.layer4.2.bn3.running_mean'].reshape(-1,1)
            # # feat = f1.extend(f2)
            # f1 = f1/(1e-9+torch.norm(f1,dim = 0))
            # f2 = f2/(1e-9+torch.norm(f2,dim =0))
            # feat_ = torch.concat([f1,f2],dim = 0)
            # feat.append(np.array(feat_.squeeze()))
            # print(f/
            # feat = np.concatenate([f1,f2],axis=0)
            feat.append(np.array(model.state_dict()['feat_embed_layer.bn.running_mean']))
            # feat.append(np.array(model.state_dict()['feat_embed_layer.bn.running_varience']))
            # feat.append(np.array(model.state_dict()['backbone_layer.layer4.2.bn3.running_mean']))
            model.state_dict()
            # exit()
            # for k, v in model.named_parameters():
            #     isBatchNorm = True if  '.bn' in k else False
            #     bn_k = '.'.join(k.split('.')[:-1])
            #     if isBatchNorm:
            #         mean = eval(f'{model}.{bn_k}.running_mean')
            #         print(bn_k)
            #         print(mean.shape)
            #         exit()
                
            # exit()
        
        feat = np.array(feat)
        feat = feat/(1e-9+np.linalg.norm(feat,axis=1,keepdims = True))
        feat_all.append(feat)
    # print(feat.shape)
    # exit()
    # feat_all = [feat_all[0],feat_all[5],feat[8]]
    # X= feat
    # y = [int(gt)+1 for gt  in domain_ids ]
    y = domain_ids
    print(y)
    # exit()
    labels = ['hierarchical','K-Means', 'Spectral Clustering', 'DBSCAN', 'Mean Shift', 'K-Medoids']
    ARI_all =[]
    # hi
    for j in range(10):
        X = feat_all[j]
        Z = hierarchy.linkage(X, method='ward')
        hierarchical_pred = fcluster(Z, 3, criterion='maxclust')
        hierarchical_pred = [int(h)-1 for h in hierarchical_pred]
        ari_hierarchical = adjusted_rand_score(y, hierarchical_pred)
        print(hierarchical_pred)
        # K-Means clustering
        kmeans = KMeans(n_clusters=3)
        kmeans_pred = kmeans.fit_predict(X)
        print(kmeans_pred)
        # Spectral Clustering
        spectral = SpectralClustering(n_clusters=3, affinity='nearest_neighbors')
        spectral_pred = spectral.fit_predict(X)
        print(spectral_pred)
        # DBSCAN
        dbscan = DBSCAN(eps=0.9, min_samples=2)
        dbscan_pred = dbscan.fit_predict(X)
        print(dbscan_pred)
        # Mean Shift
        mean_shift = MeanShift()
        mean_shift_pred = mean_shift.fit_predict(X)
        print(mean_shift_pred)
        # K-Medoids
        kmedoids = KMedoids(n_clusters=3, random_state=0)
        kmedoids_pred = kmedoids.fit_predict(X)
        print(kmedoids_pred)
        # Calculate Adjusted Rand Index (ARI) with ground truth labels
        ari_kmeans = adjusted_rand_score(y, kmeans_pred)
        ari_spectral = adjusted_rand_score(y, spectral_pred)
        ari_dbscan = adjusted_rand_score(y, dbscan_pred)
        ari_mean_shift = adjusted_rand_score(y, mean_shift_pred)
        ari_kmedoids = adjusted_rand_score(y, kmedoids_pred)

            # Plotting ARI values
        labels = ['hierarchical','K-Means', 'Spectral Clustering', 'DBSCAN', 'Mean Shift', 'K-Medoids']
        ari_values = [ari_hierarchical,ari_kmeans, ari_spectral, ari_dbscan, ari_mean_shift, ari_kmedoids]
        ARI_all.append(ari_values)
    plt.figure(figsize=(10, 6))
    cmap = plt.get_cmap('jet')
    colors = cmap(np.linspace(0, 1.0, 10))
    # plt.bar(labels, ari_values, color='skyblue')
    li = [0,5,8]
    for i in range(10):
        if i in li:
            plt.plot(labels, ARI_all[i], marker='o',label= f'C-{i}', color=colors[i], linestyle='-')
    plt.title('Adjusted Rand Index (ARI) of Clustering Algorithms')
    plt.ylabel('ARI')
    plt.ylim(0, 1)  # Adjust y-axis limits if needed
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.legend(loc = 'best')
    plt.savefig('./output/ARI_different_algo_all.pdf',dpi = 600,format = 'pdf',bbox_inches = 'tight',pad_inches = 0)
    exit()

    print(feat.shape)
    num_clusters = 3
    kmeans = faiss.Kmeans(feat.shape[1],num_clusters, niter=1000,  verbose=True,max_points_per_centroid=15)
    kmeans.train(feat)
    D, I = kmeans.index.search(feat, 1)
    asnd=[]
    for idx in I:
        asnd.append(idx[0])
    
    client_ids = np.array(client_ids)
    domain_ids = np.array(domain_ids)
    asnd = np.array(asnd)
    c0 = client_ids[domain_ids==0]
    c1 = client_ids[domain_ids==1]
    c2 = client_ids[domain_ids==2]
    print(c0,c1,c2)
    for i in range(num_clusters):
        print(f'cluster {i}',client_ids[asnd==i])
    # a0 = client_ids[asnd==0]
    # a1 = client_ids[asnd==1]
    # a2 = client_ids[asnd==2]
    # a3 = client_ids[asnd==3]
    # a4 = client_ids[asnd==4]
    # a5 = client_ids[asnd==5]
    # print(a0,a1,a2,a3,a4,a5)
    exit()
    if cfg['resume_mode'] == 1:
        
        epoch_num = 1
        cent=[]
        client_ids =[]
        domain_ids = []
        cent_info = load_Cent(epoch_num)
        for k,v in cent_info.items():
            print(k,v[2].shape)
            cent.append(np.array(v[2]))
            client_ids.append(k)
            domain_ids.append(v[0])
        cent = np.array(cent)
        print(cent.shape)
    
    num_cluster = 3
    class_cent = []
    class_labels = []
    for i in range(cent.shape[1]):
        c_i = cent[:,i,:]
        print(c_i.shape)
        c_i = np.ascontiguousarray(c_i)
        kmeans = faiss.Kmeans(c_i.shape[1], num_cluster, niter=500,  verbose=False,max_points_per_centroid=15)
        kmeans.train(c_i)
        labels =  kmeans.index.search(c_i, 1)[1].astype(int)
        centroids_i = kmeans.centroids
        # print(centroids_i.shape)
        class_cent.append(centroids_i)
        class_labels.append(labels)
    class_cent = np.array(class_cent)
    class_labels = np.array(class_labels)
    print(class_cent.shape)
    print(class_labels.shape)
    # exit()
    # Compute ARI between pairs of centroids
    num_runs = class_cent.shape[0]
    # # Calculate ARI for each set of centroids
    # for i, centroids in enumerate(class_cent):
    #     labels = faiss.vector_to_array(centroids.search(cent[i], 1)[1]).astype(int)
    #     ari = adjusted_rand_score(ground_truth_labels, labels)
    #     print(f"ARI for set {i+1}: {ari}")
    for i in range(num_runs):
        for j in range(i+1, num_runs):
            ari = adjusted_rand_score(class_labels[i].ravel(), class_labels[j].ravel())
            print(f"ARI between class {i+1} and {j+1}: {ari}")
    exit() 
    obj_ = []
    output = []
    for k in range(2,10):
        ncentroids = k
        niter = 500
        verbose = True
        kmeans = faiss.Kmeans(cent.shape[1], ncentroids, niter=niter,  verbose=verbose,max_points_per_centroid=15)
        kmeans.train(cent)
        D, I = kmeans.index.search(cent, 1)
        print(I.shape)
        labels = I.squeeze()
        score = silhouette_score(cent, labels)
        # print(kmeans.obj)
        obj_.append(kmeans.obj[-1])
        output.append(score)
    # plt.plot(list(range(2,10)),obj_)
    # plt.show()
    plt.plot(list(range(2,10)),output)
    # plt.show()
    plt.savefig('./output/elbowplot.png')
    # exit()
    kmeans = faiss.Kmeans(cent.shape[1],3, niter=500,  verbose=True,max_points_per_centroid=15)
    kmeans.train(cent)
    D, I = kmeans.index.search(cent, 1)
    asnd=[]
    for idx in I:
        asnd.append(idx[0])
    # print(I)
    # print(client_ids)
    # print(domain_ids)
    # print(asnd)
    # print(client_ids[domain_ids==0])
    client_ids = np.array(client_ids)
    domain_ids = np.array(domain_ids)
    asnd = np.array(asnd)
    c0 = client_ids[domain_ids==0]
    c1 = client_ids[domain_ids==1]
    c2 = client_ids[domain_ids==2]
    print(c0,c1,c2)
    a0 = client_ids[asnd==0]
    a1 = client_ids[asnd==1]
    a2 = client_ids[asnd==2]
    print(a0,a1,a2)
    return 

if __name__ == "__main__":
    main()
