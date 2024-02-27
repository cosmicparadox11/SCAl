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
    
    if cfg['resume_mode'] == 1:
        
        epoch_num = 1
        
        cent=[]
        client_ids =[]
        domain_ids = []
        cent_info = load_Cent(epoch_num)
        
        for k,v in cent_info.items():
            print(k,v[2].shape)
            cent.append(np.array(v[2].reshape(-1)))
            # cent.append(np.mean(np.array(v[2]),axis=0).reshape(-1))
            print(v[2].shape)
            # cent.append(v[2][20].reshape(1,-1))
            client_ids.append(k)
            domain_ids.append(v[0])
        # cent = np.concatenate(cent,axis=0)
        # print(cent.shape)
    # exit()
    cent = np.array(cent)

    print(cent.shape)
    # print(feat.shape)
    Z = hierarchy.linkage(cent, method='ward')
    # # Determine the number of clusters
    k = 5  # Example: Number of clusters

    # Assign cluster labels
    cluster_labels = fcluster(Z, k, criterion='maxclust')
    #####################################################################
    # Determine the threshold for the clustering
    # threshold = 1  # Example: Threshold for the clustering

    # # Assign cluster labels based on the threshold
    # cluster_labels = fcluster(Z, threshold, criterion='distance')
    cluster_labels = list(cluster_labels)
    # Print cluster labels
    print("Cluster Labels:", cluster_labels)
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

    # Print indices for each cluster label
    for label, indices in og_indices_by_label.items():
        print(f"Cluster Label {label} GT: Indices {indices}")
    # # cent = cent[:,1,:]
    # print(cent.shape)
    exit()
    obj_ = []
    output = []
    cent = cent/(1e-9+np.linalg.norm(cent,axis=1,keepdims = True))
    for k in range(2,10):
        ncentroids = k
        niter = 500
        verbose = True
        kmeans = faiss.Kmeans(cent.shape[1], ncentroids, niter=niter,  verbose=verbose,max_points_per_centroid=15) # try for cosine distance 
        kmeans.train(cent)
        D, I = kmeans.index.search(cent, 1)
        print(I.shape)
        labels = I.squeeze()
        # score = silhouette_score(cent, labels)
        # print(kmeans.obj)
        obj_.append(kmeans.obj[-1])
        # output.append(score)
    plt.plot(list(range(2,10)),obj_)
    # plt.show()
    plt.savefig('./output/elbowplot.png')
    # plt.plot(list(range(2,10)),output)
    # # plt.show()
    # plt.savefig('./output/so.png')
    
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
    a3 = client_ids[asnd==3]
    a4 = client_ids[asnd==4]
    a5 = client_ids[asnd==5]
    print(a0,a1,a2,a3,a4,a5)
    return 

if __name__ == "__main__":
    main()
