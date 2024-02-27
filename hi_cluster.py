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
    # print('cfg:',cfg)
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
    # model_tag = '2020_art_to_product_clipart_realworld_resnet50_1000102'
    # model_tag = '2020_art_to_product_clipart_realworld_resnet50_88888001'
    # model_tag = '2020_realworld_to_product_art_clipart_resnet50_110004'
    # model_tag = '2020_realworld_to_product_art_clipart_resnet50_110004'
    # model_tag = '2020_clipart_to_product_art_realworld_resnet50_110004'
    model_tag = '2020_product_to_clipart_art_realworld_resnet50_110004'
    load_tag = 'checkpoint1'
    result = load('./output/model/target/{}_{}.pt'.format(model_tag, load_tag))
    clients = result['client']
    model = eval('models.{}()'.format(cfg['model_name']))
    feat = []
    client_ids =[]
    domain_ids = []
    for client in clients:
        # print(client.client_id,client.domain_id)
        client_ids.append(client.client_id.item())
        domain_ids.append(client.domain_id)

        model.load_state_dict(client.model_state_dict)
        # print(model.state_dict()['feat_embed_layer.bn.running_mean'].shape)
        # print(model.state_dict().keys())
        # exit()
        # print(model.state_dict()['class_layer.fc.weight_g'].T.shape)
        # print(model.state_dict()['class_layer.fc.weight_v'].shape)
        # f = (model.state_dict()['class_layer.fc.weight_v']/(1e-9+torch.norm(model.state_dict()['class_layer.fc.weight_v'],dim = 1,keepdim=True)))*model.state_dict()['class_layer.fc.weight_g']
        
        # # print(f.shape)
        # # exit()
        # feat.append(np.array(f.reshape(-1)))
        # f1 = model.state_dict()['feat_embed_layer.bn.running_mean'].reshape(-1,1)
        # f2 = model.state_dict()['backbone_layer.layer4.2.bn3.running_mean'].reshape(-1,1)
        # # feat = f1.extend(f2)
        # f1 = f1/(1e-9+torch.norm(f1,dim = 0))
        # f2 = f2/(1e-9+torch.norm(f2,dim =0))
        # feat_ = torch.concat([f1,f2],dim = 0)
        # feat.append(np.array(feat_.squeeze()))
        feat.append(np.array(model.state_dict()['feat_embed_layer.bn.running_mean']))
        # feat.append(np.array(model.state_dict()['feat_embed_layer.bn.running_mean']))
        # feat.append(np.array(model.state_dict()['feat_embed_layer.bn.running_variance']))
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
    # print(client_ids)
    # print(domain_ids)
    # exit()
    feat = np.array(feat)
    feat = feat/(1e-9+np.linalg.norm(feat,axis=1,keepdims = True))
    print(feat.shape)
    
    # c0 = client_ids[domain_ids==0]
    # c1 = client_ids[domain_ids==1]
    # c2 = client_ids[domain_ids==2]
    # print(c0,c1,c2)
    # print(client_ids[cluster_labels==1])
    # print(client_ids[cluster_labels==2])
    # print(client_ids[cluster_labels==3])
    # print('origial Labels',domain_ids)
    # print(client_ids[domain_ids==0])
    # print(client_ids[domain_ids==1])
    # print(client_ids[domain_ids==2])
    #######################################
    # Plot dendrogram
    # plt.figure(figsize=(10, 5))
    # dn = hierarchy.dendrogram(Z)
    # plt.title('Dendrogram')
    # plt.xlabel('Samples')
    # plt.ylabel('Distance')
    # plt.savefig('./output/dendo.png')
    
    # Define the range of cluster numbers to evaluate
    min_clusters = 2
    max_clusters = 5

    # Initialize lists to store silhouette scores
    silhouette_scores = []
    method = 'ward'
    # Compute hierarchical clustering and silhouette score for each number of clusters
    for n_clusters in range(min_clusters, max_clusters + 1):
        # Z = hierarchy.linkage(feat, method='ward')
        Z = hierarchy.linkage(feat, method=method)
        cluster_labels = hierarchy.cut_tree(Z, n_clusters=n_clusters).flatten()
        silhouette_avg = silhouette_score(feat, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # # Plot Silhouette Score vs. Number of Clusters
    # plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    # plt.xlabel('Number of Clusters')
    # plt.ylabel('Average Silhouette Score')
    # plt.title('Silhouette Score vs. Number of Clusters (Hierarchical Clustering)')
    # plt.savefig('./output/Hi_SHscore.png')
    
    Z = hierarchy.linkage(feat, method=method)
    # # Determine the number of clusters
    print(silhouette_scores)
    print(max(silhouette_scores))
    k_ = silhouette_scores.index(max(silhouette_scores))  # Example: Number of clusters
    k = list(range(2,5))[k_]
    # k = 4
    print('number of clusters',k)
    # Assign cluster labels
    cluster_labels = fcluster(Z, k, criterion='maxclust')
    #####################################################################
    # Determine the threshold for the clustering
    # threshold = 1  # Example: Threshold for the clustering

    # # Assign cluster labels based on the threshold
    # cluster_labels = fcluster(Z, threshold, criterion='distance')
    cluster_labels = list(cluster_labels)
    # Print cluster labels
    # A =[]
    # cluster_labels = [2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1] #1
    # ari = adjusted_rand_score(domain_ids, cluster_labels)
    # A.append(ari)
    # cluster_labels = [2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1] #5
    # ari = adjusted_rand_score(domain_ids, cluster_labels)
    # A.append(ari)
    # cluster_labels =  [2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1] #10
    # ari = adjusted_rand_score(domain_ids, cluster_labels)
    # A.append(ari)
    # cluster_labels = [2, 2, 1, 2, 2, 2, 2, 1, 2, 1, 1, 2, 2, 2, 1] #15
    # ari = adjusted_rand_score(domain_ids, cluster_labels)
    # A.append(ari)
    print("Cluster Labels:", cluster_labels)
    print('GT Labels',domain_ids)
    # print(f'ari {A}')
    # Initialize a dictionary to store indices for each cluster label
    indices_by_label = {} 
          
    # plt.plot(list(range(len(A))), A, marker='o', linestyle='-')
    # plt.title('Adjusted Rand Index (ARI) of Clustering Algorithms')
    # plt.ylabel('ARI')
    # plt.ylim(0, 1)  # Adjust y-axis limits if needed
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()
    # # plt.legend(loc = 'best')
    # plt.savefig('./output/ARI_retraining.pdf',dpi = 600,format = 'pdf',bbox_inches = 'tight',pad_inches = 0)
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
    exit()
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
    cluster_labels = [2, 1, 2, 1, 1, 1, 2, 2, 2, 2, 2, 1, 2, 1, 2] #1
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
