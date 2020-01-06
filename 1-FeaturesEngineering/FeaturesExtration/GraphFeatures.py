import networkx as nx
import numpy as np
import os
import json
from tqdm import tqdm
from scipy.sparse.linalg import svds, eigs
import threading

author = "Kanvaly fadiga, Ecole polytechnique"


def precomputeData(G, recompute=False):
    if not recompute:
        assert os.path.isfile('precompute/precompute.json')
        with open('precompute/precompute.json', 'r') as f:
            precompute = json.load(f)

        assert os.path.isfile('precompute/katz.npy')   
        precompute["katz"] = np.load('precompute/katz.npy', allow_pickle=True)
    
        assert os.path.isfile('precompute/A_er.npy')   
        precompute["A_er"] = np.load('precompute/A_er.npy', allow_pickle=True)

        assert os.path.isfile('precompute/U.npy') and \
            os.path.isfile('precompute/V.npy') and\
            os.path.isfile('precompute/s.npy')
        precompute["svd"] = {}
        precompute["svd"]["U"] = np.load('precompute/U.npy')
        precompute["svd"]["V"] = np.load('precompute/V.npy')
        precompute["svd"]["s"] = np.load('precompute/s.npy')

        precompute['katz'] = {}
        for i in tqdm(range(1,5)):
            precompute['katz']["A_pow%i"%i] = np.load("precompute/A_pow%i.npy"%i,  allow_pickle=True)

        return precompute 
    else:
        precompute = {}
        print("preparing pagerank")
        precompute["pr"] = nx.pagerank(G, alpha=0.85)
        precompute["mean_pr"] = float(sum(precompute["pr"].values())) / len(precompute["pr"])
        print("->.....OK")


        print("preparing hits")
        precompute["hits"] = nx.hits(G, max_iter=100, tol=1e-08, nstart=None, normalized=True)
        print("->.....OK")

        #weight for source and destination of each link
        print("preparing weight")
        precompute["Weight_in"] = {}
        precompute["Weight_out"] = {}
        for i in  G.nodes():
            s1=set(G.predecessors(i))
            w_in = 1.0/(np.sqrt(1+len(s1)))
            precompute["Weight_in"][i]=w_in

            s2=set(G.successors(i))
            w_out = 1.0/(np.sqrt(1+len(s2)))
            precompute["Weight_out"][i]=w_out

        #for imputing with mean
        precompute["mean_weight_in"] = np.mean(list(precompute["Weight_in"] .values()))
        precompute["mean_weight_out"] = np.mean(list(precompute["Weight_out"] .values()))

        print("preparing katz_2")
        precompute["katz_2"] = nx.katz.katz_centrality(G,alpha=0.005, beta=1)
        precompute["mean_katz_2"] = float(sum(precompute["katz_2"].values())) / len(precompute["katz_2"])
        print("->.....OK")

        sadj_col = sorted(G.nodes())
        sadj_dict = { val:idx for idx,val in enumerate(sadj_col)}
        precompute["node2id"] = sadj_dict

        with open('precompute/precompute.json', 'w') as f:
            json.dump(precompute, f)

        Adj = nx.adjacency_matrix(G,nodelist=sorted(G.nodes()))

        print("preparing katz")
        precompute['katz'] = {}
        beta = 0.005
        Katz = np.zeros_like(Adj) 
        for i in tqdm(range(1,5)):
            precompute['katz']["A_pow%i"%i] = beta**i * np.power(Adj,i)
            np.save("precompute/A_pow%i.npy"%i, precompute['katz']["A_pow%i"%i])
        print("->.....OK")

        del Adj
        print("preparing edgerank")
        Adj = nx.adjacency_matrix(G.to_undirected(),nodelist=sorted(G.nodes()))
        und_G = G.to_undirected()
        for i in tqdm(range(len(G))):
            degree = und_G.degree(i)
            if degree == 0:
                continue
            Adj[i] = Adj[i]/degree
        precompute["A_er"] = Adj
        np.save("precompute/A_er", Adj)
        print("->.....OK")

        print("preparing svd")
        precompute["svd"] = {}
        Au = nx.adjacency_matrix(G.to_undirected(),nodelist=sorted(G.nodes()))
        Au = Au.asfptype()
        U, s, V = svds(Au,k = 80)
        del Au
        precompute["svd"]["U"] = U
        precompute["svd"]["V"] = V
        precompute["svd"]["s"] = s
        np.save("precompute/U", U)
        np.save("precompute/V", V)
        np.save("precompute/s", s)
        print("->.....OK")

        return precompute


class GraphFeaturesExtractor:

    features = {"3-level": ["jc", "cn", "aa", "cos", "katz", "simrank"],
                "1-level": ["s_out", "s_in", "t_in", "t_out", "pr_s", "pr_t", "pa",
                            "katz_s", "katz_t", "hubs_s", "hubs_t", "auto_s",
                            "auto_t", "sweight_out", "tweight_in", "weight_f1",
                            "weight_f2", "edgerank", "svd_nodes", "svd_dot", 
                            "svd_mean", "power_low", "short_path", "clust_coef_s", 
                            "clust_coef_t", "ct", "btwness_centrality"]}

    def __init__(self, G, precom):
        self.G = G
        self.undir_G = nx.to_undirected(G)
        
        self.pr = precom["pr"]
        self.mean_pr = precom["mean_pr"]
        
        self.katz = precom["katz"]
        self.katz_2 = precom["katz_2"]
        self.mean_katz_2 = precom["mean_katz_2"]

        self.hits = precom["hits"]
        
        #weight for source and destination of each link
        self.Weight_in = precom["Weight_in"]
        self.Weight_out = precom["Weight_out"]

        #for imputing with mean
        self.mean_weight_in = precom["mean_weight_in"]
        self.mean_weight_out = precom["mean_weight_out"]

        self.svd = precom['svd']
        self.node2id = precom['node2id']

        self.A_er = precom["A_er"]
    
    def set_features(self, F):
        self.features = F
    
    def G_ord1(self, nodes):
        u,v = nodes
        u_neigh = sorted(self.undir_G.neighbors(u))
        v_neigh = sorted(self.undir_G.neighbors(v))
        u_edge = [(u, w) for w in u_neigh]
        v_edge = [(v, w) for w in v_neigh]
        Guv = u_edge + v_edge
        return list(set(Guv))
        
    """ == Features for directed Graph == """

    def level1(self, name, nodes):
        return getattr(self, '_'+name)(nodes)

    def level2(self, name, nodes):
        ks_out = self.G.out_degree(nodes[0])
        nodes_s_out = sorted(self.G.successors(nodes[0]))
        if ks_out == 0:
            return 0
        return (1/ks_out) * sum([self.level1(name, [nodes[1], v]) for v in nodes_s_out])

    def level3(self, name, nodes):
        kt_in = self.G.in_degree(nodes[0])
        nodes_t_in = sorted(self.G.predecessors(nodes[1]))
        if kt_in == 0:
            return 0 
        return (1/kt_in) * sum([self.level1(name, [nodes[0], v]) for v in nodes_t_in])

    def _s_out(self, nodes):
        return self.G.out_degree(nodes[0])
    def _t_out(self, nodes):
        return self.G.out_degree(nodes[1])
    def _s_in(self, nodes):
        return self.G.in_degree(nodes[0])
    def _t_in(self, nodes):
        return self.G.in_degree(nodes[1])

    def _cn(self, nodes):
        cn = list(set(self.undir_G.neighbors(nodes[0])) & set(self.undir_G.neighbors(nodes[1])))
        return len(cn)

    def _jc(self, nodes):
        union_neigh = list(set(self.undir_G.neighbors(nodes[0])) | set(self.undir_G.neighbors(nodes[1])))
        if len(union_neigh)==0:
            return 0
        return self._cn(nodes)/len(union_neigh)

    def _aa(self, nodes):
        cn = list(set(self.undir_G.neighbors(nodes[0])) & set(self.undir_G.neighbors(nodes[1])))
        if len(cn)==0: return 0

        res = [self.undir_G.degree(v) for v in cn if self.undir_G.degree(v) !=0 and \
                                                     self.undir_G.degree(v) !=1]
        if len(res)==0: return 0
        return np.sum(1/np.log(res))

    def _pa(self, nodes):
        ks = self.undir_G.degree(nodes[0])
        kt = self.undir_G.degree(nodes[1])
        return ks * kt
    
    def _cos(self, nodes):
        ks = self.undir_G.degree(nodes[0])
        kt = self.undir_G.degree(nodes[1])
        if ks == 0 or kt == 0 :
            return 0
        return self._cn(nodes)/(ks*kt)
    
    def _ra(self, nodes):
        cn = list(set(self.undir_G.neighbors(nodes[0])) & set(self.undir_G.neighbors(nodes[1])))
        if len(cn)==0: return 0
        return sum(1/np.array([self.undir_G.in_degree(v) for v in cn if self.undir_G.degree(v) !=0 ]))
    
    def _katz(self, nodes):
        i, j = self.node2id[str(nodes[0])], self.node2id[str(nodes[1])]
        val = 0
        for idx in range(1,5):
            val+=self.katz["A_pow%i"%idx][()][i,j]
            
        return val

    def _svd_nodes(self, nodes):
        i, j = self.node2id[str(nodes[0])], self.node2id[str(nodes[1])]
        return self.svd["U"][i] @ np.diag(self.svd['s']) @ self.svd['V'][:,j]
        
    def _svd_dot(self, nodes):
        i, j = self.node2id[str(nodes[0])], self.node2id[str(nodes[1])]
        vi = self.svd["U"][i] @ np.diag(self.svd['s']) @ self.svd['V']
        vj = self.svd["U"][j] @ np.diag(self.svd['s']) @ self.svd['V']                  
        return vi.dot(vj)

    def _svd_mean(self, nodes):
        t_in = set(self.G.predecessors(nodes[1]))
        if len(t_in) == 0 : return 0
        return np.mean([self._svd_nodes((nodes[0], v)) for v in t_in])
    
    def _simrank(self, nodes):
        ord1 = self.G_ord1(nodes)
        g = nx.DiGraph()
        g.add_nodes_from(nodes.tolist())
        g.add_edges_from(ord1)
        sim = nx.simrank_similarity(g)
        return sim[nodes[0]][nodes[1]]
    
    def _edgerank(self, nodes):
        u,v = nodes
        max_iter= 500
        d= 0.5
        N = len(self.G)
        x0 = np.zeros(N)
        x0[self.node2id[str(u)]] = 1
        x = x0
        logs = []
        for i in range(max_iter):
            xlast = x
            x = (1 - d)*x0 + d * self.A_er[()]  @ xlast 
            # check convergence
            err = np.linalg.norm(xlast-x)
            if err < 1e-6:
                return x[self.node2id[str(v)]]
        print("don't converge till the end")
        return x[self.node2id[str(v)]]
    
    def _btwness_centrality(self, nodes):
        ord1 = self.G_ord1(nodes)
        g = nx.Graph()
        g.add_nodes_from(nodes.tolist())
        g.add_edges_from(ord1)
        btw = nx.betweenness_centrality(g)
        return abs(btw[nodes[0]]-btw[nodes[1]])

    def _ct(self, nodes):
        ord1 = self.G_ord1(nodes)
        g = nx.Graph()
        g.add_nodes_from(nodes.tolist())
        g.add_edges_from(ord1)
        n_list = sorted(g.nodes())
        g_id = { val:idx for idx,val in enumerate(n_list)} 

        L = nx.laplacian_matrix(g, nodelist=sorted(g.nodes))
        CTK = np.linalg.pinv(L.toarray())
        
        i,j = g_id[nodes[0]], g_id[nodes[1]]
        return len(g.edges) * (CTK[i,i] + CTK[j,j] - 2 * CTK[i,j]) 

    def _power_low(self, nodes):
        union_neigh = list(set(self.undir_G.neighbors(nodes[0])) | set(self.undir_G.neighbors(nodes[1])))
        if len(union_neigh)==0 or len(union_neigh)==1:
            return 0
        return np.log(len(self.G.edges())) / np.log(len(union_neigh))

    def _sweight_out(self, nodes):
        return self.Weight_out.get(str(nodes[0]))
    
    def _tweight_in(self, nodes):
        return self.Weight_in.get(str(nodes[1]))
    
    def _weight_f1(self, nodes):
        return self._sweight_out(nodes) + self._tweight_in(nodes)
    
    def _weight_f2(self, nodes):
        return self._sweight_out(nodes) * self._tweight_in(nodes)
    
    def _pr_s(self, nodes):
        return self.pr.get(str(nodes[0]))
    
    def _pr_t(self, nodes):
        return self.pr.get(str(nodes[1]))
    
    def _katz_s(self, nodes):
        return self.katz_2.get(str(nodes[0]))
    
    def _katz_t(self, nodes):
        return self.katz_2.get(str(nodes[1]))
    
    def _hubs_s(self, nodes):
        return self.hits[0].get(str(nodes[0]))
    
    def _hubs_t(self, nodes):
        return self.hits[0].get(str(nodes[1]))
    
    def _auto_s(self, nodes):
        return self.hits[1].get(str(nodes[0]))
    
    def _auto_t(self, nodes):
        return self.hits[1].get(str(nodes[1]))
    
    def _short_path(self, nodes):
        has_removed=False
        a,b=nodes
        try:
            if self.G.has_edge(a,b):
                self.G.remove_edge(a,b)
                has_removed = True
                p= nx.shortest_path_length(self.G,source=a,target=b)
                self.G.add_edge(a,b)
                
                return p
            else:
                p= nx.shortest_path_length(g,source=a,target=b)
                return p
        except:
            if has_removed :
                self.G.add_edge(a,b)
            return len(G)
    
    def _clust_coef_s(self, nodes):
        return nx.clustering(self.G, nodes[0])
    
    def _clust_coef_t(self, nodes):
        return nx.clustering(self.G, nodes[1])

    def _ded(self, nodes):
        if len(self.Ax)==0:
            return 0
        return len(set(self.Ax) & set(self.Dy))/len(self.Ax)
    
    def _ind(self, nodes):
        if len(self.Dx)==0:
            return 0
        return len(set(self.Dx) & set(self.Dy))/len(self.Dx)
    
    def _ded_log(self, nodes):
        if len(self.Ax)==0:
            return 0
        return len(set(self.Ax) & set(self.Ay)) * np.log(len(self.Ax)) / len(self.Ax)

    def _ind_log(self, nodes):
        if len(self.Dx)==0:
            return 0
        return len(set(self.Dx) & set(self.Dy)) * np.log(len(self.Dx)) / len(self.Dx)
    
    def _inf(self, nodes):
        return self._ded(nodes) + self._ded(nodes)

    def _inf_log(self, nodes):
        return self._ded_log(nodes) + self._ind_log(nodes)
    
    def getFeatures(self, nodes, filters=None):
        if filters == None : filters = []

        ft_1_level = [self.level1(name, nodes) for name in self.features["1-level"] if name not in filters] + \
                     [self.level1(name, nodes) for name in self.features["3-level"] if name not in filters]

        ft_2_level = [self.level2(name, nodes) for name in self.features["3-level"] if name not in filters]
        ft_3_level = [self.level3(name, nodes) for name in self.features["3-level"] if name not in filters]

        return np.array(ft_1_level + ft_2_level + ft_3_level)

if __name__ == "__main__":
    import time
    from tqdm import tqdm
    exitFlag = 0

    from LoadData import load
    G, train, test, disconnected = load()
    precompute = precomputeData(G, recompute=False)
    GFE = GraphFeaturesExtractor(G, precompute)

    import sys
    if sys.argv[1] =="train": 
        data = train
        idx = np.load('precompute/order.npy')
        save_dir = "generated_data/embs_train/"
    if sys.argv[1] =="test": 
        data = test
        idx = np.load('precompute/order_test.npy')
        save_dir = "generated_data/embs_test/"

    i = int(sys.argv[2])
    sub_size = idx.shape[0] // 4
    idx_used = idx[i*sub_size: (i+1)*sub_size]

    class myThread (threading.Thread):
        def __init__(self, k, step):
            threading.Thread.__init__(self)
            self.k = k
            self.step = step

        def run(self):
            for i in tqdm(idx_used[self.k*self.step: (self.k+1)*self.step]):
                np.save(save_dir+'%d'%i, GFE.getFeatures(data[i,:2], ["simrank", "btwness_centrality","ct"]))

    nb = 20
    size = idx_used.shape[0]//20
    thread = []
    for j in range(nb):
        thread.append(myThread(j, size))

    for j in range(nb):
        thread[j].start()
