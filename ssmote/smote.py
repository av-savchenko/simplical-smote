import numpy as np
import networkx as nx

from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.pipeline import make_pipeline

from collections import defaultdict
from itertools import combinations
from math import comb

def comb1(n, k):
    if n < k:
        return 1
    else:
        return comb(n, k)

def take_m_k_faces(cofaces, take_m, k, random_instance):
    m, n = cofaces.shape
    k_reshape = n if n < k else k 
    
    # take_m cofaces
    cofaces_idx = random_instance.choice(range(m), take_m, replace=True)
    
    # take_m faces, if n<=k remove no indices
    faces_mask = np.ones((take_m, n)).astype(bool)
    if n > k:
        # prepare and take_m faces
        combinations_nk = np.array(list(combinations(range(n), n-k))) # indices to remove!
        combinations_nk_range_idx = range(comb(n, k))
        combinations_nk_idx = random_instance.choice(combinations_nk_range_idx, take_m, replace=True)
        
        # masking
        faces_mask_idx = np.array(range(take_m)), combinations_nk[combinations_nk_idx].T
        faces_mask[faces_mask_idx] = False
    
    return cofaces[cofaces_idx][faces_mask].reshape(-1, k_reshape)

def estimate_borderline(X, y, kappa=10, n_jobs=-1):
        
    # get nearest neighbors of all positive points
    nn = NearestNeighbors(n_neighbors=kappa+1, n_jobs=n_jobs).fit(X)
    neighbor_idx = nn.kneighbors(X[y==1], return_distance=False)[:,1:]

    # get positive points on a decision boundary
    n_positive_neighbors = np.sum(y[neighbor_idx]==1, axis=1)
    borderline_idx = np.logical_and(n_positive_neighbors > 0, n_positive_neighbors <= kappa//2) # danger condition
    positive_borderline_idx = np.copy(y==1)
    positive_borderline_idx[positive_borderline_idx] = borderline_idx
    positive_borderline_int_idx, = np.where(positive_borderline_idx)

    return positive_borderline_idx, positive_borderline_int_idx


class Oversampler():

    def __init__(self):
        pass

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)

        return self

class SimplicalOversampler(Oversampler):

    def __init__(self):
        super().__init__()

    def _get_p_simplices(self, clique_complex, take_m, p, random_instance):
        
        # grade maximal simplices
        maximal_simplices = defaultdict(list)
        for simplex in clique_complex:
            maximal_simplices[len(simplex)].append(simplex)

        # simplices dimensions of a complex
        dims_sorted = sorted(maximal_simplices.keys())
        
        # the probability p_n to select a simplex of dimension n
        counts_n = np.array([len(maximal_simplices[dim]) for dim in dims_sorted])
        p_n = counts_n / counts_n.sum()

        # the probability p_sigma_n to select an n-simplex
        counts_sigma_n = np.array([comb1(dim, p) for dim in dims_sorted])
        p_sigma_n = counts_sigma_n / counts_sigma_n.sum()

        # the probability p_sigma_n to select a particular n-simplex of dimension n ~ U
        p_n_sigma_n = (p_n * p_sigma_n) / (p_n * p_sigma_n).sum()

        # sample take_i maximal simplices of dimension less or equal n ~ p_sigma_n
        nn, mm = np.unique(random_instance.choice(dims_sorted, size=take_m, replace=True, p=p_n_sigma_n), return_counts=True)

        # for each coface dimension sample n_i faces
        simplices = {}
        for i, (n_i, take_m_i) in enumerate(zip(nn, mm)):
            cofaces_n_i = np.array(maximal_simplices[n_i])
            simplices[n_i] = take_m_k_faces(cofaces_n_i, take_m_i, p, random_instance)
            
        return simplices


class Mixup(Oversampler):
    
    def __init__(self, k=2, alpha=1.0, n_jobs=None, random_state=None):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit_resample(self, X, y):

        # random number generator
        random_instance = np.random.default_rng(self.random_state)

        # sampling strategy="auto"
        m = sum(y!=1) - sum(y==1)
        n = X[y==1].shape[0]
        
        # choose n simplices to sample from
        random_states = random_instance.choice(range(int(1e6)), m)
        self.simplices = np.zeros((m, self.k)).astype(int)
        for i, a in enumerate(range(m)):
            simplex_rng = np.random.default_rng(random_states[i])
            self.simplices[i] = simplex_rng.choice(n, size=self.k, replace=None)
        
        # sample barycentric coordinates for p-simplices
        B = random_instance.dirichlet(np.ones(self.k) * self.alpha, size=m)
        
        # compute Euclidean coordinates of synthetic points w/ labels
        X_synthetic = np.einsum("ij,ijk->ik", B, X[y==1][self.simplices])
        y_synthetic = np.ones(m)
            
        return np.concatenate((X.copy(), X_synthetic)), np.concatenate((y.copy(), y_synthetic))


class SMOTE(Oversampler): 
    
    def __init__(self, k=5, alpha=1.0, n_jobs=None, random_state=None):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def fit_resample(self, X, y):
        
        # random number generator
        random_instance = np.random.default_rng(self.random_state)

        # constructing neighborhood graph
        A = kneighbors_graph(X[y==1], n_neighbors=self.k, n_jobs=-1)
        A = ((A + A.T) > 0).astype(int).A
        G = nx.from_numpy_array(A)

        # set 1-simplices
        self.simplices = np.array([list(item) for item in G.edges])
        
        # sampling strategy="auto"
        n = sum(y!=1) - sum(y==1)
        
        # choose n simplices to sample from
        idx = random_instance.choice(np.arange(0, len(self.simplices)), size=n, replace=True)
        
        # sample barycentric coordinates
        B = random_instance.dirichlet(np.ones(2) * self.alpha, size=n)
        
        # compute Euclidean coordinates of synthetic points w/ labels
        X_synthetic = np.einsum("ij,ijk->ik", B, X[y==1][self.simplices[idx]])
        y_synthetic = np.ones(n)
            
        return np.concatenate((X.copy(), X_synthetic)), np.concatenate((y.copy(), y_synthetic))


class SimplicialSMOTE(SimplicalOversampler):
    
    def __init__(self, k=5, p=5, alpha=1.0, n_jobs=None, random_state=None):
        super().__init__()
        self.k = k
        self.p = p
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def fit_resample(self, X, y):
        
        # random number generator
        random_instance = np.random.default_rng(self.random_state)
        
        # construct the clique complex of a neighborhood graph
        A = kneighbors_graph(X[y==1], n_neighbors=self.k, n_jobs=-1)
        A = ((A + A.T) > 0).astype(int).A
        G = nx.from_numpy_array(A)
        clique_complex = nx.find_cliques(G)
        
        # sampling strategy="auto"
        n = sum(y!=1) - sum(y==1)
        
        # choose w/ replacement n p-simplices to sample from
        self.simplices = self._get_p_simplices(clique_complex, n, self.p, random_instance)
        
        # sample synthetic points w/ labels
        X_synthetic = np.zeros((0, X.shape[1]))
        y_synthetic = np.ones(n)
        
        # for each simplex dimension d_i, TODO: for each strata ({d_i < p})_i, p)
        for dim in sorted(self.simplices.keys()):
            n_i, d_i = np.array(self.simplices[dim]).shape
            idx_i = self.simplices[dim]
        
            # sample barycenric coordinates for d_i-simplices
            B_i = random_instance.dirichlet(np.ones(d_i) * self.alpha, size=n_i)

            # compute Euclidean coordinates of synthetic points
            X_synthetic_i = np.einsum("ij,ijk->ik", B_i, X[y==1][idx_i])  
            X_synthetic = np.concatenate([X_synthetic, X_synthetic_i])
    
        return np.concatenate((X.copy(), X_synthetic)), np.concatenate((y.copy(), y_synthetic))


class BorderlineSMOTE(Oversampler):
    
    def __init__(self, k=5, alpha=1.0, n_jobs=None, random_state=None):
        super().__init__()
        self.k = k
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        
        # random number generator
        random_instance = np.random.default_rng(self.random_state)
        
        # sampling strategy="auto"
        n = sum(y!=1) - sum(y==1)
        m = X.shape[0]
        
        # estimate borderline+ points
        positive_borderline_idx, positive_borderline_int_idx = estimate_borderline(X, y, n_jobs=self.n_jobs)
        positive_idx = y==1
        
        # get B+ to X+ adjacency relation matrix C of size |B+|x|X+|
        nn = NearestNeighbors(n_neighbors=self.k+1).fit(X[positive_idx])
        C = nn.kneighbors_graph(X[positive_borderline_idx]).toarray()

        # set row and column indices
        row_idx, column_idx = np.where(positive_borderline_idx==True)[0], np.where(positive_idx==True)[0]
        idx = np.ix_(row_idx, column_idx)

        # set neighborhood graph adjacency matrix A
        A = np.zeros((m, m))
        diag_idx = np.diag_indices(m)
        A[idx] = C
        A[diag_idx] = 0
        A_or = (A + A.T) > 0
        G = nx.from_numpy_array(A_or)

        # set simplices
        self.simplices = np.array([list(item) for item in G.edges])
        
        # choose n simplices to sample from
        idx = random_instance.choice(np.arange(0, len(self.simplices)), size=n, replace=True)
        
        # sample barycentric coordinates
        B = random_instance.dirichlet(np.ones(2) * self.alpha, size=n)
        
        # compute Euclidean coordinates of synthetic points w/ labels
        X_synthetic = np.einsum("ij,ijk->ik", B, X[self.simplices[idx]])
        y_synthetic = np.ones(n)
            
        return np.concatenate((X.copy(), X_synthetic)), np.concatenate((y.copy(), y_synthetic))


class BorderlineSimplicialSMOTE(SimplicalOversampler):
    
    def __init__(self, k=5, p=5, alpha=1.0, n_jobs=None, random_state=None):
        super().__init__()
        self.k = k
        self.p = p
        self.alpha = alpha
        self.n_jobs = n_jobs
        self.random_state = random_state
        
    def fit_resample(self, X, y):
        
        # random number generator
        random_instance = np.random.default_rng(self.random_state)
        
        # sampling strategy="auto"
        n = sum(y!=1) - sum(y==1)
        m = X.shape[0]

        # estimate borderline+ points
        positive_borderline_idx, positive_borderline_int_idx = estimate_borderline(X, y, n_jobs=self.n_jobs)
        positive_idx = y==1
        
        # get B+ to X+ adjacency relation matrix C of size |B+|x|X+|
        nn = NearestNeighbors(n_neighbors=self.k+1).fit(X[positive_idx])
        C = nn.kneighbors_graph(X[positive_borderline_idx]).toarray()

        # set row and column indices
        row_idx, column_idx = np.where(positive_borderline_idx==True)[0], np.where(positive_idx==True)[0]
        idx = np.ix_(row_idx, column_idx)

        # set neighborhood graph adjacency matrix A
        A = np.zeros((m, m))
        diag_idx = np.diag_indices(m)
        A[idx] = C
        A[diag_idx] = 0
        A_or = (A + A.T) > 0
        G = nx.from_numpy_array(A_or)

        clique_complex = []
        self.maximal_simplices_singletons = list(nx.find_cliques(G))
        
        # filter singletons
        for simplex in self.maximal_simplices_singletons:
            if ((len(simplex)==1) & (simplex not in positive_borderline_int_idx)):
                pass
            else:
                clique_complex.append(simplex)

        # choose w/ replacement n p-simplices to sample from
        self.simplices = self._get_p_simplices(clique_complex, n, self.p, random_instance) # TODO: replace 15 w/ n
        
        # sample synthetic points w/ labels
        X_synthetic = np.zeros((0, X.shape[1]))
        y_synthetic = np.ones(n)
        
        # for each simplex dimension d_i, TODO: for each strata ({d_i < p})_i, p)
        for dim in sorted(self.simplices.keys()):
            n_i, d_i = np.array(self.simplices[dim]).shape
            idx_i = self.simplices[dim]

            #print("IDX_i", idx_i)
        
            # sample barycenric coordinates for d_i-simplices
            B_i = random_instance.dirichlet(np.ones(d_i) * self.alpha, size=n_i)

            # compute Euclidean coordinates of synthetic points
            X_synthetic_i = np.einsum("ij,ijk->ik", B_i, X[idx_i])  
            X_synthetic = np.concatenate([X_synthetic, X_synthetic_i])
            
        return np.concatenate((X.copy(), X_synthetic)), np.concatenate((y.copy(), y_synthetic))