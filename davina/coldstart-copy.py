def run_coldstart_pipeline(max_history=None, test_history_limit=None, history_from_back=False):
    """
    Run the cold start pipeline for the MIND/Kaggle dataset.
    This function sets up the environment, loads data, processes it,
    and trains a model for news recommendation.
    """
    
    # ========== 设置随机种子确保结果可复现 ==========    
# ========== 设置随机种子确保结果可复现 ==========
    import random
    import os
    import numpy as np

    # 设置随机种子
    RANDOM_SEED = 42

    # Python random
    random.seed(RANDOM_SEED)

    # NumPy
    np.random.seed(RANDOM_SEED)

    # PyTorch (在import torch之后)
    import torch
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed
    os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

    print(f"✅ Random seed set to {RANDOM_SEED} for reproducibility")
    print(f"   - Python random: {RANDOM_SEED}")
    print(f"   - NumPy: {RANDOM_SEED}")
    print(f"   - PyTorch: {RANDOM_SEED}")
    print(f"   - CUDA deterministic: True")
    print(f"   - Python hash seed: {RANDOM_SEED}")

    # %autoreload 2

    # Sentiment-Aware KGAT News Recommendation Pipeline (MIND/Kaggle Dataset Version)
    # Author: (Your Name)
    # Date: (Today)

    # 1. Data Loading (news.tsv, behaviors.tsv, entity_embedding.vec, relation_embedding.vec)
    import pandas as pd
    import numpy as np
    from textblob import TextBlob
    import ast

    # Load news.tsv (tab-separated)
    behaviour = pd.read_csv('./kaggle/input/mind-news-dataset/MINDsmall_train/behaviors.tsv', sep='\t', 
                            names=["Impression_ID", "User_ID", "Time", "History", "Impressions"])

    news = pd.read_csv('./kaggle/input/mind-news-dataset/MINDsmall_train/news.tsv', sep='\t',
                    names=["News_ID", "Category", "SubCategory", "Title", "Abstract", "URL", "Title_Entities", "Abstract_Entities"])


    print('News sample:')
    print(news.head())
    print('behaviour sample:')
    print(behaviour.head())

    # Load entity_embedding.vec
    entity_emb = {}
    with open('./kaggle/input/mind-news-dataset/MINDsmall_train/entity_embedding.vec', 'r') as f:
        for line in f:
            parts = line.strip().split()
            entity_id = parts[0]
            emb = np.array([float(x) for x in parts[1:]])
            entity_emb[entity_id] = emb
    print('Sample entity embedding:', list(entity_emb.items())[:1])




    #rayyyyyyyyyy############

    # 2. data cleaning
    # --- knobs -----------------------------------------------------------
    MAX_HISTORY       = max_history      # set to None for no cap
    HISTORY_FROM_BACK = history_from_back    # True → keep from the end (most recent); False → from the start
    # --------------------------------------------------------------------

    def parse_impressions(imp_str):
        if pd.isna(imp_str):
            return []
        items = imp_str.strip().split(' ')
        return [(i.split('-')[0], int(i.split('-')[1])) for i in items]

    behaviour['Impressions_parsed'] = behaviour['Impressions'].apply(parse_impressions)

    def parse_history(history_str, max_len=None, from_back=True):
        if pd.isna(history_str):
            return []
        seq = history_str.strip().split(' ')
        if max_len is not None and len(seq) > max_len:
            return seq[-max_len:] if from_back else seq[:max_len]
        return seq

    # apply with the knobs
    behaviour['History_parsed'] = behaviour['History'].apply(
        lambda s: parse_history(s, max_len=MAX_HISTORY, from_back=HISTORY_FROM_BACK)
    )

    def check_df_shape_and_null(df):
        print("The shape of df:", df.shape)
        df.info()
        
        print("\nThe number of missing values in each column:")
        print(df.isnull().sum())

    news_clean = news.dropna()
    news_clean.drop_duplicates(subset=['News_ID'], inplace=True)

    behaviour_clean = behaviour.dropna()
    behaviour_clean.drop_duplicates(subset=['Impression_ID'], inplace=True)

    check_df_shape_and_null(news_clean)
    check_df_shape_and_null(behaviour_clean)

    import pandas as pd
    import ast

    def has_entities(x):
        if pd.isna(x) or str(x).strip() == "":
            return False
        try:
            ents = ast.literal_eval(x)
            return isinstance(ents, list) and len(ents) > 0
        except Exception:
            return False

    # 计算“Title 无实体”和“Abstract 无实体”的比例
    frac_title_empty = 1 - news_clean['Title_Entities'].apply(has_entities).mean()
    frac_abs_empty   = 1 - news_clean['Abstract_Entities'].apply(has_entities).mean()

    print(f"Title 无实体: {frac_title_empty:.1%}")
    print(f"Abstract 无实体: {frac_abs_empty:.1%}")

    from textblob import TextBlob
    import pandas as pd

    # 2. Sentiment Analysis on News Titles & Abstracts

    def get_sentiment(text):
        if pd.isnull(text):
            return 0.0
        return TextBlob(str(text)).sentiment.polarity

    # 原来只有这一行（作用在 news）
    # news['sentiment_score'] = news['Title'].apply(get_sentiment)

    # 改成作用在 news_clean，并拆成两个字段：
    news_clean['sentiment_score_title'] = news_clean['Title']   .apply(get_sentiment)
    # news_clean['sentiment_score_abs']   = news_clean['Abstract'].apply(get_sentiment)

    print(news_clean[['Title','sentiment_score_title']].head())


    import ast
    import numpy as np

    news_clean=news_clean.copy()

    # —— 0. 先定义两个占位实体 ID
    GLOBAL_DEFAULT_TITLE = 'GLOBAL_DEFAULT_TITLE'
    GLOBAL_DEFAULT_ABS   = 'GLOBAL_DEFAULT_ABS'

    # （假设 entity_emb、get_sentiment 都已定义）

    # 1. 填充函数：标题空就用 GLOBAL_DEFAULT_TITLE；摘要空就用 GLOBAL_DEFAULT_ABS
    def fill_title_entities(row):
        try:
            ents = ast.literal_eval(row['Title_Entities'])
        except:
            ents = []
        return ents if ents else [{'Label': GLOBAL_DEFAULT_TITLE}]


    news_clean['Title_Entities_Processed']    = news_clean.apply(fill_title_entities, axis=1)
    # news_clean['Abstract_Entities_Processed'] = news_clean.apply(fill_abs_entities, axis=1)


    # 2. 分别算标题实体和摘要实体的情感字典

    # 2.1 标题实体情感
    entity_sent_title = {}
    for _, row in news_clean.iterrows():
        score = get_sentiment(row['Title'])
        for ent in row['Title_Entities_Processed']:
            eid = ent['Label']
            if eid == GLOBAL_DEFAULT_TITLE: 
                continue
            entity_sent_title.setdefault(eid, []).append(score)

    entity_sentiment_title_avg = {
        eid: sum(v)/len(v) 
        for eid, v in entity_sent_title.items()
    }
    # 默认值
    default_title_sent = np.mean(list(entity_sentiment_title_avg.values()))
    entity_sentiment_title_avg[GLOBAL_DEFAULT_TITLE] = default_title_sent




    # 3. 给两个全局实体也加上 embedding
    all_embs    = np.array(list(entity_emb.values()))
    default_emb = all_embs.mean(axis=0)
    entity_emb[GLOBAL_DEFAULT_TITLE] = default_emb
    # entity_emb[GLOBAL_DEFAULT_ABS]   = default_emb


    print(news_clean.head)

    import numpy as np

    # 采样15%用户
    sample_ratio = 0.1
    user_ids_sub = behaviour_clean['User_ID'].unique()
    sampled_users_sub = np.random.choice(user_ids_sub, size=int(len(user_ids_sub) * sample_ratio), replace=False)

    # 筛选这些用户的行为
    behaviour_clean = behaviour_clean[behaviour_clean['User_ID'].isin(sampled_users_sub)].copy()

    # 找到这些行为涉及到的所有新闻ID
    news_ids_in_history = set()
    for h in behaviour_clean['History']:
        news_ids_in_history.update(str(h).split())
    if 'Impressions' in behaviour_clean.columns:
        for imp in behaviour_clean['Impressions']:
            news_ids_in_history.update([i.split('-')[0] for i in str(imp).split()])

    # 只保留这些新闻
    news_clean = news_clean[news_clean['News_ID'].isin(news_ids_in_history)].copy()

    # 后续流程都用 *_sub 变量

    # 3. Knowledge Graph Construction (networkx illustration)
    import networkx as nx
    G = nx.Graph()
    # Add news nodes with sentiment
    for idx, row in news_clean.iterrows():
        G.add_node(row['News_ID'], node_type='news', sentiment=row['sentiment_score_title'])
    # Add entity nodes and news-entity edges
    for idx, row in news_clean.iterrows():
        try:
            entities = row['Title_Entities_Processed']
        except:
            entities = []
        for ent in entities:
            ent_label = ent.get('Label', '')
            if ent_label:
                # 节点特征：拼接实体embedding和情感分数
                ent_emb = entity_emb.get(ent_label, np.zeros(100))  # 假设embedding维度为100
                ent_sent = entity_sentiment_title_avg.get(ent_label, 0)
                G.add_node(ent_label, node_type='entity', features=np.concatenate([ent_emb, [ent_sent]]))
                G.add_edge(row['News_ID'], ent_label, edge_type='has_entity')
    # Add user-news interactions
    for idx, row in behaviour_clean.iterrows():
        user = row['User_ID']
        history = str(row['History']).split()
        for news_id in history:
            G.add_node(user, node_type='user')
            G.add_edge(user, news_id, edge_type='clicked')
    print(f'Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.')

    import torch
    import numpy as np
    from torch_geometric.data import HeteroData
    from sklearn.feature_extraction.text import TfidfVectorizer


    #4 构建信息
    # 1. 预先准备好映射和列表
    # ---------------------------------
    # 1.1 新闻 ID 列表 & 映射
    news_id_list = news_clean['News_ID'].tolist()
    news2idx     = {nid:i for i,nid in enumerate(news_id_list)}

    # 1.2 用户 ID 列表 & 映射
    user_id_list = behaviour_clean['User_ID'].unique().tolist()
    user2idx     = {uid:i for i,uid in enumerate(user_id_list)}

    # 1.3 实体 ID 列表 & embedding 列表
    entity_id_list = list(entity_emb.keys())
    emb_dim        = len(next(iter(entity_emb.values())))
    entity2idx     = {eid:i for i,eid in enumerate(entity_id_list)}

    # 1.4 新闻标题情感字典
    sent_map = news_clean.set_index('News_ID')['sentiment_score_title'].to_dict()

    # 1.5 已解析好的实体列表（Python list of dict）
    ent_lists = news_clean['Title_Entities_Processed'].tolist()


    # 2. 构造节点特征张量
    # A.news节点特征构造
    # 2.1 news_x: [N_news, 1]


    # 1. 情感分数特征
    news_sentiment = news_clean['sentiment_score_title'].values.reshape(-1, 1)  # shape: (N_news, 1)

    # 2. 类别one-hot特征
    categories = news_clean['Category'].unique().tolist()
    cat2idx = {cat: i for i, cat in enumerate(categories)}
    news_cat_feat = np.zeros((len(news_clean), len(categories)))
    for i, cat in enumerate(news_clean['Category']):
        news_cat_feat[i, cat2idx[cat]] = 1  # shape: (N_news, num_categories)

    # 3. TF-IDF文本特征
    vectorizer = TfidfVectorizer(max_features=100) #max feature 5000
    tfidf_feat = vectorizer.fit_transform(news_clean['Title'].fillna('')).toarray()  # shape: (N_news, 100)

    #4.sbert

    from sentence_transformers import SentenceTransformer
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import torch

    sbert = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    news_texts = (news_clean['Title'].fillna('') + ' ' + news_clean['Abstract'].fillna('')).tolist()
    bert_feat = sbert.encode(news_texts, show_progress_bar=True, batch_size=128)
    scaler = StandardScaler()
    bert_feat_norm = scaler.fit_transform(bert_feat)

    # from sklearn.decomposition import PCA

    # pca = PCA(n_components=128)  # 你可以试64、128、256等
    # bert_feat_pca = pca.fit_transform(bert_feat_norm)
    # news_x = np.hstack([news_sentiment, news_cat_feat, tfidf_feat, bert_feat_pca])
    # news_x = torch.tensor(news_x, dtype=torch.float)
    # 4. 拼接所有特征
    news_x = np.hstack([news_sentiment, news_cat_feat, tfidf_feat,bert_feat_norm])  # shape: (N_news, 1+num_categories+100)
    news_x = torch.tensor(news_x, dtype=torch.float)

    # 5. 赋值给HeteroData
    # data['news'].x = news_x

    # 2.2 entity_x: [N_ent, emb_dim+1]
    entity_x_np = np.array([
        np.concatenate([
            entity_emb[eid],
            [entity_sentiment_title_avg.get(eid, 0.0)]
        ])
        for eid in entity_id_list
    ])
    entity_x = torch.tensor(entity_x_np, dtype=torch.float)





    #2.3用户类别分布特征
    #2.3.1 
    news_id2cat = news_clean.set_index('News_ID')['Category'].to_dict()
    categories = list(news_clean['Category'].unique())
    cat2idx = {cat: i for i, cat in enumerate(categories)}

    user_cat_feat = np.zeros((len(user_id_list), len(categories)))
    for i, uid in enumerate(user_id_list):
        user_hist = behaviour_clean[behaviour_clean['User_ID'] == uid]['History']
        news_ids = []
        for h in user_hist:
            news_ids += str(h).split()
        for nid in news_ids:
            cat = news_id2cat.get(nid, None)
            if cat is not None:
                user_cat_feat[i, cat2idx[cat]] += 1
    # 归一化
    user_cat_feat = user_cat_feat / (user_cat_feat.sum(axis=1, keepdims=True) + 1e-8)

    #2.3.2 用户历史点击新闻embedding均值
    user_emb_feat = np.zeros((len(user_id_list), news_x.shape[1]))
    for i, uid in enumerate(user_id_list):
        user_hist = behaviour_clean[behaviour_clean['User_ID'] == uid]['History']
        news_ids = []
        for h in user_hist:
            news_ids += str(h).split()
        idxs = [news2idx[nid] for nid in news_ids if nid in news2idx]
        if idxs:
            user_emb_feat[i] = news_x[idxs].mean(dim=0).numpy()

    # 2.3.3 合并特征（横向拼接）
    user_feat = np.hstack([user_cat_feat, user_emb_feat])
    user_x = torch.tensor(user_feat, dtype=torch.float)
    # data['user'].x = user_x

    #3.塑造边
    # 3.1 news — has_entity — entity
    news_src = []
    ent_dst  = []
    for i, ents in enumerate(ent_lists):
        for ent in ents:
            # 优先用 WikidataId，因为它在 entity_emb 中是 key
            eid = ent.get('WikidataId', ent.get('Label'))
            if eid in entity2idx:
                news_src.append(i)
                ent_dst.append(entity2idx[eid])


    # 3. 构造边索引
    # ---------------------------------
    # 3.2 user — clicked — news
    user_src = []
    news_dst = []
    for row in behaviour_clean.itertuples(index=False):
        uidx = user2idx[row.User_ID]
        for nid in str(row.History).split():
            if nid in news2idx:
                user_src.append(uidx)
                news_dst.append(news2idx[nid])


    # 4. 装配 HeteroData
    # ---------------------------------
    data = HeteroData()
    data['news'].x   = news_x
    data['entity'].x = entity_x
    data['user'].x   = user_x

    data['user','clicked','news'].edge_index     = torch.tensor([user_src, news_dst], dtype=torch.long)
    data['news','has_entity','entity'].edge_index = torch.tensor([news_src, ent_dst ], dtype=torch.long)

    print(data)

    # 5. KGAT Model Skeleton (PyTorch Geometric style)
    import torch.nn.functional as F
    from torch_geometric.nn import HANConv

    # Add reverse edges for HANConv
    data['news', 'clicked_by', 'user'].edge_index = data['user', 'clicked', 'news'].edge_index[[1,0], :]
    data['entity', 'entity_of', 'news'].edge_index = data['news', 'has_entity', 'entity'].edge_index[[1,0], :]

    # 假设你已经有 news_x, entity_x, user_x

    import torch.nn as nn

    # 假设你已经有 news_x, entity_x, user_x

    class SentimentKGAT(torch.nn.Module):
        def __init__(self, hidden_dim, entity_emb_dim,dropout=0.2):
            super().__init__()
            in_channels = {
        'news': news_x.shape[1],
        'entity': entity_x.shape[1],
        'user': user_x.shape[1]
    }
            self.han = HANConv(in_channels=in_channels,out_channels=hidden_dim,metadata=data.metadata())
            self.dropout = nn.Dropout(dropout)
            self.lin = torch.nn.Linear(hidden_dim, 1)  # 推荐分数

        def forward(self, data):
            x_dict = self.han(data.x_dict, data.edge_index_dict)
            # 对每个节点类型都加Dropout
            for k in x_dict:
                x_dict[k] = self.dropout(x_dict[k])
            return x_dict

    from sklearn.model_selection import train_test_split
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    import random
    from tqdm.auto import tqdm  # NEW: progress bar
    #6 分训练集和测试集

    # ---- NEW: parameter to cap per‑user history length (None = no cap) ----
    HISTORY_LIMIT = test_history_limit  # e.g., set to 55 to keep at most 55 items by popping from the front

    # 1. 统一变量名
    user_id_list = behaviour_clean['User_ID'].unique().tolist()
    user2idx = {uid: i for i, uid in enumerate(user_id_list)}
    news_id_list = news_clean['News_ID'].tolist()
    news2idx = {nid: i for i, nid in enumerate(news_id_list)}

    # 2. 收集正例对
    # NEW: apply a front-pop limit to each History list (no other logic changed)
    hist_lists = behaviour_clean['History'].str.split()

    if HISTORY_LIMIT is not None:
        def _cap_front(lst, limit):
            if not isinstance(lst, list):
                return lst
            lst = list(lst)
            while len(lst) > limit:
                lst.pop(0)  # pop from the front (index 0)
            return lst
        hist_lists = [ _cap_front(h, HISTORY_LIMIT) for h in hist_lists ]

    user_ids = behaviour_clean['User_ID'].tolist()
    pos_pairs = [
        (user2idx[u], news2idx[nid])
        for u, hlist in zip(user_ids, hist_lists)
        for nid in hlist
        if nid in news2idx
    ]

    # 3. 划分 Train/Test
    train_pos, test_pos = train_test_split(pos_pairs, test_size=0.2, random_state=42)

    # 4. 负采样函数（只用news2idx的key采样，彻底避免KeyError）
    def generate_samples(pos_pairs, user2idx, news2idx, num_neg=4):
        news_id_candidates = list(news2idx.keys())
        pos_user_idx, pos_news_idx, neg_user_idx, neg_news_idx = [], [], [], []
        # 先按用户分组
        from collections import defaultdict
        user_hist = defaultdict(set)
        for u_idx, n_idx in pos_pairs:
            user_hist[u_idx].add(n_idx)

        # NEW: progress bar (logic unchanged; just visual feedback)
        for u_idx, pos_news_set in tqdm(user_hist.items(),
                                        total=len(user_hist),
                                        desc="Negative sampling",
                                        leave=False):
            for n_idx in pos_news_set:
                pos_user_idx.append(u_idx)
                pos_news_idx.append(n_idx)
                # 负采样：从未点击过的新闻中采样
                neg_candidates = [news2idx[nid] for nid in news_id_candidates
                                if news2idx[nid] not in pos_news_set]
                for _ in range(num_neg):
                    neg_n_idx = random.choice(neg_candidates)
                    neg_news_idx.append(neg_n_idx)
                    neg_user_idx.append(u_idx)
        return pos_user_idx, pos_news_idx, neg_user_idx, neg_news_idx

    # 5. 生成训练样本
    pos_user_idx, pos_news_idx, neg_user_idx, neg_news_idx = generate_samples(
        train_pos, user2idx, news2idx, num_neg=4
    )
    user_idx = torch.tensor(pos_user_idx + neg_user_idx, dtype=torch.long)
    news_idx = torch.tensor(pos_news_idx + neg_news_idx, dtype=torch.long)
    labels = torch.tensor([1]*len(pos_user_idx) + [0]*len(neg_user_idx), dtype=torch.float)
    train_dataset = TensorDataset(user_idx, news_idx, labels)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    # 6. 生成测试样本（num_neg可以设大一些）
    pos_user_idx, pos_news_idx, neg_user_idx, neg_news_idx = generate_samples(
        test_pos, user2idx, news2idx, num_neg=100
    )
    user_idx = torch.tensor(pos_user_idx + neg_user_idx, dtype=torch.long)
    news_idx = torch.tensor(pos_news_idx + neg_news_idx, dtype=torch.long)
    labels = torch.tensor([1]*len(pos_user_idx) + [0]*len(neg_user_idx), dtype=torch.float)
    test_dataset = TensorDataset(user_idx, news_idx, labels)
    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    from sklearn.model_selection import train_test_split
    import random, numpy as np, torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    from tqdm.notebook import tqdm 

    #7训练
    # 1. 设定 device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    # 2. 模型和图数据搬到 GPU
    entity_emb_dim = len(next(iter(entity_emb.values())))  # 取 embedding 维度
    model = SentimentKGAT(hidden_dim=16, entity_emb_dim=entity_emb_dim).to(device)
    data.to(device)   # 就地搬运，切忌写成 data = data.to(device)



    # 3. 定义优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 4. CUDA 计时器
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt   = torch.cuda.Event(enable_timing=True)



    import time
    import matplotlib.pyplot as plt
    from tqdm.notebook import tqdm
    import torch.nn.functional as F

    import copy

    class EarlyStopping:
        def __init__(self, patience=5, min_delta=0.001, restore_best_weights=True):
            self.patience = patience
            self.min_delta = min_delta
            self.restore_best_weights = restore_best_weights
            self.best_loss = None
            self.counter = 0
            self.best_weights = None
            
        def __call__(self, val_loss, model):
            if self.best_loss is None:
                self.best_loss = val_loss
                self.best_weights = copy.deepcopy(model.state_dict())
            elif val_loss < self.best_loss - self.min_delta:
                self.best_loss = val_loss
                self.counter = 0
                self.best_weights = copy.deepcopy(model.state_dict())
            else:
                self.counter += 1
                
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
            return False

    # 初始化早停
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # 在训练循环中使用


    num_epochs = 10
    train_losses = []
    test_losses  = []
    epoch_times  = []

    for epoch in range(1, num_epochs + 1):
        # 训练阶段
        model.train()
        start_time = time.time()
        total_loss = 0.0
        for u_idx, n_idx, lbl in tqdm(train_loader, desc=f"Epoch {epoch} Training", leave=False):
            u_idx, n_idx, lbl = u_idx.to(device), n_idx.to(device), lbl.to(device)
            optimizer.zero_grad()
            out = model(data)
            u_emb = out['user'][u_idx]
            n_emb = out['news'][n_idx]
            logits = (u_emb * n_emb).sum(dim=1)
            loss   = F.binary_cross_entropy_with_logits(logits, lbl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)

        # 测试阶段
        model.eval()
        total_test_loss = 0.0
        for u_idx, n_idx, lbl in tqdm(test_loader, desc=f"Epoch {epoch} Testing", leave=False):
            u_idx, n_idx, lbl = u_idx.to(device), n_idx.to(device), lbl.to(device)
            out = model(data)
            u_emb = out['user'][u_idx]
            n_emb = out['news'][n_idx]
            logits = (u_emb * n_emb).sum(dim=1)
            loss   = F.binary_cross_entropy_with_logits(logits, lbl)
            total_test_loss += loss.item()
        test_loss = total_test_loss / len(test_loader)

        elapsed = time.time() - start_time

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        epoch_times.append(elapsed)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}, Time={elapsed:.1f}s")
        if early_stopping(test_loss, model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            print(f"Best test loss: {early_stopping.best_loss:.4f}")
            break
    # 绘图
    # 绘图 - 修正：使用实际训练轮数
    actual_epochs = len(train_losses)
    epoch_range = range(1, actual_epochs + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epoch_range, train_losses, marker='o', label='Train Loss')
    axes[0].plot(epoch_range, test_losses, marker='s', label='Test Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Train vs Test Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epoch_range, epoch_times, marker='^', label='Epoch Time (s)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Time (s)')
    axes[1].set_title('Epoch Duration')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    from sklearn.metrics import roc_auc_score
    import numpy as np
    from collections import defaultdict
    #8 evaluation
    K = 10  # 你可以改成5、20等

    model.eval()
    ys, ps, us = [], [], []
    with torch.no_grad():
        for u_idx, n_idx, lbl in test_loader:
            out = model(data)
            u_emb = out['user'][u_idx]
            n_emb = out['news'][n_idx]
            logits = (u_emb * n_emb).sum(dim=1)
            ys.append(lbl.cpu())
            ps.append(torch.sigmoid(logits).cpu())
            us.append(u_idx.cpu())
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    user_idx = torch.cat(us).numpy()

    # 1. AUC
    auc = roc_auc_score(y_true, y_pred)
    print(f'AUC: {auc:.4f}')

    # 2. Top-K Precision/Recall, MRR, NDCG
    user_pred = defaultdict(list)
    user_true = defaultdict(list)
    for u, t, p in zip(user_idx, y_true, y_pred):
        user_pred[u].append(p)
        user_true[u].append(t)

    precisions, recalls, mrrs, ndcgs = [], [], [], []

    def ndcg_at_k(r, k):
        r = np.asfarray(r)[:k]
        if r.size == 0:
            return 0.0
        dcg = np.sum(r / np.log2(np.arange(2, r.size + 2)))
        idcg = np.sum(sorted(r, reverse=True)[:k] / np.log2(np.arange(2, r.size + 2)))
        return dcg / idcg if idcg > 0 else 0.0

    for u in user_true:
        pred_scores = np.array(user_pred[u])
        true_labels = np.array(user_true[u])
        if true_labels.sum() == 0:
            continue  # 跳过没有正样本的用户
        # Top-K
        topk_idx = np.argsort(pred_scores)[::-1][:K]
        hits = true_labels[topk_idx].sum()
        precisions.append(hits / K)
        recalls.append(hits / true_labels.sum())
        # MRR
        sorted_idx = np.argsort(pred_scores)[::-1]
        sorted_true = true_labels[sorted_idx]
        rank = np.where(sorted_true == 1)[0]
        if len(rank) > 0:
            mrrs.append(1.0 / (rank[0] + 1))
        else:
            mrrs.append(0.0)
        # NDCG
        ndcgs.append(ndcg_at_k(sorted_true, K))

    print(f'Precision@{K}: {np.mean(precisions):.4f}')
    print(f'Recall@{K}: {np.mean(recalls):.4f}')
    print(f'MRR: {np.mean(mrrs):.4f}')
    print(f'NDCG@{K}: {np.mean(ndcgs):.4f}')
    return K, precisions, recalls, mrrs, ndcgs