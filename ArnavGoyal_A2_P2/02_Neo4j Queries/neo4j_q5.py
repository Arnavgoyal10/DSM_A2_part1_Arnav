import pandas as pd
from neo4j import GraphDatabase
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

class GDSLinkPrediction:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="arnavlm10"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        import os
        os.makedirs('output', exist_ok=True)

    def run_link_prediction(self):
        logging.info("\n--- GDS Query 5: Bipartite Link Prediction (Restaurant Recommendations) ---")
        
        # 1. We execute FastRP to get topological graph embeddings natively in GDS
        with self.driver.session() as session:
            session.run("CALL gds.graph.drop('lpGraph', false)")
            
            proj = """
            CALL gds.graph.project.cypher(
              'lpGraph',
              'MATCH (n) WHERE (n:User AND COUNT { (n)-[:WROTE]->() } > 2) OR n:Business RETURN id(n) AS id',
              'MATCH (u)-[:WROTE]->(:Review)-[:ABOUT]->(b:Business) WHERE COUNT { (u)-[:WROTE]->() } > 2 RETURN id(u) AS source, id(b) AS target UNION MATCH (u1:User)-[:FRIENDS_WITH]-(u2:User) WHERE COUNT { (u1)-[:WROTE]->() } > 2 AND COUNT { (u2)-[:WROTE]->() } > 2 RETURN id(u1) AS source, id(u2) AS target',
              {validateRelationships: false}
            )
            """
            session.run(proj)
            logging.info("Projected multiplex graph for topological embeddings...")
            
            fastrp = """
            CALL gds.fastRP.mutate('lpGraph', {
               embeddingDimension: 16,
               randomSeed: 42,
               mutateProperty: 'fastrp_embedding'
            })
            """
            session.run(fastrp)
            logging.info("Executed FastRP embedding generation algorithm.")
            
            logging.info("Extracting topological embeddings...")
            u_query = "MATCH (u:User) WHERE COUNT { (u)-[:WROTE]->() } > 2 RETURN u.user_id AS uid, gds.util.nodeProperty('lpGraph', id(u), 'fastrp_embedding') AS u_emb, COUNT { (u)-[:FRIENDS_WITH]-() } AS user_degree, u.communityId AS u_comm, u.average_stars AS user_avg_stars"
            b_query = "MATCH (b:Business) RETURN b.business_id AS bid, gds.util.nodeProperty('lpGraph', id(b), 'fastrp_embedding') AS b_emb, b.city AS b_city, b.review_count AS biz_review_count"

            user_df = pd.DataFrame([r.values() for r in session.run(u_query)], columns=['uid', 'u_emb', 'user_degree', 'u_comm', 'user_avg_stars'])
            biz_df = pd.DataFrame([r.values() for r in session.run(b_query)], columns=['bid', 'b_emb', 'b_city', 'biz_review_count'])
            
            session.run("CALL gds.graph.drop('lpGraph', false)")
            
        import pymongo
        mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")
        db = mongo_client["yelp"]
        logging.info("Fetching chronological temporal arrays from Document Store...")
        
        # Generate chronological temporal arrays from Document Store via Sampling
        # This completely avoids BSON 16MB query limits for enormous user arrays
        mongo_pipeline = [
            {"$sample": {"size": 300000}},
            {"$project": {"uid": "$user_id", "bid": "$business_id", "date_str": "$date", "_id": 0}}
        ]
        mongo_df = pd.DataFrame(list(db.review.aggregate(mongo_pipeline)))
        
        # Merge Paradigms mathematically in RAM
        df = mongo_df.merge(user_df, on='uid', how='inner').merge(biz_df, on='bid', how='inner')

        if df.empty:
            logging.error("Failed to extract Link Prediction features.")
            return

        # Feature Engineering for Link Prediction
        df['date'] = pd.to_datetime(df['date_str'])
        df = df.sort_values(['uid', 'date'])
        
        # Determine Train/Test chronologically: User's most recent review = Test
        # To identify the last review per user:
        df['is_test'] = df.groupby('uid').cumcount(ascending=False) == 0
        
        # Cosine similarity of embeddings manually
        def cosine_sim(a, b):
            if a is None or b is None: return 0.0
            import numpy as np
            a, b = np.array(a), np.array(b)
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            
        df['cosine_sim'] = df.apply(lambda row: cosine_sim(row['u_emb'], row['b_emb']), axis=1)
        
        # Feature: friend_biz_count — review behaviour of user's friends toward the candidate business
        # For each (uid, bid), count how many of uid's friends have reviewed bid in the sampled data.
        logging.info("Loading friendship data for friend-behaviour feature...")
        sample_uids = df['uid'].unique().tolist()
        friends_map = {}  # uid -> set of friend user_ids
        batch_size_f = 500
        with self.driver.session() as fsession:
            friends_q = "UNWIND $uids AS u_id MATCH (u:User {user_id: u_id})-[:FRIENDS_WITH]-(f:User) RETURN u_id AS uid, f.user_id AS fid"
            for i in range(0, len(sample_uids), batch_size_f):
                batch = sample_uids[i:i+batch_size_f]
                for r in fsession.run(friends_q, uids=batch):
                    uid = r['uid']
                    if uid not in friends_map:
                        friends_map[uid] = set()
                    friends_map[uid].add(r['fid'])

        # Build reviewer set per business from the positive sample
        biz_reviewers = df.groupby('bid')['uid'].apply(set).to_dict()

        def get_friend_biz_count(uid, bid):
            return len(friends_map.get(uid, set()) & biz_reviewers.get(bid, set()))

        logging.info("Computing friend_biz_count for positive samples...")
        df['friend_biz_count'] = df.apply(lambda row: get_friend_biz_count(row['uid'], row['bid']), axis=1)

        # Generate Negative Samples (Edges that don't exist)
        # We'll just randomly pair users and businesses, label them 0.
        logging.info("Generating negative edge samples algebraically...")
        u_pool = df['uid'].unique()
        b_pool = df['bid'].unique()
        import random
        # Build lookup dicts so negative samples get real embedding cosine similarity
        u_emb_dict = dict(zip(user_df['uid'], user_df['u_emb']))
        b_emb_dict = dict(zip(biz_df['bid'], biz_df['b_emb']))
        u_degree_dict = dict(zip(user_df['uid'], user_df['user_degree']))
        u_comm_dict = dict(zip(user_df['uid'], user_df['u_comm']))
        u_avg_stars_dict = dict(zip(user_df['uid'], user_df['user_avg_stars']))
        b_city_dict = dict(zip(biz_df['bid'], biz_df['b_city']))
        b_rev_count_dict = dict(zip(biz_df['bid'], biz_df['biz_review_count']))
        neg_samples = []
        # Create equal number of negative samples with real cosine_sim (not random)
        for _ in range(len(df)):
            ru = random.choice(u_pool)
            rb = random.choice(b_pool)
            cs = cosine_sim(u_emb_dict.get(ru), b_emb_dict.get(rb))
            neg_samples.append({
                'uid': ru, 'bid': rb, 'label': 0,
                'cosine_sim': cs,
                'user_degree': u_degree_dict.get(ru, df['user_degree'].mean()),
                'u_comm': u_comm_dict.get(ru, 0),
                'b_city': b_city_dict.get(rb, ''),
                'biz_review_count': b_rev_count_dict.get(rb, 0),
                'user_avg_stars': u_avg_stars_dict.get(ru, 0),
                'friend_biz_count': get_friend_biz_count(ru, rb),
                'is_test': random.choice([True, False])
            })
            
        df['label'] = 1 # Positive samples
        neg_df = pd.DataFrame(neg_samples)
        
        final_df = pd.concat([df[['uid', 'bid', 'cosine_sim', 'user_degree', 'u_comm', 'b_city', 'biz_review_count', 'user_avg_stars', 'friend_biz_count', 'is_test', 'label']], neg_df], ignore_index=True)
        final_df['u_comm'] = final_df['u_comm'].fillna(-1).astype(str)
        final_df['b_city'] = final_df['b_city'].fillna('').astype(str)
        
        # Target Encoding for categories/cities
        final_df['city_hash'] = final_df['b_city'].apply(hash) % 1000
        final_df['comm_hash'] = final_df['u_comm'].apply(hash) % 1000
        
        train_df = final_df[~final_df['is_test']]
        test_df = final_df[final_df['is_test']]
        
        features = ['cosine_sim', 'user_degree', 'city_hash', 'comm_hash', 'biz_review_count', 'user_avg_stars', 'friend_biz_count']
        
        X_train = train_df[features]
        y_train = train_df['label']
        X_test = test_df[features]
        y_test = test_df['label']
        
        logging.info(f"Training DataFrame: {X_train.shape} | Test DataFrame: {X_test.shape} (Chronologically Enforced)")
        
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        
        probs = clf.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, probs)
        
        test_df = test_df.copy()
        test_df['pred_prob'] = probs
        
        # Calculate Precision@10
        def precision_at_k(group, k=10):
            # If user has fewer than k test items, we denominate by len(group)
            k_eff = min(k, len(group))
            if k_eff == 0: return 0.0
            top_k = group.nlargest(k_eff, 'pred_prob')
            return top_k['label'].sum() / k_eff
            
        p_at_10 = test_df.groupby('uid').apply(lambda x: precision_at_k(x, 10)).mean()
        
        logging.info(f"\n--- Link Prediction Results ---")
        logging.info(f"AUC-ROC Score on Chronological Test Set: {auc:.4f}")
        logging.info(f"Precision@10 on Chronological Test Set:  {p_at_10:.4f}")
        
        logging.info("\n--- Top 3 Recommended Businesses for 5 Sampled Users ---")
        sampled_users = test_df['uid'].unique()[:5]
        for su in sampled_users:
            su_preds = test_df[(test_df['uid'] == su) & (test_df['label'] == 0)].nlargest(3, 'pred_prob')
            bids = su_preds['bid'].tolist()
            logging.info(f"User {su} -> Recommended Biz IDs: {bids}")
        
        # Visualization
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.metrics import roc_curve
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Bipartite Link Prediction ROC Curve')
        plt.legend(loc="lower right")
        plt.savefig("output/stage3_roc_curve.png", dpi=300)
        plt.close()
        logging.info("\n[!] Generated visualization: output/stage3_roc_curve.png")
        
        fi = pd.DataFrame({'Feature': features, 'Importance': clf.feature_importances_}).sort_values('Importance', ascending=False)
        logging.info("\nMost Predictive Topology Features:")
        logging.info(fi.to_string(index=False))

if __name__ == "__main__":
    lp = GDSLinkPrediction()
    lp.run_link_prediction()
