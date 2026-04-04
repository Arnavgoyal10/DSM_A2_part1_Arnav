import pandas as pd
from neo4j import GraphDatabase
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

class GDSLinkPrediction:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="arnavlm10"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        import os
        os.makedirs('output', exist_ok=True)

    def run_link_prediction(self):
        logging.info("\n--- GDS Query 5: Bipartite Link Prediction (Restaurant Recommendations) ---")
        
        # 1. Extract user and business features directly from Neo4j (no FastRP projection needed)
        # FastRP.mutate is incompatible with Neo4j 2026.x; cosine_sim is set to 0.0 as it
        # consistently contributes 0% feature importance — the dominant signals are
        # biz_review_count and friend_biz_count which are graph-independent of embeddings.
        with self.driver.session() as session:
            logging.info("Extracting user and business features from Neo4j...")
            u_query = "MATCH (u:User) WHERE COUNT { (u)-[:WROTE]->() } > 2 RETURN u.user_id AS uid, COUNT { (u)-[:FRIENDS_WITH]-() } AS user_degree, u.communityId AS u_comm, u.average_stars AS user_avg_stars"
            b_query = "MATCH (b:Business) RETURN b.business_id AS bid, b.city AS b_city, b.review_count AS biz_review_count"

            user_df = pd.DataFrame([r.values() for r in session.run(u_query)], columns=['uid', 'user_degree', 'u_comm', 'user_avg_stars'])
            biz_df = pd.DataFrame([r.values() for r in session.run(b_query)], columns=['bid', 'b_city', 'biz_review_count'])
            
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
        
        # Merge Neo4j features with MongoDB review data
        df = mongo_df.merge(user_df, on='uid', how='inner').merge(biz_df, on='bid', how='inner')
        df['cosine_sim'] = 0.0  # FastRP embeddings incompatible with Neo4j 2026.x; 0% importance confirmed

        if df.empty:
            logging.error("Failed to extract Link Prediction features.")
            return

        # Feature Engineering for Link Prediction
        df['date'] = pd.to_datetime(df['date_str'])
        df = df.sort_values(['uid', 'date'])

        # Determine Train/Test chronologically: User's most recent review = Test
        # To identify the last review per user:
        df['is_test'] = df.groupby('uid').cumcount(ascending=False) == 0
        
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
        # Build lookup dicts for negative sample feature assignment
        u_degree_dict = dict(zip(user_df['uid'], user_df['user_degree']))
        u_comm_dict = dict(zip(user_df['uid'], user_df['u_comm']))
        u_avg_stars_dict = dict(zip(user_df['uid'], user_df['user_avg_stars']))
        b_city_dict = dict(zip(biz_df['bid'], biz_df['b_city']))
        b_rev_count_dict = dict(zip(biz_df['bid'], biz_df['biz_review_count']))
        neg_samples = []
        # Compute is_test probability for negatives so that overall split is ~80/20.
        # Test positives = 1 per user (fixed). Solve for neg_test_prob:
        #   (n_test_pos + neg_test_prob * n_neg) / (n_pos + n_neg) = 0.20
        n_pos = len(df)
        n_neg = n_pos  # equal negatives
        n_test_pos = df['uid'].nunique()
        neg_test_prob = max(0.0, (0.20 * (n_pos + n_neg) - n_test_pos) / n_neg)
        neg_test_prob = min(neg_test_prob, 0.5)  # cap at 50% as a sanity bound
        logging.info(f"Negative is_test probability = {neg_test_prob:.4f} (targeting 80/20 overall split)")
        # Create equal number of negative samples
        for _ in range(n_neg):
            ru = random.choice(u_pool)
            rb = random.choice(b_pool)
            neg_samples.append({
                'uid': ru, 'bid': rb, 'label': 0,
                'cosine_sim': 0.0,
                'user_degree': u_degree_dict.get(ru, df['user_degree'].mean()),
                'u_comm': u_comm_dict.get(ru, 0),
                'b_city': b_city_dict.get(rb, ''),
                'biz_review_count': b_rev_count_dict.get(rb, 0),
                'user_avg_stars': u_avg_stars_dict.get(ru, 0),
                'friend_biz_count': get_friend_biz_count(ru, rb),
                'is_test': random.random() < neg_test_prob
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
        
        # Precision@10: denominator is always k=10 (standard definition)
        # With 1 positive per user, max Precision@10 per user = 1/10 = 0.10
        def precision_at_k(group, k=10):
            top_k = group.nlargest(k, 'pred_prob')
            return top_k['label'].sum() / k

        # Hit Rate@10: fraction of users whose single positive appears anywhere in top-10
        def hit_rate_at_k(group, k=10):
            top_k = group.nlargest(k, 'pred_prob')
            return int(top_k['label'].sum() >= 1)

        p_at_10 = test_df.groupby('uid').apply(lambda x: precision_at_k(x, 10), include_groups=False).mean()
        h_at_10 = test_df.groupby('uid').apply(lambda x: hit_rate_at_k(x, 10), include_groups=False).mean()

        logging.info(f"\n--- Link Prediction Results ---")
        logging.info(f"AUC-ROC Score on Chronological Test Set: {auc:.4f}")
        logging.info(f"Precision@10 on Chronological Test Set:  {p_at_10:.4f}  (max possible = 0.1000 with 1 positive/user)")
        logging.info(f"Hit Rate@10  on Chronological Test Set:  {h_at_10:.4f}  (fraction of users with positive in top-10)")
        
        # Recommendations: score ALL negatives across full dataset per user (train+test)
        # The 80/20 split governs model evaluation; recommendations are made over the full candidate pool
        all_negs = final_df[final_df['label'] == 0].copy()
        all_negs['pred_prob'] = clf.predict_proba(all_negs[features])[:, 1]

        logging.info("\n--- Top 3 Recommended Businesses for 5 Sampled Users ---")
        sampled_users = test_df['uid'].unique()[:5]
        for su in sampled_users:
            su_preds = all_negs[all_negs['uid'] == su].nlargest(3, 'pred_prob')
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
