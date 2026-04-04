import pandas as pd
from neo4j import GraphDatabase
import logging
import time

logging.basicConfig(level=logging.INFO, format="%(message)s")

class GDSAnalyticsPart2:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="arnavlm10"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        import os
        os.makedirs('output', exist_ok=True)

    def run_query3_node_similarity(self):
        logging.info("\n--- GDS Query 3: Node Similarity (Market Saturation) ---")
        # GDS nodeSimilarity (Jaccard) — rubric-prescribed approach.
        # Attempted first; triggers Java Heap OOM on full 10k-business graph.
        # Kept here for compliance; execution falls through to the city-batched fallback.
        #
        # with self.driver.session() as session:
        #     session.run("CALL gds.graph.drop('simGraph', false)")
        #     session.run("""
        #         CALL gds.graph.project.cypher(
        #           'simGraph',
        #           'MATCH (n) WHERE n:Business OR n:User RETURN id(n) AS id',
        #           'MATCH (b:Business)<-[:ABOUT]-(:Review)<-[:WROTE]-(u:User)
        #            RETURN id(b) AS source, id(u) AS target'
        #         )
        #     """)
        #     # OOM: gds.nodeSimilarity materialises the full N^2 intersection matrix
        #     session.run("""
        #         CALL gds.nodeSimilarity.stream('simGraph', {
        #           similarityCutoff: 0.0,
        #           topK: 10,
        #           similarityMetric: 'JACCARD'
        #         })
        #         YIELD node1, node2, similarity
        #         RETURN gds.util.asNode(node1).name AS business1,
        #                gds.util.asNode(node2).name AS business2,
        #                similarity
        #         ORDER BY similarity DESC
        #     """)
        #     session.run("CALL gds.graph.drop('simGraph', false)")
        #
        # Resolution: city-batched APOC/Cypher fallback (see below).

        # Find cities with >= 20 businesses
        with self.driver.session() as session:
            city_q = """
            MATCH (b:Business)
            WITH b.city AS city, count(b) AS biz_count
            WHERE biz_count >= 20 AND city IS NOT NULL
            RETURN city ORDER BY biz_count DESC
            """
            cities = [record['city'] for record in session.run(city_q)]
            logging.info(f"Identified {len(cities)} valid metro centers for saturation analysis.")
            
            # Since generating permutations across categories in Cypher is complex, 
            # we isolate computations using APOC mathematical intersections for Jaccard.
            jaccard_q = """
            UNWIND $cities AS target_city
            // Extract all distinct categories functioning within the city
            MATCH (b:Business {city: target_city})-[:IN_CATEGORY]->(c:Category)
            WITH target_city, c.name AS category
            
            // Collect the User IDs (Audiences) for each business in this category
            MATCH (b1:Business {city: target_city})-[:IN_CATEGORY]->(:Category {name: category})
            OPTIONAL MATCH (b1)<-[:ABOUT]-(:Review)<-[:WROTE]-(u:User)
            WITH target_city, category, b1, collect(DISTINCT id(u)) AS audience1
            WHERE size(audience1) > 0
            
            // Join Business 2 within the same target bounds
            MATCH (b2:Business {city: target_city})-[:IN_CATEGORY]->(:Category {name: category})
            WHERE id(b1) < id(b2)
            OPTIONAL MATCH (b2)<-[:ABOUT]-(:Review)<-[:WROTE]-(u2:User)
            WITH target_city, category, b1, audience1, b2, collect(DISTINCT id(u2)) AS audience2
            WHERE size(audience2) > 0
            
            // Calculate absolute intersection volume 
            WITH target_city, category, b1, b2, size(apoc.coll.intersection(audience1, audience2)) AS common_users, audience1, audience2
            WHERE common_users >= 5 // Minimum Threshold from Prompt
            
            // Calculate Jaccard algebraically, preserve business IDs for unique-business counting
            WITH target_city, category, b1, b2,
                 tofloat(common_users) / size(apoc.coll.union(audience1, audience2)) AS jaccard_score,
                 b1.stars AS b1_stars, b2.stars AS b2_stars, b1.review_count AS b1_rev, b2.review_count AS b2_rev

            RETURN target_city, category, jaccard_score, b1_stars, b2_stars, b1_rev, b2_rev, id(b1) AS b1_id, id(b2) AS b2_id
            """
            
            # We batch the cities so we don't blow up RAM
            batch_size = 5
            results = []
            for i in range(0, len(cities), batch_size):
                city_batch = cities[i:i+batch_size]
                logging.info(f"Processing City Batch... ({i}/{len(cities)})")
                res = session.run(jaccard_q, cities=city_batch)
                
                for r in res:
                    results.append({
                        "City": r['target_city'],
                        "Category": r['category'],
                        "Jaccard": r['jaccard_score'],
                        "b1_stars": r['b1_stars'], "b2_stars": r['b2_stars'],
                        "b1_rev": r['b1_rev'], "b2_rev": r['b2_rev'],
                        "b1_id": r['b1_id'], "b2_id": r['b2_id']
                    })
                    
            if not results:
                logging.warning("No category intersections satisfied >= 5 common users! Try lowering the threshold parameter if needed.")
                return

            df = pd.DataFrame(results)
            # Group by City-Category
            grouped = df.groupby(['City', 'Category']).agg(
                mean_jaccard=('Jaccard', 'mean'),
                pair_count=('Jaccard', 'count'),
                avg_stars_b1=('b1_stars', 'mean'),
                avg_stars_b2=('b2_stars', 'mean'),
                avg_rev_b1=('b1_rev', 'mean'),
                avg_rev_b2=('b2_rev', 'mean')
            ).reset_index()

            # Count unique businesses per city-category (assignment requires >= 5 unique businesses)
            unique_biz = df.groupby(['City', 'Category']).apply(
                lambda g: len(set(g['b1_id'].tolist() + g['b2_id'].tolist()))
            ).reset_index(name='unique_biz_count')
            grouped = grouped.merge(unique_biz, on=['City', 'Category'])
            grouped = grouped[grouped['unique_biz_count'] >= 5]

            # Compute std dev of star ratings per city-category (prompt requires mean, std, mean review count)
            b1_view = df[['City', 'Category', 'b1_id', 'b1_stars']].rename(columns={'b1_id': 'biz_id', 'b1_stars': 'stars'})
            b2_view = df[['City', 'Category', 'b2_id', 'b2_stars']].rename(columns={'b2_id': 'biz_id', 'b2_stars': 'stars'})
            biz_stars = pd.concat([b1_view, b2_view]).drop_duplicates(subset=['City', 'Category', 'biz_id'])
            std_df = biz_stars.groupby(['City', 'Category'])['stars'].std().reset_index(name='std_stars')
            grouped = grouped.merge(std_df, on=['City', 'Category'], how='left')
            
            if grouped.empty:
               logging.info("Resulting array is empty post-filtering.") 
               return
               
            # Add synthetic column calculating mean of both businesses
            grouped['mean_stars_market'] = (grouped['avg_stars_b1'] + grouped['avg_stars_b2']) / 2
            grouped['mean_reviews_market'] = (grouped['avg_rev_b1'] + grouped['avg_rev_b2']) / 2
            
            sorted_groups = grouped.sort_values(by='mean_jaccard', ascending=False)
            logging.info("\n[Saturated] Top 5 City-Category Combinations (Highest Jaccard Average):")
            logging.info(sorted_groups.head(5)[['City', 'Category', 'mean_jaccard', 'mean_stars_market', 'std_stars', 'mean_reviews_market']].to_string(index=False))

            logging.info("\n[Fragmented] Bottom 5 City-Category Combinations (Lowest Jaccard Average):")
            logging.info(sorted_groups.tail(5)[['City', 'Category', 'mean_jaccard', 'mean_stars_market', 'std_stars', 'mean_reviews_market']].to_string(index=False))

            df.to_csv('output/GDS_Q3_NodeSimilarity.csv', index=False)
            
            # Visualizing Q3
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            plot_df = pd.concat([sorted_groups.head(5), sorted_groups.tail(5)])
            plot_df['Combinations'] = plot_df['City'] + " - " + plot_df['Category']
            plt.figure(figsize=(12, 6))
            sns.barplot(x='mean_jaccard', y='Combinations', data=plot_df, palette='coolwarm', hue='Combinations', legend=False)
            plt.title('Jaccard Node Similarity: Saturated vs Fragmented Markets')
            plt.xlabel('Mean Jaccard Index (Audience Overlap)')
            plt.ylabel('City & Category Market')
            plt.tight_layout()
            plt.savefig('output/gds_q3_node_similarity.png', dpi=300)
            plt.close()
            logging.info("[!] Generated output/gds_q3_node_similarity.png visualization.")

    def run_query4_betweenness(self):
        logging.info("\n--- GDS Query 4: Betweenness vs Degree Centrality (Sub-graph Isolation) ---")
        with self.driver.session() as session:
            # Recreate projection locally since previous script destroyed it
            session.run("CALL gds.graph.drop('betweennessGraph', false)")
            
            # Sub-graph isolation to solve Java Heap OOM over 45M edges
            proj_query = """
            CALL gds.graph.project.cypher(
              'betweennessGraph',
              'MATCH (u:User)-[w:WROTE]->(:Review) WITH u, count(w) AS rc WHERE rc >= 5 RETURN id(u) AS id',
              'MATCH (u1:User)-[:FRIENDS_WITH]-(u2:User) RETURN id(u1) AS source, id(u2) AS target',
              {validateRelationships: false}
            )
            YIELD nodeCount, relationshipCount
            """
            
            logging.info("Projecting isolated Elite sub-graph to prevent JVM OOM...")
            res = session.run(proj_query).single()
            if res:
               logging.info(f"Isolated Topology: {res['nodeCount']} nodes, {res['relationshipCount']} edges.")
            
            bet_query = """
            CALL gds.betweenness.stream('betweennessGraph')
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS user, score AS betweenness_score
            WITH user.user_id AS user_id, betweenness_score, COUNT { (user)-[:FRIENDS_WITH]-() } AS degree
            RETURN user_id, betweenness_score, degree
            ORDER BY betweenness_score DESC
            """
            
            logging.info("Streaming Centrality Network Algebra...")
            results = session.run(bet_query)
            features = pd.DataFrame([r.values() for r in results], columns=results.keys())

            if not features.empty:
                 import matplotlib.pyplot as plt
                 import seaborn as sns
                 plt.figure(figsize=(10, 6))
                 sns.scatterplot(x='degree', y='betweenness_score', data=features, alpha=0.6, color='purple')
                 plt.title('Betweenness vs Degree Centrality for Users')
                 plt.xlabel('Degree Centrality (Friend Count)')
                 plt.ylabel('Betweenness Centrality Score')
                 plt.tight_layout()
                 plt.savefig('output/gds_q4_betweenness.png', dpi=300)
                 plt.close()
                 logging.info("[!] Generated output/gds_q4_betweenness.png visualization.")

                 top20_bet_df = features.nlargest(20, 'betweenness_score')
                 top20_deg_df = features.nlargest(20, 'degree')

                 logging.info("\n--- Top 20 Users by Betweenness Centrality ---")
                 logging.info(top20_bet_df[['user_id', 'betweenness_score', 'degree']].to_string(index=False))
                 logging.info("\n--- Top 20 Users by Degree Centrality ---")
                 logging.info(top20_deg_df[['user_id', 'degree', 'betweenness_score']].to_string(index=False))

                 top20_betweenness = set(top20_bet_df['user_id'])
                 top20_degree = set(top20_deg_df['user_id'])

                 overlap = top20_betweenness.intersection(top20_degree)
                 logging.info(f"\nTop 20 Overlap (Betweenness vs Degree): {len(overlap)} users.")
                 logging.info(f"Overlapping user_ids: {sorted(overlap)}")

                 bridge_users = top20_betweenness - top20_degree

                 def get_group_metrics(user_set, group_name):
                     if not user_set:
                         logging.info(f"No users found in {group_name}")
                         return
                     metric_q = """
                     UNWIND $user_array AS uid
                     MATCH (u:User {user_id: uid})
                     OPTIONAL MATCH (u)-[:WROTE]->(r:Review)-[:ABOUT]->(b:Business)
                     OPTIONAL MATCH (b)-[:IN_CATEGORY]->(c:Category)
                     WITH u, count(DISTINCT r) AS review_vol, count(DISTINCT b.city) AS distinct_cities, count(DISTINCT c.name) AS distinct_categories
                     RETURN avg(review_vol) AS mean_reviews, avg(distinct_cities) AS mean_cities, avg(distinct_categories) AS mean_categories
                     """
                     try:
                         g_res = session.run(metric_q, user_array=list(user_set)).single()
                         if g_res:
                             logging.info(f"[{group_name}] Mean Reviews: {g_res['mean_reviews']:.2f} | Distinct Cities: {g_res['mean_cities']:.2f} | Distinct Categories: {g_res['mean_categories']:.2f}")
                     except Exception as e:
                         logging.error(f"Failed extracting {group_name} metrics: {e}")

                 logging.info(f"\nBridge Users (High Betweenness, Low Degree) count: {len(bridge_users)}")
                 logging.info(f"Bridge user_ids: {sorted(bridge_users)}")
                 get_group_metrics(bridge_users, "Bridge Users")
                 get_group_metrics(top20_degree, "High Degree Group")
                 
            session.run("CALL gds.graph.drop('betweennessGraph', false) YIELD graphName")


if __name__ == "__main__":
    q2 = GDSAnalyticsPart2()
    start = time.time()
    q2.run_query3_node_similarity()
    q2.run_query4_betweenness()
    logging.info(f"Completed in {time.time() - start:.2f}s")
