import json
import os

def subset_yelp_data(source_dir, dest_dir, num_businesses=10000):
    os.makedirs(dest_dir, exist_ok=True)
    
    kept_business_ids = set()
    kept_user_ids = set()
    
    # 1. Subset Businesses
    print(f"Subsetting businesses (target: {num_businesses})...")
    with open(os.path.join(source_dir, 'yelp_academic_dataset_business.json'), 'r', encoding='utf-8') as f_in, \
         open(os.path.join(dest_dir, 'business.json'), 'w', encoding='utf-8') as f_out:
        count = 0
        for line in f_in:
            if count >= num_businesses:
                break
            b = json.loads(line)
            kept_business_ids.add(b['business_id'])
            f_out.write(line)
            count += 1
    
    # 2. Subset Reviews
    print("Subsetting reviews...")
    with open(os.path.join(source_dir, 'yelp_academic_dataset_review.json'), 'r', encoding='utf-8') as f_in, \
         open(os.path.join(dest_dir, 'review.json'), 'w', encoding='utf-8') as f_out:
        for line in f_in:
            r = json.loads(line)
            if r['business_id'] in kept_business_ids:
                kept_user_ids.add(r['user_id'])
                f_out.write(line)
                
    # 3. Subset Users
    print(f"Subsetting users (found {len(kept_user_ids)} relevant users)...")
    with open(os.path.join(source_dir, 'yelp_academic_dataset_user.json'), 'r', encoding='utf-8') as f_in, \
         open(os.path.join(dest_dir, 'user.json'), 'w', encoding='utf-8') as f_out:
        for line in f_in:
            u = json.loads(line)
            if u['user_id'] in kept_user_ids:
                f_out.write(line)
                
    # 4. Subset Tips
    print("Subsetting tips...")
    tip_path = os.path.join(source_dir, 'yelp_academic_dataset_tip.json')
    if os.path.exists(tip_path):
        with open(tip_path, 'r', encoding='utf-8') as f_in, \
             open(os.path.join(dest_dir, 'tip.json'), 'w', encoding='utf-8') as f_out:
            for line in f_in:
                t = json.loads(line)
                if t['business_id'] in kept_business_ids and t['user_id'] in kept_user_ids:
                    f_out.write(line)

    # 5. Subset Checkins
    print("Subsetting checkins...")
    checkin_path = os.path.join(source_dir, 'yelp_academic_dataset_checkin.json')
    if os.path.exists(checkin_path):
        with open(checkin_path, 'r', encoding='utf-8') as f_in, \
             open(os.path.join(dest_dir, 'checkin.json'), 'w', encoding='utf-8') as f_out:
            for line in f_in:
                c = json.loads(line)
                if c['business_id'] in kept_business_ids:
                    f_out.write(line)
                    
    print("Done! Smaller, manageable subset generated successfully.")

if __name__ == '__main__':
    SOURCE_DIRECTORY = 'Yelp JSON/yelp_dataset'
    DESTINATION_DIRECTORY = './yelp_subset'
    
    subset_yelp_data(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, num_businesses=10000)
