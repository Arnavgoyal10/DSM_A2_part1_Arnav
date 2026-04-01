print("\n--- QUERY 1: SAFEST CITIES (BEST) ---");
printjson(db.business.aggregate([{ $unwind: "$categories" }, { $group: { _id: { city: "$city", category: "$categories" }, avg_stars: { $avg: "$stars" }, count: { $sum: 1 } } }, { $match: { count: { $gte: 5 } } }, { $sort: { avg_stars: -1 } }, { $limit: 3 }]).toArray());

print("\n--- QUERY 1: LEAST SAFE CITIES (WORST) ---");
printjson(db.business.aggregate([{ $unwind: "$categories" }, { $group: { _id: { city: "$city", category: "$categories" }, avg_stars: { $avg: "$stars" }, count: { $sum: 1 } } }, { $match: { count: { $gte: 5 } } }, { $sort: { avg_stars: 1 } }, { $limit: 3 }]).toArray());

print("\n--- QUERY 3: REVIEW VOLUME VS STARS ---");
printjson(db.business.aggregate([{ $bucket: { groupBy: "$review_count", boundaries: [0, 50, 500], default: "500+", output: { avg_stars: { $avg: "$stars" }, count: { $sum: 1 } } } }]).toArray());

print("\n--- QUERY 4: BUSINESS CATEGORY BEHAVIOR ---");
printjson(db.review.aggregate([{ $lookup: { from: "business", localField: "business_id", foreignField: "business_id", as: "biz" } }, { $unwind: "$biz" }, { $unwind: "$biz.categories" }, { $group: { _id: "$biz.categories", avg_stars: { $avg: "$stars" }, avg_useful: { $avg: "$useful" }, count: { $sum: 1 } } }, { $sort: { count: -1 } }, { $limit: 5 }]).toArray());

print("\n--- QUERY 6: ELITE VS NON-ELITE USERS ---");
printjson(db.review.aggregate([{ $lookup: { from: "user", localField: "user_id", foreignField: "user_id", as: "u" } }, { $unwind: "$u" }, { $group: { _id: { $ne: ["$u.elite", ""] }, avg_stars: { $avg: "$stars" }, avg_useful: { $avg: "$useful" }, count: { $sum: 1 } } }]).toArray());

print("\n--- QUERIES COMPLETED ---");
