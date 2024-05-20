def evaluate_model(model, model_csr, user_index, n=10):
    user_items = model_csr[user_index]
    recommendations = model.recommend(user_index, user_items, N=n)[0]
    return recommendations

def get_outfit_id_from_index(outfit_indexes, outfit_dict):
    return [outfit_dict[idx] for idx in outfit_indexes]

def evaluate_hit_rate_at_n(test_id, predicted_ids, n=10):
    predicted_ids = predicted_ids[:n]
    if test_id in predicted_ids:
        return 1
    return 0