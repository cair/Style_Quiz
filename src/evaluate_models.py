import numpy as np

def evaluate_model(model, model_csr, user_index, n=10):
    user_items = model_csr[user_index]
    recommendations = model.recommend(user_index, user_items, N=n)[0]
    return recommendations

def get_outfit_id_from_index(outfit_indexes, outfit_dict):
    return [outfit_dict[idx] for idx in outfit_indexes]

def evaluate_hit_rate_at_n(test_id, predicted_ids, n=10):
    if predicted_ids is np.nan:
        print(f"None prediction for {test_id}!")
        return 0
    predicted_ids = predicted_ids[:n]
    if type(test_id) == str or type(test_id) == np.str_:
        if test_id in predicted_ids:
            #print(f"Hit at {n} for {test_id} in {predicted_ids}")
            return 1
    elif type(test_id) == list or type(test_id) == np.ndarray:
        for outfit_id in test_id:
            if outfit_id in predicted_ids:
                return 1
    else:
        raise ValueError(f"Unknown type {type(test_id)}")
    return 0