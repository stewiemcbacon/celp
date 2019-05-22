from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import random

"""Deze functie wordt gebruikt om de ratings voor de utility matrix te berekenen"""
def get_rating(ratings,user_id,business_id):
    if ratings.loc[(ratings['user_id'] == user_id) & (ratings['business_id'] == business_id)]['stars'].any() == False:
        res = np.nan
    else:
        res = float(ratings.loc[(ratings['user_id'] == user_id) & (ratings['business_id'] == business_id),'stars'].values[0])
    return res

"""Deze functie wordt gebruikt om een utility matrix te maken"""
def pivot_ratings(df):
    """ takes a rating table as input and computes the utility matrix """
    business_ids = df['business_id'].unique()
    user_ids = df['user_id'].unique()

    # create empty data frame
    pivot_data = pd.DataFrame(np.nan, columns=user_ids, index=business_ids, dtype=float)

    # use the function get_rating to fill the matrix
    for x in pivot_data:
        for y in pivot_data.index:
            pivot_data[x][y] = get_rating(df,x,y)

    return pivot_data

"""We hebben het verschil tussen cosine en euclid similarity getest"""
# def cosine_angle(matrix, id1, id2):
#     """Compute euclid distance between two rows."""
#     if id1 == id2:
#         return 1
#     # only take the features that have values for both id1 and id2
#     selected_features = matrix.loc[id1].notna() & matrix.loc[id2].notna()
#
#     # if no matching features, return NaN
#     if not selected_features.any():
#         return 0.0
#
#     # get the features from the matrix
#     features1 = matrix.loc[id1][selected_features]
#     features2 = matrix.loc[id2][selected_features]
#     top=0
#     squared1=0
#     squared2=0
#
#     # compute the distances for the features
#     distances = features1 * features2
#     for x in distances:
#         top = top + x
#     for x in features1:
#         squared1 = squared1 + (x*x)
#     for x in features2:
#         squared2 = squared2 + (x*x)
#
#     bottom = np.sqrt(squared1) * np.sqrt(squared2)
#     if bottom == 0:
#         return 0.0
#
#     res = top/bottom
#     return res


# def create_similarity_matrix_cosine(matrix):
#     """ creates the similarity matrix based on cosine similarity """
#     similarity_matrix = pd.DataFrame(0, index=matrix.index, columns=matrix.index, dtype=float)
#     for x in similarity_matrix:
#         for y in similarity_matrix.index:
#             similarity_matrix[x][y] = cosine_angle(matrix,x,y)
#
#     return similarity_matrix

def mean(frame, group_index, avg_index):
    return frame.groupby(group_index)[avg_index].mean()


def select_neighborhood(similarity_matrix, utility_matrix, target_user, target_business):
    """selects all items with similarity > 0"""
    seen = []
    a = {}

    for i in utility_matrix.index:
        if pd.isnull(utility_matrix[target_user][i]):
            pass
        else:
            seen.append(i)


    for x in similarity_matrix:
        if similarity_matrix[target_business][x] > 0 and similarity_matrix[target_business][x] < 1 and x in seen:
            a.update({x:similarity_matrix[target_business][x]})
    res = pd.Series(a)
    return res


def weighted_mean(neighborhood, utility_matrix, business_id):
    top = 0
    bottom = 0
    res=0
    test = []

    if neighborhood.empty:
        return 0.0

    for x,y in neighborhood.iteritems():
        top = top + (utility_matrix[business_id][x] * y)
        bottom = bottom + y

    if bottom == 0:
        return 0.0

    res = top/bottom
    return res

def euclid_distance(matrix, id1, id2):
    """Compute euclid distance between two rows."""
    # only take the features that have values for both id1 and id2
    selected_features = matrix.loc[id1].notna() & matrix.loc[id2].notna()

    # if no matching features, return NaN
    if not selected_features.any():
        return np.nan

    # get the features from the matrix
    features1 = matrix.loc[id1][selected_features]
    features2 = matrix.loc[id2][selected_features]

    # compute the distances for the features
    distances = features1 - features2
    squared = 0
    # return the absolute sum
    for x in distances:
        squared = squared + x*x
        res = np.sqrt(squared)
    return res

def euclid_similarity(matrix, id1, id2):
    """Compute euclid similarity between two rows."""
    # compute distance
    distance = euclid_distance(matrix, id1, id2)

    # if no distance could be computed (no shared features) return a similarity of 0
    if distance is np.nan:
        return 0

    # else return similarity
    return 1 / (1 + distance)
# TODO


def create_similarity_matrix_euclid(matrix):
    similarity_matrix = pd.DataFrame(0, index=matrix.index, columns=matrix.index, dtype=float)
    for x in similarity_matrix:
        for y in similarity_matrix.index:
            similarity_matrix[x][y] = euclid_similarity(matrix,x,y)

    return similarity_matrix

def mean_center_rows(matrix):
    matrix1 = pd.DataFrame(matrix)
    new_matrix = pd.DataFrame(0, index=matrix.index, columns=matrix.columns,dtype=float)

    avg = matrix1.mean(axis=1)


    for x in new_matrix.index:
        for y in new_matrix:
            new_matrix[y][x] = matrix1[y][x] - avg[x]

    return new_matrix

def recommend(user_id=None, business_id=None, city=None, n=10):
    """
    Returns n recommendations as a list of dicts.
    Optionally takes in a user_id, business_id and/or city.
    A recommendation is a dictionary in the form of:
    {
    business_id:str
    stars:str
    name:str
    city:str
    adress:str
    }

    """


    user_ids = []
    user_ids2 = []
    names = []
    business_ids = []
    business_ids2 = []
    stars = []
    review_ids = []
    addresses = []
    # Puts all userids, businessids, stars, names and addresses in seperate lists.
    for cities, reviews in REVIEWS.items():
        if cities == city:
            for review in reviews:
                user_ids.append(review['user_id'])
                business_ids.append(review['business_id'])
                review_ids.append(review['review_id'])
                stars.append(review['stars'])
    for cities, users in USERS.items():
        if cities == city:
            for user in users:
                names.append(user['name'])
                user_ids2.append(user['user_id'])
    for cities, businesses in BUSINESSES.items():
        if cities == city:
            for business in businesses:
                business_ids2.append(business['business_id'])
                addresses.append(business['address'])
    data = {'user_id':user_ids,'business_id':business_ids,'stars':stars,'review_id':review_ids}

    names_dict = dict(zip(user_ids2,names))
    business_dict = dict(zip(business_ids2,addresses))

    df = pd.DataFrame(data)

    utility_matrix = pivot_ratings(df)
    utility_matrix = utility_matrix.T
    centered_utility_matrix = mean_center_rows(utility_matrix)
    similarity = create_similarity_matrix_euclid(centered_utility_matrix)

    if not city:
        city = random.choice(CITIES)
        return random.sample(BUSINESSES[city], n)
    """Creates a series consisting of the top n predicted ratings for all businesses"""
    if business_id is None:
        res = []
        rec = {}
        i = 0
        for x in utility_matrix:
            n1 = select_neighborhood(similarity, utility_matrix, x, user_id)
            p1 = weighted_mean(n1, utility_matrix, x)
            if p1 > 0:
                res.append({'business_id':x,'stars':p1,'name':names_dict[user_id],'city':city,'adress':business_dict[x]})
                rec.update({x:p1})
                i += 1

        a = rec.copy()
        for x in a:
            if pd.isnull(utility_matrix[x][user_id]):
                pass
            else:
        	    rec.pop(x)
        rec = pd.Series(rec)
        rec = rec.sort_values(ascending=False)
        rec = rec.head(n)
        return rec

    else:
        n = select_neighborhood(similarity, utility_matrix, business_id, user_id)
        p = weighted_mean(n,utility_matrix,business_id)

        res = [{'business_id':business_id,'stars':p,'name':names_dict[user_id],'city':city,'adress':business_dict[business_id]}]
        rec = {x:p1}
        return rec
print(recommend("y4a_7xpbvRvCGwMNY4iRhQ",None,'ambridge'))
