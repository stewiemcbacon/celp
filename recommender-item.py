from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
import pandas as pd
from pandas import Series, DataFrame
import numpy as np

def get_rating(ratings,user_id,business_id):
	if ratings.loc[(ratings['user_id'] == user_id) & (ratings['business_id'] == business_id)]['stars'].any() == False:
		res = np.nan
	else:
		res = float(ratings.loc[(ratings['user_id'] == user_id) & (ratings['business_id'] == business_id),'stars'].values[0])
	return res

def pivot_ratings(df):
	""" takes a rating table as input and computes the utility matrix """
	# get movie and user id's
	business_ids = df['business_id'].unique()
	user_ids = df['user_id'].unique()

	# create empty data frame
	pivot_data = pd.DataFrame(np.nan, columns=user_ids, index=business_ids, dtype=float)

	# use the function get_rating to fill the matrix
	for x in pivot_data:
		for y in pivot_data.index:
			pivot_data[x][y] = get_rating(df,x,y)

	return pivot_data


def cosine_angle(matrix, id1, id2):
	"""Compute euclid distance between two rows."""
	if id1 == id2:
		return 1
	# only take the features that have values for both id1 and id2
	selected_features = matrix.loc[id1].notna() & matrix.loc[id2].notna()

	# if no matching features, return NaN
	if not selected_features.any():
		return 0.0

	# get the features from the matrix
	features1 = matrix.loc[id1][selected_features]
	features2 = matrix.loc[id2][selected_features]
	top=0
	squared1=0
	squared2=0

	# compute the distances for the features
	distances = features1 * features2
	for x in distances:
		top = top + x
	for x in features1:
		squared1 = squared1 + (x*x)
	for x in features2:
		squared2 = squared2 + (x*x)

	bottom = np.sqrt(squared1) * np.sqrt(squared2)
	if bottom == 0:
		return 0.0

	res = top/bottom
	return res


def create_similarity_matrix_cosine(matrix):
	""" creates the similarity matrix based on cosine similarity """
	similarity_matrix = pd.DataFrame(0, index=matrix.index, columns=matrix.index, dtype=float)
	for x in similarity_matrix:
		for y in similarity_matrix.index:
			similarity_matrix[x][y] = cosine_angle(matrix,x,y)

	return similarity_matrix

def mean(frame, group_index, avg_index):
	return frame.groupby(group_index)[avg_index].mean()


def mean_center_columns(matrix):
	matrix1 = pd.DataFrame(matrix)
	new_matrix = pd.DataFrame(0, index=matrix.index, columns=matrix.columns,dtype=float)

	for x in new_matrix:

		for y in new_matrix.index:
			new_matrix[x][y] = matrix1[x][y] - matrix1[x].mean()

	return new_matrix

def select_neighborhood(similarity_matrix, utility_matrix, target_user, target_film):
	"""selects all items with similarity > 0"""
	seen = []
	a = {}

	for i in utility_matrix.index:
		if pd.isnull(utility_matrix[target_user][i]):
			pass
		else:
			seen.append(i)


	for x in similarity_matrix:
		if similarity_matrix[target_film][x] > 0 and similarity_matrix[target_film][x] < 1 and x in seen:
			a.update({x:similarity_matrix[target_film][x]})

	res = pd.Series(a)


	return res


def weighted_mean(neighborhood, utility_matrix, user_id):
	top = 0
	bottom = 0
	res=0
	test = []

	if neighborhood.empty:

		return 0.0

	for x,y in neighborhood.iteritems():
		top = top + (utility_matrix[user_id][x] * y)
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
	user_id = []
	business_id = []
	stars = []
	review_id = []
	# Puts all userids, businessids and stars in seperate lists.
	for city, reviews in REVIEWS.items():
		for review in reviews:
			user_id.append(review['user_id'])
			business_id.append(review['business_id'])
			review_id.append(review['review_id'])
			stars.append(review['stars'])

	data = {'user_id':user_id,'business_id':business_id,'stars':stars,'review_id':review_id}
	df = pd.DataFrame(data)
	utility_matrix = pivot_ratings(df)
	similarity = create_similarity_matrix_cosine(utility_matrix)

	centered_utility_matrix = mean_center_columns(utility_matrix)
	similarity = create_similarity_matrix_cosine(centered_utility_matrix)
	a = {}
	for x in utility_matrix.index:
		n1 = select_neighborhood(similarity, utility_matrix, "Y8AacNK1oloBnkTQ3CLlEA", x)
		p1 = weighted_mean(n1,utility_matrix,"Y8AacNK1oloBnkTQ3CLlEA")
	if p1 > 0:
		a.update({x:p1})

	recommended = pd.Series(a)
	if not city:
		city = random.choice(CITIES)
	return random.sample(BUSINESSES[city], n)

user_id = []
business_id = []
stars = []
review_id = []
# Puts all userids, businessids and stars in seperate lists.
for city, reviews in REVIEWS.items():
	if city == 'ambridge':
		for review in reviews:
			user_id.append(review['user_id'])
			business_id.append(review['business_id'])
			review_id.append(review['review_id'])
			stars.append(review['stars'])


data = {'user_id':user_id,'business_id':business_id,'stars':stars,'review_id':review_id}

df = pd.DataFrame(data)
# print(df)
utility_matrix = pivot_ratings(df)
# similarity = create_similarity_matrix_euclid(utility_matrix)

centered_utility_matrix = mean_center_columns(utility_matrix)
similarity = create_similarity_matrix_euclid(centered_utility_matrix)
a = {}
for x in utility_matrix.index:
	n1 = select_neighborhood(similarity, utility_matrix, "21D8GYYY-NptvXhBb9x08Q", x)
	p1 = weighted_mean(n1,utility_matrix,"21D8GYYY-NptvXhBb9x08Q")
	if p1 > 0:
		a.update({x:p1})
b = a.copy()
for x in b:
	if pd.isnull(utility_matrix["21D8GYYY-NptvXhBb9x08Q"][x]):
		pass
	else:
		a.pop(x)

recommended = pd.Series(a)
print(recommended)
