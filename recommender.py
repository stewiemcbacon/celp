from data import CITIES, BUSINESSES, USERS, REVIEWS, TIPS, CHECKINS
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import random

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


	if not city:
		city = random.choice(CITIES)
	return random.sample(BUSINESSES[city], n)

# for key, value in USERS.items():
userid = []
business_id = []
stars = []



for city, reviews in REVIEWS.items():
	for review in reviews:
		userid.append(review['user_id'])
		business_id.append(review['business_id'])
		stars.append(review['stars'])

data = {'userid':userid,'business_id':business_id,'stars':stars}
df = pd.DataFrame(data)


print(df)
