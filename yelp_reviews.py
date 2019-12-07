# from yelp.client import Client
#
#
# client = Client(api_key)
#
# print(client)

import requests
import json

from rr import start_build, predict
from es import send_loc_to_es


def get(url, params, headers):
    return requests.get(url, params=params, headers=headers)


def json_data(data):
    return json.loads(data)


def get_headers():
    return {'Authorization': 'Bearer %s' % get_api_key()}


def get_api_key():
    return open("key.txt", "r").read()


def get_business_name_url(_id):
    return "https://api.yelp.com/v3/businesses/{0}".format(_id)


def get_business_url():
    return "https://api.yelp.com/v3/businesses/search"


def get_review_url(_id):
    return "https://api.yelp.com/v3/businesses/{0}/reviews".format(_id)


def get_business_params(term="seafood", location="New York City"):
    return {'term': term, 'location': location}


def get_business_data():
    response = get(get_business_url(), get_business_params(), get_headers())
    if response.status_code != 200:
        return response.status_code, None
    data = json_data(response.text)
    return response.status_code, data


def get_business_name(_id):
    response = get(get_business_name_url(_id), {}, get_headers())
    if response.status_code != 200:
        return response.status_code, None
    data = json_data(response.text)
    return response.status_code, data.get('name')


def get_business_list():
    status, body = get_business_data()
    if status != 200:
        return None
    business_list = body.get('businesses')
    return business_list


def get_business_id_loc():
    business_id = []
    business_names = {}
    location = {}
    business_list = get_business_list()
    if business_list is None:
        return None, None
    for business in business_list:
        _id = business.get('id')
        status, name = get_business_name(_id)
        if status != 200 and name is not None:
            continue
        business_id.append(_id)
        business_names[_id] = name
        coordinates = business.get('coordinates')
        location[_id] = {'latitude': coordinates.get('latitude'), 'longitude': coordinates.get('longitude')}
    return business_id, location, business_names


def get_review_list(data):
    reviews = []
    ratings = []
    for review in data.get('reviews'):
        reviews.append(review.get('text'))
        ratings.append(review.get('rating'))
    return reviews, ratings


def get_business_review_list():
    reviews = {}
    id_list, location, names = get_business_id_loc()
    if id_list is None:
        return None, None
    for _id in id_list:
        response = get(get_review_url(_id), {}, headers=get_headers())
        data = json_data(response.text)
        reviews[_id], _ = get_review_list(data)
    return reviews, location, names


def predict_reviews():
    predicted = {}
    reviews, location, names = get_business_review_list()
    for key in reviews.keys():
        predicted[key] = avg(predict(reviews[key]))
        if predicted[key] >= 0.5:
            print("{0} - {1}".format(names[key], "Good"))
        else:
            print("{0} - {1}".format(names[key], "Not so Good"))
    return predicted, location


def avg(data):
    count = 0
    for i in data:
        count += i
    return count / (len(data) * 1.0)


def data_mappings():
    predicted, location = predict_reviews()
    for key in predicted.keys():
        if predicted[key] < 0.5:
            continue
        send_loc_to_es(location[key])


if __name__ == "__main__":
    start_build()
    data_mappings()
