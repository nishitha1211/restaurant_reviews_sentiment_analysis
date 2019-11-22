# from yelp.client import Client
#
#
# client = Client(api_key)
#
# print(client)

import requests
import json
from http import HTTPStatus

# api_key="sVuxy8O1Mkxdd4JoS_iSI35i6DUZ4Gd5Q3jfHOSH1GmUT27jN699By2USJEB7SR9-ztib6ArB7MRWeVXqajl8-wDY3NollfbKpP8IA0vTqfPUHzBrfPqwatAWQ_XXXYx"

def get(url, params, headers):
    return requests.get(url, params=params, headers=headers)


def json_data(data):
    return json.loads(data)


def get_headers():
    return {'Authorization': 'Bearer %s' % get_api_key()}


def get_api_key():
    return open("key.txt", "r").read()


def get_business_url():
    return "https://api.yelp.com/v3/businesses/search"


def get_review_url(_id):
    return "https://api.yelp.com/v3/businesses/{0}/reviews".format(_id)


def get_business_params(term="seafood", location="New York City"):
    return {'term': term, 'location': location}


def get_business_data():
    response = get(get_business_url(), get_business_params(), get_headers())
    if response.status_code != HTTPStatus.OK:
        return response.status_code, None
    data = json_data(response.text)
    return response.status_code, data


def get_business_list():
    status, body = get_business_data()
    if status != HTTPStatus.OK:
        return None
    business_list = body.get('businesses')
    return business_list


def get_business_id_loc():
    business_id = []
    location = {}
    business_list = get_business_list()
    if business_list is None:
        return None, None
    for business in business_list:
        _id = business.get('id')
        business_id.append(_id)
        coordinates = business.get('coordinates')
        location[_id] = {'latitude': coordinates.get('latitude'), 'longitude': coordinates.get('longitude')}
    return business_id, location


def get_business_review_list():
    id_list, location = get_business_id_loc()
    if id_list is None:
        return None
    for _id in id_list:
        response = get(get_review_url(_id), {}, headers=get_headers())
        print(response.text)


if __name__ == "__main__":
    get_business_review_list()
