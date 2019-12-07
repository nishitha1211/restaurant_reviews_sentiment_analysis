from elasticsearch import Elasticsearch


def elastic_search():
    return Elasticsearch([{'host': 'localhost', 'port': 9200}])


def send_tag_to_es(words):
    es = elastic_search()
    if not es.indices.exists(index="tagcloud"):
        datatype = {
            "mappings": {
                "request-info": {
                    "properties": {
                        "word": {
                            "type": "keyword"
                        }
                    }
                }
            }
        }
        es.indices.create(index="tagcloud", body=datatype)
    for i in words:
        es.index(index="tagcloud", doc_type="request-info", body={"word": str(i)})


def send_loc_to_es(loc):
    es = elastic_search()
    if not es.indices.exists(index="map"):
        datatype = {
            "mappings": {
                "request-info": {
                    "properties": {
                        "location": {
                            "type": "geo_point"
                        }
                    }
                }
            }
        }
        es.indices.create(index="map", body=datatype)
    for key in loc:
        es.index(index="map", doc_type="request-info", body={"location": {"lat": loc['latitude'], "lon": loc['longitude']}})
