def add_store_key_count(store,key,value):
    if not key in store:
        store[key]= value
        
def get_most_freq(store,key):
    if key in store:
        return store[key]
    