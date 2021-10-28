import json

def json_loader(dict_file_dir):
    with open(dict_file_dir, 'r') as data_file:
        return json.load(data_file)

dict = json_loader('BSD500_300.json')
print('done')