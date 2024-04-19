import yaml

def parse_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

def find_nevents(yaml_data, path=''):
    results = []
    if isinstance(yaml_data, dict):
        for key, value in yaml_data.items():
            new_path = f"{path}.{key}" if path else key
            if key == 'nevents':
                if isinstance(value, int):
                    results.append((new_path, value))
                else:
                    print(f"'{new_path}' value is not an integer.")
            elif isinstance(value, (dict, list)):
                results.extend(find_nevents(value, path=new_path))
    elif isinstance(yaml_data, list):
        for i, item in enumerate(yaml_data):
            new_path = f"{path}[{i}]"
            results.extend(find_nevents(item, path=new_path))

    return results

if __name__ == "__main__":
    file_path = '/work/ehettwer/KingMaker/sample_database/datasets.yaml'  # Replace with the path to your YAML file
    yaml_data = parse_yaml_file(file_path)

    if yaml_data is not None:
        nevents_values = find_nevents(yaml_data)
        if nevents_values:
            print("Found 'nevents' instances:")
            for path, value in nevents_values:
                print(f"{path}: {value}")
        else:
            print("No 'nevents' found in the YAML file.")
