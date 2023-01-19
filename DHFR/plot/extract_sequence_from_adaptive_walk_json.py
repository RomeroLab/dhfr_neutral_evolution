import json

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("json_filename",
                    help="input adaptive walk json filename")
    parser.add_argument("-f", "--json_field",
                    default="start_protein",
                    required=False,
                    help="field from json to print")

    args = parser.parse_args()
    json_filename = args.json_filename
    json_field = args.json_field

    with open(json_filename) as fh_in:
        data = json.load(fh_in)
        print(data[json_field])

