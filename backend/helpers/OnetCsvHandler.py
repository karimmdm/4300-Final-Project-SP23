import pandas as pd
from collections import defaultdict
import os

class OnetCsvHandler(object):
    
    def __init__(self) -> None:
        self._onet_dictionary = {}

    def data(self) -> dict: 
        return self._onet_dictionary
    
    def update_table_with_csv(self, job_attribute, csv_dir) -> None:
        # skills: Importance,Level,Job Zone,Code,Occupation
        # knowledge: Importance,Level,Job Zone,Code,Occupation
        file_name = csv_dir.split("/")[-1].split(".")[0]
        df = pd.read_csv(csv_dir)
        df.reset_index()
        for index, row in df.iterrows():
            code = row["Code"]
            if code in self._onet_dictionary:
                assert self._onet_dictionary[code]["occupation"] == row["Occupation"]
            else:  
                self._onet_dictionary[code] = {}
                self._onet_dictionary[code]["occupation"] = row["Occupation"]
                
            if job_attribute not in self._onet_dictionary[code]:
                self._onet_dictionary[code][job_attribute] = []
            
            self._onet_dictionary[code][job_attribute].append((file_name, row.get("Importance", None)))

    def bulk_update(self, raw_data_dir) -> None:
        if not os.path.isdir(raw_data_dir):
            print(f"missing {raw_data_dir} directory")
            exit
        else:
            for subdir in os.scandir("raw_data"):
                for file_name in os.scandir(subdir):
                    if not file_name.is_file():
                        continue
                    self.update_table_with_csv(subdir.name, file_name.path)
    
if __name__ == "__main__":
    csv_handler = OnetCsvHandler()
    csv_handler.bulk_update("raw_data")
    print(csv_handler.data())

    
