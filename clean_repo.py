""" """
from typing import List, Tuple
from dataclasses import dataclass, field
import os
import re
import pathlib
import shutil


@dataclass
class DataSanitizer:
    """ """
    folder_regex_list: List[Tuple[pathlib.Path, str]] = field(default_factory = list)
    folder_list: List[str] = field(default_factory = list)
    ignore_case: bool = False
    
    def add_item(self, file_path: pathlib.Path, regex: str) -> None:
        """ """
        self.folder_regex_list.append(tuple((file_path, regex)))
        
    def add_folder(self, folder: pathlib.Path) -> None:
        """ """
        self.folder_list.append(folder)
    
    def sanitize(self) -> None:
        """ """
        print("=============================================================")
        print("Removing intermediary and output files from the repository...")
        print("=============================================================")
        for item in self.folder_regex_list:
            folder, regex_str = item[0], item[1]
            if self.ignore_case:
                regex_pattern = re.compile(regex_str, re.IGNORECASE)
            else:
                regex_pattern = re.compile(regex_str)
            files = os.listdir(folder)
            print("\n")
            print(f"Removing files matching pattern {regex_str} from folder {folder}")
            print("-------------------------------------------------------------------------------")
            for file in files:
                if regex_pattern.match(file):
                    file_path = os.path.join(folder, file)
                    try:
                        #os.remove(file_path)
                        print(f"File removed: {file_path}")
                    except OSError:
                        print(f"Could not remove file: {file_path}")

        print("\n")
        print("=============================================================")
        print("Removing entire folder tree structures from the repository...")
        print("=============================================================")
        for folder in self.folder_list:
            try:
                #shutil.rmtree(folder)
                print(f"Folder {folder} removed")
            except FileNotFoundError:
                print(f"Folder {folder} not removed. NOT FOUND")



if __name__ == "__main__":
    sanitizer = DataSanitizer()
    sanitizer.add_item(pathlib.Path('inputs/reemission'), r'reemission_inputs_[a-zA-Z]*_[a-zA-Z]*_[a-zA-Z]*\.json')
    sanitizer.add_item(pathlib.Path('outputs/reemission'), r'outputs_[a-zA-Z]*_[a-zA-Z]*_[a-zA-Z]*\.(?:xlsx?|json)')
    sanitizer.add_item(pathlib.Path('outputs/reemission/combined'), r'combined_outputs\.(?:xlsx?|csv)')
    sanitizer.add_item(pathlib.Path('outputs/pywr_hp'), r'.*\.(csv|xlsx)$')
    sanitizer.add_item(pathlib.Path('figures/ghg_visualisation'), r'.*\.(svg|png|jpe?g)$')
    sanitizer.add_item(pathlib.Path('figures/ifc_pywr_power_comparison'), r'.*\.(svg|png|jpe?g)$')
    sanitizer.add_item(pathlib.Path('figures/maps'), r'.*\.(svg|png|jpe?g)$')
    sanitizer.add_item(pathlib.Path('figures/data_exploration'), r'.*\.(svg|png|jpe?g)$')
    sanitizer.add_item(pathlib.Path('figures/model_explanation'), r'.*\.(svg|png|jpe?g)$')
    sanitizer.add_item(pathlib.Path('intermediate'), r'.*\.(csv|xlsx)$')
    sanitizer.add_item(pathlib.Path(''), r'.*\.(csv|pickle)$')
    sanitizer.add_folder(pathlib.Path('outputs/model_explanations'))
    sanitizer.add_folder(pathlib.Path('saved_models'))
    sanitizer.add_folder(pathlib.Path("figures/clustering"))
    sanitizer.sanitize()
