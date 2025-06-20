import pandas as pd

def get_class_label(class_name):
    """
    Get the label name for a given class name from the Open Images dataset class descriptions.

    :param class_name: The display name of the class to look up.
    :return: The corresponding label name if found, otherwise None.
    """
    df = pd.read_csv("./prepare_data/files_from_OpenImages/oidv7-class-descriptions-boxable.csv")
    if class_name in df['DisplayName'].values:
        return df['LabelName'][(df['DisplayName'] == class_name)].values[0]
    else:
        return None
