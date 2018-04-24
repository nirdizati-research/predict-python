import json


def label_task(df):
    """Calculates the distribution of labels in the data frame

    :return Dict of string and int {'label1': label1_count, 'label2': label2_count}
    """
    # Stupid but it works
    # True must be turned into 'true'
    json_value = df.label.value_counts().to_json()
    return json.loads(json_value)
