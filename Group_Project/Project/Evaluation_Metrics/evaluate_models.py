import pandas as pd

from summ_eval.data_stats_metric import DataStatsMetric
from summ_eval.rouge_we_metric   import RougeWeMetric
from meteor_metric               import MeteorMetric
# from summ_eval.meteor_metric   import MeteorMetric

DATA_PATH = "~/data"
MODEL_OUTPUT_FILE_PATH = DATA_PATH + "/model_outputs/M22/paired/outputs_cnndm.aligned.paired.jsonl"

# Model output data needs to have the following format:
# {"reference": "This is summary 1", "filepath": "cnndm/cnn/stories/469c6ac05092ca5997728c9dfc19f9ab6b936e40.story"}
# Then with pair_data the summaries can be paired with the original stories (via filepath)

# Java can't find the meteor-1.5.jar if its path contains spaces!

def main():
    
    # Read data into dataframes
    model_output_data = pd.read_json(MODEL_OUTPUT_FILE_PATH, lines = True)

    # Summaries, References & Original Texts of the Test instances
    summaries      = model_output_data["reference"].to_list()  # Summaries generated by our model
    original_texts = model_output_data["text"].to_list()       # Full text retrieved by pairing data with original texts
    references     = model_output_data["references"].to_list() # Reference summaries of this original text also retrieved by pairing data

    # Placeholders
    summaries      = ["This is the summary for text 1 right", "This is the summary for text 2"]
    references     = [["This is the reference summary for text 1", "This is another reference summary for text 1"], ["This is the reference summary for text 2"]]
    original_texts = ["This is the original text 1", "This is the original text 2"]

    # Currently not working
    evaluate_model(summaries, references, original_texts)


def evaluate_model(summaries, references, original_texts):

    ## Initialize output
    metric_outputs = []

    ## Compare to original texts
    print("Starting with DataStatsMetric")
    data_stats_metric = DataStatsMetric()
    data_stats_output = data_stats_metric.evaluate_batch(summaries, original_texts)
    metric_outputs = metric_outputs + [["DataStats", data_stats_output]]
    # Counter({'summary_length': 60.03959965187119, 'compression': 14.255931705255252, 'density': 3.8189903896763187, 'coverage': 0.8828704948022259, 'percentage_novel_3-gram': 0.7125468921646818, 'percentage_novel_2-gram': 0.5177580908057711, 'percentage_novel_1-gram': 0.159994130255124, 'percentage_repeated_1-gram_in_summ': 0.15454390142190308, 'percentage_repeated_2-gram_in_summ': 0.014050349207946686, 'percentage_repeated_3-gram_in_summ': 0.0025787665845277597})
    
    ## Compare to references
    print("Starting with MeteorMetric")
    meteor_metric = MeteorMetric()
    meteor_output = meteor_metric.evaluate_batch(summaries, references)
    metric_outputs = metric_outputs + [["Meteor", meteor_output]]
    # {'meteor': 0.4392209049699441}

    print("Starting with RougeWeMetric")
    rouge_we_metric = RougeWeMetric()
    rouge_we_output = rouge_we_metric.evaluate_batch(summaries, references)
    metric_outputs = metric_outputs + [["RougeWeMetric", rouge_we_output]]
    # Counter({'rouge_we_3_p': 0.010823020654666329, 'rouge_we_3_f': 0.009876746518628769, 'rouge_we_3_r': 0.009445517249164976})

    ## Write output to file
    metric_outputs_df = pd.DataFrame(data = metric_outputs, columns = ["Metric", "Output"])
    metric_outputs_df.to_csv("Metrics_output.csv")


if __name__ == "__main__":
    main()