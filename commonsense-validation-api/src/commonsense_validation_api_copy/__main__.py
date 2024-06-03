import click
from commonsense_validation_api.common_evaluations import multi_evals
from commonsense_validation_api.data_processing import load_question_answer_clusters_from_jsonl_file
from commonsense_validation_api.data_processing import load_predictions_from_jsonl_file


@click.group()
def main():
    """Evaluation and data processing for ProtoQA common sense QA dataset"""
    pass


@main.group()
def convert():
    """Functions for loading, converting, and saving data files"""
    pass


@main.command()
@click.option("--target",
              default="data/targets.jsonl",
              type=click.Path())

@click.option("--prediction",
              default="data/annotated_predictions.json",
              type=click.Path())

@click.option(
    "--similarity_function",
    default="fasttext",
    type=click.Choice([
        "wordnet",
        "fasttext"
    ], case_sensitive=False),
)
@click.option(
    "--clustering_algo",
    default="hac",
    type=click.Choice([
        "gmeans",
        "xmeans",
        "hac",
    ], case_sensitive=False),
)
@click.option('--num_sampling_steps', default=30, show_default=True, type=int, required=True)
@click.option('--save_files', default=True, show_default=True, type=bool, required=True)
@click.option('--starting_component', type=int, default=1, help="Different cluster algorithm has different components."
                                                                "xmeans: 7-14 (1)"
                                                                "gmeans: 1-7 (1)"
                                                                "hac: 0.2-0.5 (0.05)")

def evaluate(
        target, prediction,
        similarity_function, clustering_algo,
        num_sampling_steps, save_files,
        starting_component
):
    print("*"*20)
    print("Using {} as similarity function and {} as clustering algorithm".format(similarity_function, clustering_algo))
    
    function_name = get_function_name(clustering_algo, similarity_function)
    targets = load_question_answer_clusters_from_jsonl_file(target) # len(targets)=52
    all_predictions = load_predictions_from_jsonl_file(prediction) #all_predictions['annotated_prediction_data'] & ['model_answers']
    if "annotated_prediction_data" in all_predictions:
        annotated_prediction_answers = all_predictions["annotated_prediction_data"]
        predictions = {k: all_predictions["annotated_prediction_data"][k] for k in targets}
    else:
        annotated_prediction_answers = None
        predictions = {k: all_predictions["model_answers"][k] for k in targets}

    multi_evals(
        eval_type=function_name,
        question_data=targets,
        prediction_data=predictions,
        embedding="/home/qic69/Desktop/CFC/commonsense-validation-api/embeddings/new_embeddings_100.jsonl",
        annotated_prediction_answers=annotated_prediction_answers,
        clustering_algo=clustering_algo,
        num_sampling_steps=num_sampling_steps,
        save_files=save_files,
        n_component=starting_component
    )


def get_function_name(clustering_algo: str, similarity_function: str):  
    return clustering_algo+"_cluster_and_"+similarity_function


