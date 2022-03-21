import argparse

from experiment_design import *


if __name__ == '__main__':

    my_parser = argparse.ArgumentParser(description='Arguments for the model.')

    my_parser.add_argument(
        '-s',
        metavar='--sample',
        type=int,
        default=None,
        help='Informs the sample size for the experiment.'
        )
    
    my_parser.add_argument(
        '-r',
        metavar='--replicate',
        type=int,
        default=None,
        help='Informs the replicate number for the experiment.'
        )

    my_parser.add_argument(
        '-f',
        metavar='--folder',
        type=str,
        default='./benchmark/pdf_files/',
        help='Path to a folder containing the benchmark PDF files.'
        )

    my_parser.add_argument(
        '-b',
        metavar='--benchmark',
        type=str,
        default='./benchmark/reference_dataset/dataset_benchmark.xlsx',
        help='Path to benchmark dataset (i.e., the reference dataset).'
        )

    my_parser.add_argument(
        '-o',
        metavar='--output',
        type=str,
        default='./output/',
        help='Path to the folder where experiments output should be saved.'
        )

    my_parser.add_argument(
        '-m',
        metavar='--method',
        type=str,
        default=None,
        help='Method for table extraction. Not available for user.'
        )


    args = my_parser.parse_args()


    folder_path = args.f
    benchmark_dataset_path = args.b
    max_sample_size = args.s
    replicate = args.r
    csv_path = args.o
    method = args.m
    if method is None:
        method = ['camelot', 'tabula']
    else:
        method = [method]


    experiment = Experiment(
        folder_path = folder_path,
        benchmark_dataset_path = benchmark_dataset_path,
        max_sample_size = max_sample_size,
        replicate = replicate,
        method = method
    )

    experiment_output = experiment.run(
        csv_path = csv_path,
        save_copy = True
        )
