import pandas as pd

from parameters import *
from datasetbuilder import *
from benchmarkexperiment import *


class Experiment:
    
    def __init__(self, folder_path, benchmark_dataset_path, \
                 max_sample_size, replicate, method=['camelot', 'tabula']):

        self.benchmark_dataset_path = benchmark_dataset_path
        self.databuilder = DatasetBuilder(folder_path)
        self.table_extraction_methods = method
        self.max_sample_size = max_sample_size
        self.replicate = replicate
        self.file_paths = os.listdir(folder_path)


    def run(self, csv_path:str = None, return_df:bool = True, save_copy:bool = True):

        start_time_experiment = time.localtime()
        formated_start_time_experiment = time.strftime("%Y-%m-%d %H:%M:%S", start_time_experiment)
        print(f'\n[STATUS] ### Running experiment. Start time: {formated_start_time_experiment} ###\n')

        output = pd.DataFrame()

        for table_extraction_method in self.table_extraction_methods:

            for sample_size in range(1, self.max_sample_size + 1):

                for r in range(1, self.replicate + 1):

                    print(
            f'[STATUS] Running for method "{table_extraction_method}", sample size "{sample_size}" and replicate "{r}".'
                    )

                    # sampled_paths = np.random.choice(file_paths, size=sample_size, replace=False)
                    sampled_files = np.random.choice(self.file_paths, size=sample_size, replace=False)

                    for sampled_file in sampled_files:

                        start_time = time.time()
                    
                        dataset, number_of_tables = self.databuilder.run(
                            # file_name = sampled_files,
                            file_name = [sampled_file],
                            method = table_extraction_method
                            )

                        end_time = time.time()
                        time_lapse = end_time - start_time
                        time_lapse = round(time_lapse, 5)
                        print(f'[STATUS] Data extracted in: {time_lapse} seconds')

                        benchmark = BenchmarkExperiment(
                            reference_dataset_path = self.benchmark_dataset_path,
                            test_dataset = dataset,
                            # ordering = sampled_files
                            ordering = [sampled_file]
                            )

                        if save_copy is True:

                            if csv_path is not None:

                                path_for_copy = os.path.join(
                                    csv_path,
                                    f'test_dataset_{table_extraction_method}S{sample_size}R{r}.csv'
                                    )

                            else:

                                path_for_copy = os.path.join(
                                    f'./test_dataset_{table_extraction_method}S{sample_size}R{r}.csv'
                                    )

                            benchmark.test_dataset.to_csv(path_for_copy, sep='$')

                            print(f'[STATUS] A copy of the test dataset were saved at: {path_for_copy}')

                        print(f'[STATUS] Benchmark dataset shape: {benchmark.benchmark_dataset.shape}')
                        print(f'[STATUS] Test dataset shape: {benchmark.test_dataset.shape}')


                        benchmark_output = benchmark.run()

                        print('\n-----------------------------------------------------------------\n')

                        for k in benchmark_output.keys():

                            for metric in EXPERIMENT_METRICS:

                                if metric == 'levenshtein':

                                    value = benchmark_output[k]['levenshtein']

                                elif metric == 'number_of_characters':

                                    value = benchmark_output[k]['number_of_characters']
                                
                                elif metric == 'percentual_error':

                                    value = (
                                        benchmark_output[k]['levenshtein'] / benchmark_output[k]['number_of_characters']
                                            ) * 100

                                elif metric == 'time':

                                    value = time_lapse

                                else:

                                    raise ValueError(f'Incosistence found in `metric`, for the value {metric}.')


                                output = output.append(
                                    {
                                        'method': table_extraction_method,
                                        'sample_size': sample_size,
                                        'replicate': r,
                                        # 'sampled_files': (sampled_files 
                                        #                 if len(sampled_files) == 1 
                                        #                 else [x.replace(',', ' ') + '|' 
                                        #                         for x in sampled_files]),
                                        'sampled_file': sampled_file,
                                        'feature': k,
                                        'metric': metric,
                                        'value': value
                                    },
                                    ignore_index=True
                                )

                        output = output.append(
                            {
                                'method': table_extraction_method,
                                'sample_size': sample_size,
                                'replicate': r,
                                'sampled_file': sampled_file,
                                'feature': 'number_of_tables',
                                'metric': 'number_of_tables',
                                'value': number_of_tables
                            },
                            ignore_index=True
                        )

                        if csv_path:
            
                            output.to_csv(os.path.join(csv_path, 'experiment_output.csv'), index=False) #, mode='a')

        # output = output[['method', 'sample_size', 'replicate', 'sampled_files', 'feature', 'metric', 'value']]
        output = output[['method', 'sample_size', 'replicate', 'sampled_file', 'feature', 'metric', 'value']]

        end_time_experiment = time.localtime()
        formated_end_time_experiment = time.strftime("%Y-%m-%d %H:%M:%S", end_time_experiment)
        print(f'[STATUS] ### Experiment finished succefully! End time: {formated_end_time_experiment} ###')

        if csv_path:
    
            print(f'[STATUS] Output file saved at {os.path.join(csv_path, "experiment_output.csv")}.')

            if return_df is True:

                return output

        else:

            return output
