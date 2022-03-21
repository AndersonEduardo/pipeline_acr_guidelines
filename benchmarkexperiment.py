import unidecode
import pandas as pd
from parameters import *
from datasetbuilder import *


class BenchmarkExperiment:

    def __init__(self, reference_dataset_path:str, test_dataset:pd.DataFrame, ordering:list):

        self.benchmark_dataset = self._setup_benchmark_dataset(reference_dataset_path, ordering) 
        self.test_dataset = self._setup_test_dataset(test_dataset.copy(), ordering)


    def _setup_benchmark_dataset(self, file_path, ordering):

        df = pd.read_excel(file_path)
        df.columns = [x.replace(' ', '_').rstrip().lower() for x in df.columns]
        df = DatasetBuilder._get_appropriateness_category(df)

        rows_ordering = [os.path.basename(x).split('.')[0] for x in ordering]

        df = df[df['category'].apply(lambda x: x in rows_ordering)]
        df = df[COLUMNS_ORDERING]
        df = df.reset_index(drop=True)

        return df


    def _setup_test_dataset(self, df_dict, ordering):

        for i, k in enumerate(ordering):

            if i == 0:

                df = df_dict[k].copy()

            else:

                df = df.append(df_dict[k].copy())

        df = df[COLUMNS_ORDERING]
        df = df.reset_index(drop=True)

        return df


    def run(self):

        print('[STATUS] Comparing reference and test datasets...')
        
        output = dict()

        for col in self.benchmark_dataset.columns:

            # .sort_values(ascending=True)\
            benchmark_data = self.benchmark_dataset[col]\
                                .astype('str')\
                                .values.tolist()
            # .sort_values(ascending=True)\
            test_data = self.test_dataset[col]\
                            .astype('str')\
                            .values\
                            .tolist()

            benchmark_data_concat = ''.join([''.join(unidecode.unidecode(str(x)).strip().lower().split()) 
                                             for x in benchmark_data])
            test_data_concat = ''.join([''.join(unidecode.unidecode(str(x)).strip().lower().split()) 
                                        for x in test_data])

            d = Levenshtein.distance(
                benchmark_data_concat, 
                test_data_concat
                )

            print(f'[STATUS]       - Levenshtein distance for "{col}": {d}')

            if output.get(col):

                output[col]['levenshtein'] += d
                output[col]['number_of_characters'] += len(benchmark_data_concat)

            else:

                output[col] = {
                    'levenshtein': d, 
                    'number_of_characters': len(benchmark_data_concat),
                    }


        print('[STATUS] Dataset comparison complete.')

        return output