import os
# import easyocr
import pickle
import time

import camelot
import cv2
import Levenshtein
import numpy as np
import pandas as pd
import PyPDF2
import pytesseract
import tabula
import unidecode
from pdf2image import convert_from_path

from parameters import *

# pytesseract setup (apenas para Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


class DatasetBuilder:
    

    def __init__(self, folder:str):

        self.folder = folder
        self.dpi = 600
        self.first_page = None
        self.last_page = None
        self.grayscale = True


    def _get_unique_variants(self, variant_list):
        
        # output = list()
        idx_list = list()

        for i in range(len(variant_list)):

            for j in range(i+1, len(variant_list)):

                # distancia de Levenshtein para o texto como um todo
                d1 = Levenshtein.distance(
                    variant_list[i].replace('\n', ' ').strip().lower(),
                    variant_list[j].replace('\n', ' ').strip().lower()
                )

                # distancia de Levenshtein para detectar variants com mesmo numero
                # (por exemplo: Variant 3 e Variant 3, em dois textos diferentes da lista)
                d2 = Levenshtein.distance(
                    variant_list[i].split(':')[0].strip().replace('  ', ' ').lower(),
                    variant_list[j].split(':')[0].strip().replace('  ', ' ').lower()
                    )

                if (d1 < LEVENSHTEIN_THRESHOLD) or (d2 == 0):
                    idx_list.append(j)

        return [variant_list[i] for i in range(len(variant_list)) if i not in idx_list]


    def _ensure_begining_pattern(self, variant_list):
        
        classifier = pickle.load(open(NAIVE_BAYES_CLASSIFIER_PATH, 'rb'))
        tfidf = pickle.load(open(TFIDF_PATH, 'rb'))

        pred_list = list()
        output = list()

        for variant_text in variant_list:

            text = ' '.join(variant_text.split()[:SAMPLE_SIZE_VARIANT_TOLKENS])
            text = tfidf.transform([text])

            first_part = variant_text.replace('\n', ' ').replace('  ', ' ').split()[0].rstrip().lower()
            second_part = variant_text.replace('\n', ' ').replace('  ', ' ').split()[1].strip()
            second_part = ''.join([x for x in second_part if x.isdigit()])

            pred = classifier.predict(text)
            pred_list.append(pred[0])

            # predicao correta de 1
            if (pred == 1) and (variant_text.lower().replace('\n', ' ').replace('  ', ' ').rstrip().startswith('variant')):

                if isinstance(first_part, str) and second_part.isdigit():
    
                    output.append(variant_text)
            
            # predicao correta de 0
            elif (pred == 0) and not (variant_text.lower().replace('\n', ' ').replace('  ', ' ').rstrip().startswith('variant')):

                pass

            # predicao incorreta de 1
            elif (pred == 1) and not (variant_text.lower().replace('\n', ' ').replace('  ', ' ').rstrip().startswith('variant')):

                pass

            # predicao incorreta de 0
            elif (pred == 0) and (variant_text.lower().replace('\n', ' ').replace('  ', ' ').rstrip().startswith('variant')):

                if isinstance(first_part, str) and second_part.isdigit():
    
                    output.append(variant_text)

        return output


    # def _ensure_begining_pattern(self, variant_list):

    #     # idx_list = list()
    #     output = list()

    #     for i in range(len(variant_list)):

    #         prefix_text = variant_list[i].split(':')[0].split()
    #         first_part = prefix_text[0].strip().lower()
    #         second_part = prefix_text[1].strip()

    #         if  first_part == BEGINING_VARIANT_PATTERN and second_part.isdigit():

    #                 output.append(variant_list[i])

    #     #     if not variant_list[i].strip().lower().startswith(BEGINING_VARIANT_PATTERN):
    
    #     #         idx_list.append(i)

    #     # return [variant_list[i] for i in range(len(variant_list)) if i not in idx_list]

    #     return output


    def _get_by_integrity(self, variant_list):

        output = list()

        for variant_text in variant_list:

            if len(variant_text.split('\n')) < THRESHOLD_FOR_VARIANT_INTEGRITY:

                output.append(variant_text)

        return output


    def _sort_variants(self, variant_list):

        output = list()
        variant_numbers = list()

        for variant_text in variant_list:

            text_roi = ' '.join(variant_text.split()[:3])
            variant_numbers.append(int(''.join([x for x in text_roi if x.isdigit()])))
            # variant_numbers.append(int(''.join([x for x in variant_text.split(':')[0] if x.isdigit()])))


        n_max = variant_numbers[-1] if variant_numbers[-1] > len(variant_numbers) else len(variant_numbers)

        missing_numbers = [int(x) for x in range(1, n_max+1) if x not in variant_numbers]

        missing_numbers_df = pd.DataFrame(
            {
                'variant_number': missing_numbers,
                'variant_text': np.nan
            }
        )

        output = pd.DataFrame(
            {
                'variant_number': variant_numbers,
                'variant_text': variant_list
            }
        ).append(missing_numbers_df).sort_values(by='variant_number', ascending=True).variant_text.to_list()

        return output

        # output = list()
        # variant_list.sort()

        # for i in range(len(variant_list)):

        #     if [i + 1] == [int(j.strip()) for j in variant_list[i].split(':')[0].split() if j.isdigit()]:
    
        #         output.append(variant_list[i])
            
        #     else:

        #         output.append(np.nan)

        # return output


    def _get_clean_variant_text(self, variant_list):
        
        output = list()

        for variant_text in variant_list:

            if variant_text is not np.nan:

                output.append(variant_text.replace('\n', ' ').replace('  ', ' ').strip())

            else:

                output.append(variant_text)

        return output


    def df_sieve(self, df_list, method):

        output = list()

        for df_object in df_list:

            if method == 'camelot':
    
                df = df_object.df
                df.columns = df.loc[0]
                df = df.drop(0)

            elif method == 'tabula':
    
                df = df_object
    
            else:
    
                raise ValueError('`method` must be either `camelot` or `tabula`.')


            cols = list(df.columns.str.strip().str.lower().str.replace(' ', '_'))

            if cols == CANONICAL_COLUMN_NAMES_TEMPLATE_1:

                df.columns = CANONICAL_COLUMN_NAMES_TEMPLATE_1

            elif cols == CANONICAL_COLUMN_NAMES_TEMPLATE_2:

                df.columns = CANONICAL_COLUMN_NAMES_TEMPLATE_2
                df = df.drop(['comments'], axis=1)
                
                if df.iloc[-1,:].values[0].lower().strip().replace(' ', '_').startswith('*relative') or \
                   df.iloc[-1,:].values[0].lower().strip().replace(' ', '_').startswith('rating_scale:'):

                    df = df.drop(df.index[-1], axis=0)

                df.columns = CANONICAL_COLUMN_NAMES_TEMPLATE_1

            else:
                continue

            output.append(df)

        return output


    def _build_variants_output(self, idx_list, contents):

        output = list()

        for idx in idx_list:
            output.append(' '.join(contents.replace('\n', '').replace('\\', '').split()[idx['start']:idx['end']]))

        return output


    def get_file_contents(self, file_path):

        contents = str()
        pdfFileObj = open(file_path, 'rb')
        pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

        for i in range(pdfReader.numPages):
        
            page = pdfReader.getPage(i)
            contents += page.extractText()

        return contents


    def _clear_jpg(self):
        
        files_in_directory = os.listdir(TEMPORARY_DIRECTORY)
        filtered_files = [file for file in files_in_directory if file.endswith(".jpg")]

        for file in filtered_files:

            path_to_file = os.path.join(TEMPORARY_DIRECTORY, file)
            os.remove(path_to_file)


    def _gvm1(self, file_path):
        '''
        Variants extraction using pytesseract.
        '''

        output = list()

        if not os.path.exists(TEMPORARY_DIRECTORY):
        
            os.mkdir(TEMPORARY_DIRECTORY)

            print(f'[STATUS] ...temporary directory "{TEMPORARY_DIRECTORY}" created...')
        
        pages = convert_from_path(
            file_path, 
            dpi = self.dpi,
            first_page = self.first_page,
            last_page = self.last_page,
            grayscale = self.grayscale
            )

        for (i, page) in enumerate(pages):
        # for page in enumerate(pages):
            
            filename = os.path.join(TEMPORARY_DIRECTORY, f"page_" + str(i + 1) + ".jpg")
            page.save(filename, 'JPEG')
        
            img = cv2.imread(filename, 0)
            img_data_raw = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            # img_data_raw = pytesseract.image_to_data(
            #     np.array(page), 
            #     output_type=pytesseract.Output.DICT
            #     )
            img_data_df = pd.DataFrame(img_data_raw)
        
            # n_boxes = len(img_data_raw['level'])
            idx = img_data_df.text.str.lower().str.contains(KEYTERM_FOR_VARIANTS)
        
            blocks_num_list = img_data_df[idx].block_num.tolist()
            block_list = list()

            for j in blocks_num_list:
                idx_bock_j = img_data_df.block_num == j
                block_list.append(
                    img_data_df[idx_bock_j].iloc[0][['left', 'top', 'width', 'height']].to_dict()
                    )

            for block_i in block_list:
                (x, y, w, h) = (block_i['left'],
                                block_i['top'], 
                                block_i['width'], 
                                block_i['height'])

                output.append(pytesseract.image_to_string(img[y:y+h, x:x+w]))
        
        self._clear_jpg()

        return output


    def _gvm2(self, file_path):
        '''
        Variants extraction using direct manipulation of the text extracted from PDF file.
        '''
    
        idx_list = list()
        found_variant = False

        contents = self.get_file_contents(file_path)
        contents = contents.lower().split('summary of literature review')[0]

        for i, text in enumerate(contents.replace('\n', '').replace('\\', '').split()):
            if ('variant' in text.lower()) and (found_variant == False):
                idx_list.append({'start': i})
                found_variant = True
            else:
                if (text.lower() in ['procedure', 'appropriateness', 'radiation']) and (found_variant == True):
                    idx_list[-1].update({'end': i})
                    found_variant = False


        return self._build_variants_output(idx_list, contents)


    def _gvm3(self, file_path):
        '''
        Variants extraction using easyocr.
        '''

        output = list()
        reader = easyocr.Reader(['en'])

        pages = convert_from_path(
            file_path, 
            dpi = self.dpi,
            first_page = self.first_page,
            last_page = self.last_page,
            grayscale = self.grayscale
            )

        for (i, page) in enumerate(pages):

            border_eocr = reader.readtext(
                np.array(pages[i]),
                paragraph=True
                )
            
            for j in range(len(border_eocr)):

                if ('variant' in border_eocr[j - 1][1].strip().lower()) and\
                (len(border_eocr[j - 1][1].strip().lower()) <= MAXSIZE_FOR_VARIANTS) and\
                (len(border_eocr[j - 1][1].strip().lower()) > MINSIZE_FOR_VARIANTS):

                    x, y = border_eocr[j][0][0]
                    w = border_eocr[j][0][2][0] - x
                    h  = border_eocr[j][0][2][1] - y

                    output.append(border_eocr[j - 1][1] + ' ' + border_eocr[j][1])

        return output


    def extract_variants(self, file_path, method=1):

        if method == 1:
        
            output = self._gvm1(file_path)
            # print('#####################')
            # print('self._gvm1(file_path)', output)
            # print('#####################')

            output = self._get_unique_variants(output)  # deve vir PRIMEIRO
            # print('#####################')
            # print('self._get_unique_variants(file_path)', output)
            # print('#####################')

            output = self._ensure_begining_pattern(output)  # deve vir ANTES do _sort_variants
            # print('#####################')
            # print('self._ensure_begining_pattern(file_path)', output)
            # print('#####################')

            output = self._get_by_integrity(output)
            # print('#####################')
            # print('self._get_by_integrity(file_path)', output)
            # print('#####################')

            output = self._sort_variants(output)  # deve vir DEPOIS do _ensure_begining_pattern
            # print('#####################')
            # print('self._sort_variants(file_path)', output)
            # print('#####################')

            output = self._get_clean_variant_text(output)
            # print('#####################')
            # print('self._get_clean_variant_text(file_path)', output)
            # print('#####################')

            return output

        elif method == 2:

            output = self._gvm2(file_path)
            output = self._get_unique_variants(output)  # deve vir PRIMEIRO
            output = self._ensure_begining_pattern(output)  # deve vir ANTES do _sort_variants
            output = self._get_by_integrity(output)
            output = self._sort_variants(output)  # deve vir DEPOIS do _ensure_begining_pattern
            output = self._get_clean_variant_text(output)

            return output
        
        elif method == 3:
    
            output = self._gvm3(file_path)
            output = self._get_unique_variants(output)  # deve vir PRIMEIRO
            output = self._ensure_begining_pattern(output)  # deve vir ANTES do _sort_variants
            output = self._get_by_integrity(output)
            output = self._sort_variants(output)  # deve vir DEPOIS do _ensure_begining_pattern
            output = self._get_clean_variant_text(output)

            return output

        else:

            raise ValueError('`method` should be either 1 or 2.')


    @staticmethod
    def _concatenate_df_list(df_list):

        df_output = pd.DataFrame()

        for df in df_list:

            df_output = df_output.append(df).reset_index(drop=True)

        return df_output


    @staticmethod
    def _build_dataset(tables, category, subcategory):

        for i in range(len(tables)):

            na_index = tables[i].loc[tables[i].isna().any(axis=1)].index

            tables[i]['subcategory'] = subcategory[i]
            tables[i]['category'] = category


            if len(na_index) == tables[i].shape[0]:

                continue

            else:

                for j in na_index:

                    try:

                        tables[i].loc[j+1, 'procedure'] = \
                            tables[i].loc[j, 'procedure'] + ' ' + tables[i].loc[j+1, 'procedure']

                    except:

                        pass

                tables[i] = tables[i].dropna().reset_index(drop=True)


        return DatasetBuilder._concatenate_df_list(tables)


    def _listdir_fullpath(self, d):
        
        return [os.path.join(d, f) for f in os.listdir(d)]


    def check_verbose(self, verbore):

        if verbore not in [0, 1]:

            raise ValueError('`verbose` must be an Python int, either 0 or 1.')

        else:

            return True


    def _get_numeric_radiation_value(self, df):
        
        for i in df.index:

            label = df.loc[i, 'relative_radiation_level']

            if label.strip() == 'O':

                df.loc[i, 'relative_radiation_level'] = 0

            elif isinstance(label, str) and \
                ''.join(unidecode.unidecode(str(label)).strip().lower().split()) == 'varies':
                #label.lower().strip() == 'varies':

                df.loc[i, 'relative_radiation_level'] = 'Varies'

            else:

                # df.loc[i, 'relative_radiation_level'] = len(label)
                df.loc[i, 'relative_radiation_level'] = len(''.join(label.split()))
        
        return df

    @staticmethod
    def _get_appropriateness_category(df:pd.DataFrame):

        df = df.copy()

        for i in df.index:

            appropriateness = df.loc[i, 'appropriateness_category']

            try:
                appropriateness = int(appropriateness)
        
                for label, values in APPROPRIATENESS_DICTIONARY.items():

                    if appropriateness in list(values):

                        df.loc[i, 'appropriateness_category'] = label
                        break

                if not isinstance(df.loc[i, 'appropriateness_category'], str):

                    raise ValueError(
            f'No appropriateness category found for the input value `{appropriateness}`.'
                    )

            except:

                # continue
                if '(disagreement)' in appropriateness.replace('/n', '').lower():

                    df.loc[i, 'appropriateness_category'] = \
                        appropriateness.\
                        replace('/n', '').\
                        lower().\
                        replace('(disagreement)', '').\
                        title()

                else:

                    continue


        return df


    def extract_tables(self, file_path:str, method:str, **kwargs):
        
        if method == 'tabula':

            pages = kwargs.get('pages')

            return tabula.read_pdf(file_path, pages='all')
        
        elif method == 'camelot':

            pages = kwargs.get('pages')
            resolution = kwargs.get('resolution')

            return camelot.read_pdf(file_path, pages=pages, resolution=resolution, flavor='lattice')

        else:

            raise ValueError('`method` must be either `tabula` or `camelot`.')


    @staticmethod
    def _clear_line_breaks(df:pd.DataFrame):

        for col in df.columns:

            df[col] = df[col].\
                        apply(
                            lambda x: 
                            x.replace('\n', ' ').replace('  ', ' ') 
                            if isinstance(x, str) else x
                        )

        return df


    def run(self, file_name:list=None, method:str=None, verbose:int=1):
    
        if not isinstance(file_name, list):

            raise TypeError('`file_name` must be a list.')

        dataset = dict()

        if file_name is not None:

            file_name = [os.path.basename(x) for x in file_name]
            file_list = os.listdir(self.folder)
            idx = [x in file_name for x in file_list]
            file_paths = [os.path.join(self.folder, y) for x,y in zip(idx, file_list) if x is True]

        else:

            file_paths = self._listdir_fullpath(self.folder)


        for file_path in file_paths:

            if self.check_verbose(verbose) and verbose == 1:
                print('[STATUS] Extracting data from "', 
                os.path.basename(file_path), '"...', sep='')
                # file_path.split('\\')[-1], '"...', sep='')
            
            variants = self.extract_variants(file_path)

            tables = self.extract_tables(file_path, 
                                         method=method, 
                                         pages='all', 
                                         resolution=self.dpi)
            tables = self.df_sieve(tables, method=method)

            number_of_tables = len(tables)

            if number_of_tables == 0:
                tables = [
                    pd.DataFrame(
                        columns = CANONICAL_COLUMN_NAMES_TEMPLATE_1
                    )
                ]

            key = os.path.basename(file_path) #file_path.split('\\')[-1]
            dataset[key] = self._build_dataset(tables, 
                                               category=key.split('.')[0], 
                                               subcategory=variants)

            dataset[key] = self._get_numeric_radiation_value(dataset[key])
            dataset[key] = self._get_appropriateness_category(dataset[key])
            dataset[key] = self._clear_line_breaks(dataset[key])

            if self.check_verbose(verbose) and verbose == 1:
                # print(' done!')
                print(f'[STATUS] ...found {len(tables)} tables and...')
                print(f'[STATUS] ...{len(variants)} variants in "{key}".')

        return dataset, number_of_tables
