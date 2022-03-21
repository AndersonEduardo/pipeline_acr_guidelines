
# CANONICAL_COLUMN_NAMES = ['radiological_procedure', 'procedure', 'appropriateness_category', 'relative_radiation_level']
CANONICAL_COLUMN_NAMES_TEMPLATE_1 = ['procedure', 'appropriateness_category', 'relative_radiation_level']
CANONICAL_COLUMN_NAMES_TEMPLATE_2 = ['radiologic_procedure', 'rating', 'comments', 'rrl*']
COLUMNS_ORDERING = ['procedure', 'appropriateness_category', 'relative_radiation_level', 'subcategory', 'category']
CANONICAL_NUMBER_OF_COLUMNS = 3
COLUMN_NAMES_THRESHOLD = 2
KEYTERM_FOR_VARIANTS = 'variant'
APPROPRIATENESS_DICTIONARY = {
    'Usually Not Appropriate': range(1,3+1), 
    'May Be Appropriate': range(4,6+1), 
    'Usually Appropriate': range(7,9+1)}
TEMPORARY_DIRECTORY = './tmpdir'
ACR_GUIDELINES_URL = 'https://acsearch.acr.org'
MAXSIZE_FOR_VARIANTS = 300
MINSIZE_FOR_VARIANTS = 10
BEGINING_VARIANT_PATTERN = 'variant'
LEVENSHTEIN_THRESHOLD = 3
THRESHOLD_FOR_VARIANT_INTEGRITY = 10
SAMPLE_SIZE_VARIANT_TOLKENS = 5
NAIVE_BAYES_CLASSIFIER_PATH = r'./models/naive_bayes_classifier.pkl'
TFIDF_PATH = r'./models/tfidf.pkl'
EXPERIMENT_METRICS = ['levenshtein', 'number_of_characters', 'percentual_error', 'time']
