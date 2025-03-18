# Model parameters
DATASET = "../datasets/ctpa_xray/"
ROOT = "../datasets/ctpa_xray/"

OUTPUT_DIR="./results"

DINO_IMAGE_SIZE = [128, 224, 224]

# Hounsfield Units for Air
AIR_HU_VAL = -1000.

# Statistics for Hounsfield Units
CONTRAST_HU_MIN = -100.     # Min value for loading contrast
CONTRAST_HU_MAX = 900.      # Max value for loading contrast
CONTRAST_HU_MEAN = 0.15897  # Mean voxel value after normalization and clipping
CONTRAST_HU_STD = 0.19974   # Standard deviation of voxel values after normalization and clipping

# Dataset parameters
POS_WEIGHT = (843 / 305)
LABEL_COL = 'PE'
CT_ACCESSION_COL = 'CT_Accession_number'
XRAY_ACCESSION_COL = 'cxr_Accession_number'
LIDC_CT_ACCESSION_COL = 'Subject ID'
DEVICE = "cuda"

TRAIN_LABELS = "labels_csv/cross_validation_ct_xray/fold0/y_train.csv"
TEST_LABELS =  "labels_csv/cross_validation_ct_xray/fold0/y_test.csv"
VALID_LABELS =  "labels_csv/cross_validation_ct_xray/fold0/y_valid.csv"

GEN_TRAIN_LABELS = "labels_csv/cross_validation_ct_generation/fold0/y_train.csv"
GEN_TEST_LABELS =  "labels_csv/cross_validation_ct_generation/fold0/y_test.csv"
GEN_VALID_LABELS =  "labels_csv/cross_validation_ct_generation/fold0/y_valid.csv"

LIDC_TRAIN_LABELS = "cross_validation_ct_xray/fold0/train.csv"
LIDC_TEST_LABELS =  "cross_validation_ct_xray/fold0/test.csv"

RSPECT_TRAIN_LABELS = "../datasets/RSPECT/labels_csv_split_64/train.csv"
RSPECT_VALID_LABELS = "../datasets/RSPECT/labels_csv_split_64/test.csv"

RSPECT_CT_STUDY_COL = "StudyInstanceUID"
RSPECT_CT_SERIES_COL = "SeriesInstanceUID"
RSPECT_LABEL_COL = "negative_exam_for_pe"

MEAN = 76.15645329742716
STD = 67.03809638194153

normMu = [MEAN]
normSigma = [STD]
