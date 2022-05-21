n_trials = 1
n_gram = True
val_rate = 0.2
num_words = 25
batch_size = 1024
epochs = 50
val_threshold = 0.5     # 検証用データに対する陽性・陰性の閾値
threshold = 0.99        # モデルの評価を行うときの陽性・陰性の閾値
head_num = 8            # Transformerの並列化に関するパラメータ
beta = 0.5				# Fベータスコアの引数

lengths = $(shell seq 14 2 30)
viruses = $(notdir $(basename $(wildcard data/interim/*.fasta)))

fasta_dir = data/interim
processed_dir = data/processed
tfrecord_dir = data/tfrecord
model_dir = models
result_dir = reports/result

INPUT = $(foreach virus,$(viruses),data/interim/$(virus).fasta)
PROCESSED = $(foreach length,$(lengths),\
		$(foreach virus,$(viruses),data/processed/length$(length)/$(virus).pickle))
VOCAB_FILE = references/vocab.pickle
TRAIN_TFRECORD = $(foreach length,$(lengths),\
		data/tfrecord/length$(length)/train_dataset.tfrecord)
TEST_TFRECORD = $(foreach length,$(lengths),\
		data/tfrecord/length$(length)/test_dataset.tfrecord)
STUDY = reports/result/study.pickle
RESULT = reports/result/tuning_result.txt

all: $(STUDY)

$(PROCESSED): $(INPUT)
	mkdir -p $(@D)
	python3 -m src.make_dataset $(@D) $(fasta_dir) \
		--n_gram $(n_gram)

$(VOCAB_FILE): $(PROCESSED)
	python3 src/fit_vocab.py $(lastword $(PROCESSED)) \
		$(VOCAB_FILE) $(num_words)

$(TRAIN_TFRECORD): $(PROCESSED) $(VOCAB_FILE)
	mkdir -p $(@D)
	python3 src/convert_dataset.py $(processed_dir) $@ \
		$(VOCAB_FILE) $(processed_dir) --val_rate $(val_rate)

$(TEST_TFRECORD): $(PROCESSED) $(VOCAB_FILE)
	mkdir -p $(@D)
	python3 src/convert_dataset.py $(processed_dir) $@ \
		$(VOCAB_FILE) $(processed_dir) --val_rate $(val_rate)

$(STUDY): $(TRAIN_TFRECORD) $(TEST_TFRECORD)
	python3 src/optimize_parameter.py $(n_trials) \
		$(num_words) $(batch_size) $(epochs) $(head_num) \
		$(val_threshold) $(tfrecord_dir) $(processed_dir) \
		$(STUDY) $(RESULT)

$(RESULT): $(TRAIN_TFRECORD) $(TEST_TFRECORD)
	python3 src/optimize_parameter.py $(n_trials) \
		$(num_words) $(batch_size) $(epochs) $(head_num) \
		$(val_threshold) $(tfrecord_dir) $(processed_dir) \
		$(STUDY) $(RESULT)
