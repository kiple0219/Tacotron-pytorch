max_char = 105
sample_rate = 16000
n_fft = 2048
hop_length = 256
win_length = 1024
preemphasis = 0.97
ref_db = 20
max_db = 100
mel_dim = 80
max_length = 780
reduction = 5
embedding_dim = 256
encoder_dim = 128
decoder_dim = 256
symbol_length = 70
batch_size = 32
batch_group = batch_size * batch_size
checkpoint_step = 2000
max_iter = 200
cleaners = ['english_cleaners']