from clean_up_data import generate_csv
from my_model import train_bert

if __name__ == '__main__':
    generate_csv(should_generate_file_system=False)
    train_bert(epochs=5, init_lr=2e-5, use_lstm=False, load_ckpt=False)
    train_bert(epochs=5, init_lr=5e-5, use_lstm=False, load_ckpt=True)
