from clean_up_data import generate_csv
from my_model import train_bert

if __name__ == '__main__':
    generate_csv(should_generate_file_system=False)
    use_lstm = False
    train_bert(epochs=5, init_lr=2e-5, use_lstm=use_lstm, load_ckpt=False)
    train_bert(epochs=5, init_lr=5e-5, use_lstm=use_lstm, load_ckpt=True)
