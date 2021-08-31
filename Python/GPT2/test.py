from datasets import load_dataset
from happytransformer.happy_generation import HappyGeneration


def main():
    train_txt_path = "train.txt"
    eval_txt_path = "eval.txt"

    train_dataset = load_dataset('billsum', split='train[0:1999]')
    eval_dataset = load_dataset('billsum', split='test[0:499]')

    generate_txt(train_txt_path, train_dataset)
    generate_txt(eval_txt_path, eval_dataset)

def generate_txt(txt_path, dataset):
    with open(txt_path, 'w', newline='', encoding="utf-8") as text_file:
        for case in dataset:
            text = case["summary"]
            text_file.write(text + "\n")

def produce_examples(starter_texts, happy_gen):
    for start in starter_texts:
        output = happy_gen.generate_text(start)
        print(start, output.text)


if __name__ == "__main__":
    main()