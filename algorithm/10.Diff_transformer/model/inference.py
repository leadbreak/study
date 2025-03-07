import torch, json
from transformers import MarianTokenizer
from modern_transformer import ModelArgs, ModernTransformer, LLaMA
from pathlib import Path
import click

@click.command()
@click.option('--checkpoint', required=True, help='모델 체크포인트 경로')
@click.option('--tokenizer_name', default="Helsinki-NLP/opus-mt-ko-en", help='토크나이저 이름 또는 경로')
@click.option('--device', default='cuda:0', help='디바이스')
@click.option('--prompt', prompt="번역할 원문 입력", help="번역할 원문")
def main(checkpoint, tokenizer_name, device, prompt):
    params_path = Path(checkpoint).parent / "params.json"
    with open(params_path, "r") as f:
        params_dict = json.load(f)
    args = ModelArgs(**params_dict)
    args.device = device
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
    args.vocab_size = tokenizer.vocab_size
    model = ModernTransformer(args).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded model from {checkpoint}")
    llama = LLaMA(model, tokenizer, args)
    translation = llama.generate(prompt, max_new_tokens=100, temperature=0.7, top_p=0.9, repetition_penalty=1.1)
    print("Translation:", translation)

if __name__ == "__main__":
    main()
