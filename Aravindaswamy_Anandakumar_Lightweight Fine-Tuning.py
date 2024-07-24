import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import evaluate

model_identifier = "gpt2"
classification_model = AutoModelForSequenceClassification.from_pretrained(model_identifier, num_labels=2)
tokenizer_tool = AutoTokenizer.from_pretrained(model_identifier)

data = load_dataset("imdb", split="train[:1000]")

def process_data(examples):
    return tokenizer_tool(examples["text"], padding="max_length", truncation=True)

processed_data = data.map(process_data, batched=True)

training_config = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    num_train_epochs=1,
)

trainer_obj = Trainer(
    model=classification_model,
    args=training_config,
    train_dataset=processed_data,
    eval_dataset=processed_data,
)

initial_evaluation = trainer_obj.evaluate()
print("Initial model evaluation:", initial_evaluation)

peft_configuration = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

peft_enhanced_model = get_peft_model(classification_model, peft_configuration)

fine_tune_config = TrainingArguments(
    output_dir="./peft_results",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    learning_rate=1e-4,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

fine_tune_trainer = Trainer(
    model=peft_enhanced_model,
    args=fine_tune_config,
    train_dataset=processed_data,
    eval_dataset=processed_data,
)

fine_tune_trainer.train()

peft_enhanced_model.save_pretrained("./peft_model")

peft_model_directory = "./peft_model"
model_config = PeftConfig.from_pretrained(peft_model_directory)
reloaded_model = AutoModelForSequenceClassification.from_pretrained(model_config.base_model_name_or_path)
reloaded_peft_model = PeftModel.from_pretrained(reloaded_model, peft_model_directory)

eval_trainer_obj = Trainer(
    model=reloaded_peft_model,
    args=training_config,
    eval_dataset=processed_data,
)

final_evaluation = eval_trainer_obj.evaluate()
print("Fine-tuned model evaluation:", final_evaluation)

print("\nEvaluation Comparison:")
print("Initial model accuracy:", initial_evaluation['eval_accuracy'])
print("Fine-tuned model accuracy:", final_evaluation['eval_accuracy'])
