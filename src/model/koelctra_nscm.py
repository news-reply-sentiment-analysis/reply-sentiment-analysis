import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, ElectraForSequenceClassification, AdamW
from tqdm.notebook import tqdm

device = torch.device("cuda") # gpu 사용

# dataset 만들어서 불러오기
class NSMCDataset(Dataset):

    def __init__(self, csv_file):
        # 일부 값중에 NaN이 있음...
        self.dataset = pd.read_csv(csv_file, sep='\t').dropna(axis=0)
        # 중복제거
        self.dataset.drop_duplicates(subset=['document'], inplace=True)
        # 입력 데이터 토큰화하기(의미단어단위로 자르고 정수 인코딩)
        self.tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-small-v2-discriminator")

        print(self.dataset.describe())

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset.iloc[idx, 1:3].values # 한 줄의 필요한 데이터만 가져옴
        text = row[0] # 텍스트 데이터
        y = row[1] # 감정 라벨 데이터

        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=256, # 문장의 최대 길이
            pad_to_max_length=True,
            add_special_tokens=True
        )

        input_ids = inputs['input_ids'][0] # 모델의 입력 문장
        attention_mask = inputs['attention_mask'][0] # 입력문장의 어텐션 마스크 값

        return input_ids, attention_mask, y


# 데이터 셋 만들기

train_dataset = NSMCDataset("ratings_train.txt")
test_dataset = NSMCDataset("ratings_test.txt")

model = ElectraForSequenceClassification.from_pretrained("monologg/koelectra-small-v2-discriminator").to(device)

# 모델이 있을 경우 불러오기
# model.load_state_dict(torch.load("model.pt"))

epochs = 3
batch_size = 128
optimizer = AdamW(model.parameters(), lr=1e-5)

# 배치 크기만큼씩 데이터 만들기
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

losses = []
accuracies = []

for i in range(epochs): # 원하는 숫자만큼 학습진행
    total_loss = 0.0
    correct = 0
    total = 0
    batches = 0

    model.train() # 순방향 학습

    for input_ids_batch, attention_masks_batch, y_batch in tqdm(train_loader): # 역방향 학습(파라미터 수정)
        optimizer.zero_grad()
        y_batch = y_batch.to(device)
        y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
        loss = F.cross_entropy(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        _, predicted = torch.max(y_pred, 1)
        correct += (predicted == y_batch).sum()
        total += len(y_batch)

        batches += 1
        if batches % 100 == 0:
            print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

    losses.append(total_loss)
    accuracies.append(correct.float() / total)
    print("Train Loss:", total_loss, "Accuracy:", correct.float() / total)

# 모델의 정확도 측정( 매 epoch마다)
model.eval()

test_correct = 0
test_total = 0

for input_ids_batch, attention_masks_batch, y_batch in tqdm(test_loader):
  y_batch = y_batch.to(device)
  y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
  _, predicted = torch.max(y_pred, 1)
  test_correct += (predicted == y_batch).sum()
  test_total += len(y_batch)

print("Accuracy:", test_correct.float() / test_total)

#모델 저장하기
torch.save(model.state_dict(), "model.pt")
