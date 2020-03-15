# DeepACS
Automating Jacob Frye walking around the busy environment of 1868 London in Assassin's Creed Syndicate, using end to end deep learning architecture (AlexNet), implemented with fastai.

### Disclaimer:
Probability of proper working might vary depending upon specifications

Clone the repository, then:

```
python -r requirements.txt
wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
python main.py
```

Do check out v1 for end to end AlexNet model
