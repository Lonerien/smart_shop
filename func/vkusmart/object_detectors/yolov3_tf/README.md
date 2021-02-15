# YOLOv3 (TensorFlow)

## Installation

Чтобы запустить код из данного репозитория, нужно:  

1. Иметь установленную CUDA (тестировалось на 9.0)  
Проверить можно командой `nvcc --version`  

2. Иметь установленный cuDNN  

3. Иметь установленный tensorflow (с GPU, иначе будет слишком медленно)  

## Training pipeline

Подробное описание подготовки данных и настройки config'ов данных и модели:  

https://neuruslab.atlassian.net/wiki/spaces/GRT/pages/45580291/YOLOv3+darknet+custom+dataset+Ubuntu+16.04+LTS+Nvidia+Tesla+P100

Пример запуска (если следовать вышенаписанной инструкции и всё, что связано с darknet, положить в папку `darknet` данного репозитория) обучения на датасете "Игровые карты":  
`
```
python train.py --data_file cfg/cards.data --model_cfg cfg/yolov3-cards.cfg --start_weights darknet53.conv.74  
```


## Test (prediction)  
  
  
Общий синтаксис:  
`
``` 
python predict.py  

	--input_img <путь к папке с картинками/к одной картинке>  

	--output_img <имя папки результата/картинки результата>  

	--weights_file <файл с весами для yolov3, лежит в папке weights>  

	--names_file <файл с именами классов, обычно лежит в папке /names>
```
  
  
Пример запуска на изображении с именем *./cards_test/cam_image2.jpg*:  
`
``` 
python predict.py --input_img cards_test/cam_image2.jpg 
                  --output_img 2.jpg
				  --names_file names/cards.names 
				  --weights_file weights/yolov3-cards.backup
```

Пример запуска на папке с картинками:  
`
```
python predict.py --input_path /mnt/nfs/08_11_21classes/real_test 
				  --output_path /mnt/nfs/08_11_21classes/real_test_preds 
				  --names_file /mnt/nfs/08_11_21classes/goods21.names 
				  --weights_file ./weights/yolov3-goods21_12_11_69200.weights
```