# translation-task
pip install -r requirements.txt    

В процессе экспериментирования данные было удобнее разбить на train/val в отдельных файлах:     
python split_data.py --path /path/to/train_data.tsv

Запуск тренировочной процедуры:    
python train.py    

Запуск предсказания test-set'а:    
python predict.py --test-path /path/to/test_data.tsv --checkpoint-path /path/to/checkpoint.ckpt